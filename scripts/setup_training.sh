#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-$ROOT/third_party/verl}"
VERL_COMMIT="${VERL_COMMIT:-184687bb3b7c601627eed3660283ecabd2bb75ed}"
VERL_PATCH="${VERL_PATCH:-$ROOT/patches/verl-qwen35-sft-compat.patch}"
VERL_ROUND_ROBIN_PATCH="${VERL_ROUND_ROBIN_PATCH:-$ROOT/patches/verl-agent-worker-round-robin.patch}"
VERL_EXPORT_PATCH="${VERL_EXPORT_PATCH:-$ROOT/patches/verl-export-checkpoint-at-start.patch}"
TAU2_ROOT="${TAU2_ROOT:-$ROOT/third_party/tau2-bench}"
TAU2_COMMIT="${TAU2_COMMIT:-19f13ea3f5fba8b4c4d272d1353590f29dc53062}"

if ! command -v uv >/dev/null 2>&1; then
  if [[ ! -x "$ROOT/.tools/bin/uv" ]]; then
    mkdir -p "$ROOT/.tools"
    curl -LsSf https://astral.sh/uv/install.sh -o "$ROOT/.tools/install-uv.sh"
    UV_INSTALL_DIR="$ROOT/.tools/bin" UV_NO_MODIFY_PATH=1 \
      sh "$ROOT/.tools/install-uv.sh"
  fi
fi
UV_BIN="$(command -v uv || echo "$ROOT/.tools/bin/uv")"

if [[ ! -d "$VERL_ROOT/.git" ]]; then
  git clone --depth 1 https://github.com/verl-project/verl.git "$VERL_ROOT"
fi
git -C "$VERL_ROOT" sparse-checkout disable >/dev/null 2>&1 || true
git -C "$VERL_ROOT" fetch --depth 1 origin "$VERL_COMMIT"
git -C "$VERL_ROOT" checkout "$VERL_COMMIT"
for patch in "$VERL_PATCH" "$VERL_ROUND_ROBIN_PATCH" "$VERL_EXPORT_PATCH"; do
  if git -C "$VERL_ROOT" apply --reverse --check "$patch" >/dev/null 2>&1; then
    echo "veRL patch already applied: $(basename "$patch")"
  else
    git -C "$VERL_ROOT" apply --check "$patch"
    git -C "$VERL_ROOT" apply "$patch"
  fi
done

if [[ ! -d "$TAU2_ROOT/.git" ]]; then
  git clone --depth 1 \
    https://github.com/sierra-research/tau2-bench.git \
    "$TAU2_ROOT"
fi
git -C "$TAU2_ROOT" sparse-checkout disable >/dev/null 2>&1 || true
git -C "$TAU2_ROOT" fetch --depth 1 origin "$TAU2_COMMIT"
git -C "$TAU2_ROOT" checkout "$TAU2_COMMIT"

cd "$VERL_ROOT"
UV_LINK_MODE=copy "$UV_BIN" sync --extra vllm --extra fsdp

"$UV_BIN" pip install --python "$VERL_ROOT/.venv/bin/python" \
  -e "$TAU2_ROOT[gym]"
"$UV_BIN" pip install --python "$VERL_ROOT/.venv/bin/python" \
  --no-deps -e "$ROOT"

required_paths=(
  "$ROOT/models/Qwen3.5-2B/config.json"
  "$ROOT/models/Qwen3.5-2B/model.safetensors-00001-of-00001.safetensors"
  "$ROOT/results/data/processed/sft/sft_uniform.parquet"
  "$ROOT/results/data/processed/sft/sft_failure_aware.parquet"
  "$ROOT/results/data/processed/rl/rl_uniform.parquet"
  "$ROOT/results/data/processed/rl/rl_failure_aware.parquet"
)
for required_path in "${required_paths[@]}"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Training bundle is incomplete: missing $required_path" >&2
    exit 1
  fi
done

echo "Training environment ready."
echo "Model directory: $ROOT/models/Qwen3.5-2B."
echo "Configure .env securely, then run scripts/preflight.sh."
