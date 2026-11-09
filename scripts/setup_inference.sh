#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TAU2_URL="https://github.com/sierra-research/tau2-bench.git"
TAU2_COMMIT="19f13ea3f5fba8b4c4d272d1353590f29dc53062"

mkdir -p .tools third_party

if ! command -v gcc >/dev/null 2>&1 || [[ ! -f /usr/include/python3.12/Python.h ]]; then
  echo "Missing compiler or Python headers."
  echo "Run first: sudo bash scripts/install_system_deps.sh"
  exit 1
fi

if [[ ! -x .tools/bin/uv ]]; then
  curl -LsSf https://astral.sh/uv/install.sh -o .tools/install-uv.sh
  UV_INSTALL_DIR="$ROOT/.tools/bin" UV_NO_MODIFY_PATH=1 sh .tools/install-uv.sh
fi

if [[ ! -d third_party/tau2-bench/.git ]]; then
  git clone --depth 1 --filter=blob:none --no-checkout "$TAU2_URL" \
    third_party/tau2-bench
  git -C third_party/tau2-bench sparse-checkout init --cone
  git -C third_party/tau2-bench sparse-checkout set \
    src data/tau2/domains/telecom data/tau2/multilingual/en \
    data/tau2/task_subsets data/tau2/judge_catalog docs \
    pyproject.toml uv.lock README.md LICENSE
fi

git -C third_party/tau2-bench fetch --depth 1 origin "$TAU2_COMMIT"
git -C third_party/tau2-bench checkout "$TAU2_COMMIT"

UV_LINK_MODE=copy .tools/bin/uv sync --group dev --extra collect

echo "Inference environment ready."
echo "Edit .env with DEEPSEEK_API_KEY and DEEPSEEK_MODEL before collection."
