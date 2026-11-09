#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$ROOT/.env" ]]; then
  echo "Missing $ROOT/.env" >&2
  exit 1
fi
if [[ ! -x "$ROOT/third_party/verl/.venv/bin/python" ]]; then
  echo "Run scripts/setup_training.sh first" >&2
  exit 1
fi

set -a
source "$ROOT/.env"
source "$ROOT/scripts/lib_profile.sh"
load_training_profile "$ROOT"
set +a

exec "$ROOT/third_party/verl/.venv/bin/python" \
  "$ROOT/scripts/preflight.py" \
  --root "$ROOT" \
  "$@"
