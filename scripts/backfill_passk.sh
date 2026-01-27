#!/bin/bash
set -euo pipefail

ROOT_DIR="${1:-/home/caixq/project/LLM_EVAL/eval_results}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/eval/bin/python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "[INFO] Backfilling pass@k under: ${ROOT_DIR}"
"${PYTHON_BIN}" tools/backfill_passk.py --root "${ROOT_DIR}"
