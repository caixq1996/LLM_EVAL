#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

RUN_EVAL_SCRIPT="$PROJECT_ROOT/LLM_EVAL/scripts/run_eval_all.sh"
if [ ! -f "$RUN_EVAL_SCRIPT" ]; then
  echo "[ERROR] run_eval_all.sh not found: $RUN_EVAL_SCRIPT"
  exit 1
fi

PROJECT_NAME="${PROJECT_NAME:-OPRA}"
EXP_NAMES="${EXP_NAMES:-OPRA-K-ABLATION}"
MODEL_PATH="${MODEL_PATH:-checkpoints}"
PROMPT_TYPE="${PROMPT_TYPE:-think-boxed}"
MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-256}"

echo "[INFO] OPRA k-ablation eval"
echo "  PROJECT_NAME   : ${PROJECT_NAME}"
echo "  EXP_NAMES      : ${EXP_NAMES}"
echo "  MODEL_PATH     : ${MODEL_PATH}"
echo "  PROMPT_TYPE    : ${PROMPT_TYPE}"
echo "  MAX_SAMPLE_NUMS: ${MAX_SAMPLE_NUMS}"

env \
  PROJECT_NAME="${PROJECT_NAME}" \
  EXP_NAMES="${EXP_NAMES}" \
  MODEL_PATH="${MODEL_PATH}" \
  PROMPT_TYPE="${PROMPT_TYPE}" \
  MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS}" \
  "$RUN_EVAL_SCRIPT" "$@"
