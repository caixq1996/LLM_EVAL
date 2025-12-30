#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"$PROJECT_ROOT/OPRA/checkpoints/OPRA-LoRA"}
BASE_MODEL=${BASE_MODEL:-}
RUNS=${RUNS:-}
RUN_FILTER=${RUN_FILTER:-}
PRINCIPAL_RANK=${PRINCIPAL_RANK:-16}
OUT_DIR=${OUT_DIR:-"$PROJECT_ROOT/LLM_EVAL/eval_log/opra_layerwise"}

PYTHON=${PYTHON:-$HOME/miniconda3/envs/eval/bin/python}
export CHECKPOINT_ROOT RUN_FILTER

if [[ -z "${RUNS:-}" ]]; then
  RUNS=$($PYTHON - <<'PY'
import os
import re
from pathlib import Path
root = Path(os.environ["CHECKPOINT_ROOT"])
pattern = os.environ.get("RUN_FILTER", "").strip()
regex = re.compile(pattern) if pattern else None
paths = root.rglob("adapter_config.json")
runs = sorted({p.parents[3].name for p in paths})
if regex:
    runs = [r for r in runs if regex.search(r)]
print("\n".join(runs))
PY
)
else
  RUNS=$(printf '%s\n' "$RUNS" | tr ',' '\n')
fi

mapfile -t RUN_LIST < <(printf '%s\n' "$RUNS" | sed '/^$/d')
if [[ ${#RUN_LIST[@]} -eq 0 ]]; then
  echo "[ERROR] No LoRA adapter checkpoints found under $CHECKPOINT_ROOT"
  exit 1
fi

CMD=($PYTHON "$PROJECT_ROOT/LLM_EVAL/tools/plot_opra_layerwise_leakage.py" \
  --checkpoint_root "$CHECKPOINT_ROOT" \
  --principal_rank "$PRINCIPAL_RANK" \
  --out_dir "$OUT_DIR")

for RUN_NAME in "${RUN_LIST[@]}"; do
  CMD+=(--run "${RUN_NAME}=${RUN_NAME}")
done

if [[ -n "$BASE_MODEL" ]]; then
  CMD+=(--base_model "$BASE_MODEL")
fi

"${CMD[@]}"
