#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"$PROJECT_ROOT/OPRA/checkpoints/OPRA-LoRA"}
BASE_MODEL=${BASE_MODEL:-}
RUNS=${RUNS:-}
RUN_FILTER=${RUN_FILTER:-}
PROMPT_FILE=${PROMPT_FILE:-"$PROJECT_ROOT/LLM_EVAL/data/gsm8k/test.jsonl"}
PROMPT_FIELD=${PROMPT_FIELD:-question}
ANSWER_FIELD=${ANSWER_FIELD:-answer}
MASK_PROMPT=${MASK_PROMPT:-0}
NUM_SAMPLES=${NUM_SAMPLES:-64}
GRID_SIZE=${GRID_SIZE:-9}
RADIUS=${RADIUS:-0.5}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_LENGTH=${MAX_LENGTH:-512}
STEPS=${STEPS:-}
OUT_DIR=${OUT_DIR:-"$PROJECT_ROOT/LLM_EVAL/eval_log/opra_loss_landscape"}

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
runs = set()
for p in paths:
    try:
        rel = p.relative_to(root)
    except ValueError:
        continue
    if rel.parts:
        runs.add(rel.parts[0])
runs = sorted(runs)
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

CMD=($PYTHON "$PROJECT_ROOT/LLM_EVAL/tools/opra_plot/plot_opra_loss_landscape.py" \
  --checkpoint_root "$CHECKPOINT_ROOT" \
  --prompt_file "$PROMPT_FILE" \
  --prompt_field "$PROMPT_FIELD" \
  --answer_field "$ANSWER_FIELD" \
  --num_samples "$NUM_SAMPLES" \
  --grid_size "$GRID_SIZE" \
  --radius "$RADIUS" \
  --batch_size "$BATCH_SIZE" \
  --max_length "$MAX_LENGTH" \
  --out_dir "$OUT_DIR")

if [[ "$MASK_PROMPT" == "1" ]]; then
  CMD+=(--mask_prompt)
fi

for RUN_NAME in "${RUN_LIST[@]}"; do
  CMD+=(--run "${RUN_NAME}=${RUN_NAME}")
done

if [[ -n "$BASE_MODEL" ]]; then
  CMD+=(--base_model "$BASE_MODEL")
fi

if [[ -n "$STEPS" ]]; then
  CMD+=(--steps "$STEPS")
fi

if [[ -n "${REPLOT:-}" && "$REPLOT" == "1" ]]; then
  CMD+=(--replot)
fi

"${CMD[@]}"
