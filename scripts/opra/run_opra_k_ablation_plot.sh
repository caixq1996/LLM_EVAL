#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-python3}"

EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/OPRA/eval_results/OPRA-K-ABLATION_think-boxed}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/LLM_EVAL/eval_log/opra/k_ablation}"
PROMPT_TYPE="${PROMPT_TYPE:-think-boxed}"

PERF_DATASETS="${PERF_DATASETS:-aime24x8,aime25x8,amc23x8}"
KNOW_DATASETS="${KNOW_DATASETS:-minerva_math,olympiadbench,math500}"
PERF_METRIC="${PERF_METRIC:-pass@1}"
KNOW_METRIC="${KNOW_METRIC:-acc}"
RUN_FILTER="${RUN_FILTER:-}"
TITLE="${TITLE:-OPRA k Ablation}"

COLLECT_SCRIPT="$PROJECT_ROOT/LLM_EVAL/tools/opra/collect_opra_k_ablation_metrics.py"
PLOT_SCRIPT="$PROJECT_ROOT/LLM_EVAL/tools/opra/plot_opra_k_ablation.py"

JSON_OUT="$OUT_DIR/opra_k_ablation_metrics.json"
CSV_OUT="$OUT_DIR/opra_k_ablation_metrics.csv"

echo "[INFO] Collecting metrics from ${EVAL_ROOT}"

"$PYTHON_BIN" "$COLLECT_SCRIPT" \
  --eval-root "$EVAL_ROOT" \
  --prompt-type "$PROMPT_TYPE" \
  --performance-datasets "$PERF_DATASETS" \
  --knowledge-datasets "$KNOW_DATASETS" \
  --performance-metric "$PERF_METRIC" \
  --knowledge-metric "$KNOW_METRIC" \
  --run-filter "$RUN_FILTER" \
  --out-json "$JSON_OUT" \
  --out-csv "$CSV_OUT"

echo "[INFO] Plotting to ${OUT_DIR}"

"$PYTHON_BIN" "$PLOT_SCRIPT" \
  --input "$JSON_OUT" \
  --out-dir "$OUT_DIR" \
  --title "$TITLE"
