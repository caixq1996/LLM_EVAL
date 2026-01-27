#!/bin/bash
# Script to generate variance decomposition plots: kept vs full within same model.
#
# This script validates Theorem 3.1 by showing that curriculum selection
# reduces variance compared to using the full dataset.
#
# Usage:
#   bash scripts/vi_curl/run_plot_variance_kept_vs_full.sh
#   FILTER=ver_rule bash scripts/vi_curl/run_plot_variance_kept_vs_full.sh  # Oracle only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${EVAL_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/eval/bin/python3}"

# Input/output directories
GRAD_VARIANCE_DIR="${GRAD_VARIANCE_DIR:-${EVAL_ROOT}/eval_log/vi_curl/grad_variance}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/eval_log/vi_curl/variance_kept_vs_full}"

# Options
FILTER="${FILTER:-}"
NO_GRID="${NO_GRID:-false}"
RATIO_ONLY="${RATIO_ONLY:-false}"

echo "[INFO] Grad variance dir: ${GRAD_VARIANCE_DIR}"
echo "[INFO] Output dir: ${OUT_DIR}"
echo "[INFO] Filter: ${FILTER:-<all>}"

mkdir -p "${OUT_DIR}"

args=(
    --grad_variance_dir "${GRAD_VARIANCE_DIR}"
    --out_dir "${OUT_DIR}"
)

if [[ -n "${FILTER}" ]]; then
    args+=( --filter "${FILTER}" )
fi

if [[ "${NO_GRID}" == "true" ]]; then
    args+=( --no_grid )
fi

if [[ "${RATIO_ONLY}" == "true" ]]; then
    args+=( --ratio_only )
fi

"$PYTHON_BIN" -u "${EVAL_ROOT}/tools/vi_curl_plot/plot_variance_kept_vs_full.py" "${args[@]}"

echo "[DONE] Plots saved to ${OUT_DIR}"
