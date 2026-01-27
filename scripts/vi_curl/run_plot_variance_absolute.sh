#!/bin/bash
# Script to generate absolute variance comparison plots from existing grad_variance data.
#
# Usage:
#   bash scripts/vi_curl/run_plot_variance_absolute.sh
#
# This script reads the existing JSON files in eval_log/vi_curl/grad_variance
# and generates comparison plots showing absolute variance values for curl vs nocurl.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${EVAL_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/eval/bin/python3}"

# Input/output directories
GRAD_VARIANCE_DIR="${GRAD_VARIANCE_DIR:-${EVAL_ROOT}/eval_log/vi_curl/grad_variance}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/eval_log/vi_curl/variance_absolute_plots}"

# Options
USE_FULL="${USE_FULL:-false}"
NO_GRID="${NO_GRID:-false}"

echo "[INFO] Grad variance dir: ${GRAD_VARIANCE_DIR}"
echo "[INFO] Output dir: ${OUT_DIR}"
echo "[INFO] Use full data: ${USE_FULL}"

mkdir -p "${OUT_DIR}"

args=(
    --grad_variance_dir "${GRAD_VARIANCE_DIR}"
    --out_dir "${OUT_DIR}"
)

if [[ "${USE_FULL}" == "true" ]]; then
    args+=( --use_full )
fi

if [[ "${NO_GRID}" == "true" ]]; then
    args+=( --no_grid )
fi

"$PYTHON_BIN" -u "${EVAL_ROOT}/tools/vi_curl_plot/plot_variance_absolute.py" "${args[@]}"

echo "[DONE] Plots saved to ${OUT_DIR}"
