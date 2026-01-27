#!/bin/bash
# Generate variance decomposition plots and copy to VI-CuRL paper figures directory.
#
# Usage:
#   bash scripts/vi_curl/gen_variance_figures_for_paper.sh
#
# This script:
# 1. Generates variance comparison plots from existing grad_variance data
# 2. Copies the figures to the VI-CURL paper figures directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${EVAL_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/eval/bin/python3}"

# Input/output directories
GRAD_VARIANCE_DIR="${GRAD_VARIANCE_DIR:-${EVAL_ROOT}/eval_log/vi_curl/grad_variance}"
PAPER_FIGURES_DIR="${PAPER_FIGURES_DIR:-${HOME}/works/RLVR/VI-CURL/figures}"

# Temporary output directory
TEMP_OUT_DIR="${EVAL_ROOT}/eval_log/vi_curl/variance_paper"

# Filter (e.g., "ver_rule" for Oracle only, "" for all)
FILTER="${FILTER:-}"

echo "[INFO] Grad variance dir: ${GRAD_VARIANCE_DIR}"
echo "[INFO] Paper figures dir: ${PAPER_FIGURES_DIR}"
echo "[INFO] Filter: ${FILTER:-<all>}"

mkdir -p "${TEMP_OUT_DIR}"
mkdir -p "${PAPER_FIGURES_DIR}"

# Generate plots
args=(
    --grad_variance_dir "${GRAD_VARIANCE_DIR}"
    --out_dir "${TEMP_OUT_DIR}"
)

if [[ -n "${FILTER}" ]]; then
    args+=( --filter "${FILTER}" )
fi

"$PYTHON_BIN" -u "${EVAL_ROOT}/tools/vi_curl_plot/plot_variance_for_paper.py" "${args[@]}"

# Copy to paper figures directory
echo ""
echo "[INFO] Copying figures to paper directory..."

for pdf_file in "${TEMP_OUT_DIR}"/*.pdf; do
    if [[ -f "${pdf_file}" ]]; then
        filename=$(basename "${pdf_file}")
        cp "${pdf_file}" "${PAPER_FIGURES_DIR}/${filename}"
        echo "       Copied: ${filename}"
    fi
done

# Also generate the absolute variance comparison plots
echo ""
echo "[INFO] Generating additional absolute variance plots..."
"$PYTHON_BIN" -u "${EVAL_ROOT}/tools/vi_curl_plot/plot_variance_absolute.py" \
    --grad_variance_dir "${GRAD_VARIANCE_DIR}" \
    --out_dir "${TEMP_OUT_DIR}"

echo ""
echo "[DONE] Figures available in:"
echo "       ${TEMP_OUT_DIR}"
echo "       ${PAPER_FIGURES_DIR}"
