#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=00:20:00
#PBS -W group_list=gq50
#PBS -N opra_plot_v2
#PBS -j oe
#PBS -V

cd "${PBS_O_WORKDIR:-$HOME}"

# 环境激活
if [ -z "${WORK_HOME}" ]; then
    export WORK_HOME="/work/gq50/$USER"
fi
source "${WORK_HOME}/miniconda3/bin/activate" eval

# === 配置区域 ===
PROJECT_NAME="OPRA"
EXP_NAME="${EXP_NAME:-OPRA-LoRA}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen25-math-cot}"
EVAL_HOME="${WORK_HOME}/project/LLM_EVAL"

# 数据源路径
OUT_ROOT="${WORK_HOME}/project/${PROJECT_NAME}/eval_results/${EXP_NAME}_${PROMPT_TYPE}"
# 输出路径
PLOT_DIR="${OUT_ROOT}/plots_visualization"

# --- 绘图选项 ---
# "step" : 画 Pass@k 随 Step 变化的图 (横轴 Step)
# "k"    : 画 最佳 Score 随 Pass@k 变化的图 (横轴 k)
# "all"  : 两种都画
PLOT_MODE="${PLOT_MODE:-k}"
MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-128}"
if [[ -z "${PASS_AT_KS:-}" ]]; then
  default_pass_ks=(1 8 16 32 64 128 256)
  pass_ks=()
  for k in "${default_pass_ks[@]}"; do
    if (( k > 0 && k <= MAX_SAMPLE_NUMS )) && [[ " ${pass_ks[*]} " != *" $k "* ]]; then
      pass_ks+=("$k")
    fi
  done
  PASS_AT_KS=$(IFS=,; echo "${pass_ks[*]}")
fi
export PASS_AT_KS

echo "[INFO] Plotting Mode: ${PLOT_MODE}"
echo "[INFO] Source: ${OUT_ROOT}"
echo "[INFO] Output: ${PLOT_DIR}"

python ${EVAL_HOME}/tools/plot_pass_k.py \
    --eval_root "${OUT_ROOT}" \
    --output_dir "${PLOT_DIR}" \
    --target_ks "${PASS_AT_KS}" \
    --plot_mode "${PLOT_MODE}"

echo "[INFO] Done."