#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -W group_list=gq50
#PBS -N opra_merge_only
#PBS -j oe
#PBS -V

########################################
# 0. 初始化 & 日志
########################################

cd "${PBS_O_WORKDIR:-$HOME}"
echo "[INFO] PBS_O_WORKDIR: ${PBS_O_WORKDIR}"

# 设置日志文件
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="./miyabi_logs"
mkdir -p "${LOG_DIR}"
DEST_LOG="${LOG_DIR}/merge_only_${TIMESTAMP}.log"

echo "[INFO] Logging to: ${DEST_LOG}"
exec > >(tee -a "${DEST_LOG}") 2>&1

set -e

########################################
# 1. 路径 & 参数配置 (需与评测脚本保持一致)
########################################

if [ -z "${WORK_HOME}" ]; then
    echo "[WARN] WORK_HOME is not set, fallback to /work/gq50/$USER"
    export WORK_HOME="/work/gq50/$USER"
fi

# === 请根据实际情况修改以下变量 ===
PROJECT_NAME="OPRA"
EXP_NAME="${EXP_NAME:-OPRA-LoRA}"        # 你的实验名称
PROMPT_TYPE="${PROMPT_TYPE:-qwen25-math-cot}" # 你的 Prompt 类型
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-false}"
MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-128}" # 用于 Pass@k 计算
# =================================

ROOT_DIR="${WORK_HOME}/project/${PROJECT_NAME}"
EVAL_DIR="${ROOT_DIR}"
WORK_DIR="${WORK_HOME}/project/LLM_EVAL"
MODEL_ROOT="${MODEL_ROOT:-${ROOT_DIR}/checkpoints/${EXP_NAME}}"
BASE_ROOT="${BASE_ROOT:-${WORK_HOME}/model}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/eval_results/${EXP_NAME}_${PROMPT_TYPE}}"

# 确保 Python 路径包含项目根目录，防止 import 错误
export PYTHONPATH="${ROOT_DIR}:${EVAL_DIR}:${PYTHONPATH}"

echo "[INFO] OUT_ROOT: ${OUT_ROOT}"

########################################
# 2. 环境激活 & 依赖设置
########################################

source "${WORK_HOME}/miniconda3/bin/activate" eval
PYTHON="${WORK_HOME}/miniconda3/envs/eval/bin/python"

# 设置 Pass@k 环境变量 (merge_results.py 中 evaluate 可能会用到)
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
echo "[INFO] PASS_AT_KS: ${PASS_AT_KS}"

########################################
# 3. 执行 Merge 逻辑
########################################

if [ ! -d "${OUT_ROOT}" ]; then
    echo "[ERROR] OUT_ROOT directory does not exist: ${OUT_ROOT}"
    exit 1
fi

echo "[INFO] Starting merge process..."

# ---- 3.1 Base 模型 Merge ----
if [ "${SKIP_BASE_EVAL}" != "true" ]; then
    echo "[INFO] Looking for base__* runs under: ${OUT_ROOT}"

    shopt -s nullglob
    # 优先找跟 BASE_ROOT 名字一致的目录
    BASE_CANDIDATE_DIR="${OUT_ROOT}/base__$(basename "${BASE_ROOT}")"
    BASE_RUN_DIR=""
    
    if [ -d "${BASE_CANDIDATE_DIR}" ]; then
        BASE_RUN_DIR="${BASE_CANDIDATE_DIR}"
    else
        # 找不到特定的，就找任意一个 base__ 开头的
        for d in "${OUT_ROOT}"/base__*; do
            if [ -d "${d}" ]; then
                BASE_RUN_DIR="${d}"
                break
            fi
        done
    fi
    shopt -u nullglob

    if [ -n "${BASE_RUN_DIR}" ]; then
        BASE_RUN_NAME="$(basename "${BASE_RUN_DIR}")"
        echo "[INFO] Merging results for base run: ${BASE_RUN_NAME} ..."
        "${PYTHON}" "${WORK_DIR}/tools/merge_results.py" \
            --out_root "${OUT_ROOT}" \
            --run_name "${BASE_RUN_NAME}" \
            --prompt_type "${PROMPT_TYPE}"
    else
        echo "[WARN] No base__* run directory found under ${OUT_ROOT}, skip base merge."
    fi
else
    echo "[INFO] SKIP_BASE_EVAL=true, skipping base merge."
fi

# ---- 3.2 Checkpoint 模型 Merge (fine-tuned) ----
echo "[INFO] Looking for *__global_step_* runs under: ${OUT_ROOT}"

shopt -s nullglob
RUN_DIRS=( "${OUT_ROOT}"/*__global_step_* )
shopt -u nullglob

if [ "${#RUN_DIRS[@]}" -gt 0 ]; then
    for RUN_DIR in "${RUN_DIRS[@]}"; do
        [ -d "${RUN_DIR}" ] || continue
        RUN_NAME="$(basename "${RUN_DIR}")"
        echo "[INFO] Merging results for ${RUN_NAME} ..."
        "${PYTHON}" "${WORK_DIR}/tools/merge_results.py" \
            --out_root "${OUT_ROOT}" \
            --run_name "${RUN_NAME}" \
            --prompt_type "${PROMPT_TYPE}"
    done
else
    echo "[WARN] No *__global_step_* directories found under ${OUT_ROOT}, skip fine-tuned merges."
fi

echo "[INFO] Merge process finished successfully."