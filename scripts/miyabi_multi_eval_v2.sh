#!/bin/bash
#PBS -q regular-g
#PBS -l select=64
#PBS -l walltime=08:00:00
#PBS -W group_list=gq50
#PBS -N opra_multi_eval
#PBS -j oe
#PBS -V

########################################
# 0. 回到提交目录 + 日志初始化
########################################

# 推荐：作业一开始就切回提交目录（用户指南 5.4.2）
cd "${PBS_O_WORKDIR:-$HOME}"
echo "[INFO] PBS_O_WORKDIR: ${PBS_O_WORKDIR}"
echo "[INFO] CWD: $(pwd)"

# Miyabi 的 stdout/stderr 默认会写到提交目录下 jobname.o<jobid>:contentReference[oaicite:1]{index=1}
JOBID_SHORT="${PBS_JOBID%%.*}"
PBS_STDOUT_FILE="${PBS_JOBNAME}.o${JOBID_SHORT}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="./miyabi_logs"
mkdir -p "${LOG_DIR}"
DEST_LOG="${LOG_DIR}/opra_eval_${TIMESTAMP}.log"

echo "[INFO] Logging to: ${DEST_LOG}"
# 所有 stdout/stderr 同时写到 PBS 默认输出和 DEST_LOG
exec > >(tee -a "${DEST_LOG}") 2>&1

set -e

########################################
# 1. 基础路径 & 参数
########################################

if [ -z "${WORK_HOME}" ]; then
    echo "[WARN] WORK_HOME is not set, fallback to /work/gq50/$USER"
    export WORK_HOME="/work/gq50/$USER"
fi

PROJECT_NAME="OPRA"
ROOT_DIR="${WORK_HOME}/project/${PROJECT_NAME}"
EVAL_DIR="${ROOT_DIR}"
WORK_DIR="${WORK_HOME}/project/LLM_EVAL"

EXP_NAME="${EXP_NAME:-OPRA-LoRA}"
MODEL_ROOT="${MODEL_ROOT:-${ROOT_DIR}/checkpoints/${EXP_NAME}}"
BASE_ROOT="${BASE_ROOT:-${WORK_HOME}/model}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen25-math-cot}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/eval_results/${EXP_NAME}_${PROMPT_TYPE}_v2}"
MAX_TOKENS="${MAX_TOKENS:-4096}"

NUM_GPUS_PER_NODE=1
MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-1024}"
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-false}"
TEMP_G1="${TEMP_G1:-0.6}"
TEMP_G2="${TEMP_G2:-0.0}"
NSAMP_G1="${NSAMP_G1:-${MAX_SAMPLE_NUMS}}"
NSAMP_G2="${NSAMP_G2:-${MAX_SAMPLE_NUMS}}"

mkdir -p "${OUT_ROOT}"

########################################
# 2. 节点探测
########################################

NUM_NODES=1
NODE_LIST_FILE="${OUT_ROOT}/node_list.txt"

if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
    sort -u "${PBS_NODEFILE}" > "${NODE_LIST_FILE}"
    NUM_NODES="$(wc -l < "${NODE_LIST_FILE}")"
    echo "[INFO] Detected PBS Job. Allocated Node Count: ${NUM_NODES}"
else
    echo "[WARN] PBS_NODEFILE not found. Assuming local single node execution."
    hostname > "${NODE_LIST_FILE}"
fi

########################################
# 3. 生成每节点 worker wrapper
########################################

WRAPPER_SCRIPT="${OUT_ROOT}/pbs_worker_wrapper.sh"
echo "[INFO] Generating wrapper script at: ${WRAPPER_SCRIPT}"

cat << 'EOF_WRAPPER' > "${WRAPPER_SCRIPT}"
#!/bin/bash

# --------- 固定环境 ----------
export TZ="Asia/Tokyo"
export CUDA_HOME="/work/opt/local/aarch64/cores/cuda/12.8.1"
export CUDA_ROOT="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDA_LIB_PATH="${CUDA_HOME}/targets/sbsa-linux/lib"

# 注意：这里的 WORK_HOME / ROOT_DIR / EVAL_DIR 等都在生成 wrapper 时已嵌入
# 在外层脚本会用 envsubst 的方式注入（见下方实际替换）
EOF_WRAPPER

# 用 cat 再次追加需要变量展开的部分（用双引号，让外层变量展开一次）
cat << EOF_WRAPPER >> "${WRAPPER_SCRIPT}"
export LD_LIBRARY_PATH="\${CUDA_LIB_PATH}:${WORK_HOME}/miniconda3/envs/eval/lib:\${LD_LIBRARY_PATH}"

export WORK_HOME="${WORK_HOME}"
export PROJECT_NAME="${PROJECT_NAME}"
export HF_HOME="${WORK_HOME}/cache/hf"
export CUDA_CACHE_PATH="${WORK_HOME}/cache/cuda"
export HF_HUB_OFFLINE=0
export RAY_TMPDIR=/tmp/ray
mkdir -p "\${RAY_TMPDIR}"

export PYTHONPATH="${ROOT_DIR}:${EVAL_DIR}:\${PYTHONPATH}"

# ---------- Pass@k ----------
if [[ -z "\${PASS_AT_KS:-}" ]]; then
  default_pass_ks=(1 8 16 32 64 128 256 512 1024 2048 4096)
  pass_ks=()
  for k in "\${default_pass_ks[@]}"; do
    if (( k > 0 && k <= ${MAX_SAMPLE_NUMS} )) && [[ " \${pass_ks[*]} " != *" \$k "* ]]; then
      pass_ks+=("\$k")
    fi
  done
  PASS_AT_KS=\$(IFS=,; echo "\${pass_ks[*]}")
fi
export PASS_AT_KS

export TORCH_CPP_LOG_LEVEL=ERROR
export CUDA_VISIBLE_DEVICES="0"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export EVAL_ONE_MODEL_TIMEOUT="\${EVAL_ONE_MODEL_TIMEOUT:-21600}"

# 激活环境
source "${WORK_HOME}/miniconda3/bin/activate" eval
export PATH="${WORK_HOME}/miniconda3/envs/eval/bin:\${PATH}"
PYTHON="${WORK_HOME}/miniconda3/envs/eval/bin/python"

NODE_LIST_FILE="${NODE_LIST_FILE}"
MY_HOSTNAME=\$(hostname)

if [ -f "\${NODE_LIST_FILE}" ]; then
    LINE_NUM=\$(grep -n "^\${MY_HOSTNAME}\$" "\${NODE_LIST_FILE}" | cut -d: -f1)
    if [ -z "\${LINE_NUM}" ]; then
        MY_RANK=0
        TOTAL_NODES=1
    else
        MY_RANK=\$((LINE_NUM - 1))
        TOTAL_NODES=\$(wc -l < "\${NODE_LIST_FILE}")
    fi
else
    MY_RANK=0
    TOTAL_NODES=1
fi

echo "[Wrapper] Host: \${MY_HOSTNAME}, Rank: \${MY_RANK}, Total: \${TOTAL_NODES}"

ARGS=(
  --model_root "${MODEL_ROOT}"
  --out_root "${OUT_ROOT}"
  --base_root "${BASE_ROOT}"
  --prompt_type "${PROMPT_TYPE}"
  --max_tokens_per_call "${MAX_TOKENS}"
  --nproc ${NUM_GPUS_PER_NODE}
  --use_vllm
  --pipeline_parallel_size 1
  --vllm_batch_size 0
  --temperature_g1 "${TEMP_G1}"
  --temperature_g2 "${TEMP_G2}"
  --n_sampling_g1 "${NSAMP_G1}"
  --n_sampling_g2 "${NSAMP_G2}"
  --shard_id "\${MY_RANK}"
  --num_shards "\${TOTAL_NODES}"
)

if [ "${SKIP_BASE_EVAL}" = "true" ]; then
  ARGS+=( --skip_base_eval )
fi

cd "${WORK_DIR}"
echo "[Wrapper] Working directory: ${WORK_DIR}"
echo "[Wrapper] Executing Python script..."

exec "\$PYTHON" -u "${WORK_DIR}/tools/run_qwen_eval_all_shared.py" "\${ARGS[@]}"
EOF_WRAPPER

chmod +x "${WRAPPER_SCRIPT}"

########################################
# 4. 调度执行（单节点 / 多节点）
########################################

if [ "${NUM_NODES}" -gt 1 ]; then
    echo "[INFO] >>> MULTI-NODE MODE (${NUM_NODES} nodes) <<<"
    echo "[INFO] Using pbsdsh to distribute tasks."
    pbsdsh "${WRAPPER_SCRIPT}"
else
    echo "[INFO] >>> SINGLE-NODE MODE <<<"
    echo "[INFO] Running wrapper locally."
    "${WRAPPER_SCRIPT}"
fi

EXIT_CODE=$?

########################################
# 5. merge 结果
########################################

if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "[INFO] Tasks completed. Starting merge process..."
    # 重新激活环境（确保有 python）
    source "${WORK_HOME}/miniconda3/bin/activate" eval
    PYTHON="${WORK_HOME}/miniconda3/envs/eval/bin/python"

    # ---- 5.1 base__* 目录 ----
    if [ "${SKIP_BASE_EVAL}" != "true" ]; then
        echo "[INFO] Looking for base__* runs under: ${OUT_ROOT}"

        shopt -s nullglob
        # 优先找跟 BASE_ROOT 名字一致的 base__<basename>
        BASE_CANDIDATE_DIR="${OUT_ROOT}/base__$(basename "${BASE_ROOT}")"
        BASE_RUN_DIR=""
        if [ -d "${BASE_CANDIDATE_DIR}" ]; then
            BASE_RUN_DIR="${BASE_CANDIDATE_DIR}"
        else
            # 退而求其次：找第一个 base__*
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
        echo "[INFO] SKIP_BASE_EVAL=true, skip base merge."
    fi

    # ---- 5.2 所有 *__global_step_* 目录 ----
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

    echo "[INFO] Evaluation + merge finished successfully."
else
    echo "[ERROR] Evaluation failed with code ${EXIT_CODE}. Skip merge. Check log above."
fi

exit "${EXIT_CODE}"
