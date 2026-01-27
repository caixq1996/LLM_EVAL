#!/bin/bash
#PBS -q regular-g
#PBS -l select=16
#PBS -l walltime=16:00:00
#PBS -W group_list=gq50
#PBS -N opra_multi_eval
#PBS -j oe
#PBS -V

set -e

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SUBMIT_DIR="${PBS_O_WORKDIR:-$PWD}"

########################################
# 0. CLI 参数（支持直接运行/自提交）
########################################

RUN_EVAL_SUBMIT="${RUN_EVAL_SUBMIT:-0}"
RUN_EVAL_MULTI_SUBMIT="${RUN_EVAL_MULTI_SUBMIT:-0}"
EVAL_STEP_FILTER="${EVAL_STEP_FILTER:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)
      RUN_EVAL_SUBMIT=1
      shift
      ;;
    --multi-submit)
      RUN_EVAL_MULTI_SUBMIT=1
      RUN_EVAL_SUBMIT=1
      shift
      ;;
    --steps|--eval-steps)
      if [[ -n "${2:-}" ]]; then
        EVAL_STEP_FILTER="$2"
        shift 2
      else
        echo "[ERROR] --steps requires a value (e.g. 20,40,100 or 100-300 or 100-300:50)."
        exit 2
      fi
      ;;
    *)
      shift
      ;;
  esac
done

########################################
# 0. 回到提交目录 + 日志初始化
########################################

# 推荐：作业一开始就切回提交目录（用户指南 5.4.2）
cd "${SUBMIT_DIR}"
echo "[INFO] PBS_O_WORKDIR: ${PBS_O_WORKDIR:-}"
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
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/eval_results/${EXP_NAME}_${PROMPT_TYPE}_v4}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
EXPORT_ROOT="${EXPORT_ROOT:-${WORK_HOME}/export}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${WORK_DIR}/data}"
EVAL_NUM_TEST_SAMPLE="${EVAL_NUM_TEST_SAMPLE:-}"
EVAL_GROUP1_DATASETS="${EVAL_GROUP1_DATASETS:-}"
EVAL_GROUP2_DATASETS="${EVAL_GROUP2_DATASETS:-}"
EVAL_STEP_FILTER="${EVAL_STEP_FILTER:-}"

NUM_GPUS_PER_NODE=1
MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-1024}"
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-true}"
TEMP_G1="${TEMP_G1:-0.6}"
TEMP_G2="${TEMP_G2:-0.0}"
NSAMP_G1="${NSAMP_G1:-${MAX_SAMPLE_NUMS}}"
NSAMP_G2="${NSAMP_G2:-${MAX_SAMPLE_NUMS}}"

mkdir -p "${OUT_ROOT}"

########################################
# 1.5 qsub 提交（登录节点触发）
########################################

if [[ -z "${PBS_JOBID:-}" && -z "${RUN_EVAL_SUBMITTED:-}" && "${RUN_EVAL_SUBMIT}" == "1" ]]; then
    MIYABI_SELECT_NODES="${MIYABI_SELECT_NODES:-16}"
    MIYABI_QUEUE="${MIYABI_QUEUE:-regular-g}"
    MIYABI_WALLTIME="${MIYABI_WALLTIME:-16:00:00}"
    MIYABI_GROUP="${MIYABI_GROUP:-gq50}"

    if [[ "${RUN_EVAL_MULTI_SUBMIT}" == "1" || "${RUN_EVAL_MULTI_SUBMIT}" == "true" ]]; then
        if [[ ! -d "${MODEL_ROOT}" ]]; then
            echo "[ERROR] MODEL_ROOT not found: ${MODEL_ROOT}"
            exit 1
        fi

        SUBDIRS=()
        while IFS= read -r -d '' dir; do
            SUBDIRS+=("$dir")
        done < <(find "${MODEL_ROOT}" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)

        if [[ ${#SUBDIRS[@]} -eq 0 ]]; then
            echo "[WARN] No subdirectories found under ${MODEL_ROOT}, falling back to single submit."
        else
            echo "[INFO] Multi-submit mode: ${#SUBDIRS[@]} subdirectories under ${MODEL_ROOT}"
            for subdir in "${SUBDIRS[@]}"; do
                sub_name="$(basename "${subdir}")"
                job_tag="${sub_name//[^A-Za-z0-9_]/_}"
                job_tag="${job_tag:0:60}"
                job_name="EVAL_${PROJECT_NAME}_${job_tag}"
                job_name="${job_name//[^A-Za-z0-9_]/_}"
                job_name="${job_name:0:120}"
                out_root_sub="${OUT_ROOT}/${sub_name}"

                echo "[INFO] Submitting: ${job_name} (MODEL_ROOT=${subdir}) nodes=${MIYABI_SELECT_NODES}"
                qsub -N "${job_name}" \
                     -q "${MIYABI_QUEUE}" \
                     -l "select=${MIYABI_SELECT_NODES}" \
                     -l "walltime=${MIYABI_WALLTIME}" \
                     -W "group_list=${MIYABI_GROUP}" \
                     -v RUN_EVAL_SUBMITTED=1 \
                     -v MODEL_ROOT="${subdir}" \
                     -v OUT_ROOT="${out_root_sub}" \
                     -v EXP_NAME="${EXP_NAME}" \
                     -V \
                     "${SCRIPT_PATH}"
            done
            echo "[INFO] Submitted ${#SUBDIRS[@]} jobs."
            exit 0
        fi
    fi

    job_tag="${EXP_NAME//[^A-Za-z0-9_]/_}"
    job_tag="${job_tag:0:60}"
    job_name="${MIYABI_JOB_NAME:-EVAL_${PROJECT_NAME}_${job_tag}}"
    job_name="${job_name//[^A-Za-z0-9_]/_}"
    job_name="${job_name:0:120}"

    echo "[INFO] Submitting eval job: ${job_name} nodes=${MIYABI_SELECT_NODES}"
    qsub -N "${job_name}" \
         -q "${MIYABI_QUEUE}" \
         -l "select=${MIYABI_SELECT_NODES}" \
         -l "walltime=${MIYABI_WALLTIME}" \
         -W "group_list=${MIYABI_GROUP}" \
         -v RUN_EVAL_SUBMITTED=1 \
         -V \
         "${SCRIPT_PATH}"
    exit 0
fi

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
EVAL_FAIL_FAST="${EVAL_FAIL_FAST:-1}"
VLLM_GPU_MEMORY_UTILIZATION="\${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_GPU_WAIT_TIMEOUT="\${VLLM_GPU_WAIT_TIMEOUT:-600}"
VLLM_GPU_WAIT_INTERVAL="\${VLLM_GPU_WAIT_INTERVAL:-30}"
VLLM_MIN_FREE_GB="\${VLLM_MIN_FREE_GB:-20}"
VLLM_MIN_FREE_RATIO="\${VLLM_MIN_FREE_RATIO:-0.20}"
if command -v nvidia-smi >/dev/null 2>&1; then
  start_ts=\$(date +%s)
  while true; do
    smi_out=\$(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | head -n1)
    total=\$(echo "\${smi_out}" | awk -F',' '{gsub(/ /,"",$1); print $1}')
    free=\$(echo "\${smi_out}" | awk -F',' '{gsub(/ /,"",$2); print $2}')
    if [[ -n "\${total}" && -n "\${free}" ]]; then
      free_gb=\$(awk -v f="\${free}" 'BEGIN{printf "%.2f", f/1024}')
      ratio=\$(awk -v f="\${free}" -v t="\${total}" 'BEGIN{printf "%.4f", f/t}')
      ok_ratio=\$(awk -v r="\${ratio}" -v minr="\${VLLM_MIN_FREE_RATIO}" 'BEGIN{print (r>=minr)?1:0}')
      ok_gb=\$(awk -v g="\${free_gb}" -v ming="\${VLLM_MIN_FREE_GB}" 'BEGIN{print (g>=ming)?1:0}')
      if [[ "\${ok_ratio}" == "1" && "\${ok_gb}" == "1" ]]; then
        util_max=\$(awk -v r="\${ratio}" 'BEGIN{u=r-0.02; if(u<0.05) u=0.05; if(u>0.95) u=0.95; printf "%.2f", u}')
        if [[ "\$(awk -v a="\${util_max}" -v b="\${VLLM_GPU_MEMORY_UTILIZATION}" 'BEGIN{print (a<b)?1:0}')" == "1" ]]; then
          echo "[WARN] Adjusting VLLM_GPU_MEMORY_UTILIZATION from \${VLLM_GPU_MEMORY_UTILIZATION} to \${util_max} (free \${free_gb} GiB, total \${total} MiB)."
          VLLM_GPU_MEMORY_UTILIZATION="\${util_max}"
        fi
        break
      fi
      now_ts=\$(date +%s)
      elapsed=\$((now_ts - start_ts))
      if (( elapsed >= VLLM_GPU_WAIT_TIMEOUT )); then
        echo "[ERROR] GPU free memory too low (\${free_gb} GiB free, ratio=\${ratio}). Timeout after \${elapsed}s."
        exit 1
      fi
      echo "[WARN] GPU memory low (\${free_gb} GiB free, ratio=\${ratio}); waiting \${VLLM_GPU_WAIT_INTERVAL}s..."
      sleep "\${VLLM_GPU_WAIT_INTERVAL}"
    else
      echo "[WARN] Unable to query GPU memory via nvidia-smi; skipping preflight check."
      break
    fi
  done
fi
export VLLM_GPU_MEMORY_UTILIZATION
export EVAL_FAIL_FAST
export EVAL_ONE_MODEL_TIMEOUT="\${EVAL_ONE_MODEL_TIMEOUT:-21600}"
export EXPORT_ROOT="${EXPORT_ROOT}"
export EVAL_DATA_DIR="${EVAL_DATA_DIR}"
if [ -n "${EVAL_NUM_TEST_SAMPLE}" ]; then
  export EVAL_NUM_TEST_SAMPLE="${EVAL_NUM_TEST_SAMPLE}"
else
  unset EVAL_NUM_TEST_SAMPLE
fi
if [ -n "${EVAL_GROUP1_DATASETS}" ]; then
  export EVAL_GROUP1_DATASETS="${EVAL_GROUP1_DATASETS}"
else
  unset EVAL_GROUP1_DATASETS
fi
if [ -n "${EVAL_GROUP2_DATASETS}" ]; then
  export EVAL_GROUP2_DATASETS="${EVAL_GROUP2_DATASETS}"
else
  unset EVAL_GROUP2_DATASETS
fi
if [ -n "${EVAL_STEP_FILTER}" ]; then
  export EVAL_STEP_FILTER="${EVAL_STEP_FILTER}"
else
  unset EVAL_STEP_FILTER
fi

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
