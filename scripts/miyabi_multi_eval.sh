#!/bin/bash
#PBS -q regular-g
#PBS -l select=8
#PBS -l walltime=08:00:00
#PBS -W group_list=gq50
#PBS -N opra_multi_eval
#PBS -j oe
#PBS -V
#PBS -o ./pbs_opra_multi_eval.o$PBS_JOBID

# 推荐：作业一开始就切回提交目录（用户指南 5.4.2 的写法）
cd "$PBS_O_WORKDIR"
echo "[INFO] CWD: $(pwd)"
# ================= 配置日志 =================
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_DIR="./miyabi_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/opra_eval_${TIMESTAMP}.log"

echo "[INFO] Start logging to: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

set -e

# ... (环境变量检查部分保持不变) ...
if [ -z "$WORK_HOME" ]; then
    echo "[WARN] WORK_HOME is not set, fallback to /work/gq50/$USER"
    export WORK_HOME="/work/gq50/$USER"
fi

PROJECT_NAME="OPRA"
ROOT_DIR="$WORK_HOME/project/${PROJECT_NAME}"
EVAL_DIR="${ROOT_DIR}"
WORK_DIR="${WORK_HOME}/project/LLM_EVAL"

EXP_NAME="${EXP_NAME:-OPRA-LoRA}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/checkpoints/${EXP_NAME}}"
BASE_ROOT="${BASE_ROOT:-$WORK_HOME/model}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen25-math-cot}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/eval_results/${EXP_NAME}_${PROMPT_TYPE}}"
MAX_TOKENS="${MAX_TOKENS:-4096}"

# ... (参数设置部分保持不变) ...
NUM_GPUS_PER_NODE=1
MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-128}"
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-false}"
TEMP_G1="${TEMP_G1:-0.6}"
TEMP_G2="${TEMP_G2:-0.0}"
NSAMP_G1="${NSAMP_G1:-${MAX_SAMPLE_NUMS}}"
NSAMP_G2="${NSAMP_G2:-${MAX_SAMPLE_NUMS}}"

mkdir -p "${OUT_ROOT}"

# Node Detection
NUM_NODES=1
NODE_LIST_FILE="${OUT_ROOT}/node_list.txt"
if [ -n "$PBS_NODEFILE" ]; then
    sort -u "$PBS_NODEFILE" > "$NODE_LIST_FILE"
    NUM_NODES=$(wc -l < "$NODE_LIST_FILE")
    echo "[INFO] Detected PBS Job. Allocated Node Count: ${NUM_NODES}"
else
    echo "[WARN] PBS_NODEFILE not found. Assuming local single node execution."
    hostname > "$NODE_LIST_FILE"
fi

WRAPPER_SCRIPT="${OUT_ROOT}/pbs_worker_wrapper.sh"
echo "[INFO] Generating wrapper script..."

# ================= 生成 Wrapper 脚本 (重点修改区域) =================
cat << EOF > "${WRAPPER_SCRIPT}"
#!/bin/bash
export TZ="Asia/Tokyo"
export CUDA_HOME="/work/opt/local/aarch64/cores/cuda/12.8.1"
export CUDA_ROOT="\${CUDA_HOME}"
export PATH="\${CUDA_HOME}/bin:\${PATH}"
export CUDA_LIB_PATH="\${CUDA_HOME}/targets/sbsa-linux/lib"
export LD_LIBRARY_PATH="\${CUDA_LIB_PATH}:${WORK_HOME}/miniconda3/envs/eval/lib:\${LD_LIBRARY_PATH}"

export WORK_HOME="${WORK_HOME}"
export PROJECT_NAME="${PROJECT_NAME}"
export HF_HOME="${WORK_HOME}/cache/hf"
export CUDA_CACHE_PATH="${WORK_HOME}/cache/cuda"
export HF_HUB_OFFLINE=0
export RAY_TMPDIR=/tmp/ray
mkdir -p \$RAY_TMPDIR

export PYTHONPATH="${ROOT_DIR}:${EVAL_DIR}:\${PYTHONPATH}"

# Pass@k 逻辑
if [[ -z "${PASS_AT_KS:-}" ]]; then
  default_pass_ks=(1 8 16 32 64 128 256)
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
export EVAL_ONE_MODEL_TIMEOUT="${EVAL_ONE_MODEL_TIMEOUT:-21600}"

source "${WORK_HOME}/miniconda3/bin/activate" eval
export PATH="${WORK_HOME}/miniconda3/envs/eval/bin:\${PATH}"

# Fix: 这里的 PYTHON 是在 wrapper 运行时定义的，不是生成时
PYTHON="${WORK_HOME}/miniconda3/envs/eval/bin/python"

NODE_LIST_FILE="${NODE_LIST_FILE}"
MY_HOSTNAME=\$(hostname)

if [ -f "\$NODE_LIST_FILE" ]; then
    LINE_NUM=\$(grep -n "^\${MY_HOSTNAME}$" "\$NODE_LIST_FILE" | cut -d: -f1)
    if [ -z "\$LINE_NUM" ]; then
        MY_RANK=0
        TOTAL_NODES=1
    else
        MY_RANK=\$((LINE_NUM - 1))
        TOTAL_NODES=\$(wc -l < "\$NODE_LIST_FILE")
    fi
else
    MY_RANK=0
    TOTAL_NODES=1
fi

echo "[Wrapper] Host: \${MY_HOSTNAME}, Rank: \${MY_RANK}, Total: \${TOTAL_NODES}"

# Fix: 使用 \${...} 防止 ARGS 在生成文件时被展开
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

# Fix: 关键修改！使用 \$PYTHON 而不是 $PYTHON
# $PYTHON 在生成脚本时是空的，导致变成了 exec -u ...
exec \$PYTHON -u "${WORK_DIR}/tools/run_qwen_eval_all_shared.py" "\${ARGS[@]}"
EOF

chmod +x "${WRAPPER_SCRIPT}"

# ================= 执行阶段 =================
if [ "$NUM_NODES" -gt 1 ]; then
    echo "[INFO] >>> MULTI-NODE MODE (${NUM_NODES} nodes) <<<"
    echo "[INFO] Using pbsdsh to distribute tasks."
    # 不需要重定向，因为当前 shell 已经做了全局重定向
    pbsdsh "${WRAPPER_SCRIPT}"
else
    echo "[INFO] >>> SINGLE-NODE MODE <<<"
    echo "[INFO] Running wrapper locally."
    "${WRAPPER_SCRIPT}"
fi

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[INFO] Tasks completed. Starting merge process..."
    source "${WORK_HOME}/miniconda3/bin/activate" eval
    PYTHON="${WORK_HOME}/miniconda3/envs/eval/bin/python"

    if [ "$SKIP_BASE_EVAL" != "true" ]; then
        BASE_NAME=$(basename "$BASE_ROOT")
        echo "[INFO] Merging results for base model..."
        $PYTHON ${WORK_DIR}/tools/merge_results.py --out_root "$OUT_ROOT" --run_name "base__${BASE_NAME}" --prompt_type "$PROMPT_TYPE"
    fi

    # 使用 ls 避免 glob 展开为空时的报错
    RUN_DIRS=$(ls -d "$OUT_ROOT"/"$EXP_NAME"__global_step_* 2>/dev/null)
    if [ -n "$RUN_DIRS" ]; then
        for RUN_DIR in $RUN_DIRS; do
            if [ -d "$RUN_DIR" ]; then
                RUN_NAME=$(basename "$RUN_DIR")
                echo "[INFO] Merging results for $RUN_NAME..."
                $PYTHON ${WORK_DIR}/tools/merge_results.py --out_root "$OUT_ROOT" --run_name "$RUN_NAME" --prompt_type "$PROMPT_TYPE"
            fi
        done
    fi
    echo "[INFO] Evaluation finished successfully."
else
    echo "[ERROR] Evaluation failed with code $EXIT_CODE. Check log above."
fi

exit $EXIT_CODE