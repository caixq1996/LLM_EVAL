#!/bin/bash
#PBS -q regular-g
#PBS -l select=8
#PBS -l walltime=08:00:00
#PBS -W group_list=gq50
#PBS -j oe
#PBS -N opra_multi_eval
#PBS -o ./miyabi_logs/opra_multi_eval.log

set -e
# set -x 

# ======================================================================
# 1. 基础配置
# ======================================================================

if [ -z "$WORK_HOME" ]; then
    echo "[Error] WORK_HOME is not set."
    exit 1
fi

PROJECT_NAME="OPRA"
ROOT_DIR="$WORK_HOME/project/${PROJECT_NAME}"
EVAL_DIR="${ROOT_DIR}"

EXP_NAME="${EXP_NAME:-OPRA-LoRA}" 
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/checkpoints/${EXP_NAME}}"
BASE_ROOT="${BASE_ROOT:-$WORK_HOME/model}"

PROMPT_TYPE="${PROMPT_TYPE:-qwen25-math-cot}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/eval_results/${EXP_NAME}_${PROMPT_TYPE}}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
NUM_GPUS_PER_NODE=1 
MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-128}"
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-false}"

TEMP_G1="${TEMP_G1:-0.6}"
TEMP_G2="${TEMP_G2:-0.0}"
NSAMP_G1="${NSAMP_G1:-${MAX_SAMPLE_NUMS}}"
NSAMP_G2="${NSAMP_G2:-${MAX_SAMPLE_NUMS}}"

mkdir -p "${OUT_ROOT}"
LOG_FILE="${OUT_ROOT}/eval_$(date +%Y%m%d_%H%M%S).log"

# ======================================================================
# 2. 节点检测与列表生成
# ======================================================================

NUM_NODES=1
NODE_LIST_FILE="${OUT_ROOT}/node_list.txt"

if [ -n "$PBS_NODEFILE" ]; then
    # 生成节点列表文件 (去重)
    sort -u "$PBS_NODEFILE" > "$NODE_LIST_FILE"
    NUM_NODES=$(wc -l < "$NODE_LIST_FILE")
    echo "[INFO] Detected PBS Job. Allocated Node Count: ${NUM_NODES}" | tee -a "$LOG_FILE"
else
    echo "[WARN] PBS_NODEFILE not found. Assuming local single node execution." | tee -a "$LOG_FILE"
    hostname > "$NODE_LIST_FILE"
fi

# ======================================================================
# 3. 生成 Wrapper Script
# ======================================================================

WRAPPER_SCRIPT="${OUT_ROOT}/pbs_worker_wrapper.sh"
echo "[INFO] Generating wrapper script..." | tee -a "$LOG_FILE"

cat << EOF > "${WRAPPER_SCRIPT}"
#!/bin/bash

# --- 1. Environment Setup (Miyabi Specific - NO MODULE LOAD) ---
export TZ="Asia/Tokyo"

# [Fix] 硬编码路径
export CUDA_HOME="/work/opt/local/aarch64/cores/cuda/12.8.1"
export CUDA_ROOT="\${CUDA_HOME}"
export PATH="\${CUDA_HOME}/bin:\${PATH}"
export CUDA_LIB_PATH="\${CUDA_HOME}/targets/sbsa-linux/lib"
export LD_LIBRARY_PATH="\${CUDA_LIB_PATH}:${WORK_HOME}/miniconda3/envs/eval/lib:\${LD_LIBRARY_PATH}"

# 导出关键变量
export WORK_HOME="${WORK_HOME}"
export PROJECT_NAME="${PROJECT_NAME}"
export HF_HOME="${WORK_HOME}/cache/hf"
export CUDA_CACHE_PATH="${WORK_HOME}/cache/cuda"
export HF_HUB_OFFLINE=0
export RAY_TMPDIR=/tmp/ray
mkdir -p \$RAY_TMPDIR

export PYTHONPATH="${ROOT_DIR}:${EVAL_DIR}:\${PYTHONPATH}"
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
export TORCH_CPP_LOG_LEVEL=ERROR
export CUDA_VISIBLE_DEVICES="0"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export EVAL_ONE_MODEL_TIMEOUT="${EVAL_ONE_MODEL_TIMEOUT:-21600}"

# 激活 Conda
source "${WORK_HOME}/miniconda3/bin/activate" eval
export PATH="${WORK_HOME}/miniconda3/envs/eval/bin:\${PATH}"
PYTHON="${WORK_HOME}/miniconda3/envs/eval/bin/python"

# --- 2. 动态计算 Rank ---
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

# --- 3. Python Arguments ---
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

cd "$(pwd)"
echo "[Wrapper] Working directory: \$(pwd)"

# --- 4. Execute ---
echo "[Wrapper] Executing Python script..."
exec \$PYTHON -u "$(pwd)/tools/run_qwen_eval_all_shared.py" "\${ARGS[@]}" 2>&1
EOF

chmod +x "${WRAPPER_SCRIPT}"

# ======================================================================
# 4. 执行分发
# ======================================================================

if [ "$NUM_NODES" -gt 1 ]; then
    echo "[INFO] >>> MULTI-NODE MODE (${NUM_NODES} nodes) <<<" | tee -a "$LOG_FILE"
    echo "[INFO] Using pbsdsh to distribute tasks." | tee -a "$LOG_FILE"
    pbsdsh "${WRAPPER_SCRIPT}" >> "$LOG_FILE" 2>&1
else
    echo "[INFO] >>> SINGLE-NODE MODE <<<" | tee -a "$LOG_FILE"
    echo "[INFO] Running wrapper locally." | tee -a "$LOG_FILE"
    "${WRAPPER_SCRIPT}" >> "$LOG_FILE" 2>&1
fi

EXIT_CODE=$?

# ======================================================================
# 5. 合并结果 (Merge Shards)
# ======================================================================

if [ $EXIT_CODE -eq 0 ]; then
    echo "[INFO] Tasks completed. Starting merge process..." | tee -a "$LOG_FILE"
    
    # 获取 Python 路径 (本地)
    source "${WORK_HOME}/miniconda3/bin/activate" eval
    PYTHON="${WORK_HOME}/miniconda3/envs/eval/bin/python"
    
    # 合并 Base 结果
    if [ "$SKIP_BASE_EVAL" != "true" ]; then
        BASE_NAME=$(basename "$BASE_ROOT")
        echo "[INFO] Merging results for base model..." | tee -a "$LOG_FILE"
        $PYTHON tools/merge_results.py --out_root "$OUT_ROOT" --run_name "base__${BASE_NAME}" --prompt_type "$PROMPT_TYPE" >> "$LOG_FILE" 2>&1
    fi
    
    # 合并 Checkpoint 结果
    # 注意：这里需要遍历所有 global_step 目录
    # 假设 run_qwen_eval_all_shared.py 中的 run_name 格式为 EXP_NAME__global_step_XXX
    # 我们这里简单粗暴地遍历 OUT_ROOT 下的所有 run
    for RUN_DIR in "$OUT_ROOT"/"$EXP_NAME"__global_step_*; do
        if [ -d "$RUN_DIR" ]; then
            RUN_NAME=$(basename "$RUN_DIR")
            echo "[INFO] Merging results for $RUN_NAME..." | tee -a "$LOG_FILE"
            $PYTHON tools/merge_results.py --out_root "$OUT_ROOT" --run_name "$RUN_NAME" --prompt_type "$PROMPT_TYPE" >> "$LOG_FILE" 2>&1
        fi
    done
    
    echo "[INFO] Evaluation finished successfully." | tee -a "$LOG_FILE"
else
    echo "[ERROR] Evaluation failed with code $EXIT_CODE. Check log above." | tee -a "$LOG_FILE"
fi

exit $EXIT_CODE