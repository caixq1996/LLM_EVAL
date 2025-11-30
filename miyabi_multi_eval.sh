#!/bin/bash

set -e
# set -x # 生产环境建议关闭 debug 输出

# ======================================================================
# 1. 基础配置 (Configuration)
# ======================================================================

# 检查 WORK_HOME
if [ -z "$WORK_HOME" ]; then
    echo "[Error] WORK_HOME is not set. Please set it or source your .bash_profile."
    exit 1
fi

PROJECT_NAME="OPRA"
ROOT_DIR="$WORK_HOME/project/${PROJECT_NAME}"
EVAL_DIR="${ROOT_DIR}"

# 路径定义
EXP_NAME="${EXP_NAME:-OPRA-LoRA}" 
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/checkpoints/${EXP_NAME}}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/eval_results/${EXP_NAME}}"
BASE_ROOT="${BASE_ROOT:-$WORK_HOME/model}"

# 评测参数
PROMPT_TYPE="${PROMPT_TYPE:-qwen25-math-cot}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
NUM_GPUS_PER_NODE=1 
MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-128}"
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-false}"

# 采样温度
TEMP_G1="${TEMP_G1:-0.6}"
TEMP_G2="${TEMP_G2:-0.0}"
NSAMP_G1="${NSAMP_G1:-${MAX_SAMPLE_NUMS}}"
NSAMP_G2="${NSAMP_G2:-${MAX_SAMPLE_NUMS}}"

# 创建输出目录
mkdir -p "${OUT_ROOT}"
LOG_FILE="${OUT_ROOT}/eval_$(date +%Y%m%d_%H%M%S).log"

# ======================================================================
# 2. 节点检测逻辑
# ======================================================================

NUM_NODES=1
if [ -n "$PBS_NODEFILE" ]; then
    NUM_NODES=$(sort -u "$PBS_NODEFILE" | wc -l)
    echo "[INFO] Detected PBS Job. Allocated Node Count: ${NUM_NODES}" | tee -a "$LOG_FILE"
else
    echo "[WARN] PBS_NODEFILE not found. Assuming local single node execution." | tee -a "$LOG_FILE"
fi

# ======================================================================
# 3. 核心修复：生成 Wrapper Script
# ======================================================================
# 我们将根据 miyabi_multi_run.sh 中的路径硬编码 CUDA_HOME，
# 确保 flashinfer 绝对能找到 nvcc。

WRAPPER_SCRIPT="${OUT_ROOT}/pbs_worker_wrapper.sh"

echo "[INFO] Generating wrapper script with hardcoded CUDA paths..." | tee -a "$LOG_FILE"

cat << EOF > "${WRAPPER_SCRIPT}"
#!/bin/bash

# --- 1. Environment Setup (Miyabi Specific) ---
export TZ="Asia/Tokyo"
module purge
module load cuda/12.8

# [CRITICAL FIX] 硬编码路径，参考自 miyabi_multi_run.sh
# miyabi_multi_run.sh 中的 CUDA_LIB_PATH 是:
# /work/opt/local/aarch64/cores/cuda/12.8.1/targets/sbsa-linux/lib
#由此推导 CUDA_HOME 为:
export CUDA_HOME="/work/opt/local/aarch64/cores/cuda/12.8.1"

# Flashinfer 和 vLLM 需要这变量来定位 nvcc
export CUDA_ROOT="\${CUDA_HOME}"

# 强制将 nvcc 加入 PATH
export PATH="\${CUDA_HOME}/bin:\${PATH}"

# 设置库路径 (完全参考 miyabi_multi_run.sh)
export CUDA_LIB_PATH="\${CUDA_HOME}/targets/sbsa-linux/lib"
# 注意：这里需要确保 WORK_HOME 在远程节点被正确解析
export LD_LIBRARY_PATH="\${CUDA_LIB_PATH}:${WORK_HOME}/miniconda3/envs/eval/lib:\${LD_LIBRARY_PATH}"

# 验证 nvcc 是否可用 (调试用)
echo "[Wrapper] Host: \$(hostname)"
echo "[Wrapper] CUDA_HOME: \${CUDA_HOME}"
if which nvcc > /dev/null; then
    echo "[Wrapper] Found nvcc at: \$(which nvcc)"
else
    echo "[Wrapper] [ERROR] nvcc still not found in PATH!"
fi

# 重新导出其他关键变量
export WORK_HOME="${WORK_HOME}"
export PROJECT_NAME="${PROJECT_NAME}"
export HF_HOME="${WORK_HOME}/cache/hf"
export CUDA_CACHE_PATH="${WORK_HOME}/cache/cuda"
export HF_HUB_OFFLINE=0
export RAY_TMPDIR=/tmp/ray
mkdir -p \$RAY_TMPDIR

# Python Path
export PYTHONPATH="${ROOT_DIR}:${EVAL_DIR}:\${PYTHONPATH}"

# vLLM 设置
export PASS_AT_KS="${PASS_AT_KS:-1}"
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

# --- 2. Construct Python Arguments ---
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
)

if [ "${SKIP_BASE_EVAL}" = "true" ]; then
  ARGS+=( --skip_base_eval )
fi

cd "$(pwd)"
echo "[Wrapper] Working directory: \$(pwd)"

# --- 3. Execute ---
echo "[Wrapper] Executing Python script..."
exec \$PYTHON -u "$(pwd)/tools/run_qwen_eval_all_shared.py" "\${ARGS[@]}" 2>&1
EOF

# 赋予执行权限
chmod +x "${WRAPPER_SCRIPT}"

# ======================================================================
# 4. 执行分发
# ======================================================================

if [ "$NUM_NODES" -gt 1 ]; then
    echo "[INFO] >>> MULTI-NODE MODE (${NUM_NODES} nodes) <<<" | tee -a "$LOG_FILE"
    echo "[INFO] Using pbsdsh to distribute tasks." | tee -a "$LOG_FILE"
    
    # 核心：使用 pbsdsh 在所有节点执行上面生成的脚本
    pbsdsh "${WRAPPER_SCRIPT}" >> "$LOG_FILE" 2>&1
else
    echo "[INFO] >>> SINGLE-NODE MODE <<<" | tee -a "$LOG_FILE"
    echo "[INFO] Running wrapper locally." | tee -a "$LOG_FILE"
    
    # 单节点直接执行 Wrapper
    "${WRAPPER_SCRIPT}" >> "$LOG_FILE" 2>&1
fi

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "[INFO] Evaluation finished successfully." | tee -a "$LOG_FILE"
else
    echo "[ERROR] Evaluation failed with code $EXIT_CODE. Check log above." | tee -a "$LOG_FILE"
fi

exit $EXIT_CODE