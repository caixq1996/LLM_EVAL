#!/bin/bash

set -e
set -x

# ======================================================================
# 1. Environment Setup (Miyabi Specific - Copied from miyabi_run.sh)
# ======================================================================
export TZ="Asia/Tokyo"
module purge
module load cuda/12.8

# [Critical Fix] Set LD_LIBRARY_PATH for libnvrtc.so and other CUDA libs
export CUDA_LIB_PATH="/work/opt/local/aarch64/cores/cuda/12.8.1/targets/sbsa-linux/lib"
export LD_LIBRARY_PATH="${CUDA_LIB_PATH}:$WORK_HOME/miniconda3/envs/eval/lib:$LD_LIBRARY_PATH"

# Activate Conda
source $WORK_HOME/miniconda3/bin/activate eval
export PATH=$PATH:$WORK_HOME/miniconda3/envs/eval/bin
PYTHON=$WORK_HOME/miniconda3/envs/eval/bin/python

# Directory definitions
PROJECT_NAME="OPRA"
ROOT_DIR="$WORK_HOME/project/${PROJECT_NAME}"
EVAL_DIR="${ROOT_DIR}"

# Set PYTHONPATH to include the evaluation directory so imports work
export PYTHONPATH="${ROOT_DIR}:${EVAL_DIR}:${PYTHONPATH}"

# Ray & vLLM Temp dirs
export RAY_TMPDIR=/tmp/ray
mkdir -p $RAY_TMPDIR

# HF Cache & Network
export HF_HUB_OFFLINE=0
export HF_HOME=$WORK_HOME/cache/hf
export CUDA_CACHE_PATH=$WORK_HOME/cache/cuda

# ======================================================================
# 2. Evaluation Configuration
# ======================================================================

# --- Input/Output Paths ---
# The name of the experiment folder in your checkpoints directory
# Matches the 'trainer.experiment_name' from the training script
EXP_NAME="${EXP_NAME:-OPRA-LoRA}" 

# Directory containing the FSDP checkpoints (e.g. global_step_*)
# The shared script expects a folder structure like: $MODEL_ROOT/$EXP_NAME/global_step_xxx
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/checkpoints/${EXP_NAME}}"

# Directory where evaluation results (and exported HF models) will be saved
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/eval_results/${EXP_NAME}}"

# Path to the base model (required for merging LoRA/FSDP weights)
BASE_ROOT="${BASE_ROOT:-$WORK_HOME/model}" # Directory containing the base model folder

# --- Evaluation Parameters ---
PROMPT_TYPE="${PROMPT_TYPE:-qwen25-math-cot}" # Check parser.py for valid types (e.g. cot, tool-integrated)
MAX_TOKENS="${MAX_TOKENS:-4096}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-128}"
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-false}"

# Temperature settings for different dataset groups (G1: difficult, G2: easier/standard)
# G1 typically: AIME, AMC
# G2 typically: MATH, GSM8k, Minerva
TEMP_G1="${TEMP_G1:-0.6}"
TEMP_G2="${TEMP_G2:-0.0}"
NSAMP_G1="${NSAMP_G1:-${MAX_SAMPLE_NUMS}}"
NSAMP_G2="${NSAMP_G2:-${MAX_SAMPLE_NUMS}}"

# Timeout per model evaluation (seconds)
export EVAL_ONE_MODEL_TIMEOUT="${EVAL_ONE_MODEL_TIMEOUT:-21600}" # 6 hours

# Environment variables for vLLM and Evaluation
export PASS_AT_KS="${PASS_AT_KS:-1}" # Compute Pass@1. Add more like "1,8" if NSAMP > 1
export TORCH_CPP_LOG_LEVEL=ERROR
export CUDA_VISIBLE_DEVICES="0" # Explicitly set for Miyabi single-GPU usage
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=1

# ======================================================================
# 3. Execution
# ======================================================================

mkdir -p "${OUT_ROOT}"
LOG_FILE="${OUT_ROOT}/eval_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] Starting Evaluation..." | tee -a "$LOG_FILE"
echo "[INFO] Model Root: $MODEL_ROOT" | tee -a "$LOG_FILE"
echo "[INFO] Output Root: $OUT_ROOT" | tee -a "$LOG_FILE"
echo "[INFO] Base Root: $BASE_ROOT" | tee -a "$LOG_FILE"

# Construct arguments
args=(
  --model_root "$MODEL_ROOT"
  --out_root "$OUT_ROOT"
  --base_root "$BASE_ROOT"
  --prompt_type "$PROMPT_TYPE"
  --max_tokens_per_call "$MAX_TOKENS"
  --nproc "$NUM_GPUS"
  --use_vllm
  --pipeline_parallel_size 1
  --vllm_batch_size 0
  --temperature_g1 "$TEMP_G1"
  --temperature_g2 "$TEMP_G2"
  --n_sampling_g1 "$NSAMP_G1"
  --n_sampling_g2 "$NSAMP_G2"
  # Uncomment the following line if you want to save space by deleting exported HF models after eval
  # --cleanup_exported 
)

if [ "$SKIP_BASE_EVAL" = "true" ]; then
  args+=( --skip_base_eval )
fi

# Run the shared evaluation script
# This script handles:
# 1. Scanning MODEL_ROOT for runs
# 2. Converting FSDP/DTensor checkpoints to HuggingFace format (using base_root)
# 3. Running vLLM inference
# 4. Scoring results
$PYTHON -u tools/run_qwen_eval_all_shared.py "${args[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "[INFO] Done. Log saved to $LOG_FILE"