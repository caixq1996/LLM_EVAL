#!/bin/bash
set -x
set -e

PROJECT_NAME="noisy-RLVR"
EXP_NAMES="${EXP_NAMES:-noise_rlvr_Qwen2.5-1.5B}"
BASE_ROOT="${BASE_ROOT:-/hss/giil/caixq/model}"
PROMPT_TYPE="${PROMPT_TYPE:-think-boxed}" # qwen25-math-cot / think-boxed
MAX_TOKENS="${MAX_TOKENS:-3072}"
# 自动探测 GPU 数量，允许通过环境变量覆盖
if [[ -z "${NUM_GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --list-gpus | grep -c '^GPU')
    [[ "${NUM_GPUS}" -ge 1 ]] || NUM_GPUS=1
  else
    NUM_GPUS=1
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/eval/bin/python3}"
OUT_ROOT="${OUT_ROOT:-$HOME/project/${PROJECT_NAME}/new_eval_${EXP_NAMES}_${PROMPT_TYPE}}"
# MODEL_ROOT="${MODEL_ROOT:-$HOME/project/${PROJECT_NAME}/checkpoints/${EXP_NAMES}}"
MODEL_ROOT="${MODEL_ROOT:-/data/giil/caixq/ckpts/${EXP_NAMES}}"
MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-8}"
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-false}"

TEMP_G1="${TEMP_G1:-0.6}"
TEMP_G2="${TEMP_G2:-0.0}"
NSAMP_G1="${NSAMP_G1:-${MAX_SAMPLE_NUMS}}"
NSAMP_G2="${NSAMP_G2:-${MAX_SAMPLE_NUMS}}"

# 关键：给“每个模型的子进程”一个硬超时，避免卡住整个循环
# 例如 6 小时（按需调整）。也可用 CLI: --per_model_timeout 传入。
export EVAL_ONE_MODEL_TIMEOUT="${EVAL_ONE_MODEL_TIMEOUT:-21600}"

export PASS_AT_KS="${PASS_AT_KS:-1,${MAX_SAMPLE_NUMS}}"
export TORCH_CPP_LOG_LEVEL=ERROR
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS-1)))"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=1  # vLLM 建议开启以获得最佳采样性能

args=(
  --model_root "$MODEL_ROOT"
  --out_root "$OUT_ROOT"
  --prompt_type "$PROMPT_TYPE"
  --max_tokens_per_call "$MAX_TOKENS"
  --nproc "$NUM_GPUS"
  --base_root "$BASE_ROOT"
  --use_vllm
  --pipeline_parallel_size 1
  --vllm_batch_size 0
  --temperature_g1 "$TEMP_G1"
  --temperature_g2 "$TEMP_G2"
  --n_sampling_g1 "$NSAMP_G1"
  --n_sampling_g2 "$NSAMP_G2"
  --cleanup_exported
  # 也可以在这里显式传超时（秒），优先级高于环境变量
  # --per_model_timeout 21600
)

if [ "$SKIP_BASE_EVAL" = "true" ]; then
  args+=( --skip_base_eval )
fi

echo "[INFO] Running: $PYTHON_BIN -u tools/run_qwen_eval_all_shared.py ${args[*]}"
"$PYTHON_BIN" -u tools/run_qwen_eval_all_shared.py "${args[@]}"

echo "[INFO] Done at $(date)"
echo "[INFO] Saved log -> $LOG_FILE"
