#!/bin/bash

# 配置部分 (根据你的 SGE 脚本环境修改)
PROJECT_NAME="noisy-RLVR"
EXP_NAMES="${EXP_NAMES:-noise_rlvr_1_5b_128batchsize_deepscaler_v2}"
PROMPT_TYPE="${PROMPT_TYPE:-think-boxed}"
OUT_ROOT="${OUT_ROOT:-$HOME/project/${PROJECT_NAME}/eval_results/${EXP_NAMES}_${PROMPT_TYPE}}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/eval/bin/python3}"

# 如果你需要计算 Pass@K，请设置这个环境变量
export PASS_AT_KS="1,4,8,16" 

echo "[INFO] Starting manual merge process for root: $OUT_ROOT"

# 进入代码根目录（确保能引用到 tools/merge_results.py）
# 假设脚本在 scripts/ 目录下，我们需要回退一层到根目录
cd "$(dirname "$0")/.." || exit

# 1. Merge Base Models (if any)
BASE_RUN_DIRS=$(ls -d "$OUT_ROOT"/base__* 2>/dev/null || true)
if [ -n "$BASE_RUN_DIRS" ]; then
    for RUN_DIR in $BASE_RUN_DIRS; do
        if [ -d "$RUN_DIR" ]; then
            RUN_NAME=$(basename "$RUN_DIR")
            echo "[INFO] Merging results for base run: $RUN_NAME ..."
            "$PYTHON_BIN" tools/merge_results.py \
              --out_root "$OUT_ROOT" \
              --run_name "$RUN_NAME" \
              --prompt_type "$PROMPT_TYPE"
        fi
    done
else
    echo "[WARN] No base__* runs found."
fi

# 2. Merge Checkpoint Models
CKPT_RUN_DIRS=$(ls -d "$OUT_ROOT"/*__global_step_* 2>/dev/null || true)
if [ -n "$CKPT_RUN_DIRS" ]; then
    for RUN_DIR in $CKPT_RUN_DIRS; do
        if [ -d "$RUN_DIR" ]; then
            RUN_NAME=$(basename "$RUN_DIR")
            echo "[INFO] Merging results for checkpoint run: $RUN_NAME ..."
            "$PYTHON_BIN" tools/merge_results.py \
              --out_root "$OUT_ROOT" \
              --run_name "$RUN_NAME" \
              --prompt_type "$PROMPT_TYPE"
        fi
    done
else
    echo "[WARN] No *__global_step_* runs found."
fi

echo "[INFO] Merge finished."