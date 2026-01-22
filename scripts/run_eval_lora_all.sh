#!/bin/bash
# Multi-LoRA Evaluation Script
# Evaluates multiple LoRA adapters sharing the same base model efficiently.
#
# Usage:
#   MODEL_ROOT=/path/to/checkpoints ./scripts/run_eval_lora_all.sh
#
# Key features:
# - One job per adapter type (e.g., lora, adalora, pissa)
# - Each job evaluates all global_steps of that adapter using LoRARequest
# - Supports single GPU (g1) evaluation

set -x
set -e

PROJECT_NAME="${PROJECT_NAME:-OPRA}"
EXP_NAMES="${EXP_NAMES:-OPRA-LoRA}"
MODEL_PATH="${MODEL_PATH:-checkpoints}" # checkpoints | giil

# ========================================
# Configuration
# ========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="${EVAL_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LOG_ROOT="${REPO_ROOT}"

# Base model lookup paths
BASE_MODEL_ROOTS=(
    "/home/caixq/models"
    "/data/giil/caixq/models"
)

# Output directories
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_results/${EXP_NAMES}_multi_lora}"

# GPU settings (default, can be overridden by scheduler)
NUM_GPUS="${NUM_GPUS:-1}"

# Allow single GPU evaluation
P90_ALLOW_G1="${P90_ALLOW_G1:-1}"
export P90_ALLOW_G1
EVAL_STEPS="${EVAL_STEPS:-100,200,300,313}"

# ========================================
# Helper Functions
# ========================================

find_base_model() {
    local adapter_name="$1"
    local base_name=""
    
    if [[ "$adapter_name" =~ ^(Qwen2\.5-Math-[0-9.]+B) ]]; then
        base_name="${BASH_REMATCH[1]}"
    elif [[ "$adapter_name" =~ ^(Qwen2\.5-math-[0-9.]+B) ]]; then
        base_name="${BASH_REMATCH[1]}"
    elif [[ "$adapter_name" =~ ^(DeepSeek-R1-Distill-Qwen-[0-9._]+B) ]]; then
        base_name="${BASH_REMATCH[1]}"
    else
        base_name="${adapter_name%%_*}"
    fi
    
    for root in "${BASE_MODEL_ROOTS[@]}"; do
        if [[ -d "${root}/${base_name}" ]]; then
            echo "${root}/${base_name}"
            return 0
        fi
    done
    
    echo ""
    return 1
}
# EVAL_STEPS 过滤函数
parse_eval_steps_to_array() {
    local spec="$1"
    local result=()
    if [[ -z "$spec" ]]; then
        echo ""
        return
    fi
    if [[ "$spec" =~ ^([0-9]+)-([0-9]+)(:([0-9]+))?$ ]]; then
        local start="${BASH_REMATCH[1]}"
        local end="${BASH_REMATCH[2]}"
        local step="${BASH_REMATCH[4]:-1}"
        for ((i=start; i<=end; i+=step)); do
            result+=("$i")
        done
    else
        IFS=',' read -ra tokens <<< "$spec"
        for t in "${tokens[@]}"; do
            t="${t//[^0-9]/}"
            if [[ -n "$t" ]]; then
                result+=("$t")
            fi
        done
    fi
    echo "${result[*]}"
}

step_in_filter() {
    local step_num="$1"
    local -a allowed=($2)
    if [[ ${#allowed[@]} -eq 0 ]]; then
        return 0
    fi
    for a in "${allowed[@]}"; do
        if [[ "$step_num" == "$a" ]]; then
            return 0
        fi
    done
    return 1
}

# ========================================
# Main Logic
# ========================================

# Determine MODEL_ROOT
if [[ -z "${MODEL_ROOT:-}" ]]; then
    if [[ "$MODEL_PATH" == "checkpoints" ]]; then
        MODEL_ROOT="${REPO_ROOT}/../OPRA/${MODEL_PATH}/${EXP_NAMES}"
    else
        MODEL_ROOT="/data/${MODEL_PATH}/caixq/${EXP_NAMES}"
    fi
fi

if [[ ! -d "$MODEL_ROOT" ]]; then
    echo "[ERROR] MODEL_ROOT not found: $MODEL_ROOT"
    exit 1
fi

echo "=============================================="
echo "Multi-LoRA Evaluation (Per Adapter Type)"
echo "MODEL_ROOT: $MODEL_ROOT"
echo "OUT_ROOT: $OUT_ROOT"
echo "=============================================="

# Already submitted as qsub job?
if [[ -n "${RUN_LORA_EVAL_SUBMITTED:-}" ]]; then
    echo "[INFO] Running multi-LoRA evaluation..."
    
    cd "$REPO_ROOT"
    
    python tools/run_lora_eval_shared.py \
        --base_model "${BASE_MODEL_PATH}" \
        --lora_adapters "${LORA_ADAPTERS}" \
        --out_root "${OUT_ROOT}" \
        --num_gpus "${NUM_GPUS}" \
        --datasets "${EVAL_DATASETS:-aime24,amc23,aime25,math500,minerva_math,olympiadbench}"
    
    exit 0
fi

# ========================================
# Discovery and Job Submission
# ========================================

echo "[INFO] Discovering LoRA adapters in $MODEL_ROOT..."

# Parse EVAL_STEPS filter
ALLOWED_STEPS="$(parse_eval_steps_to_array "${EVAL_STEPS:-}")"
if [[ -n "$ALLOWED_STEPS" ]]; then
    echo "[INFO] EVAL_STEPS filter: $ALLOWED_STEPS"
fi

# Find all adapters (global_step_* directories containing adapter_config.json)
ADAPTER_LIST=()
while IFS= read -r -d '' adapter_dir; do
    ADAPTER_LIST+=("$adapter_dir")
done < <(find "$MODEL_ROOT" -name "adapter_config.json" -printf '%h\0' | sort -zu)

echo "[INFO] Found ${#ADAPTER_LIST[@]} LoRA adapter checkpoints"

if [[ ${#ADAPTER_LIST[@]} -eq 0 ]]; then
    echo "[WARN] No LoRA adapters found, exiting"
    exit 0
fi

# Group by adapter type (algorithm directory name)
# Key: "base_model::adapter_type", Value: list of global_step paths
declare -A ADAPTER_GROUPS
for adapter in "${ADAPTER_LIST[@]}"; do
    # adapter is like: /path/to/Qwen2.5-Math-1.5B_lora/global_step_100/actor
    # or: /path/to/Qwen2.5-Math-1.5B_lora/global_step_100
    
    # Get adapter type (algorithm) directory
    adapter_parent="$adapter"
    # Walk up to find the algorithm directory (contains global_step_*)
    while [[ "$(basename "$adapter_parent")" == "actor" || "$(basename "$adapter_parent")" =~ ^global_step_ ]]; do
        adapter_parent="$(dirname "$adapter_parent")"
    done
    
    adapter_type="$(basename "$adapter_parent")"
    
    # Extract step number for EVAL_STEPS filtering
    step_dir_name="$(echo "$adapter" | grep -oP 'global_step_\d+' | head -1)"
    if [[ -n "$step_dir_name" ]]; then
        step_num="${step_dir_name#global_step_}"
        if ! step_in_filter "$step_num" "$ALLOWED_STEPS"; then
            echo "[INFO] Skipping $adapter (step $step_num not in EVAL_STEPS)"
            continue
        fi
    fi
    
    base_model=$(find_base_model "$adapter_type")
    if [[ -z "$base_model" ]]; then
        echo "[WARN] Skipping $adapter - cannot find base model for $adapter_type"
        continue
    fi
    
    base_key="$(basename "$base_model")"
    group_key="${base_key}::${adapter_type}"
    
    if [[ -n "${ADAPTER_GROUPS[$group_key]}" ]]; then
        ADAPTER_GROUPS[$group_key]="${ADAPTER_GROUPS[$group_key]}|${adapter}"
    else
        ADAPTER_GROUPS[$group_key]="$adapter"
    fi
done

echo "[INFO] Grouped into ${#ADAPTER_GROUPS[@]} adapter type groups"

# Source scheduler
SCHEDULER_TOOL="${SCHEDULER_TOOL:-$HOME/tools/qsub_gpu_scheduler_p90.sh}"
if [[ -f "$SCHEDULER_TOOL" ]]; then
    source "$SCHEDULER_TOOL"
else
    echo "[WARN] Scheduler not found: $SCHEDULER_TOOL"
fi

# Submit one job per adapter type
for group_key in "${!ADAPTER_GROUPS[@]}"; do
    adapters="${ADAPTER_GROUPS[$group_key]}"
    adapter_count=$(echo "$adapters" | tr '|' '\n' | wc -l)
    
    # Parse group key: base_model::adapter_type
    base_key="${group_key%%::*}"
    adapter_type="${group_key##*::}"
    
    # Find full base model path
    base_model_path=""
    for root in "${BASE_MODEL_ROOTS[@]}"; do
        if [[ -d "${root}/${base_key}" ]]; then
            base_model_path="${root}/${base_key}"
            break
        fi
    done
    
    if [[ -z "$base_model_path" ]]; then
        echo "[ERROR] Base model path not found for: $base_key"
        continue
    fi
    
    job_name="LORA_${PROJECT_NAME}_${adapter_type}"
    job_name="${job_name//[^A-Za-z0-9_]/_}"
    job_name="${job_name:0:120}"
    
    echo "=============================================="
    echo "Submitting: $job_name"
    echo "Adapter Type: $adapter_type"
    echo "Base Model: $base_model_path"
    echo "Checkpoints: $adapter_count"
    echo "=============================================="
    
    # Determine resources (allow g1 for single GPU)
    if type -t select_resources_for_job >/dev/null 2>&1; then
        read -r jc_base n_gpus < <(select_resources_for_job "$PROJECT_NAME" "$job_name")
        jc_full="$(full_jclass_from_base "$jc_base")"
        d_shm="$(get_d_shm_for_jc "$jc_base")"
    else
        jc_full="gtn-container_g1.24h"
        n_gpus=1
        d_shm="128g"
    fi
    
    echo "[INFO] Using $n_gpus GPU(s), jc=$jc_full"
    
    qsub -N "$job_name" \
         -jc "$jc_full" \
         -ac "d=nvcr-cuda-12.4.1-ubuntu22.04,d_shm=${d_shm}" \
         -v NUM_GPUS="${n_gpus}" \
         -v BASE_MODEL_PATH="${base_model_path}" \
         -v LORA_ADAPTERS="${adapters}" \
         -v OUT_ROOT="${OUT_ROOT}" \
         -v RUN_LORA_EVAL_SUBMITTED=1 \
         -V \
         "$SCRIPT_PATH"
done

echo "[INFO] All jobs submitted"
