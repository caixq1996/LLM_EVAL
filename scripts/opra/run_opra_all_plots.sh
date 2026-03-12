#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
SCRIPTS_DIR="$PROJECT_ROOT/LLM_EVAL/scripts/opra"

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"$PROJECT_ROOT/OPRA/checkpoints/OPRA-LoRA"}
BASE_MODEL=${BASE_MODEL:-}
RUNS=${RUNS:-}
RUN_FILTER=${RUN_FILTER:-}
MODEL_FILTER=${MODEL_FILTER:-}  # Optional: filter to specific base model(s)
PROMPT_FIELD=${PROMPT_FIELD:-question}
ANSWER_FIELD=${ANSWER_FIELD:-answer}
PROMPT_FILE_MATH=${PROMPT_FILE_MATH:-"$PROJECT_ROOT/LLM_EVAL/data/math500/test.jsonl"}
PROMPT_FILE_GSM8K=${PROMPT_FILE_GSM8K:-"$PROJECT_ROOT/LLM_EVAL/data/gsm8k/test.jsonl"}

PRINCIPAL_RANK=${PRINCIPAL_RANK:-16}
PRINCIPAL_RANK_SPECTRAL=${PRINCIPAL_RANK_SPECTRAL:-1}
RANDOM_SEED=${RANDOM_SEED:-1234}
NUM_SAMPLES=${NUM_SAMPLES:-64}
GRID_SIZE=${GRID_SIZE:-9}
RADIUS=${RADIUS:-0.5}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_LENGTH=${MAX_LENGTH:-512}
MASK_PROMPT=${MASK_PROMPT:-0}
USE_LOWRANK=${USE_LOWRANK:-0}
DIRECTION_SEED=${DIRECTION_SEED:-1234}
SHARED_DIRS=${SHARED_DIRS:-1}
STEPS=${STEPS:-}
ETA_SOURCE=${ETA_SOURCE:-both}
WANDB_ROOT=${WANDB_ROOT:-"$PROJECT_ROOT/OPRA/wandb"}
WANDB_METRIC_GRAD=${WANDB_METRIC_GRAD:-actor/grad_eta}
WANDB_METRIC_PARAM=${WANDB_METRIC_PARAM:-actor/delta_w_eta}
WANDB_PLOT_BOTH=${WANDB_PLOT_BOTH:-1}

OUT_DIR_ROOT=${OUT_DIR_ROOT:-"$PROJECT_ROOT/LLM_EVAL/eval_log/opra"}

RUN_ALIGNMENT=${RUN_ALIGNMENT:-1}
RUN_ALIGNMENT_RANDOM=${RUN_ALIGNMENT_RANDOM:-0}
RUN_ACTIVATION_DRIFT=${RUN_ACTIVATION_DRIFT:-0}
RUN_LAYERWISE=${RUN_LAYERWISE:-0}
RUN_SPECTRAL_DECOUPLING=${RUN_SPECTRAL_DECOUPLING:-0}
RUN_LOSS=${RUN_LOSS:-0}
RUN_LOSS_SPECTRAL=${RUN_LOSS_SPECTRAL:-0}

echo "[INFO] checkpoint_root=$CHECKPOINT_ROOT"
if [[ -n "$BASE_MODEL" ]]; then
  echo "[INFO] base_model=$BASE_MODEL"
fi
if [[ -n "$RUNS" ]]; then
  echo "[INFO] runs=$RUNS"
elif [[ -n "$RUN_FILTER" ]]; then
  echo "[INFO] run_filter=$RUN_FILTER"
fi
if [[ -n "$MODEL_FILTER" ]]; then
  echo "[INFO] model_filter=$MODEL_FILTER"
fi

# Extract unique base model names from checkpoint directories
# Format: <BaseModel>_<algorithm> -> extract <BaseModel>
get_base_models() {
  local base_models=()
  for dir in "$CHECKPOINT_ROOT"/*/; do
    local name=$(basename "$dir")
    # Extract base model name by removing the algorithm suffix (last _xxx part)
    local base=$(echo "$name" | sed 's/_[^_]*$//')
    # Handle cases like "opra_opra" suffix
    if [[ "$base" == *"_opra" ]]; then
      base=$(echo "$base" | sed 's/_opra$//')
    fi
    base_models+=("$base")
  done
  # Return unique sorted base models
  printf '%s\n' "${base_models[@]}" | sort -u
}

run_step () {
  local name="$1"
  shift
  echo "[INFO] ===== ${name} ====="
  "$@"
}

# Function to run all plots for a specific base model
run_plots_for_model() {
  local base_model_name="$1"
  local model_run_filter="${base_model_name}_*"
  
  # Set model-specific output directories
  local out_dir_alignment="$OUT_DIR_ROOT/alignment/$base_model_name"
  local out_dir_alignment_random="$OUT_DIR_ROOT/alignment_random/$base_model_name"
  local out_dir_activation="$OUT_DIR_ROOT/activation_drift/$base_model_name"
  local out_dir_layerwise="$OUT_DIR_ROOT/layerwise_leakage/$base_model_name"
  local out_dir_spectral="$OUT_DIR_ROOT/spectral_decoupling/$base_model_name"
  local out_dir_loss="$OUT_DIR_ROOT/loss_landscape/$base_model_name"
  local out_dir_loss_spectral="$OUT_DIR_ROOT/loss_landscape_spectral/$base_model_name"
  
  echo ""
  echo "======================================"
  echo "[INFO] Processing base model: $base_model_name"
  echo "======================================"

run_step () {
  local name="$1"
  shift
  echo "[INFO] ===== ${name} ====="
  "$@"
}

  if [[ "$RUN_ALIGNMENT" == "1" ]]; then
    run_step "alignment_curve" env \
      CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
      BASE_MODEL="$BASE_MODEL" \
      RUNS="$RUNS" \
      RUN_FILTER="$model_run_filter" \
      PRINCIPAL_RANK="$PRINCIPAL_RANK" \
      ETA_SOURCE="$ETA_SOURCE" \
      WANDB_ROOT="$WANDB_ROOT" \
      WANDB_METRIC_GRAD="$WANDB_METRIC_GRAD" \
      WANDB_METRIC_PARAM="$WANDB_METRIC_PARAM" \
      WANDB_PLOT_BOTH="$WANDB_PLOT_BOTH" \
      OUT_DIR="$out_dir_alignment" \
      "$SCRIPTS_DIR/run_opra_alignment_curve.sh"
  fi

  if [[ "$RUN_ALIGNMENT_RANDOM" == "1" ]]; then
    run_step "alignment_curve_random" env \
      CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
      BASE_MODEL="$BASE_MODEL" \
      RUNS="$RUNS" \
      RUN_FILTER="$model_run_filter" \
      PRINCIPAL_RANK="$PRINCIPAL_RANK" \
      RANDOM_SEED="$RANDOM_SEED" \
      OUT_DIR="$out_dir_alignment_random" \
      "$SCRIPTS_DIR/run_opra_alignment_curve_random.sh"
  fi

  if [[ "$RUN_ACTIVATION_DRIFT" == "1" ]]; then
    run_step "activation_drift" env \
      CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
      BASE_MODEL="$BASE_MODEL" \
      RUNS="$RUNS" \
      RUN_FILTER="$model_run_filter" \
      PROMPT_FILE="$PROMPT_FILE_MATH" \
      PROMPT_FIELD="$PROMPT_FIELD" \
      PRINCIPAL_RANK="$PRINCIPAL_RANK" \
      OUT_DIR="$out_dir_activation" \
      "$SCRIPTS_DIR/run_opra_activation_drift.sh"
  fi

  if [[ "$RUN_LAYERWISE" == "1" ]]; then
    run_step "layerwise_leakage" env \
      CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
      BASE_MODEL="$BASE_MODEL" \
      RUNS="$RUNS" \
      RUN_FILTER="$model_run_filter" \
      PRINCIPAL_RANK="$PRINCIPAL_RANK" \
      OUT_DIR="$out_dir_layerwise" \
      "$SCRIPTS_DIR/run_opra_layerwise_leakage.sh"
  fi

  if [[ "$RUN_SPECTRAL_DECOUPLING" == "1" ]]; then
    run_step "spectral_decoupling" env \
      CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
      BASE_MODEL="$BASE_MODEL" \
      RUNS="$RUNS" \
      RUN_FILTER="$model_run_filter" \
      PROMPT_FILE="$PROMPT_FILE_MATH" \
      PROMPT_FIELD="$PROMPT_FIELD" \
      PRINCIPAL_RANK="$PRINCIPAL_RANK" \
      OUT_DIR="$out_dir_spectral" \
      "$SCRIPTS_DIR/run_opra_spectral_decoupling.sh"
  fi

  if [[ "$RUN_LOSS" == "1" ]]; then
    run_step "loss_landscape" env \
      CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
      BASE_MODEL="$BASE_MODEL" \
      RUNS="$RUNS" \
      RUN_FILTER="$model_run_filter" \
      PROMPT_FILE="$PROMPT_FILE_GSM8K" \
      PROMPT_FIELD="$PROMPT_FIELD" \
      ANSWER_FIELD="$ANSWER_FIELD" \
      NUM_SAMPLES="$NUM_SAMPLES" \
      GRID_SIZE="$GRID_SIZE" \
      RADIUS="$RADIUS" \
      BATCH_SIZE="$BATCH_SIZE" \
      MAX_LENGTH="$MAX_LENGTH" \
      MASK_PROMPT="$MASK_PROMPT" \
      STEPS="$STEPS" \
      OUT_DIR="$out_dir_loss" \
      "$SCRIPTS_DIR/run_opra_loss_landscape.sh"
  fi

  if [[ "$RUN_LOSS_SPECTRAL" == "1" ]]; then
    run_step "loss_landscape_spectral" env \
      CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
      BASE_MODEL="$BASE_MODEL" \
      RUNS="$RUNS" \
      RUN_FILTER="$model_run_filter" \
      PROMPT_FILE="$PROMPT_FILE_GSM8K" \
      PROMPT_FIELD="$PROMPT_FIELD" \
      ANSWER_FIELD="$ANSWER_FIELD" \
      NUM_SAMPLES="$NUM_SAMPLES" \
      GRID_SIZE="$GRID_SIZE" \
      RADIUS="$RADIUS" \
      BATCH_SIZE="$BATCH_SIZE" \
      MAX_LENGTH="$MAX_LENGTH" \
      MASK_PROMPT="$MASK_PROMPT" \
      PRINCIPAL_RANK="$PRINCIPAL_RANK_SPECTRAL" \
      USE_LOWRANK="$USE_LOWRANK" \
      DIRECTION_SEED="$DIRECTION_SEED" \
      SHARED_DIRS="$SHARED_DIRS" \
      STEPS="$STEPS" \
      OUT_DIR="$out_dir_loss_spectral" \
      "$SCRIPTS_DIR/run_opra_loss_landscape_spectral.sh"
  fi
}

# Main execution: iterate over base models
if [[ -n "$RUN_FILTER" ]]; then
  # If RUN_FILTER is set, use legacy single-run behavior
  echo "[INFO] Using legacy single-run mode with RUN_FILTER=$RUN_FILTER"
  # Extract base model name from RUN_FILTER for output directory
  BASE_MODEL_NAME=$(echo "$RUN_FILTER" | sed 's/_[^_]*$//' | sed 's/_opra$//' | sed 's/\*$//')
  run_plots_for_model "$BASE_MODEL_NAME"
else
  # Iterate over all base models
  BASE_MODELS=$(get_base_models)
  echo "[INFO] Found base models:"
  echo "$BASE_MODELS" | while read -r m; do echo "  - $m"; done
  
  for base_model_name in $BASE_MODELS; do
    # Apply MODEL_FILTER if set
    if [[ -n "$MODEL_FILTER" ]] && [[ "$base_model_name" != *"$MODEL_FILTER"* ]]; then
      echo "[INFO] Skipping $base_model_name (does not match MODEL_FILTER=$MODEL_FILTER)"
      continue
    fi
    run_plots_for_model "$base_model_name"
  done
fi

echo "[INFO] All requested OPRA plots finished"
