#!/usr/bin/env bash
# Auto-evaluate LoRA adapters on gtn dev nodes (g1/g4), respecting 4-hour dev limits.
# This script runs as a controller on the login node and spawns dev sessions via qrsh.
# Worker logic runs on the dev node and executes evaluations directly (no qsub).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
EVAL_ALL_SCRIPT="${SCRIPT_DIR}/run_eval_lora_all.sh"

log(){ printf "[%(%F %T)T] %s\n" -1 "$*" >&2; }
need(){ command -v "$1" &>/dev/null || { echo "missing: $1"; exit 1; }; }

# ============================================================
# Shared defaults (align with run_eval_lora_all.sh where possible)
# ============================================================
PROJECT_NAME="${PROJECT_NAME:-OPRA}"
EXP_NAMES="${EXP_NAMES:-OPRA-K-ABLATION}" # OPRA-LoRA | OPRA-K-ABLATION
MODEL_PATH="${MODEL_PATH:-checkpoints}" # checkpoints | giil
MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-32}"
DEV_TARGET="${DEV_TARGET:-g1}" # auto | g4 | g1 (overrides DEV_JOB_ORDER when set to g4/g1)
PROMPT_TYPE="${PROMPT_TYPE:-think-boxed}"
MAX_TOKENS="${MAX_TOKENS:-3072}"
EVAL_GROUP1_DATASETS="${EVAL_GROUP1_DATASETS:-aime25x8,amc23x8,aime24x8}"
EVAL_GROUP2_DATASETS="${EVAL_GROUP2_DATASETS:-minerva_math,olympiadbench,math500}"
DEFAULT_EVAL_DATASETS="${EVAL_GROUP1_DATASETS},${EVAL_GROUP2_DATASETS}"
EVAL_DATASETS="${EVAL_DATASETS:-${DEFAULT_EVAL_DATASETS}}"
EVAL_STEPS="${EVAL_STEPS:-100,200,300,313}"

# Base model lookup paths (same as run_eval_lora_all.sh)
BASE_MODEL_ROOTS=(
  "/hss/giil/caixq/model"
)

# ============================================================
# Dev session settings
# ============================================================
G1_JOBCLASS_NORM="${G1_JOBCLASS_NORM:-gtn-container_g1_dev}"
G4_JOBCLASS_NORM="${G4_JOBCLASS_NORM:-gtn-container_g4_dev}"
AC_OPTS_G1="${AC_OPTS_G1:-d=nvcr-cuda-12.4.1-ubuntu22.04,d_shm=64g}"
AC_OPTS_G4="${AC_OPTS_G4:-d=nvcr-cuda-12.4.1-ubuntu22.04,d_shm=256g}"

DEV_MAX_HOURS="${DEV_MAX_HOURS:-4}"
DEV_GRACE_SECONDS="${DEV_GRACE_SECONDS:-300}"
DEV_WAIT_START_SEC="${DEV_WAIT_START_SEC:-60}"
DEV_COOLDOWN_SECONDS="${DEV_COOLDOWN_SECONDS:-30}"
DEV_MIN_REMAIN_SECONDS="${DEV_MIN_REMAIN_SECONDS:-300}"
DEV_JOB_ORDER="${DEV_JOB_ORDER:-g4,g1}" # order to try: g4,g1 or g1,g4

PROXY="${PROXY:-}"

# ============================================================
# Worker mode (runs on dev node)
# ============================================================
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

adjust_tp_for_model() {
  local model_path="$1"
  local desired_gpus="$2"
  if [[ "${FORCE_NUM_GPUS:-}" =~ ^[0-9]+$ ]]; then
    echo "${FORCE_NUM_GPUS}"
    return 0
  fi
  if [[ -z "$model_path" || -z "$desired_gpus" ]]; then
    echo "$desired_gpus"
    return 0
  fi
  local config_path="${model_path}/config.json"
  if [[ ! -f "$config_path" ]]; then
    echo "$desired_gpus"
    return 0
  fi
  local adjusted
  adjusted="$(CFG_PATH="$config_path" DESIRED="$desired_gpus" $PYTHON_BIN - <<'PY'
import json, os, sys
cfg_path = os.environ.get("CFG_PATH", "")
desired = int(os.environ.get("DESIRED", "1"))
try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
except Exception:
    print(desired)
    sys.exit(0)

heads = None
for key in ("num_attention_heads", "n_head", "num_heads", "n_heads"):
    if key in cfg:
        try:
            heads = int(cfg[key])
            break
        except Exception:
            pass
if not heads or heads <= 0:
    print(desired)
    sys.exit(0)

valid = [d for d in range(1, heads + 1) if heads % d == 0 and d <= desired]
if not valid:
    print(1)
else:
    print(max(valid))
PY
  )"
  echo "${adjusted:-$desired_gpus}"
}

check_group_complete() {
  local adapters="$1"
  LORA_ADAPTERS="${adapters}" \
  EVAL_DATASETS="${EVAL_DATASETS}" \
  EVAL_GROUP1_DATASETS="${EVAL_GROUP1_DATASETS}" \
  EVAL_GROUP2_DATASETS="${EVAL_GROUP2_DATASETS}" \
  OUT_ROOT="${OUT_ROOT}" \
  ${PYTHON_BIN} - <<'PY'
from pathlib import Path
import json
import os
import sys

def adapter_meta(path: str):
    p = Path(path)
    if p.name == "actor" and p.parent.name.startswith("global_step_"):
        step_name = p.parent.name
        run_name = p.parent.parent.name
    else:
        step_name = p.name
        run_name = p.parent.name
    safe_run_name = run_name.replace(".", "_").replace("-", "_")
    run_tag = f"{run_name}__{step_name}"
    return safe_run_name, run_tag

def group_idx(name: str) -> int:
    g1 = set(os.environ.get("EVAL_GROUP1_DATASETS", "").split(","))
    g2 = set(os.environ.get("EVAL_GROUP2_DATASETS", "").split(","))
    if name in g1:
        return 1
    if name in g2:
        return 2
    return 0

def _ensure_metrics_std(metrics_path: Path) -> bool:
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    acc_std = data.get("acc_std")
    total_acc_std = data.get("total_acc_std")
    pass_std = data.get("pass_at_k_std")
    has_pass = isinstance(data.get("pass_at_k_percent"), dict)
    need_std = (acc_std is None) or (total_acc_std is None) or (has_pass and pass_std is None)
    if not need_std:
        return True
    try:
        from tools.merge_results import compute_pass_at_k_std
    except Exception:
        return False
    try:
        acc_std, total_acc_std, pass_std = compute_pass_at_k_std(metrics_path)
    except Exception:
        return False
    data["acc_std"] = acc_std
    data["total_acc_std"] = total_acc_std
    if has_pass and pass_std is not None:
        data["pass_at_k_std"] = pass_std
    metrics_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    return True

adapters = [a.strip() for a in os.environ.get("LORA_ADAPTERS", "").split("|") if a.strip()]
datasets = [d.strip() for d in os.environ.get("EVAL_DATASETS", "").split(",") if d.strip()]
out_root = Path(os.environ.get("OUT_ROOT", "."))

missing = []
total = 0
for adapter in adapters:
    safe_run_name, run_tag = adapter_meta(adapter)
    for data_name in datasets:
        total += 1
        out_dir = out_root / safe_run_name / run_tag / f"g{group_idx(data_name)}" / data_name
        metrics_files = [p for p in out_dir.glob("*_metrics.json") if "_part" not in p.name]
        if not metrics_files:
            missing.append(str(out_dir))
            continue
        mpath = sorted(metrics_files)[0]
        try:
            data = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            missing.append(str(mpath))
            continue
        if not isinstance(data, dict) or ("acc" not in data and "num_samples" not in data):
            missing.append(str(mpath))
            continue
        has_pass = isinstance(data.get("pass_at_k_percent"), dict)
        if "acc_std" not in data or "total_acc_std" not in data or (has_pass and "pass_at_k_std" not in data):
            if not _ensure_metrics_std(mpath):
                missing.append(str(mpath))

if missing:
    print(f"[CHECK] Incomplete metrics: {len(missing)}/{total}")
    for item in missing[:5]:
        print(f"[CHECK] Missing: {item}")
    sys.exit(1)

print(f"[CHECK] All metrics complete: {total}/{total}")
sys.exit(0)
PY
}

guess_gpu_count_from_jc() {
  local jc="${1:-}"
  case "$jc" in
    *_g1*) echo 1 ;;
    *_g4*) echo 4 ;;
    *_g8*) echo 8 ;;
    *) echo 1 ;;
  esac
}

worker_time_left() {
  local now
  now="$(date +%s)"
  if [[ -z "${DEV_DEADLINE_EPOCH:-}" || ! "${DEV_DEADLINE_EPOCH}" =~ ^[0-9]+$ ]]; then
    echo ""
    return 0
  fi
  echo $((DEV_DEADLINE_EPOCH - now))
}

run_worker() {
  local start_ts deadline_ts
  start_ts="$(date +%s)"
  if [[ -n "${DEV_DEADLINE_EPOCH:-}" && "${DEV_DEADLINE_EPOCH}" =~ ^[0-9]+$ ]]; then
    deadline_ts="${DEV_DEADLINE_EPOCH}"
  elif [[ -n "${DEV_RUN_SECONDS:-}" && "${DEV_RUN_SECONDS}" =~ ^[0-9]+$ ]]; then
    deadline_ts=$((start_ts + DEV_RUN_SECONDS))
    export DEV_DEADLINE_EPOCH="${deadline_ts}"
  fi

  # Resolve REPO_ROOT (same logic as run_eval_lora_all.sh)
  local REPO_ROOT_LOCAL="${EVAL_REPO_ROOT:-${REPO_ROOT:-}}"
  if [[ -z "${REPO_ROOT_LOCAL}" ]]; then
    if [[ -f "${SCRIPT_DIR}/../tools/run_qwen_eval_all_shared.py" ]]; then
      REPO_ROOT_LOCAL="$(cd "${SCRIPT_DIR}/.." && pwd)"
    elif [[ -f "${PWD}/tools/run_qwen_eval_all_shared.py" ]]; then
      REPO_ROOT_LOCAL="$(pwd)"
    elif [[ -f "${HOME}/project/LLM_EVAL/tools/run_qwen_eval_all_shared.py" ]]; then
      REPO_ROOT_LOCAL="${HOME}/project/LLM_EVAL"
    elif [[ -n "${WORK_HOME:-}" && -f "${WORK_HOME}/project/LLM_EVAL/tools/run_qwen_eval_all_shared.py" ]]; then
      REPO_ROOT_LOCAL="${WORK_HOME}/project/LLM_EVAL"
    else
      REPO_ROOT_LOCAL="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
  fi
  if [[ -d "${REPO_ROOT_LOCAL}" ]]; then
    cd "${REPO_ROOT_LOCAL}"
  else
    log "[WARN] REPO_ROOT not found or not a directory: ${REPO_ROOT_LOCAL}"
  fi

  local LOG_ROOT_LOCAL="${EVAL_LOG_ROOT:-${LOG_ROOT:-${REPO_ROOT_LOCAL}}}"
  if [[ -n "${LOG_ROOT_LOCAL}" && ! -w "${LOG_ROOT_LOCAL}" ]]; then
    if [[ -n "${WORK_HOME:-}" && -w "${WORK_HOME}" ]]; then
      LOG_ROOT_LOCAL="${WORK_HOME}"
    elif [[ -w "${HOME}" ]]; then
      LOG_ROOT_LOCAL="${HOME}"
    else
      LOG_ROOT_LOCAL="${PWD}"
    fi
  fi
  local log_dir="${LOG_ROOT_LOCAL}/eval_log/eval_all/dev_auto"
  mkdir -p "${log_dir}"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local job_tag="${EVAL_DEV_JOB_TAG:-dev}"
  local log_file="${log_dir}/${ts}.${job_tag}.log"
  exec > >(tee -a "${log_file}") 2>&1

  log "Worker started (job=${job_tag}, deadline=${DEV_DEADLINE_EPOCH:-none})"

  export EVAL_GROUP1_DATASETS EVAL_GROUP2_DATASETS EVAL_DATASETS
  export EVAL_MP_START_METHOD=spawn
  export SYMBOLIC_TIMEOUT_MODE="${SYMBOLIC_TIMEOUT_MODE:-auto}"
  export SYMBOLIC_TIMEOUT="${SYMBOLIC_TIMEOUT:-1.0}"

  local MODEL_ROOT_LOCAL="${MODEL_ROOT:-}"
  if [[ -z "${MODEL_ROOT_LOCAL}" ]]; then
    if [[ "${MODEL_PATH}" == "checkpoints" ]]; then
      MODEL_ROOT_LOCAL="${REPO_ROOT_LOCAL}/../OPRA/${MODEL_PATH}/${EXP_NAMES}"
    else
      MODEL_ROOT_LOCAL="/data/${MODEL_PATH}/caixq/${EXP_NAMES}"
    fi
  fi

  if [[ ! -d "${MODEL_ROOT_LOCAL}" ]]; then
    log "[ERROR] MODEL_ROOT not found: ${MODEL_ROOT_LOCAL}"
    return 1
  fi

  OUT_ROOT="${OUT_ROOT:-${REPO_ROOT_LOCAL}/eval_results/${EXP_NAMES}_${PROMPT_TYPE}}"

  PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/eval/bin/python3}"
  TEMP_G1="${TEMP_G1:-0.6}"
  TEMP_G2="${TEMP_G2:-0.8}"
  NSAMP_G1="${NSAMP_G1:-${MAX_SAMPLE_NUMS}}"
  NSAMP_G2="${NSAMP_G2:-${MAX_SAMPLE_NUMS}}"

  if [[ -z "${PASS_AT_KS:-}" ]]; then
    local default_pass_ks=(1 8 16 32 64 128 256 512 1024 2048)
    local pass_ks=()
    local k
    for k in "${default_pass_ks[@]}"; do
      if (( k > 0 && k <= MAX_SAMPLE_NUMS )) && [[ " ${pass_ks[*]} " != *" $k "* ]]; then
        pass_ks+=("$k")
      fi
    done
    PASS_AT_KS=$(IFS=,; echo "${pass_ks[*]}")
  fi
  export PASS_AT_KS

  export TORCH_CPP_LOG_LEVEL=ERROR
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export PYTHONUNBUFFERED=1
  export VLLM_USE_FLASHINFER_SAMPLER=1
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

  local allowed_steps
  allowed_steps="$(parse_eval_steps_to_array "${EVAL_STEPS:-100,200,300,313}")"
  if [[ -n "${allowed_steps}" ]]; then
    log "[INFO] EVAL_STEPS filter: ${allowed_steps}"
  fi

  log "[INFO] Discovering LoRA adapters in ${MODEL_ROOT_LOCAL}..."
  local ADAPTER_LIST=()
  while IFS= read -r -d '' adapter_dir; do
    ADAPTER_LIST+=("$adapter_dir")
  done < <(find "${MODEL_ROOT_LOCAL}" -name "adapter_config.json" -printf '%h\0' | sort -zu)

  log "[INFO] Found ${#ADAPTER_LIST[@]} LoRA adapter checkpoints"
  if [[ ${#ADAPTER_LIST[@]} -eq 0 ]]; then
    log "[WARN] No LoRA adapters found, exiting"
    return 0
  fi

  declare -A ADAPTER_GROUPS
  declare -A GROUP_BASE_MODEL
  for adapter in "${ADAPTER_LIST[@]}"; do
    if [[ "$(basename "$adapter")" == *_vllm ]]; then
      log "[INFO] Skipping converted adapter dir: $adapter"
      continue
    fi

    local adapter_parent="$adapter"
    while [[ "$(basename "$adapter_parent")" == "actor" || "$(basename "$adapter_parent")" =~ ^global_step_ ]]; do
      adapter_parent="$(dirname "$adapter_parent")"
    done
    local adapter_type
    adapter_type="$(basename "$adapter_parent")"

    local step_num
    step_num=""
    if [[ "$adapter" =~ global_step_([0-9]+) ]]; then
      step_num="${BASH_REMATCH[1]}"
    fi
    if [[ -n "$step_num" ]]; then
      if ! step_in_filter "$step_num" "$allowed_steps"; then
        log "[INFO] Skipping $adapter (step $step_num not in EVAL_STEPS)"
        continue
      fi
    fi

    if [[ -f "${adapter}/adapter_config.json" ]]; then
      local peft_type
      peft_type="$(ADAPTER_CONFIG="${adapter}/adapter_config.json" $PYTHON_BIN - <<'PY'
import json
import os

cfg_path = os.environ.get("ADAPTER_CONFIG", "")
try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(str(cfg.get("peft_type", "")).upper())
except Exception:
    print("")
PY
)"
      if [[ "${peft_type}" == "OFT" ]]; then
        log "[WARN] Skipping OFT adapter (unsupported by vLLM LoRA): $adapter"
        continue
      fi
    fi

    local base_model
    base_model="$(find_base_model "$adapter_type" || true)"
    if [[ -z "$base_model" ]]; then
      log "[WARN] Skipping $adapter - cannot find base model for $adapter_type"
      continue
    fi

    local base_key group_key
    base_key="$(basename "$base_model")"
    group_key="${base_key}::${adapter_type}"

    if [[ -n "${ADAPTER_GROUPS[$group_key]:-}" ]]; then
      ADAPTER_GROUPS["$group_key"]="${ADAPTER_GROUPS[$group_key]}|${adapter}"
    else
      ADAPTER_GROUPS["$group_key"]="$adapter"
    fi
    GROUP_BASE_MODEL["$group_key"]="$base_model"
  done

  mapfile -t GROUP_KEYS < <(printf '%s\n' "${!ADAPTER_GROUPS[@]}" | sort)
  log "[INFO] Grouped into ${#GROUP_KEYS[@]} adapter type groups"

  local default_gpu_count
  default_gpu_count="${EVAL_DEV_GPU_COUNT:-}"
  if [[ -z "${default_gpu_count}" ]]; then
    default_gpu_count="$(guess_gpu_count_from_jc "${EVAL_DEV_JOBCLASS:-}")"
  fi

  local any_pending=0
  for group_key in "${GROUP_KEYS[@]}"; do
    local adapters base_model_path
    adapters="${ADAPTER_GROUPS[$group_key]}"
    base_model_path="${GROUP_BASE_MODEL[$group_key]}"
    if [[ -z "$adapters" || -z "$base_model_path" ]]; then
      continue
    fi

    if check_group_complete "$adapters"; then
      log "[INFO] ${group_key} complete"
      continue
    fi

    any_pending=1
    local tl
    tl="$(worker_time_left)"
    if [[ -n "$tl" && "$tl" -le "$DEV_MIN_REMAIN_SECONDS" ]]; then
      log "[INFO] Time left (${tl}s) too short, stop for re-queue"
      return 2
    fi

    local gpu_list=()
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      IFS=',' read -ra _GPU_RAW <<< "${CUDA_VISIBLE_DEVICES}"
      for g in "${_GPU_RAW[@]}"; do
        g="${g//[[:space:]]/}"
        [[ -n "$g" ]] && gpu_list+=("$g")
      done
    elif [[ -n "${default_gpu_count:-}" && "${default_gpu_count}" =~ ^[0-9]+$ ]]; then
      for ((i=0; i<default_gpu_count; i++)); do
        gpu_list+=("$i")
      done
    elif command -v nvidia-smi >/dev/null 2>&1; then
      local _gpu_count
      _gpu_count=$(nvidia-smi --list-gpus | grep -c '^GPU' || true)
      if [[ -z "${_gpu_count}" || "${_gpu_count}" -lt 1 ]]; then
        _gpu_count=1
      fi
      for ((i=0; i<_gpu_count; i++)); do
        gpu_list+=("$i")
      done
    else
      gpu_list=("0")
    fi

    local available_gpu_count="${#gpu_list[@]}"
    if (( available_gpu_count < 1 )); then
      gpu_list=("0")
      available_gpu_count=1
    fi

    local shard_count
    if [[ -n "${LORA_NUM_SHARDS:-}" ]]; then
      shard_count="${LORA_NUM_SHARDS}"
    else
      shard_count="${available_gpu_count}"
    fi
    if (( shard_count < 1 )); then
      shard_count=1
    fi
    if (( shard_count > available_gpu_count )); then
      shard_count="${available_gpu_count}"
    fi

    export ALLOCATED_GPUS="${available_gpu_count}"
    local tp_gpus
    tp_gpus="${EVAL_DEV_TP_GPUS:-1}"
    if [[ -z "${tp_gpus}" || ! "${tp_gpus}" =~ ^[0-9]+$ || "${tp_gpus}" -lt 1 ]]; then
      tp_gpus=1
    fi
    if (( tp_gpus > available_gpu_count )); then
      tp_gpus="${available_gpu_count}"
    fi
    tp_gpus="$(adjust_tp_for_model "$base_model_path" "$tp_gpus")"
    if [[ -z "${tp_gpus}" || ! "${tp_gpus}" =~ ^[0-9]+$ || "${tp_gpus}" -lt 1 ]]; then
      tp_gpus=1
    fi

    export TP_NUM_GPUS="${tp_gpus}"
    if (( tp_gpus > 1 )); then
      if [[ -n "${LORA_NUM_SHARDS:-}" && "${LORA_NUM_SHARDS}" != "1" ]]; then
        log "[WARN] TP_NUM_GPUS>1 requires LORA_NUM_SHARDS=1; overriding ${LORA_NUM_SHARDS} -> 1"
      fi
      export LORA_NUM_SHARDS=1
      export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpu_list[*]:0:${tp_gpus}}")"
    else
      export LORA_NUM_SHARDS="${shard_count}"
      export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpu_list[*]}")"
    fi

    # Dynamically set eval parallelism based on nproc/shards (unless user overrides)
    local total_cpus per_shard chunk_size
    if command -v nproc >/dev/null 2>&1; then
      total_cpus="$(nproc)"
    else
      total_cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
    fi
    if [[ -z "${total_cpus}" || ! "${total_cpus}" =~ ^[0-9]+$ ]]; then
      total_cpus=1
    fi
    per_shard=$(( total_cpus / LORA_NUM_SHARDS ))
    if (( per_shard < 1 )); then
      per_shard=1
    fi
    if [[ -z "${EVAL_MP_WORKERS:-}" ]]; then
      export EVAL_MP_WORKERS="${per_shard}"
    fi
    if [[ -z "${EVAL_THREAD_WORKERS:-}" ]]; then
      export EVAL_THREAD_WORKERS=1
    fi
    if [[ -z "${EVAL_MP_CHUNK_SIZE:-}" ]]; then
      chunk_size=$(( per_shard * 2 ))
      if (( chunk_size < 4 )); then
        chunk_size=4
      elif (( chunk_size > 128 )); then
        chunk_size=128
      fi
      export EVAL_MP_CHUNK_SIZE="${chunk_size}"
    fi

    log "[INFO] Eval parallelism: nproc=${total_cpus}, shards=${LORA_NUM_SHARDS}, per_shard=${per_shard} -> MP=${EVAL_MP_WORKERS}, THREADS=${EVAL_THREAD_WORKERS}, CHUNK=${EVAL_MP_CHUNK_SIZE}"
    log "[INFO] Running group ${group_key} with shards=${LORA_NUM_SHARDS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, TP=${TP_NUM_GPUS})"

    set +e
    local rc
    if command -v timeout >/dev/null 2>&1; then
      tl="$(worker_time_left)"
      if [[ -n "$tl" && "$tl" -gt "$DEV_MIN_REMAIN_SECONDS" ]]; then
        local per_job_timeout=$((tl - DEV_MIN_REMAIN_SECONDS))
        timeout --signal=TERM --kill-after=60 "${per_job_timeout}s" \
          env \
          RUN_LORA_EVAL_SUBMITTED=1 \
          NUM_GPUS="${TP_NUM_GPUS}" \
          BASE_MODEL_PATH="${base_model_path}" \
          LORA_ADAPTERS="${adapters}" \
          OUT_ROOT="${OUT_ROOT}" \
          PROMPT_TYPE="${PROMPT_TYPE}" \
          MAX_TOKENS="${MAX_TOKENS}" \
          MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS}" \
          TEMP_G1="${TEMP_G1}" \
          TEMP_G2="${TEMP_G2}" \
          NSAMP_G1="${NSAMP_G1}" \
          NSAMP_G2="${NSAMP_G2}" \
          "${EVAL_ALL_SCRIPT}"
        rc=$?
      else
        RUN_LORA_EVAL_SUBMITTED=1 \
          NUM_GPUS="${TP_NUM_GPUS}" \
          BASE_MODEL_PATH="${base_model_path}" \
          LORA_ADAPTERS="${adapters}" \
          OUT_ROOT="${OUT_ROOT}" \
          PROMPT_TYPE="${PROMPT_TYPE}" \
          MAX_TOKENS="${MAX_TOKENS}" \
          MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS}" \
          TEMP_G1="${TEMP_G1}" \
          TEMP_G2="${TEMP_G2}" \
          NSAMP_G1="${NSAMP_G1}" \
          NSAMP_G2="${NSAMP_G2}" \
          "${EVAL_ALL_SCRIPT}"
        rc=$?
      fi
    else
      RUN_LORA_EVAL_SUBMITTED=1 \
        NUM_GPUS="${TP_NUM_GPUS}" \
        BASE_MODEL_PATH="${base_model_path}" \
        LORA_ADAPTERS="${adapters}" \
        OUT_ROOT="${OUT_ROOT}" \
        PROMPT_TYPE="${PROMPT_TYPE}" \
        MAX_TOKENS="${MAX_TOKENS}" \
        MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS}" \
        TEMP_G1="${TEMP_G1}" \
        TEMP_G2="${TEMP_G2}" \
        NSAMP_G1="${NSAMP_G1}" \
        NSAMP_G2="${NSAMP_G2}" \
        "${EVAL_ALL_SCRIPT}"
      rc=$?
    fi
    set -e

    if [[ "$rc" -eq 124 ]]; then
      log "[INFO] Group run timed out, stop for re-queue"
      return 2
    fi
    if [[ "$rc" -ne 0 ]]; then
      log "[ERROR] Group run failed (rc=${rc})"
      return "$rc"
    fi
  done

  if [[ "$any_pending" -eq 0 ]]; then
    log "[INFO] All groups complete"
    return 0
  fi

  log "[INFO] Worker finished loop; pending groups remain"
  return 2
}

# ============================================================
# Controller mode (runs on login node)
# ============================================================
normalize_jclass(){
  local x="$1"
  x="${x%.default}"; x="${x%.24h}"; x="${x%.72h}"; x="${x%.168h}"
  echo "$x"
}

my_jobs(){
  qstat -u "$USER" 2>/dev/null \
   | awk '
     /^[[:space:]]*[0-9]+[[:space:]]/{
       jid=$1; name=$3; st=$5; line=$0;
       jc="";
       if (match(line, /[A-Za-z0-9_-]+-container_[^[:space:]]+/, m)) { jc=m[0]; }
       gsub(/\.(default|24h|72h|168h)$/, "", jc);
       printf "%s|%s|%s|%s\n", jid, name, st, jc;
     }'
}

qdel_by_name(){
  local name="$1"
  while IFS='|' read -r jid jname st jc; do
    [[ "$jname" == "$name" ]] || continue
    log "qdel $jid (name=$jname state=$st jc=$jc)"
    qdel "$jid" || true
  done < <(my_jobs)
}

run_on_dev() {
  local jc="$1"
  local gpus="$2"
  local ac_opts="$3"
  local tag="$4"

  local max_seconds
  if [[ -n "${DEV_MAX_SECONDS:-}" && "${DEV_MAX_SECONDS}" =~ ^[0-9]+$ ]]; then
    max_seconds="${DEV_MAX_SECONDS}"
  else
    max_seconds=$((DEV_MAX_HOURS*3600))
  fi
  local run_seconds=$((max_seconds - DEV_GRACE_SECONDS))
  if (( run_seconds < 600 )); then
    run_seconds=600
  fi
  local deadline
  deadline=$(( $(date +%s) + run_seconds ))

  local job_name="DEV_EVAL_${tag}_$(date +%H%M%S)"

  local env_prefix=""
  local pass_vars=(PROJECT_NAME EXP_NAMES MODEL_PATH MODEL_ROOT PROMPT_TYPE MAX_TOKENS MAX_SAMPLE_NUMS \
    EVAL_GROUP1_DATASETS EVAL_GROUP2_DATASETS EVAL_DATASETS EVAL_STEPS \
    REPO_ROOT EVAL_REPO_ROOT LOG_ROOT EVAL_LOG_ROOT WORK_HOME PYTHON_BIN \
    TOKENIZERS_PARALLELISM LORA_NUM_SHARDS FORCE_NUM_GPUS)
  local v
  for v in "${pass_vars[@]}"; do
    if [[ -n "${!v:-}" ]]; then
      env_prefix+="${v}=$(printf '%q' "${!v}") "
    fi
  done

  local cmd
  cmd="${env_prefix}DEV_EVAL_WORKER=1 EVAL_DEV_GPU_COUNT=${gpus} EVAL_DEV_JOBCLASS=${jc} EVAL_DEV_JOB_TAG=${job_name} DEV_RUN_SECONDS=${run_seconds} DEV_DEADLINE_EPOCH=${deadline} \"${SCRIPT_PATH}\""

  local http_env=()
  if [[ -n "${PROXY}" ]]; then
    http_env+=(-v "http_proxy=${PROXY},https_proxy=${PROXY},HTTP_PROXY=${PROXY},HTTPS_PROXY=${PROXY}")
  fi

  log "Try allocate ${jc} (gpus=${gpus}), allow queueing until start..."
  set +e
  qrsh -now n -pty n \
       -jc "${jc}" \
       -ac "${ac_opts}" \
       -N "${job_name}" \
       "${http_env[@]}" \
       bash -lc "${cmd}" &
  local qpid=$!
  set -e

  log "Waiting for ${job_name} to start and complete..."
  wait "$qpid"
  return $?
}

controller_main() {
  for c in qrsh qstat qdel awk grep sed; do need "$c"; done
  if [[ ! -x "${EVAL_ALL_SCRIPT}" ]]; then
    log "[ERROR] Missing eval script: ${EVAL_ALL_SCRIPT}"
    exit 1
  fi

  local order=()
  case "${DEV_TARGET}" in
    g4|G4)
      order=(g4)
      ;;
    g1|G1)
      order=(g1)
      ;;
    auto|AUTO|"")
      IFS=',' read -ra order <<< "${DEV_JOB_ORDER}"
      if [[ ${#order[@]} -eq 0 ]]; then
        order=(g4 g1)
      fi
      ;;
    *)
      log "[WARN] Unknown DEV_TARGET=${DEV_TARGET}, fallback to DEV_JOB_ORDER"
      IFS=',' read -ra order <<< "${DEV_JOB_ORDER}"
      if [[ ${#order[@]} -eq 0 ]]; then
        order=(g4 g1)
      fi
      ;;
  esac

  while :; do
    local tried=0
    local rc=0
    local token
    for token in "${order[@]}"; do
      case "$token" in
        g4)
          tried=1
          if run_on_dev "${G4_JOBCLASS_NORM}" 4 "${AC_OPTS_G4}" "g4"; then
            return 0
          else
            rc=$?
          fi
          ;;
        g1)
          tried=1
          if run_on_dev "${G1_JOBCLASS_NORM}" 1 "${AC_OPTS_G1}" "g1"; then
            return 0
          else
            rc=$?
          fi
          ;;
        *)
          ;;
      esac

      if [[ "${rc}" -eq 0 ]]; then
        return 0
      fi
      if [[ "${rc}" -ne 3 ]]; then
        log "dev session ended (rc=${rc}), retrying next round"
        sleep "${DEV_COOLDOWN_SECONDS}"
      fi
    done

    if (( tried == 0 )); then
      log "[ERROR] DEV_JOB_ORDER is empty or invalid: ${DEV_JOB_ORDER}"
      exit 1
    fi

    log "No dev session started this round; cooldown ${DEV_COOLDOWN_SECONDS}s then retry"
    sleep "${DEV_COOLDOWN_SECONDS}"
  done
}

if [[ "${DEV_EVAL_WORKER:-0}" == "1" ]]; then
  run_worker
  exit $?
fi

controller_main
