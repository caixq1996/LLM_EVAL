#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gtn-container_g8.24h
#$ -ac d=nvcr-cuda-12.4.1-ubuntu22.04,d_shm=256g
#$ -j y

set -x
set -e   # 如果希望有 worker 挂掉就整 job 失败，可以打开

ORIG_ARGS=("$@")

PROJECT_NAME="LLM_EVAL"
EXP_NAMES="${EXP_NAMES:-grpo_baselines}" # VI-CURL_deepscaler_diff | OPRA-LoRA
MODEL_PATH="${MODEL_PATH:-giil}" # giil | checkpoints
MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-1024}"
MAX_TOKENS="${MAX_TOKENS:-}"
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-3072}"
DEEPSEEK_MAX_TOKENS="${DEEPSEEK_MAX_TOKENS:-3072}"
MAX_JOBS="${MAX_JOBS:-8}"
D_SHM_DEFAULT="${D_SHM_DEFAULT:-256g}"
D_SHM_G1="${D_SHM_G1:-64g}"
D_SHM_G4="${D_SHM_G4:-${D_SHM_DEFAULT}}"
D_SHM_G8="${D_SHM_G8:-${D_SHM_DEFAULT}}"
EVAL_JOB_SCALE="${EVAL_JOB_SCALE:-g1}"
EVAL_G1_FAMILY_DEFAULT="${EVAL_G1_FAMILY_DEFAULT:-auto}"
QSUB_RUNTIME_REFRESH="${QSUB_RUNTIME_REFRESH:-false}"
export QSUB_RUNTIME_REFRESH

# 特殊 adapter 算法配置（需要特殊 base model 的算法）
# 格式: "algorithm1:suffix1,algorithm2:suffix2,..."
# 例如: "pissa:_pissa_base,qpissa:_qpissa_base"
export SPECIAL_ADAPTER_ALGORITHMS="${SPECIAL_ADAPTER_ALGORITHMS:-pissa:_pissa_base,qpissa:_qpissa_base}"

get_d_shm_for_jc() {
  local jc="$1"
  case "$jc" in
    *_g1) echo "${D_SHM_G1}" ;;
    *_g4) echo "${D_SHM_G4}" ;;
    *_g8) echo "${D_SHM_G8}" ;;
    *) echo "${D_SHM_DEFAULT}" ;;
  esac
}

is_deepseek_1_5b() {
  local name="$1"
  [[ -n "$name" ]] || return 1
  [[ "$name" == *DeepSeek-R1-Distill-Qwen-1.5B* || "$name" == *DeepSeek_R1_Distill_Qwen_1_5B* ]]
}

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

# -------------------------------------------------------
# Adaptive submit (use p90 scheduler on submit node)
#   - default: direct run
#   - submit: RUN_EVAL_SUBMIT=1 or --submit
# -------------------------------------------------------
# 解析命令行参数
KEEP_EXPORTED_HF="${KEEP_EXPORTED_HF:-false}"
RUN_EVAL_MULTI_SUBMIT="${RUN_EVAL_MULTI_SUBMIT:-false}"
MULTI_SUBMIT_BASE_ONCE="${MULTI_SUBMIT_BASE_ONCE:-true}"
RUN_EVAL_MULTI_SUBMIT_DEEP="${RUN_EVAL_MULTI_SUBMIT_DEEP:-false}"
RUN_EVAL_BASE_ONLY="${RUN_EVAL_BASE_ONLY:-false}"
RUN_EVAL_PLAN_ONLY="${RUN_EVAL_PLAN_ONLY:-false}"
FORCE_G1_FAMILY="${FORCE_G1_FAMILY:-}" # gtn, gtb, or gtn|gtb (auto-select within A100 g1)
EVAL_STEPS="${EVAL_STEPS:-313}" # 100,200,300,313
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)
      RUN_EVAL_SUBMIT=1
      shift
      ;;
    --keep-exported)
      KEEP_EXPORTED_HF="true"
      shift
      ;;
    --multi-submit)
      RUN_EVAL_MULTI_SUBMIT="true"
      shift
      ;;
    --no-base-once)
      MULTI_SUBMIT_BASE_ONCE="false"
      shift
      ;;
    --multi-submit-deep)
      RUN_EVAL_MULTI_SUBMIT_DEEP="true"
      shift
      ;;
    --base-only)
      RUN_EVAL_BASE_ONLY="true"
      shift
      ;;
    --force-g1-family)
      FORCE_G1_FAMILY="$2"
      shift 2
      ;;
    --steps)
      EVAL_STEPS="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
export EVAL_STEPS
export FORCE_G1_FAMILY
export EVAL_G1_FAMILY_DEFAULT
export RUN_EVAL_BASE_ONLY
EVAL_GEN_CHUNK_SIZE="${EVAL_GEN_CHUNK_SIZE:-128}"
export EVAL_GEN_CHUNK_SIZE
EVAL_REQUIRED_PASS_K="${EVAL_REQUIRED_PASS_K:-${MAX_SAMPLE_NUMS}}"
export EVAL_REQUIRED_PASS_K
EVAL_OVERWRITE="${EVAL_OVERWRITE:-false}"
export EVAL_OVERWRITE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_NAME="run_eval_all_tmux.sh"
LIB_PATH="${SCRIPT_DIR}/lib/run_eval_all_tmux_lib.sh"
if [[ ! -f "${LIB_PATH}" ]]; then
  declare -a _SCRIPT_DIR_CANDIDATES=()
  [[ -n "${RUN_EVAL_SCRIPT_DIR:-}" ]] && _SCRIPT_DIR_CANDIDATES+=("${RUN_EVAL_SCRIPT_DIR}")
  [[ -n "${PWD:-}" ]] && _SCRIPT_DIR_CANDIDATES+=("${PWD}/scripts")
  [[ -n "${REPO_ROOT:-}" ]] && _SCRIPT_DIR_CANDIDATES+=("${REPO_ROOT}/scripts")
  _SCRIPT_DIR_CANDIDATES+=("${HOME}/project/${PROJECT_NAME}/scripts")
  [[ -n "${WORK_HOME:-}" ]] && _SCRIPT_DIR_CANDIDATES+=("${WORK_HOME}/project/${PROJECT_NAME}/scripts")

  for _candidate_dir in "${_SCRIPT_DIR_CANDIDATES[@]}"; do
    [[ -n "${_candidate_dir}" ]] || continue
    if [[ -f "${_candidate_dir}/lib/run_eval_all_tmux_lib.sh" ]]; then
      SCRIPT_DIR="${_candidate_dir}"
      SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"
      LIB_PATH="${SCRIPT_DIR}/lib/run_eval_all_tmux_lib.sh"
      break
    fi
  done
  unset _candidate_dir
  unset _SCRIPT_DIR_CANDIDATES
fi
if [[ -f "${LIB_PATH}" ]]; then
  # shellcheck source=/home/caixq/project/LLM_EVAL/scripts/lib/run_eval_all_tmux_lib.sh
  source "${LIB_PATH}"
else
  echo "[ERROR] Missing helper library: ${LIB_PATH}"
  exit 1
fi
export RUN_EVAL_SCRIPT_DIR="${SCRIPT_DIR}"
export RUN_EVAL_SCRIPT_PATH="${SCRIPT_PATH}"

case "${RUN_EVAL_BOOTSTRAP_ONLY:-false}" in
  1|true|TRUE|yes|YES)
  echo "BOOTSTRAP_OK SCRIPT_DIR=${SCRIPT_DIR} LIB_PATH=${LIB_PATH} SCRIPT_PATH=${SCRIPT_PATH}"
  exit 0
  ;;
esac

# Launch tmux monitor unless already running in qsub or monitor mode.
if [[ -z "${RUN_EVAL_SUBMITTED:-}" && -z "${RUN_EVAL_MONITOR:-}" ]]; then
  if command -v tmux >/dev/null 2>&1; then
    _TMUX_SESSION="${TMUX_SESSION_NAME:-$(run_eval_default_tmux_session_name "${EXP_NAMES:-LLM_EVAL}" "${RUN_EVAL_BASE_ONLY}")}"
    tmux_cmd=(RUN_EVAL_MONITOR=1 "${SCRIPT_PATH}" "${ORIG_ARGS[@]}")
    tmux new-session -d -s "${_TMUX_SESSION}" "$(printf '%q ' "${tmux_cmd[@]}")"
    echo "[INFO] Started tmux session: ${_TMUX_SESSION}"
    echo "[INFO] Attach with: tmux attach -t ${_TMUX_SESSION}"
    exit 0
  else
    echo "[WARN] tmux not found; running monitor in current shell."
    RUN_EVAL_MONITOR=1
  fi
fi

REPO_ROOT="${EVAL_REPO_ROOT:-${REPO_ROOT:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
  if [[ -f "${SCRIPT_DIR}/../tools/run_qwen_eval_all_shared.py" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  elif [[ -f "${PWD}/tools/run_qwen_eval_all_shared.py" ]]; then
    REPO_ROOT="$(pwd)"
  elif [[ -f "${HOME}/project/LLM_EVAL/tools/run_qwen_eval_all_shared.py" ]]; then
    REPO_ROOT="${HOME}/project/LLM_EVAL"
  elif [[ -n "${WORK_HOME:-}" && -f "${WORK_HOME}/project/LLM_EVAL/tools/run_qwen_eval_all_shared.py" ]]; then
    REPO_ROOT="${WORK_HOME}/project/LLM_EVAL"
  else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  fi
fi
if [[ -d "${REPO_ROOT}" ]]; then
  cd "${REPO_ROOT}"
else
  echo "[WARN] REPO_ROOT not found or not a directory: ${REPO_ROOT}"
fi
LOG_ROOT="${EVAL_LOG_ROOT:-${LOG_ROOT:-${REPO_ROOT}}}"
if [[ -n "${LOG_ROOT}" && ! -w "${LOG_ROOT}" ]]; then
  if [[ -n "${WORK_HOME:-}" && -w "${WORK_HOME}" ]]; then
    LOG_ROOT="${WORK_HOME}"
  elif [[ -w "${HOME}" ]]; then
    LOG_ROOT="${HOME}"
  else
    LOG_ROOT="${PWD}"
  fi
fi

if [[ $RUN_EVAL_MULTI_SUBMIT_DEEP == "true" ]]; then
  RUN_EVAL_MULTI_SUBMIT="true"
  RUN_EVAL_SUBMIT=1
elif [[ $RUN_EVAL_MULTI_SUBMIT == "true" ]]; then
  RUN_EVAL_SUBMIT=1
fi

# =======================================================
# 日志目录设置（在所有 qsub 逻辑之前）
# =======================================================
export TZ='JST-9'
_TS="$(date +%Y%m%d_%H%M%S)"

# 判断日志目录：直接运行 vs qsub 提交后
if [[ -n "${RUN_EVAL_SUBMITTED:-}" ]]; then
  _LOG_BASE="${LOG_ROOT}/eval_log/eval_all/qsub_submit"
else
  _LOG_BASE="${LOG_ROOT}/eval_log/eval_all/main"
fi
mkdir -p "${_LOG_BASE}" "${LOG_ROOT}/eval_log/eval_all/eval_gpus"

# 主脚本日志文件
# 如果是 multi-submit 子任务，使用 SUB_EXP_NAME（具体算法名）
if [[ -n "${SUB_EXP_NAME:-}" ]]; then
  _EXP_TAG="${SUB_EXP_NAME}"
else
  _EXP_TAG="${EXP_NAMES:-default}"
fi
_MAIN_LOG="${_LOG_BASE}/run_eval_all.${_TS}.${_EXP_TAG}.log"
echo "[INFO] Main script log: ${_MAIN_LOG}"

# 使用 exec 和 tee 将输出同时打印到终端和日志文件
exec > >(tee -a "${_MAIN_LOG}") 2>&1

# =======================================================
# Auto submit + monitor (tmux)
#   - Always submit qsub jobs and keep monitoring until complete
#   - No submit/multi-submit flags needed
#   - Explicit submit modes are also monitored here once the tmux session starts
# =======================================================
if [[ -z "${RUN_EVAL_SUBMITTED:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/eval/bin/python3}"
  BASE_ROOT="${BASE_ROOT:-/hss/giil/caixq/model}"
  PROMPT_TYPE="${PROMPT_TYPE:-think-boxed}"

  # Resolve model root
  if [[ $MODEL_PATH == "giil" ]]; then
    _MODEL_ROOT="${MODEL_ROOT:-/data/giil/caixq/ckpts/${EXP_NAMES}}"
  else
    _MODEL_ROOT="${MODEL_ROOT:-$HOME/project/${PROJECT_NAME}/checkpoints/${EXP_NAMES}}"
  fi

  if [[ ! -d "$_MODEL_ROOT" ]]; then
    echo "[ERROR] MODEL_ROOT not found: $_MODEL_ROOT"
    exit 1
  fi

  # Output root (base); per-job out_root may append SUB_EXP_NAME if OUT_ROOT is not set
  BASE_OUT_ROOT="${OUT_ROOT:-$HOME/project/${PROJECT_NAME}/eval_results/${EXP_NAMES}_${PROMPT_TYPE}}"

  # Scheduler tool
  SCHEDULER_TOOL="${SCHEDULER_TOOL:-$HOME/tools/qsub_gpu_scheduler_p90.sh}"
  if [ -f "$SCHEDULER_TOOL" ]; then
    # shellcheck source=/home/caixq/tools/qsub_gpu_scheduler_p90.sh
    source "$SCHEDULER_TOOL"
  else
    echo "[WARN] Scheduler tool not found: $SCHEDULER_TOOL"
  fi

  MULTI_SUBMIT_SKIP_BASE_EVAL="${MULTI_SUBMIT_SKIP_BASE_EVAL:-true}"

  job_out_root() {
    local sub_exp_name="$1"
    if [[ -n "${OUT_ROOT:-}" ]]; then
      echo "${OUT_ROOT}"
      return
    fi
    if [[ -n "$sub_exp_name" ]]; then
      echo "${BASE_OUT_ROOT}/${sub_exp_name}"
    else
      echo "${BASE_OUT_ROOT}"
    fi
  }

  choose_max_tokens() {
    local name="$1"
    if [[ -n "${MAX_TOKENS}" ]]; then
      echo "${MAX_TOKENS}"
      return
    fi
    if is_deepseek_1_5b "$name"; then
      echo "${DEEPSEEK_MAX_TOKENS}"
    else
      echo "${DEFAULT_MAX_TOKENS}"
    fi
  }

  submit_qsub_eval() {
    local job_name="$1"
    local model_root="$2"
    local sub_exp_name="$3"
    local deep_step_filter="$4"
    local max_tokens="$5"
    local skip_base="$6"
    local skip_step="$7"
    local n_gpus="$8"
    local jc_full="$9"
    local d_shm="${10}"
    local submit_out
    submit_out="$(
      qsub -terse \
           -N "$job_name" \
           -jc "$jc_full" \
           -ac "d=nvcr-cuda-12.4.1-ubuntu22.04,d_shm=${d_shm}" \
           -v NUM_GPUS="${n_gpus}" \
           -v MODEL_ROOT="${model_root}" \
           -v EXP_NAMES="${EXP_NAMES}" \
           -v PROMPT_TYPE="${PROMPT_TYPE}" \
           -v SUB_EXP_NAME="${sub_exp_name}" \
           -v DEEP_STEP_FILTER="${deep_step_filter}" \
           -v MAX_TOKENS="${max_tokens}" \
           -v SKIP_BASE_EVAL="${skip_base}" \
           -v SKIP_STEP_EVAL="${skip_step}" \
           -v RUN_EVAL_BASE_ONLY="${RUN_EVAL_BASE_ONLY}" \
           -v RUN_EVAL_SUBMITTED=1 \
           -v KEEP_EXPORTED_HF="${KEEP_EXPORTED_HF}" \
           -V \
           "$SCRIPT_PATH"
    )"
    echo "[INFO] qsub accepted ${job_name} as job ${submit_out}"
  }

  job_complete() {
    local model_root="$1"
    local out_root="$2"
    local deep_step_filter="$3"
    local skip_base="$4"
    local skip_step="$5"
    local passive_check="false"
    if run_eval_is_truthy "${RUN_EVAL_SUBMIT:-0}"; then
      passive_check="true"
    fi

    PYTHONPATH="${REPO_ROOT}" \
    MODEL_ROOT="${model_root}" \
    OUT_ROOT="${out_root}" \
    BASE_ROOT="${BASE_ROOT}" \
    PROMPT_TYPE="${PROMPT_TYPE}" \
    EVAL_STEPS="${EVAL_STEPS}" \
    DEEP_STEP_FILTER="${deep_step_filter}" \
    SKIP_BASE_EVAL="${skip_base}" \
    SKIP_STEP_EVAL="${skip_step}" \
    RUN_EVAL_PASSIVE_CHECK="${passive_check}" \
    "$PYTHON_BIN" - <<'PY'
import os
import sys
import importlib
import re
from pathlib import Path

sys.path.insert(0, os.environ.get("PYTHONPATH", ""))

try:
    from tools.eval_result_state import check_missing_by_group
except Exception as exc:
    print(f"[WARN] import failed: {exc}")
    sys.exit(1)

def _filter_steps(step_dirs, steps_spec):
    if not steps_spec:
        return step_dirs
    wanted = set()
    for token in steps_spec.split(","):
        t = token.strip()
        if not t:
            continue
        if t.isdigit():
            wanted.add(f"global_step_{t}")
        else:
            wanted.add(t)
    if not wanted:
        return step_dirs
    return [p for p in step_dirs if p.name in wanted]

def _norm(s):
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def has_hf_weights(hf_dir):
    if not hf_dir or not hf_dir.exists():
        return False
    if list(hf_dir.glob('*.safetensors')):
        return True
    if list(hf_dir.glob('pytorch_model*.bin')) or (hf_dir / 'pytorch_model.bin.index.json').exists():
        return True
    return False

def find_base_model_dir(base_root, run_name):
    if not base_root or not base_root.exists():
        return None
    run_key = _norm(run_name)
    best = None
    for d in base_root.iterdir():
        if not d.is_dir():
            continue
        key = _norm(d.name)
        if not key:
            continue
        if key in run_key or run_key in key:
            if best is None or len(key) > len(_norm(best.name)):
                best = d
    return best

def _step_num_from_dir(step_dir):
    match = re.search(r'global_step_(\d+)$', step_dir.name)
    return int(match.group(1)) if match else -1

def list_step_dirs(run_dir, only_latest=False):
    step_dirs = [p for p in run_dir.glob('global_step_*') if p.is_dir()]
    if not step_dirs:
        return []
    if only_latest:
        return [max(step_dirs, key=_step_num_from_dir)]
    return sorted(step_dirs, key=_step_num_from_dir)

model_root = Path(os.environ.get("MODEL_ROOT", "."))
out_root = Path(os.environ.get("OUT_ROOT", "."))
base_root = Path(os.environ.get("BASE_ROOT", "/hss/giil/caixq/model"))
prompt_type = os.environ.get("PROMPT_TYPE", "think-boxed")
steps_spec = os.environ.get("EVAL_STEPS", "")
deep_step_filter = os.environ.get("DEEP_STEP_FILTER", "").strip()
if deep_step_filter:
    steps_spec = deep_step_filter
skip_base = os.environ.get("SKIP_BASE_EVAL", "false").lower() == "true"
skip_step = os.environ.get("SKIP_STEP_EVAL", "false").lower() == "true"
passive_check = os.environ.get("RUN_EVAL_PASSIVE_CHECK", "false").lower() == "true"
finalize_fn = None
if not passive_check:
    eval_state = importlib.import_module("tools.eval_result_state")
    finalize_fn = getattr(eval_state, "finalize" + "_run")

if not model_root.exists():
    print("INCOMPLETE missing MODEL_ROOT")
    sys.exit(1)

all_subdirs = [p for p in model_root.iterdir() if p.is_dir()]
is_single_run = any(p.name.startswith("global_step_") for p in all_subdirs)

if is_single_run:
    run_name_for_base = model_root.name
    runs = sorted([p for p in all_subdirs if p.name.startswith("global_step_")])
    runs = _filter_steps(runs, steps_spec)
else:
    run_name_for_base = None
    runs = sorted(all_subdirs)

if not runs:
    print("INCOMPLETE no runs")
    sys.exit(1)

base_done = {}
for run in runs:
    run_name = run.name
    lookup_name = run_name_for_base if run_name_for_base else run_name
    base_dir = find_base_model_dir(base_root, lookup_name)
    if base_dir is None or not has_hf_weights(base_dir):
        continue

    base_key = base_dir.name
    if not base_done.get(base_key, False):
        if not skip_base:
            if finalize_fn is not None:
                finalize_fn(out_root=out_root, run_name=f"base__{base_key}", prompt_type=prompt_type)
            missing = check_missing_by_group(out_root=out_root, run_name=f"base__{base_key}", final_required=True)
            if any(missing[g] for g in missing):
                print(f"INCOMPLETE base__{base_key}")
                sys.exit(1)
        base_done[base_key] = True

    if skip_step:
        continue

    if is_single_run:
        step_dirs = [run]
    else:
        step_dirs = list_step_dirs(run, only_latest=False)
        step_dirs = _filter_steps(step_dirs, steps_spec)

    if not step_dirs:
        print(f"INCOMPLETE no steps for {run_name}")
        sys.exit(1)

    for step_dir in step_dirs:
        safe_run_name = run_name.replace(".", "_").replace("-", "_")
        if is_single_run:
            tag = f"{safe_run_name}/{lookup_name}__{step_dir.name}"
        else:
            tag = f"{safe_run_name}/{run_name}__{step_dir.name}"
        if finalize_fn is not None:
            finalize_fn(out_root=out_root, run_name=tag, prompt_type=prompt_type)
        missing = check_missing_by_group(out_root=out_root, run_name=tag, final_required=True)
        if any(missing[g] for g in missing):
            print(f"INCOMPLETE {tag}")
            sys.exit(1)

print("COMPLETE")
sys.exit(0)
PY
  }

  QSTAT_NAME_TOOL="${QSTAT_NAME_TOOL:-$HOME/tools/qstat_name.sh}"
  declare -A RUNNING_JOBS
  refresh_running_jobs() {
    RUNNING_JOBS=()
    local names=""
    if [[ -x "$QSTAT_NAME_TOOL" ]]; then
      names="$("$QSTAT_NAME_TOOL" 2>/dev/null | awk 'NR>1 {print $NF}' || true)"
    else
      names="$(qstat 2>/dev/null | awk 'NR>2 {print $3}' || true)"
    fi
    while IFS= read -r name; do
      [[ -n "$name" ]] && RUNNING_JOBS["$name"]=1
    done <<< "$names"
    return 0
  }

  job_name_running() {
    local name="$1"
    [[ -n "${RUNNING_JOBS[$name]:-}" ]]
  }

  # Determine target jobs
  declare -a JOB_KEYS
  declare -A JOB_NAME
  declare -A JOB_MODEL_ROOT
  declare -A JOB_SUB_EXP
  declare -A JOB_DEEP_STEP
  declare -A JOB_SKIP_BASE
  declare -A JOB_SKIP_STEP
  declare -A JOB_MAX_TOKENS
  declare -A JOB_OUT_ROOT

  # detect single-run mode (global_step_* at top level)
  if find "$_MODEL_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' -print -quit | grep -q .; then
    SINGLE_RUN=1
  else
    SINGLE_RUN=0
  fi

  # Base-only job (optional)
  if run_eval_is_truthy "${RUN_EVAL_BASE_ONLY}" || [[ "${MULTI_SUBMIT_BASE_ONCE}" == "true" ]]; then
    base_job_name="$(run_eval_base_job_name "${PROJECT_NAME}" "${EXP_NAMES}" "${RUN_EVAL_BASE_ONLY}")"
    JOB_KEYS+=("BASE_ONLY")
    JOB_NAME["BASE_ONLY"]="${base_job_name}"
    JOB_MODEL_ROOT["BASE_ONLY"]="${_MODEL_ROOT}"
    JOB_SUB_EXP["BASE_ONLY"]=""
    JOB_DEEP_STEP["BASE_ONLY"]=""
    JOB_SKIP_BASE["BASE_ONLY"]="false"
    JOB_SKIP_STEP["BASE_ONLY"]="true"
    JOB_MAX_TOKENS["BASE_ONLY"]="${MAX_TOKENS:-${DEFAULT_MAX_TOKENS}}"
    JOB_OUT_ROOT["BASE_ONLY"]="$(job_out_root "")"
  else
    echo "[INFO] MULTI_SUBMIT_BASE_ONCE=false, skipping base-only job."
  fi

  if run_eval_is_truthy "${RUN_EVAL_BASE_ONLY}"; then
    echo "[INFO] RUN_EVAL_BASE_ONLY=true, only monitoring/submitting base-model eval."
  elif [[ "${RUN_EVAL_MULTI_SUBMIT_DEEP}" == "true" ]]; then
    ALLOWED_STEPS="$(parse_eval_steps_to_array "${EVAL_STEPS:-}")"
    if [[ -n "${ALLOWED_STEPS}" ]]; then
      echo "[INFO] Monitor EVAL_STEPS filter parsed: ${ALLOWED_STEPS}"
    fi
    while IFS= read -r -d '' stepdir; do
      step_name="$(basename "$stepdir")"
      step_num="${step_name#global_step_}"
      if ! step_in_filter "$step_num" "$ALLOWED_STEPS"; then
        continue
      fi

      parent_dir="$(dirname "$stepdir")"
      parent_name="$(basename "$parent_dir")"
      sub_rel="${parent_name}/${step_name}"
      sub_tag="${sub_rel//[^A-Za-z0-9_]/_}"
      sub_tag="${sub_tag:0:120}"
      job_tag="${sub_tag:0:60}"
      job_name="EVAL_${PROJECT_NAME}_${job_tag}"
      job_name="${job_name//[^A-Za-z0-9_]/_}"
      job_name="${job_name:0:120}"

      key="${sub_rel}"
      JOB_KEYS+=("$key")
      JOB_NAME["$key"]="$job_name"
      JOB_MODEL_ROOT["$key"]="$parent_dir"
      JOB_SUB_EXP["$key"]="$sub_tag"
      JOB_DEEP_STEP["$key"]="$step_name"
      JOB_SKIP_BASE["$key"]="${MULTI_SUBMIT_SKIP_BASE_EVAL}"
      JOB_SKIP_STEP["$key"]="false"
      JOB_MAX_TOKENS["$key"]="$(choose_max_tokens "$parent_name")"
      JOB_OUT_ROOT["$key"]="$(job_out_root "$sub_tag")"
    done < <(find "$_MODEL_ROOT" -type d -name 'global_step_*' -print0 | sort -zV)
  elif [[ "${SINGLE_RUN}" -eq 1 ]]; then
    # Deep-like mode: submit per global_step_* under a single run dir
    while IFS= read -r -d '' stepdir; do
      step_name="$(basename "$stepdir")"
      parent_dir="${_MODEL_ROOT}"
      parent_name="$(basename "$parent_dir")"
      sub_rel="${parent_name}/${step_name}"
      sub_tag="${sub_rel//[^A-Za-z0-9_]/_}"
      sub_tag="${sub_tag:0:120}"
      job_tag="${sub_tag:0:60}"
      job_name="EVAL_${PROJECT_NAME}_${job_tag}"
      job_name="${job_name//[^A-Za-z0-9_]/_}"
      job_name="${job_name:0:120}"

      key="${sub_rel}"
      JOB_KEYS+=("$key")
      JOB_NAME["$key"]="$job_name"
      JOB_MODEL_ROOT["$key"]="$parent_dir"
      JOB_SUB_EXP["$key"]="$sub_tag"
      JOB_DEEP_STEP["$key"]="$step_name"
      JOB_SKIP_BASE["$key"]="${MULTI_SUBMIT_SKIP_BASE_EVAL}"
      JOB_SKIP_STEP["$key"]="false"
      JOB_MAX_TOKENS["$key"]="$(choose_max_tokens "$parent_name")"
      JOB_OUT_ROOT["$key"]="$(job_out_root "$sub_tag")"
    done < <(find "$_MODEL_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' -print0 | sort -zV)
  else
    # Multi-run mode: submit per run directory
    while IFS= read -r -d '' subdir; do
      sub_name="$(basename "$subdir")"
      sub_rel="$sub_name"
      sub_tag="${sub_rel//[^A-Za-z0-9_]/_}"
      sub_tag="${sub_tag:0:120}"
      job_tag="${sub_tag:0:60}"
      job_name="EVAL_${PROJECT_NAME}_${job_tag}"
      job_name="${job_name//[^A-Za-z0-9_]/_}"
      job_name="${job_name:0:120}"

      key="${sub_rel}"
      JOB_KEYS+=("$key")
      JOB_NAME["$key"]="$job_name"
      JOB_MODEL_ROOT["$key"]="$subdir"
      JOB_SUB_EXP["$key"]="$sub_tag"
      JOB_DEEP_STEP["$key"]=""
      if [[ "${MULTI_SUBMIT_SKIP_BASE_EVAL}" == "true" ]]; then
        JOB_SKIP_BASE["$key"]="true"
      else
        JOB_SKIP_BASE["$key"]="false"
      fi
      JOB_SKIP_STEP["$key"]="false"
      JOB_MAX_TOKENS["$key"]="$(choose_max_tokens "$sub_name")"
      JOB_OUT_ROOT["$key"]="$(job_out_root "$sub_tag")"
    done < <(find "$_MODEL_ROOT" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
  fi

  if [[ ${#JOB_KEYS[@]} -eq 0 ]]; then
    echo "[WARN] No subdirectories found under $_MODEL_ROOT."
    exit 0
  fi

  if run_eval_is_truthy "${RUN_EVAL_PLAN_ONLY}"; then
    echo "PLAN_TMUX_SESSION $(run_eval_default_tmux_session_name "${EXP_NAMES:-LLM_EVAL}" "${RUN_EVAL_BASE_ONLY}")"
    echo "PLAN_JOB_COUNT ${#JOB_KEYS[@]}"
    for key in "${JOB_KEYS[@]}"; do
      echo "PLAN_JOB ${key} ${JOB_NAME[$key]} ${JOB_SKIP_BASE[$key]} ${JOB_SKIP_STEP[$key]} ${JOB_OUT_ROOT[$key]}"
    done
    exit 0
  fi

  poll_interval="${QSTAT_POLL_INTERVAL:-60}"
  refresh_running_jobs

  echo "[INFO] Precheck: submitting unfinished jobs without running jobs"
  for key in "${JOB_KEYS[@]}"; do
    job_name="${JOB_NAME[$key]}"
    model_root="${JOB_MODEL_ROOT[$key]}"
    sub_exp="${JOB_SUB_EXP[$key]}"
    deep_step="${JOB_DEEP_STEP[$key]}"
    skip_base="${JOB_SKIP_BASE[$key]}"
    skip_step="${JOB_SKIP_STEP[$key]}"
    max_tokens="${JOB_MAX_TOKENS[$key]}"
    out_root="${JOB_OUT_ROOT[$key]}"

    if job_complete "$model_root" "$out_root" "$deep_step" "$skip_base" "$skip_step"; then
      echo "[INFO] ${job_name} complete (precheck)"
      continue
    fi
    if job_name_running "$job_name"; then
      echo "[INFO] ${job_name} already running (precheck)"
      continue
    fi

    read -r jc_base jc_full n_gpus d_shm_val < <(resolve_eval_resources "$PROJECT_NAME" "$job_name")

    echo "[INFO] Precheck submit ${job_name} (MODEL_ROOT=${model_root}, STEP=${deep_step:-all})"
    submit_qsub_eval "$job_name" "$model_root" "$sub_exp" "$deep_step" "$max_tokens" "$skip_base" "$skip_step" "$n_gpus" "$jc_full" "$d_shm_val" >/dev/null
  done

  while true; do
    all_done=1
    refresh_running_jobs
    for key in "${JOB_KEYS[@]}"; do
      job_name="${JOB_NAME[$key]}"
      model_root="${JOB_MODEL_ROOT[$key]}"
      sub_exp="${JOB_SUB_EXP[$key]}"
      deep_step="${JOB_DEEP_STEP[$key]}"
      skip_base="${JOB_SKIP_BASE[$key]}"
      skip_step="${JOB_SKIP_STEP[$key]}"
      max_tokens="${JOB_MAX_TOKENS[$key]}"
      out_root="${JOB_OUT_ROOT[$key]}"

      if job_complete "$model_root" "$out_root" "$deep_step" "$skip_base" "$skip_step"; then
        continue
      fi

      all_done=0
      if job_name_running "$job_name"; then
        continue
      fi

      read -r jc_base jc_full n_gpus d_shm_val < <(resolve_eval_resources "$PROJECT_NAME" "$job_name")

      echo "[INFO] Submitting ${job_name} (MODEL_ROOT=${model_root}, STEP=${deep_step:-all})"
      submit_qsub_eval "$job_name" "$model_root" "$sub_exp" "$deep_step" "$max_tokens" "$skip_base" "$skip_step" "$n_gpus" "$jc_full" "$d_shm_val" >/dev/null
    done

    if [[ "$all_done" -eq 1 ]]; then
      echo "[INFO] All jobs complete"
      break
    fi

    sleep "$poll_interval"
  done

  exit 0
fi

# =======================================================
# Multi-submit mode: 为 MODEL_ROOT 下每个子目录单独提交 qsub
# =======================================================
if [[ -z "${JOB_ID:-}" && -z "${RUN_EVAL_SUBMITTED:-}" && "${RUN_EVAL_MULTI_SUBMIT:-false}" == "true" && "${RUN_EVAL_SUBMIT:-0}" == "1" ]]; then
  # 需要先确定 MODEL_ROOT
  if [[ $MODEL_PATH == "giil" ]]; then
    _MODEL_ROOT="${MODEL_ROOT:-/data/giil/caixq/ckpts/${EXP_NAMES}}"
  else
    _MODEL_ROOT="${MODEL_ROOT:-/home/caixq/project/${PROJECT_NAME}/checkpoints/${EXP_NAMES}}"
  fi
  
  if [[ ! -d "$_MODEL_ROOT" ]]; then
    echo "[ERROR] MODEL_ROOT not found: $_MODEL_ROOT"
    exit 1
  fi
  
  # 解析 EVAL_STEPS 为有效的 step 数字列表
  # 支持格式: "100,200,300" 或 "100-300:50" 或 "100-300"
  parse_eval_steps_to_array() {
    local spec="$1"
    local result=()
    if [[ -z "$spec" ]]; then
      echo ""
      return
    fi
    # 处理范围格式 "start-end:step" 或 "start-end"
    if [[ "$spec" =~ ^([0-9]+)-([0-9]+)(:([0-9]+))?$ ]]; then
      local start="${BASH_REMATCH[1]}"
      local end="${BASH_REMATCH[2]}"
      local step="${BASH_REMATCH[4]:-1}"
      for ((i=start; i<=end; i+=step)); do
        result+=("$i")
      done
    else
      # 处理逗号分隔格式 "100,200,300"
      IFS=',' read -ra tokens <<< "$spec"
      for t in "${tokens[@]}"; do
        t="${t//[^0-9]/}"  # 只保留数字
        if [[ -n "$t" ]]; then
          result+=("$t")
        fi
      done
    fi
    echo "${result[*]}"
  }
  
  # 检查 step 数字是否在允许列表中
  step_in_filter() {
    local step_num="$1"
    local -a allowed=($2)
    if [[ ${#allowed[@]} -eq 0 ]]; then
      return 0  # 无过滤器，允许所有
    fi
    for a in "${allowed[@]}"; do
      if [[ "$step_num" == "$a" ]]; then
        return 0
      fi
    done
    return 1
  }
  
  # 查找要提交的子目录
  SUBDIRS=()
  DEEP_STEP_NAMES=()  # 用于 multi-submit-deep 记录每个 step 的名称
  if [[ "${RUN_EVAL_MULTI_SUBMIT_DEEP}" == "true" ]]; then
    # 解析 EVAL_STEPS 过滤器
    ALLOWED_STEPS="$(parse_eval_steps_to_array "${EVAL_STEPS:-}")"
    if [[ -n "$ALLOWED_STEPS" ]]; then
      echo "[INFO] EVAL_STEPS filter parsed: $ALLOWED_STEPS"
    fi
    
    # 深入到每个 global_step_* 目录单独提交
    while IFS= read -r -d '' stepdir; do
      step_name="$(basename "$stepdir")"
      # 提取 step 数字
      step_num="${step_name#global_step_}"
      if step_in_filter "$step_num" "$ALLOWED_STEPS"; then
        SUBDIRS+=("$stepdir")
        DEEP_STEP_NAMES+=("$step_name")
      else
        echo "[INFO] Skipping $step_name (not in EVAL_STEPS filter)"
      fi
    done < <(find "$_MODEL_ROOT" -type d -name 'global_step_*' -print0 | sort -zV)
  fi
  if [[ ${#SUBDIRS[@]} -eq 0 ]]; then
    while IFS= read -r -d '' dir; do
      SUBDIRS+=("$dir")
    done < <(find "$_MODEL_ROOT" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
  fi
  
  if [[ ${#SUBDIRS[@]} -eq 0 ]]; then
    echo "[WARN] No subdirectories found under $_MODEL_ROOT, falling back to single submit."
  else
    SCHEDULER_TOOL="${SCHEDULER_TOOL:-$HOME/tools/qsub_gpu_scheduler_p90.sh}"
    if [ ! -f "$SCHEDULER_TOOL" ]; then
      echo "[ERROR] Scheduler tool not found: $SCHEDULER_TOOL"
      exit 1
    fi
    source "$SCHEDULER_TOOL"
    
    echo "[INFO] Multi-submit mode: found ${#SUBDIRS[@]} subdirectories under $_MODEL_ROOT"

    if [[ "${MULTI_SUBMIT_BASE_ONCE}" == "true" ]]; then
      base_job_tag="${EXP_NAMES//[^A-Za-z0-9_]/_}"
      base_job_tag="${base_job_tag:0:60}"
      base_job_name="EVAL_${PROJECT_NAME}_${base_job_tag}_BASE"
      base_job_name="${base_job_name//[^A-Za-z0-9_]/_}"
      base_job_name="${base_job_name:0:120}"
      read -r base_jc_base base_n_gpus < <(select_resources_for_job "$PROJECT_NAME" "$base_job_name")
      base_jc_full="$(full_jclass_from_base "$base_jc_base")"
      base_d_shm="$(get_d_shm_for_jc "$base_jc_base")"
      base_max_tokens="${MAX_TOKENS:-${DEFAULT_MAX_TOKENS}}"
      echo "[INFO] Submitting base-only eval job: ${base_job_name} (MODEL_ROOT=${_MODEL_ROOT})"
      qsub -N "$base_job_name" \
           -jc "$base_jc_full" \
           -ac "d=nvcr-cuda-12.4.1-ubuntu22.04,d_shm=${base_d_shm}" \
           -v NUM_GPUS="${base_n_gpus}" \
           -v MODEL_ROOT="${_MODEL_ROOT}" \
           -v EXP_NAMES="${EXP_NAMES}" \
           -v MAX_TOKENS="${base_max_tokens}" \
           -v RUN_EVAL_SUBMITTED=1 \
           -v SKIP_STEP_EVAL=true \
           -v SKIP_BASE_EVAL=false \
           -v RUN_EVAL_MULTI_SUBMIT=false \
           -v KEEP_EXPORTED_HF="${KEEP_EXPORTED_HF}" \
           -V \
           "$SCRIPT_PATH"
    else
      echo "[INFO] MULTI_SUBMIT_BASE_ONCE=false, skipping base-only job."
    fi
    
    for i in "${!SUBDIRS[@]}"; do
      subdir="${SUBDIRS[$i]}"
      sub_name="$(basename "$subdir")"
      
      # 对于 multi-submit-deep，subdir 是 global_step_* 目录
      # 需要使用其父目录作为 MODEL_ROOT，并传递 step 名称作为过滤器
      if [[ "${RUN_EVAL_MULTI_SUBMIT_DEEP}" == "true" ]]; then
        step_name="$sub_name"  # e.g., global_step_100
        parent_dir="$(dirname "$subdir")"  # 父目录（算法目录）
        parent_name="$(basename "$parent_dir")"
        sub_rel="${parent_name}/${step_name}"
        actual_model_root="$parent_dir"
        deep_step_filter="$step_name"
      else
        sub_rel="$sub_name"
        actual_model_root="$subdir"
        deep_step_filter=""
      fi
      
      sub_tag="${sub_rel//[^A-Za-z0-9_]/_}"
      sub_tag="${sub_tag:0:120}"
      job_tag="$sub_tag"
      job_tag="${job_tag:0:60}"
      job_name="EVAL_${PROJECT_NAME}_${job_tag}"
      job_name="${job_name//[^A-Za-z0-9_]/_}"
      job_name="${job_name:0:120}"

      if [[ -n "${MAX_TOKENS}" ]]; then
        sub_max_tokens="${MAX_TOKENS}"
      else
        sub_max_tokens="${DEFAULT_MAX_TOKENS}"
        if is_deepseek_1_5b "$sub_rel" || is_deepseek_1_5b "$sub_name" || is_deepseek_1_5b "$parent_name"; then
          sub_max_tokens="${DEEPSEEK_MAX_TOKENS}"
        fi
      fi

      if [[ "${RUN_EVAL_MULTI_SUBMIT_DEEP}" == "true" ]]; then
        _prev_allow_g1="${P90_ALLOW_G1:-}"
        P90_ALLOW_G1=1
      fi
      read -r jc_base n_gpus < <(select_resources_for_job "$PROJECT_NAME" "$job_name")
      if [[ "${RUN_EVAL_MULTI_SUBMIT_DEEP}" == "true" ]]; then
        if [[ -n "${_prev_allow_g1}" ]]; then
          P90_ALLOW_G1="${_prev_allow_g1}"
        else
          unset P90_ALLOW_G1
        fi
        unset _prev_allow_g1
      fi
      jc_full="$(full_jclass_from_base "$jc_base")"
      d_shm_val="$(get_d_shm_for_jc "$jc_base")"
      
      echo "[INFO] Submitting: ${job_name} (MODEL_ROOT=${actual_model_root}, DEEP_STEP_FILTER=${deep_step_filter:-none})"
      qsub -N "$job_name" \
           -jc "$jc_full" \
           -ac "d=nvcr-cuda-12.4.1-ubuntu22.04,d_shm=${d_shm_val}" \
           -v NUM_GPUS="${n_gpus}" \
           -v MODEL_ROOT="${actual_model_root}" \
           -v EXP_NAMES="${EXP_NAMES}" \
           -v SUB_EXP_NAME="${sub_tag}" \
           -v DEEP_STEP_FILTER="${deep_step_filter}" \
           -v MAX_TOKENS="${sub_max_tokens}" \
           -v RUN_EVAL_SUBMITTED=1 \
           -v KEEP_EXPORTED_HF="${KEEP_EXPORTED_HF}" \
           -V \
           "$SCRIPT_PATH"
    done
    
    echo "[INFO] Submitted ${#SUBDIRS[@]} jobs."
    exit 0
  fi
fi

# =======================================================
# Single-submit mode
# =======================================================
if [[ -z "${JOB_ID:-}" && -z "${RUN_EVAL_SUBMITTED:-}" && "${RUN_EVAL_SUBMIT:-0}" == "1" ]]; then
  SCHEDULER_TOOL="${SCHEDULER_TOOL:-$HOME/tools/qsub_gpu_scheduler_p90.sh}"
  if [ ! -f "$SCHEDULER_TOOL" ]; then
    echo "[ERROR] Scheduler tool not found: $SCHEDULER_TOOL"
    exit 1
  fi
  # shellcheck source=/home/caixq/tools/qsub_gpu_scheduler_p90.sh
  source "$SCHEDULER_TOOL"

  job_tag="${EXP_NAMES//[^A-Za-z0-9_]/_}"
  job_tag="${job_tag:0:60}"
  job_name="EVAL_${PROJECT_NAME}_${job_tag}"
  job_name="${job_name//[^A-Za-z0-9_]/_}"
  job_name="${job_name:0:120}"

  read -r jc_base n_gpus < <(select_resources_for_job "$PROJECT_NAME" "$job_name")
  jc_full="$(full_jclass_from_base "$jc_base")"
  d_shm_val="$(get_d_shm_for_jc "$jc_base")"

  echo "[INFO] Submitting eval job: name=${job_name} jc=${jc_full} n_gpus=${n_gpus}"
  qsub -N "$job_name" \
       -jc "$jc_full" \
       -ac "d=nvcr-cuda-12.4.1-ubuntu22.04,d_shm=${d_shm_val}" \
       -v NUM_GPUS="${n_gpus}" \
       -v RUN_EVAL_SUBMITTED=1 \
       -V \
       "$SCRIPT_PATH"
  exit 0
fi

# 这里是 base 模型所在的「根目录」，里面有很多模型子目录
BASE_ROOT="${BASE_ROOT:-/hss/giil/caixq/model}"

PROMPT_TYPE="${PROMPT_TYPE:-think-boxed}"

# 1. 自动探测 GPU 数量
GPU_LIST=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _GPU_RAW <<< "${CUDA_VISIBLE_DEVICES}"
  for g in "${_GPU_RAW[@]}"; do
    g="${g//[[:space:]]/}"
    [[ -n "$g" ]] && GPU_LIST+=("$g")
  done
  NUM_GPUS="${#GPU_LIST[@]}"
  [[ "${NUM_GPUS}" -ge 1 ]] || NUM_GPUS=1
else
  if [[ -z "${NUM_GPUS:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      NUM_GPUS=$(nvidia-smi --list-gpus | grep -c '^GPU')
      [[ "${NUM_GPUS}" -ge 1 ]] || NUM_GPUS=1
    else
      NUM_GPUS=1
    fi
  fi
  for ((i=0; i<NUM_GPUS; i++)); do
    GPU_LIST+=("$i")
  done
fi

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/eval/bin/python3}"

# 如果是 multi-submit 模式下的子任务，SUB_EXP_NAME 表示具体算法名
# 输出路径将是 ${EXP_NAMES}_${PROMPT_TYPE}/${SUB_EXP_NAME}/
if [[ -n "${SUB_EXP_NAME:-}" ]]; then
  OUT_ROOT="${OUT_ROOT:-$HOME/project/${PROJECT_NAME}/eval_results/${EXP_NAMES}_${PROMPT_TYPE}/${SUB_EXP_NAME}}"
else
  OUT_ROOT="${OUT_ROOT:-$HOME/project/${PROJECT_NAME}/eval_results/${EXP_NAMES}_${PROMPT_TYPE}}"
fi

# MODEL_ROOT 应该是「这一堆 checkpoint 的根目录」，下面有多个 run 目录
# 例如: /data/.../ckpts/noise_rlvr_llama-3.2-3B-Instruct/<run_name>/global_step_xxx
if [[ $MODEL_PATH == "giil" ]]; then
  MODEL_ROOT="${MODEL_ROOT:-/data/giil/caixq/ckpts/${EXP_NAMES}}"
else
  MODEL_ROOT="${MODEL_ROOT:-$HOME/project/${PROJECT_NAME}/checkpoints/${EXP_NAMES}}"
fi

# Default MAX_TOKENS: 1024 normally, 2048 for DeepSeek-R1-Distill-Qwen-1.5B
if [[ -z "${MAX_TOKENS}" ]]; then
  MAX_TOKENS="${DEFAULT_MAX_TOKENS}"
  if is_deepseek_1_5b "${MODEL_ROOT}" || is_deepseek_1_5b "${EXP_NAMES}" || is_deepseek_1_5b "${SUB_EXP_NAME:-}"; then
    MAX_TOKENS="${DEEPSEEK_MAX_TOKENS}"
  else
    if [[ -d "${MODEL_ROOT}" ]]; then
      if find "${MODEL_ROOT}" -maxdepth 1 -mindepth 1 -type d \( -name '*DeepSeek-R1-Distill-Qwen-1.5B*' -o -name '*DeepSeek_R1_Distill_Qwen_1_5B*' \) | head -n 1 | grep -q .; then
        MAX_TOKENS="${DEEPSEEK_MAX_TOKENS}"
      fi
    fi
  fi
fi

# Multi-submit subjobs: skip base eval by default to avoid duplication.
MULTI_SUBMIT_SKIP_BASE_EVAL="${MULTI_SUBMIT_SKIP_BASE_EVAL:-true}"
if [[ -n "${SUB_EXP_NAME:-}" && "${MULTI_SUBMIT_SKIP_BASE_EVAL}" == "true" ]]; then
  if [[ -z "${SKIP_BASE_EVAL+x}" ]]; then
    SKIP_BASE_EVAL=true
  fi
fi
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-false}"
# Skip global_step evaluation; useful for base-only runs.
SKIP_STEP_EVAL="${SKIP_STEP_EVAL:-false}"

if run_eval_is_truthy "${RUN_EVAL_BASE_ONLY}"; then
  SKIP_BASE_EVAL=false
  SKIP_STEP_EVAL=true
fi

# 导出目录（与 Python 代码中的 EXPORT_ROOT 一致）
EXPORT_ROOT="${EXPORT_ROOT:-${WORK_HOME:-/data/giil/caixq}/export}"

# Worker 超时设置（秒），防止单个 worker 卡死。默认 8 小时
WORKER_TIMEOUT="${WORKER_TIMEOUT:-28800}"

# 使用前面已设置的时间戳
TS="${_TS}"

echo "[INFO] Job started at ${TS}. Detected ${NUM_GPUS} GPUs."

# Optional cap on concurrent GPU workers
if [[ -n "${MAX_JOBS}" ]]; then
  if [[ "${MAX_JOBS}" =~ ^[0-9]+$ && "${MAX_JOBS}" -ge 1 ]]; then
    if [[ "${MAX_JOBS}" -lt "${NUM_GPUS}" ]]; then
      NUM_GPUS="${MAX_JOBS}"
      GPU_LIST=("${GPU_LIST[@]:0:${NUM_GPUS}}")
      echo "[INFO] MAX_JOBS set. Capping workers to ${NUM_GPUS} GPU(s)."
    fi
  else
    echo "[WARN] MAX_JOBS is not a positive integer: ${MAX_JOBS}. Ignoring."
  fi
fi

TEMP_G1="${TEMP_G1:-0.6}"
TEMP_G2="${TEMP_G2:-0.8}"
NSAMP_G1="${NSAMP_G1:-${MAX_SAMPLE_NUMS}}"
NSAMP_G2="${NSAMP_G2:-${MAX_SAMPLE_NUMS}}"

export EVAL_ONE_MODEL_TIMEOUT="${EVAL_ONE_MODEL_TIMEOUT:-21600}"

# PASS@k 列表（受 MAX_SAMPLE_NUMS 限制）
if [[ -z "${PASS_AT_KS:-}" ]]; then
  default_pass_ks=(1 8 16 32 64 128 256 512 1024 2048)
  # default_pass_ks=(1 8)
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
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=1

# =======================================================
# 2. 评测：单机多卡 -> 多进程，每个进程 1 GPU + 1 shard
# =======================================================

base_args=(
  --model_root "$MODEL_ROOT"
  --out_root "$OUT_ROOT"
  --prompt_type "$PROMPT_TYPE"
  --max_tokens_per_call "$MAX_TOKENS"
  --base_root "$BASE_ROOT"
  --use_vllm
  --pipeline_parallel_size 1
  --vllm_batch_size 0
  --temperature_g1 "$TEMP_G1"
  --temperature_g2 "$TEMP_G2"
  --n_sampling_g1 "$NSAMP_G1"
  --n_sampling_g2 "$NSAMP_G2"
  # 注意：cleanup_exported 会在脚本结束时统一处理，避免多个 shard 抢着删模型
)

if [[ -n "${EVAL_STEPS}" ]]; then
  base_args+=( --steps "$EVAL_STEPS" )
fi

if [ "$SKIP_BASE_EVAL" = "true" ]; then
  base_args+=( --skip_base_eval )
fi
if [ "$SKIP_STEP_EVAL" = "true" ]; then
  base_args+=( --skip_step_eval )
fi

pids=()
start_times=()
failed_pids=()

for ((i=0; i<NUM_GPUS; i++)); do
    gpu_id="${GPU_LIST[$i]}"
    # 使用 _EXP_TAG 保持日志命名一致（multi-submit 时为算法名）
    LOG_FILE="${LOG_ROOT}/eval_log/eval_all/eval_gpus/gpu_worker.${TS}.${_EXP_TAG}.rank_${i}.log"
    echo "[INFO] Starting Worker $i/$NUM_GPUS on GPU $gpu_id... Log: $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$gpu_id "$PYTHON_BIN" -u tools/run_qwen_eval_all_shared.py \
      "${base_args[@]}" \
      --nproc 1 \
      --shard_id "$i" \
      --num_shards "$NUM_GPUS" \
      > "$LOG_FILE" 2>&1 &

    pids+=($!)
    start_times+=($(date +%s))
    sleep 10
done

echo "[INFO] All workers started. Monitoring with timeout=${WORKER_TIMEOUT}s..."

# 改进的等待逻辑：支持超时检测和进程监控
check_interval=60  # 每 60 秒检查一次
while true; do
    all_done=true
    current_time=$(date +%s)
    
    for ((i=0; i<${#pids[@]}; i++)); do
        pid=${pids[$i]}
        # 检查进程是否还在运行
        if kill -0 "$pid" 2>/dev/null; then
            all_done=false
            elapsed=$((current_time - start_times[$i]))
            
            # 检查是否超时
            if [[ "$elapsed" -ge "$WORKER_TIMEOUT" ]]; then
                echo "[WARN] Worker $i (PID $pid) exceeded timeout (${elapsed}s >= ${WORKER_TIMEOUT}s). Killing..."
                # 先尝试优雅终止
                kill -TERM "$pid" 2>/dev/null || true
                sleep 5
                # 强制终止
                kill -9 "$pid" 2>/dev/null || true
                failed_pids+=("$pid")
            fi
        fi
    done
    
    if $all_done; then
        break
    fi
    
    sleep "$check_interval"
done

# 收集所有子进程的退出状态
exit_status=0
for ((i=0; i<${#pids[@]}; i++)); do
    pid=${pids[$i]}
    if ! wait "$pid" 2>/dev/null; then
        echo "[WARN] Worker $i (PID $pid) exited with non-zero status"
        exit_status=1
    fi
done

if [[ ${#failed_pids[@]} -gt 0 ]]; then
    echo "[WARN] The following workers were terminated due to timeout: ${failed_pids[*]}"
    exit_status=1
fi

echo "[INFO] All eval workers finished."

if [[ "$exit_status" -ne 0 ]]; then
    echo "[ERROR] One or more workers failed. Skip merge."
    exit "$exit_status"
fi

# =======================================================
# 3. 合并结果 (Merge Shards)
# =======================================================

echo "[INFO] Tasks completed. Starting merge process..."

cd "${REPO_ROOT}"  # 确保还在 repo 根目录（有 tools/merge_results.py）

echo "[INFO] Finalizing merged results recursively under $OUT_ROOT ..."
"$PYTHON_BIN" tools/eval_result_state.py \
  finalize \
  --out-root "$OUT_ROOT" \
  --prompt-type "$PROMPT_TYPE"

# =======================================================
# 4. 清理导出的 HF 模型（可选）
# =======================================================

if [ "$KEEP_EXPORTED_HF" != "true" ]; then
    echo "[INFO] Cleaning up exported HF models in $EXPORT_ROOT ..."
    if [ -d "$EXPORT_ROOT" ]; then
        # 只删除本次评测可能生成的目录（根据 EXP_NAMES 模式匹配）
        find "$EXPORT_ROOT" -maxdepth 1 -type d -name "*${EXP_NAMES}*" -exec rm -rf {} + 2>/dev/null || true
        echo "[INFO] Exported HF models cleaned."
    fi
else
    echo "[INFO] KEEP_EXPORTED_HF=true, exported HF models retained in $EXPORT_ROOT"
fi

echo "[INFO] Done at $(date). All workers finished and results merged."
