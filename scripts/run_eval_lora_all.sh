#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gtn-container_g1.24h
#$ -ac d=nvcr-cuda-12.4.1-ubuntu22.04,d_shm=128g
#$ -j y

# Multi-LoRA Evaluation Script
# Evaluates multiple LoRA adapters sharing the same base model efficiently.
#
# Usage:
#   MODEL_ROOT=/path/to/checkpoints ./scripts/run_eval_lora_all.sh

# Only enable debug output if DEBUG=1
[[ "${DEBUG:-0}" == "1" ]] && set -x
set -e

PROJECT_NAME="${PROJECT_NAME:-OPRA}"
EXP_NAMES="${EXP_NAMES:-OPRA-LoRA}" # OPRA-LoRA | OPRA-K-ABLATION
MODEL_PATH="${MODEL_PATH:-checkpoints}" # checkpoints | giil
MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-1024}"
PROMPT_TYPE="${PROMPT_TYPE:-think-boxed}"
MAX_TOKENS="${MAX_TOKENS:-3072}"
EVAL_GROUP1_DATASETS="${EVAL_GROUP1_DATASETS:-aime25x8,amc23x8,aime24x8}"
EVAL_GROUP2_DATASETS="${EVAL_GROUP2_DATASETS:-minerva_math,olympiadbench,math500}"
DEFAULT_EVAL_DATASETS="${EVAL_GROUP1_DATASETS},${EVAL_GROUP2_DATASETS}"
EVAL_DATASETS="${EVAL_DATASETS:-${DEFAULT_EVAL_DATASETS}}"
export EVAL_GROUP1_DATASETS EVAL_GROUP2_DATASETS EVAL_DATASETS
export EVAL_MP_START_METHOD=spawn
export SYMBOLIC_TIMEOUT_MODE="${SYMBOLIC_TIMEOUT_MODE:-auto}"
export SYMBOLIC_TIMEOUT="${SYMBOLIC_TIMEOUT:-1.0}"
export EVAL_ITEM_TIMEOUT="${EVAL_ITEM_TIMEOUT:-5}"
EVAL_BASE_FT_ONLY="${EVAL_BASE_FT_ONLY:-0}"
FULL_FT_ROOT="${FULL_FT_ROOT:-/data/giil/caixq/ckpts/VI-CURL_deepscaler_diff}"
FULL_FT_PREFIX="${FULL_FT_PREFIX:-ver_rule_grpo_nocurl_}"
FULL_FT_OUT_ROOT="${FULL_FT_OUT_ROOT:-}"
FULL_FT_MODEL_FILTER="${FULL_FT_MODEL_FILTER:-}"

# ========================================
# Configuration
# ========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
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
if [[ -z "${FULL_FT_OUT_ROOT}" ]]; then
  FULL_FT_OUT_ROOT="${REPO_ROOT}/eval_results/VI-CURL_deepscaler_diff_${PROMPT_TYPE}"
fi

# Base model lookup paths
BASE_MODEL_ROOTS=(
    "/hss/giil/caixq/model"
)

# Output directories (align with run_eval_all.sh)
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_results/${EXP_NAMES}_${PROMPT_TYPE}}"

export TZ='JST-9'
_TS="$(date +%Y%m%d_%H%M%S)"

# 判断日志目录：直接运行 vs qsub 提交后
if [[ -n "${RUN_EVAL_SUBMITTED:-}" ]]; then
  _LOG_BASE="${LOG_ROOT}/eval_log/eval_all/qsub_submit"
else
  _LOG_BASE="${LOG_ROOT}/eval_log/eval_all/main"
fi
mkdir -p "${_LOG_BASE}" "${LOG_ROOT}/eval_log/eval_all/lora_jobs"
LOG_DIR="${LOG_ROOT}/eval_log/eval_all/lora_jobs"

# GPU settings
NUM_GPUS="${NUM_GPUS:-1}"
# Force tensor-parallel size to 1 regardless of allocated GPUs
TP_NUM_GPUS="${TP_NUM_GPUS:-1}"
FORCE_NUM_GPUS="${FORCE_NUM_GPUS:-}"
export FORCE_NUM_GPUS

# Allow single GPU evaluation
if [[ -z "${P90_ALLOW_G1:-}" ]]; then
  if [[ "${MAX_SAMPLE_NUMS}" =~ ^[0-9]+$ ]] && (( MAX_SAMPLE_NUMS >= 512 )); then
    P90_ALLOW_G1=0
  else
    P90_ALLOW_G1=1
  fi
fi
export P90_ALLOW_G1
EVAL_STEPS="${EVAL_STEPS:-100,200,300,313}"
export EVAL_STEPS
OPRA_K_FILTERS="${OPRA_K_FILTERS:-"Qwen2.5-math-1.5B:k16;Llama-3.2-3B-Instruct:k16;DeepSeek-R1-Distill-Qwen-1.5B:k16"}" # e.g. "Qwen2.5-math-1.5B:k0,k8;Llama-3.2-3B-Instruct:k0"
export OPRA_K_FILTERS
EVAL_NON_OPRA_MAX_STEP_ONLY="${EVAL_NON_OPRA_MAX_STEP_ONLY:-1}"
EVAL_MP_MAX_TOTAL="${EVAL_MP_MAX_TOTAL:-64}"
EVAL_ALGO_BLACKLIST="${EVAL_ALGO_BLACKLIST:-qlora,qpissa}"

if [[ -z "${EVAL_GEN_CHUNK_SIZE:-}" ]]; then
  if [[ "${MAX_SAMPLE_NUMS}" =~ ^[0-9]+$ ]] && (( MAX_SAMPLE_NUMS >= 1024 )); then
    EVAL_GEN_CHUNK_SIZE=8
  elif [[ "${MAX_SAMPLE_NUMS}" =~ ^[0-9]+$ ]] && (( MAX_SAMPLE_NUMS >= 512 )); then
    EVAL_GEN_CHUNK_SIZE=16
  else
    EVAL_GEN_CHUNK_SIZE=64
  fi
fi
EVAL_GEN_TARGET_OUTPUTS="${EVAL_GEN_TARGET_OUTPUTS:-8192}"
export EVAL_GEN_CHUNK_SIZE EVAL_GEN_TARGET_OUTPUTS

# Shared memory configuration for job classes
D_SHM_G1="${D_SHM_G1:-64g}"
D_SHM_G4="${D_SHM_G4:-256g}"
D_SHM_G8="${D_SHM_G8:-256g}"
D_SHM_DEFAULT="${D_SHM_DEFAULT:-256g}"

get_d_shm_for_jc() {
  local jc="$1"
  case "$jc" in
    *_g1) echo "${D_SHM_G1}" ;;
    *_g4) echo "${D_SHM_G4}" ;;
    *_g8) echo "${D_SHM_G8}" ;;
    *) echo "${D_SHM_DEFAULT}" ;;
  esac
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

algo_in_blacklist() {
    local algo="${1,,}"
    local list="${EVAL_ALGO_BLACKLIST:-}"
    if [[ -z "$list" ]]; then
        return 1
    fi
    local item
    IFS=',' read -ra _items <<< "$list"
    for item in "${_items[@]}"; do
        item="${item//[[:space:]]/}"
        [[ -z "$item" ]] && continue
        if [[ "$algo" == *"${item,,}"* ]]; then
            return 0
        fi
    done
    return 1
}

parse_opra_k_filters() {
    local spec="$1"
    if [[ -z "$spec" ]]; then
        return 0
    fi
    IFS=';' read -ra pairs <<< "$spec"
    for pair in "${pairs[@]}"; do
        pair="${pair//[[:space:]]/}"
        [[ -z "$pair" ]] && continue
        local model="${pair%%:*}"
        local ks="${pair#*:}"
        if [[ -z "$model" || "$ks" == "$pair" ]]; then
            echo "[WARN] Invalid OPRA_K_FILTERS entry (expect Model:k0,k1): $pair"
            continue
        fi
        local -a k_list=()
        IFS=',' read -ra tokens <<< "$ks"
        for t in "${tokens[@]}"; do
            t="${t//[^0-9]/}"
            if [[ -n "$t" ]]; then
                k_list+=("$t")
            fi
        done
        if [[ ${#k_list[@]} -eq 0 ]]; then
            echo "[WARN] OPRA_K_FILTERS entry has no valid k values: $pair"
            continue
        fi
        local uniq=()
        local seen=""
        for k in "${k_list[@]}"; do
            if [[ " $seen " != *" $k "* ]]; then
                uniq+=("$k")
                seen+=" $k"
            fi
        done
        OPRA_K_FILTERS_MAP["$model"]="$(printf '%s ' "${uniq[@]}")"
    done
}

k_in_filter() {
    local k_val="$1"
    local -a allowed=($2)
    if [[ ${#allowed[@]} -eq 0 ]]; then
        return 0
    fi
    for a in "${allowed[@]}"; do
        if [[ "$k_val" == "$a" ]]; then
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

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/eval/bin/python3}"

TEMP_G1="${TEMP_G1:-0.6}"
TEMP_G2="${TEMP_G2:-0.8}"
NSAMP_G1="${NSAMP_G1:-${MAX_SAMPLE_NUMS}}"
NSAMP_G2="${NSAMP_G2:-${MAX_SAMPLE_NUMS}}"

# PASS@k 列表（受 MAX_SAMPLE_NUMS 限制）
if [[ -z "${PASS_AT_KS:-}" ]]; then
  default_pass_ks=(1 8 16 32 64 128 256 512 1024 2048)
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
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Launch tmux monitor unless already running in qsub or monitor mode
if [[ -z "${RUN_LORA_EVAL_SUBMITTED:-}" && -z "${RUN_LORA_EVAL_MONITOR:-}" && "${EVAL_BASE_FT_ONLY}" != "1" ]]; then
  if command -v tmux >/dev/null 2>&1; then
    if [[ "${EVAL_BASE_FT_ONLY}" == "1" ]]; then
      _TMUX_SESSION="BASE_FT_ONLY_${EXP_NAMES}"
    else
      _TMUX_SESSION="${EXP_NAMES}"
    fi
    tmux new-session -d -s "${_TMUX_SESSION}" "RUN_LORA_EVAL_MONITOR=1 ${SCRIPT_PATH}"
    echo "[INFO] Started tmux session: ${_TMUX_SESSION}"
    echo "[INFO] Attach with: tmux attach -t ${_TMUX_SESSION}"
    exit 0
  else
    echo "[WARN] tmux not found; running monitor in current shell."
    RUN_LORA_EVAL_MONITOR=1
  fi
fi

# Already submitted as qsub job?
if [[ -n "${RUN_LORA_EVAL_SUBMITTED:-}" ]]; then
    echo "[INFO] Running multi-LoRA evaluation..."
    NUM_GPUS="${TP_NUM_GPUS}"
    export NUM_GPUS
    echo "[INFO] Forcing tensor_parallel_size to ${NUM_GPUS} (TP_NUM_GPUS)"

    if [[ -n "${JOB_ID:-}" ]]; then
      JOB_LOG_FILE="${LOG_DIR}/${_TS}.${JOB_NAME}.job${JOB_ID}.log"
      exec > >(tee -a "${JOB_LOG_FILE}") 2>&1
      echo "[INFO] Job log (with ID): ${JOB_LOG_FILE}"
    fi
    
    cd "$REPO_ROOT"

    if [[ -n "${LORA_ADAPTERS:-}" ]]; then
        set +e
        LORA_ADAPTERS="${LORA_ADAPTERS}" \
        EVAL_DATASETS="${EVAL_DATASETS}" \
        EVAL_GROUP1_DATASETS="${EVAL_GROUP1_DATASETS}" \
        EVAL_GROUP2_DATASETS="${EVAL_GROUP2_DATASETS}" \
        NSAMP_G1="${NSAMP_G1:-}" \
        NSAMP_G2="${NSAMP_G2:-}" \
        MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-}" \
        EVAL_NUM_TEST_SAMPLE="${EVAL_NUM_TEST_SAMPLE:-}" \
        EVAL_DATA_DIR="${EVAL_DATA_DIR:-}" \
        OUT_ROOT="${OUT_ROOT}" \
        $PYTHON_BIN - <<'PY'
from pathlib import Path
import json
import os

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

def _resolve_data_dir(out_root: Path) -> Path:
    env_dir = os.environ.get("EVAL_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    try:
        parts = out_root.resolve().parts
        if "eval_results" in parts:
            base = Path(*parts[:parts.index("eval_results")])
            candidate = base / "data"
            if candidate.exists():
                return candidate
    except Exception:
        pass
    return Path("./data")

def _count_dataset_samples(data_dir: Path, data_name: str, split: str = "test"):
    jsonl_path = data_dir / data_name / f"{split}.jsonl"
    if jsonl_path.exists():
        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return None
    json_path = data_dir / data_name / f"{split}.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return len(data)
        except Exception:
            return None
    return None

def _expected_samples(data_name: str, data_dir: Path) -> int | None:
    raw = os.environ.get("EVAL_NUM_TEST_SAMPLE", "")
    num = None
    if raw.strip():
        try:
            num = int(raw)
        except ValueError:
            num = None
    if num is not None and num <= 0:
        num = None
    ds_count = _count_dataset_samples(data_dir, data_name)
    if ds_count is None:
        return num
    if num is None:
        return ds_count
    return min(num, ds_count)

def _parse_int(val):
    try:
        return int(val)
    except Exception:
        return None

def _target_n_for_dataset(data_name: str) -> int | None:
    nsamp_g1 = _parse_int(os.environ.get("NSAMP_G1") or os.environ.get("MAX_SAMPLE_NUMS") or "")
    nsamp_g2 = _parse_int(os.environ.get("NSAMP_G2") or os.environ.get("MAX_SAMPLE_NUMS") or "")
    g = group_idx(data_name)
    if g == 1:
        return nsamp_g1
    if g == 2:
        return nsamp_g2
    return nsamp_g1 or nsamp_g2

def _iter_jsonl_files(out_dir: Path):
    files = sorted(out_dir.glob("*.jsonl"))
    if not files:
        return []
    non_part = [p for p in files if "_part" not in p.name]
    return non_part if non_part else files

def _count_jsonl_samples(out_dir: Path) -> int:
    files = _iter_jsonl_files(out_dir)
    if not files:
        return 0
    seen = set()
    count = 0
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and "idx" in obj:
                        idx = obj.get("idx")
                        if idx in seen:
                            continue
                        seen.add(idx)
                        count += 1
                    else:
                        count += 1
        except Exception:
            continue
    return count

def _min_sample_len(out_dir: Path) -> int:
    files = _iter_jsonl_files(out_dir)
    if not files:
        return 0
    by_idx = {}
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    idx_val = obj.get("idx")
                    if idx_val is None:
                        continue
                    sample_len = 0
                    for key in ("pred", "code", "score"):
                        val = obj.get(key)
                        if isinstance(val, list):
                            sample_len = len(val)
                            break
                    prev = by_idx.get(idx_val, 0)
                    if sample_len > prev:
                        by_idx[idx_val] = sample_len
        except Exception:
            continue
    if not by_idx:
        return 0
    return min(by_idx.values())

def _find_jsonl(metrics_path: Path) -> Path | None:
    parent = metrics_path.parent
    jsonl = list(parent.glob("*.jsonl"))
    if not jsonl:
        return None
    jsonl.sort()
    return jsonl[0]

def _load_score_lists(jsonl_path: Path):
    import json
    try:
        return [json.loads(line).get("score", []) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []

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
    jsonl_path = _find_jsonl(metrics_path)
    if not jsonl_path:
        return False
    score_mat = _load_score_lists(jsonl_path)
    if not score_mat:
        return False
    try:
        from evaluate import _compute_sample_std_fields
        pass_keys = list((data.get("pass_at_k_percent") or {}).keys())
        acc_std, total_std, pass_std = _compute_sample_std_fields(
            score_mat=score_mat,
            pass_at_k_keys=pass_keys,
            decimals=1,
        )
    except Exception:
        return False
    if acc_std is None and total_std is None and pass_std is None:
        return False
    if acc_std is not None:
        data["acc_std"] = acc_std
    if total_std is not None:
        data["total_acc_std"] = total_std
    if pass_std is not None:
        data["pass_at_k_std"] = pass_std
    metrics_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    return True

adapters = [a.strip() for a in os.environ.get("LORA_ADAPTERS", "").split("|") if a.strip()]
datasets = [d.strip() for d in os.environ.get("EVAL_DATASETS", "").split(",") if d.strip()]
out_root = Path(os.environ.get("OUT_ROOT", "."))
data_dir = _resolve_data_dir(out_root)

missing = []
total = 0
for adapter in adapters:
    safe_run_name, run_tag = adapter_meta(adapter)
    for data_name in datasets:
        total += 1
        out_dir = out_root / safe_run_name / run_tag / f"g{group_idx(data_name)}" / data_name
        expected = _expected_samples(data_name, data_dir)
        if expected is not None and expected > 0:
            have = _count_jsonl_samples(out_dir)
            if have < expected:
                missing.append(f"{out_dir} (jsonl {have}/{expected})")
                continue
        target_n = _target_n_for_dataset(data_name)
        if target_n is not None and target_n > 0:
            min_len = _min_sample_len(out_dir)
            if min_len < target_n:
                missing.append(f"{out_dir} (n={min_len}<{target_n})")
                continue
        metrics_files = [p for p in out_dir.glob("*_metrics.json") if "_part" not in p.name]
        if not metrics_files:
            missing.append(str(out_dir))
            continue
        mpath = sorted(metrics_files)[0]
        if not _ensure_metrics_std(mpath):
            missing.append(str(mpath))
            continue

if not missing and total > 0:
    print("[INFO] All metrics already complete; skip evaluation.")
    raise SystemExit(0)
raise SystemExit(1)
PY
        precheck_rc=$?
        set -e
        if [[ "$precheck_rc" -eq 0 ]]; then
            exit 0
        fi
    fi

    GPU_LIST=()
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -ra _GPU_RAW <<< "${CUDA_VISIBLE_DEVICES}"
        for g in "${_GPU_RAW[@]}"; do
            g="${g//[[:space:]]/}"
            [[ -n "$g" ]] && GPU_LIST+=("$g")
        done
    elif [[ -n "${ALLOCATED_GPUS:-}" && "${ALLOCATED_GPUS}" =~ ^[0-9]+$ ]]; then
        for ((i=0; i<ALLOCATED_GPUS; i++)); do
            GPU_LIST+=("$i")
        done
    elif command -v nvidia-smi >/dev/null 2>&1; then
        _GPU_COUNT=$(nvidia-smi --list-gpus | grep -c '^GPU' || true)
        if [[ -z "${_GPU_COUNT}" || "${_GPU_COUNT}" -lt 1 ]]; then
            _GPU_COUNT=1
        fi
        for ((i=0; i<_GPU_COUNT; i++)); do
            GPU_LIST+=("$i")
        done
    else
        GPU_LIST=("0")
    fi

    if [[ -n "${LORA_NUM_SHARDS:-}" ]]; then
        SHARD_COUNT="${LORA_NUM_SHARDS}"
    else
        SHARD_COUNT="${#GPU_LIST[@]}"
    fi
    if (( SHARD_COUNT < 1 )); then
        SHARD_COUNT=1
    fi
    if (( SHARD_COUNT > ${#GPU_LIST[@]} )); then
        SHARD_COUNT=${#GPU_LIST[@]}
    fi
    if [[ -z "${NUM_GPUS:-}" || "${NUM_GPUS}" == "0" ]]; then
        NUM_GPUS="${#GPU_LIST[@]}"
        export NUM_GPUS
    fi

    echo "[INFO] Visible GPUs: ${CUDA_VISIBLE_DEVICES:-auto} (resolved=${GPU_LIST[*]:-0}, shards=${SHARD_COUNT})"

    if [[ "${EVAL_BASE_FT_ONLY}" == "1" ]]; then
        echo "[$(date '+%F %T')] [INFO] EVAL_BASE_FT_ONLY=1: base model + full finetune last step only"
        if [[ ! -d "${FULL_FT_ROOT}" ]]; then
            echo "[ERROR] FULL_FT_ROOT not found: ${FULL_FT_ROOT}"
            exit 1
        fi
        mkdir -p "${FULL_FT_OUT_ROOT}"

        model_filters=()
        if [[ -n "${FULL_FT_MODEL_FILTER}" ]]; then
            IFS=',' read -ra model_filters <<< "${FULL_FT_MODEL_FILTER}"
        fi

        runs=()
        for d in "${FULL_FT_ROOT}/${FULL_FT_PREFIX}"*; do
            [[ -d "$d" ]] || continue
            run_name="$(basename "$d")"
            if [[ ${#model_filters[@]} -gt 0 ]]; then
                matched=0
                for m in "${model_filters[@]}"; do
                    m="${m//[[:space:]]/}"
                    if [[ -n "$m" && "$run_name" == *"$m"* ]]; then
                        matched=1
                        break
                    fi
                done
                [[ "$matched" -eq 1 ]] || continue
            fi
            runs+=("$d")
        done

        if [[ ${#runs[@]} -eq 0 ]]; then
            echo "[WARN] No full finetune runs found under ${FULL_FT_ROOT}/${FULL_FT_PREFIX}*"
            exit 0
        fi

        for run_dir in "${runs[@]}"; do
            run_name="$(basename "$run_dir")"
            last_step=""
            if [[ -f "${run_dir}/latest_checkpointed_iteration.txt" ]]; then
                last_step="$(tr -cd '0-9' < "${run_dir}/latest_checkpointed_iteration.txt")"
            fi
            if [[ -z "$last_step" ]]; then
                last_step="$(ls -d "${run_dir}"/global_step_* 2>/dev/null | sed -E 's|.*/global_step_([0-9]+)|\\1|' | sort -n | tail -n 1)"
            fi
            if [[ -z "$last_step" ]]; then
                echo "[WARN] No global_step_* found in ${run_dir}, skip"
                continue
            fi
            echo "[$(date '+%F %T')] [INFO] Eval full finetune: ${run_name}, last_step=${last_step}"
            CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(IFS=,; echo "${GPU_LIST[*]}")}" \
            $PYTHON_BIN tools/run_qwen_eval_all_shared.py \
                --model_root "${run_dir}" \
                --out_root "${FULL_FT_OUT_ROOT}" \
                --prompt_type "${PROMPT_TYPE}" \
                --max_tokens_per_call "${MAX_TOKENS}" \
                --nproc "${NUM_GPUS}" \
                --worker_concurrency 1 \
                --use_vllm \
                --temperature_g1 "${TEMP_G1}" \
                --temperature_g2 "${TEMP_G2}" \
                --n_sampling_g1 "${NSAMP_G1}" \
                --n_sampling_g2 "${NSAMP_G2}" \
                --steps "${last_step}"
        done
        echo "[$(date '+%F %T')] [INFO] Base+full-ft(last step) eval done"
        exit 0
    fi

    # Dynamically set eval parallelism based on nproc/shards (unless user overrides)
    if command -v nproc >/dev/null 2>&1; then
        total_cpus="$(nproc)"
    else
        total_cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
    fi
    if [[ -z "${total_cpus}" || ! "${total_cpus}" =~ ^[0-9]+$ ]]; then
        total_cpus=1
    fi
    max_total="${EVAL_MP_MAX_TOTAL}"
    if [[ -z "${max_total}" || ! "${max_total}" =~ ^[0-9]+$ ]]; then
        max_total=64
    fi
    if (( total_cpus > max_total )); then
        total_cpus="${max_total}"
    fi
    per_shard=$(( total_cpus / SHARD_COUNT ))
    if (( per_shard < 1 )); then
        per_shard=1
    fi
    max_per_shard="${per_shard}"
    if [[ -z "${EVAL_MP_WORKERS:-}" || "${EVAL_MP_WORKERS}" -gt "${max_per_shard}" ]]; then
        export EVAL_MP_WORKERS="${max_per_shard}"
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
    echo "[INFO] Eval parallelism: nproc=${total_cpus}, shards=${SHARD_COUNT}, per_shard=${per_shard} -> MP=${EVAL_MP_WORKERS}, THREADS=${EVAL_THREAD_WORKERS}, CHUNK=${EVAL_MP_CHUNK_SIZE}"

    if (( SHARD_COUNT <= 1 )); then
        $PYTHON_BIN tools/run_lora_eval_shared.py \
            --base_model "${BASE_MODEL_PATH}" \
            --lora_adapters "${LORA_ADAPTERS}" \
            --out_root "${OUT_ROOT}" \
            --num_gpus "${NUM_GPUS}" \
            --datasets "${EVAL_DATASETS}" \
            --prompt_type "${PROMPT_TYPE}" \
            --max_tokens "${MAX_TOKENS}" \
            --temperature_g1 "${TEMP_G1}" \
            --temperature_g2 "${TEMP_G2}" \
            --n_sampling_g1 "${NSAMP_G1}" \
            --n_sampling_g2 "${NSAMP_G2}" \
            --shard_id 0 \
            --num_shards 1
    else
        pids=()
        shard_log_dir="${LORA_SHARD_LOG_DIR:-${LOG_DIR}/shards}"
        mkdir -p "${shard_log_dir}"
        cache_root_base="${VLLM_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/vllm}"
        safe_job_name="${JOB_NAME:-lora_eval}"
        safe_job_name="${safe_job_name//[^A-Za-z0-9_.-]/_}"
        for ((sid=0; sid<SHARD_COUNT; sid++)); do
            gpu_id="${GPU_LIST[$sid]}"
            shard_cache_root="${cache_root_base}/shard_${safe_job_name}_${JOB_ID:-$$}_${sid}"
            shard_inductor_cache="${shard_cache_root}/inductor_cache"
            shard_triton_cache="${shard_cache_root}/triton_cache"
            shard_log="${shard_log_dir}/${_TS}.${safe_job_name}.shard${sid}.log"
            mkdir -p "${shard_cache_root}" "${shard_inductor_cache}" "${shard_triton_cache}"

            CUDA_VISIBLE_DEVICES="$gpu_id" \
            FORCE_EXIT_AFTER_EVAL=1 \
            VLLM_CACHE_ROOT="${shard_cache_root}" \
            TORCHINDUCTOR_CACHE_DIR="${shard_inductor_cache}" \
            TRITON_CACHE_DIR="${shard_triton_cache}" \
            $PYTHON_BIN tools/run_lora_eval_shared.py \
                --base_model "${BASE_MODEL_PATH}" \
                --lora_adapters "${LORA_ADAPTERS}" \
                --out_root "${OUT_ROOT}" \
                --num_gpus "${NUM_GPUS}" \
                --datasets "${EVAL_DATASETS}" \
                --prompt_type "${PROMPT_TYPE}" \
                --max_tokens "${MAX_TOKENS}" \
                --temperature_g1 "${TEMP_G1}" \
                --temperature_g2 "${TEMP_G2}" \
                --n_sampling_g1 "${NSAMP_G1}" \
                --n_sampling_g2 "${NSAMP_G2}" \
                --shard_id "${sid}" \
                --num_shards "${SHARD_COUNT}" \
                > "${shard_log}" 2>&1 &
            pids+=($!)
            echo "[INFO] Shard ${sid} log: ${shard_log}"
        done
        fail=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                fail=1
            fi
        done
        if [[ "$fail" -ne 0 ]]; then
            echo "[ERROR] One or more shard workers failed."
            exit 1
        fi

        echo "[$(date '+%F %T')] [INFO] Merging shard outputs..."
        LORA_ADAPTERS="${LORA_ADAPTERS}" OUT_ROOT="${OUT_ROOT}" PROMPT_TYPE="${PROMPT_TYPE}" \
        $PYTHON_BIN - <<'PY'
from pathlib import Path
import os
from tools.merge_results import merge_shard_files

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

out_root = Path(os.environ.get("OUT_ROOT", "."))
prompt_type = os.environ.get("PROMPT_TYPE", "think-boxed")
adapters = [a.strip() for a in os.environ.get("LORA_ADAPTERS", "").split("|") if a.strip()]

for adapter in adapters:
    safe_run_name, run_tag = adapter_meta(adapter)
    run_name = f"{safe_run_name}/{run_tag}"
    merge_shard_files(out_root, run_name, prompt_type)
PY
        echo "[$(date '+%F %T')] [INFO] Merge done"
    fi
    
    exit 0
fi

# ========================================
# Discovery and Job Submission
# ========================================

if [[ "${EVAL_BASE_FT_ONLY}" == "1" && -z "${RUN_LORA_EVAL_SUBMITTED:-}" ]]; then
    if [[ ! -d "${FULL_FT_ROOT}" ]]; then
        echo "[ERROR] FULL_FT_ROOT not found: ${FULL_FT_ROOT}"
        exit 1
    fi
    mkdir -p "${FULL_FT_OUT_ROOT}"

    model_filters=()
    if [[ -n "${FULL_FT_MODEL_FILTER}" ]]; then
        IFS=',' read -ra model_filters <<< "${FULL_FT_MODEL_FILTER}"
    fi

    runs=()
    for d in "${FULL_FT_ROOT}/${FULL_FT_PREFIX}"*; do
        [[ -d "$d" ]] || continue
        run_name="$(basename "$d")"
        if [[ ${#model_filters[@]} -gt 0 ]]; then
            matched=0
            for m in "${model_filters[@]}"; do
                m="${m//[[:space:]]/}"
                if [[ -n "$m" && "$run_name" == *"$m"* ]]; then
                    matched=1
                    break
                fi
            done
            [[ "$matched" -eq 1 ]] || continue
        fi
        runs+=("$d")
    done

    if [[ ${#runs[@]} -eq 0 ]]; then
        echo "[WARN] No full finetune runs found under ${FULL_FT_ROOT}/${FULL_FT_PREFIX}*"
        exit 0
    fi

    for run_dir in "${runs[@]}"; do
        run_name="$(basename "$run_dir")"
        model_name="${run_name#${FULL_FT_PREFIX}}"
        job_name="BASEFT_${run_name}"
        job_name="${job_name//[^A-Za-z0-9_.-]/_}"
        last_step=""
        if [[ -f "${run_dir}/latest_checkpointed_iteration.txt" ]]; then
            last_step="$(tr -cd '0-9' < "${run_dir}/latest_checkpointed_iteration.txt")"
        fi
        if [[ -z "$last_step" ]]; then
            last_step="$(ls -d "${run_dir}"/global_step_* 2>/dev/null | sed -E 's|.*/global_step_([0-9]+)|\\1|' | sort -n | tail -n 1)"
        fi
        if [[ -z "$last_step" ]]; then
            echo "[WARN] No global_step_* found in ${run_dir}, skip"
            continue
        fi
        if RUN_DIR="${run_dir}" OUT_ROOT="${FULL_FT_OUT_ROOT}" BASE_ROOT="/hss/giil/caixq/model" LAST_STEP="${last_step}" \
           EVAL_GROUP1_DATASETS="${EVAL_GROUP1_DATASETS}" EVAL_GROUP2_DATASETS="${EVAL_GROUP2_DATASETS}" \
           $PYTHON_BIN - <<'PY'
import os
import re
import json
from pathlib import Path

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def find_base_model_dir(base_root: Path, run_name: str):
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

def _split_ds_list(datasets: str):
    return [d.strip() for d in datasets.split(',') if d.strip()]

def _expected_samples(ds_name: str) -> int:
    defaults = {
        'aime24x8': 240,
        'aime25x8': 240,
        'amc23x8': 320,
        'math500': 500,
        'minerva_math': 272,
        'olympiadbench': 675,
    }
    return defaults.get(ds_name, 0)

def _is_dataset_complete(ds_dir: Path, ds_name: str) -> bool:
    if not ds_dir.exists():
        return False
    expected = _expected_samples(ds_name)
    if expected <= 0:
        return bool(list(ds_dir.glob('*metrics.json')))
    final_metrics = [f for f in ds_dir.glob('*metrics.json') if '_part' not in f.name]
    if final_metrics:
        try:
            data = json.loads(final_metrics[0].read_text())
            if data.get('num_samples', 0) >= expected:
                return True
        except Exception:
            pass
    part_metrics = sorted(ds_dir.glob('*_part*_*metrics.json'))
    if not part_metrics:
        return False
    total_samples = 0
    for pm in part_metrics:
        try:
            data = json.loads(pm.read_text())
            total_samples += data.get('num_samples', 0)
        except Exception:
            pass
    return total_samples >= expected

def check_missing_by_group(out_root: Path, run_name: str, group1: list, group2: list):
    missing = {1: [], 2: []}
    run_out = out_root / run_name
    for group_idx, ds_list in enumerate([group1, group2], start=1):
        gdir = run_out / f'g{group_idx}'
        for ds in ds_list:
            ds_dir = gdir / ds
            if not _is_dataset_complete(ds_dir, ds):
                missing[group_idx].append(ds)
    return missing

run_dir = Path(os.environ.get("RUN_DIR", ""))
out_root = Path(os.environ.get("OUT_ROOT", ""))
base_root = Path(os.environ.get("BASE_ROOT", "/hss/giil/caixq/model"))
last_step = os.environ.get("LAST_STEP", "").strip()
group1 = _split_ds_list(os.environ.get("EVAL_GROUP1_DATASETS", ""))
group2 = _split_ds_list(os.environ.get("EVAL_GROUP2_DATASETS", ""))

if not run_dir or not run_dir.exists() or not last_step:
    raise SystemExit(1)

lookup_name = run_dir.name
base_dir = find_base_model_dir(base_root, lookup_name)
if base_dir is None:
    raise SystemExit(1)

safe_run_name = f"global_step_{last_step}".replace(".", "_").replace("-", "_")
tag = f"{safe_run_name}/{lookup_name}__global_step_{last_step}"

missing_base = check_missing_by_group(out_root=out_root, run_name=f"base__{base_dir.name}", group1=group1, group2=group2)
missing_step = check_missing_by_group(out_root=out_root, run_name=tag, group1=group1, group2=group2)

need_base = any(missing_base[g] for g in missing_base)
need_step = any(missing_step[g] for g in missing_step)

raise SystemExit(0 if (not need_base and not need_step) else 1)
PY
        then
            echo "[$(date '+%F %T')] [INFO] Skip ${run_name} (base + last step already complete)"
            continue
        fi
        if type -t select_resources_for_job >/dev/null 2>&1; then
            read -r jc_base n_gpus < <(select_resources_for_job "$PROJECT_NAME" "$job_name")
            jc_full="$(full_jclass_from_base "$jc_base")"
            d_shm="$(get_d_shm_for_jc "$jc_base")"
        else
            jc_full="gtn-container_g1.24h"
            n_gpus=1
            d_shm="64g"
        fi
        echo "[$(date '+%F %T')] [INFO] Submit base+ft job: ${run_name} (jc=${jc_full}, gpus=${n_gpus})"
        qsub_out=$(qsub -N "$job_name" \
            -jc "${jc_full}" \
            -ac "d=nvcr-cuda-12.4.1-ubuntu22.04,d_shm=${d_shm}" \
            -o /dev/null \
            -v NUM_GPUS="${n_gpus}" \
            -v ALLOCATED_GPUS="${n_gpus}" \
            -v RUN_LORA_EVAL_SUBMITTED=1 \
            -v EVAL_BASE_FT_ONLY=1 \
            -v FULL_FT_ROOT="${FULL_FT_ROOT}" \
            -v FULL_FT_PREFIX="${FULL_FT_PREFIX}" \
            -v FULL_FT_OUT_ROOT="${FULL_FT_OUT_ROOT}" \
            -v FULL_FT_MODEL_FILTER="${model_name}" \
            -v PROMPT_TYPE="${PROMPT_TYPE}" \
            -v MAX_TOKENS="${MAX_TOKENS}" \
            -v MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS}" \
            -v TEMP_G1="${TEMP_G1}" \
            -v TEMP_G2="${TEMP_G2}" \
            -v NSAMP_G1="${NSAMP_G1}" \
            -v NSAMP_G2="${NSAMP_G2}" \
            -V \
            "${SCRIPT_PATH}" 2>&1)
        echo "[INFO] qsub output: ${qsub_out}"
    done
    exit 0
fi

echo "[INFO] Discovering LoRA adapters in $MODEL_ROOT..."

# Parse EVAL_STEPS filter
ALLOWED_STEPS="$(parse_eval_steps_to_array "${EVAL_STEPS:-}")"
if [[ -n "$ALLOWED_STEPS" ]]; then
    echo "[INFO] EVAL_STEPS filter: $ALLOWED_STEPS"
fi

# Parse OPRA-K-ABLATION per-model k filters
declare -A OPRA_K_FILTERS_MAP
OPRA_K_FILTER_ACTIVE=0
if [[ "${EXP_NAMES}" == "OPRA-K-ABLATION" && -n "${OPRA_K_FILTERS:-}" ]]; then
    parse_opra_k_filters "${OPRA_K_FILTERS}"
    if [[ ${#OPRA_K_FILTERS_MAP[@]} -gt 0 ]]; then
        OPRA_K_FILTER_ACTIVE=1
        echo "[INFO] OPRA_K_FILTERS enabled: ${OPRA_K_FILTERS}"
    else
        echo "[WARN] OPRA_K_FILTERS set but no valid entries parsed"
    fi
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

# Precompute max global_step per non-OPRA adapter type if enabled
declare -A MAX_STEP_BY_TYPE
if [[ "${EVAL_NON_OPRA_MAX_STEP_ONLY}" == "1" ]]; then
    for adapter in "${ADAPTER_LIST[@]}"; do
        if [[ "$(basename "$adapter")" == *_vllm ]]; then
            continue
        fi
        adapter_parent="$adapter"
        while [[ "$(basename "$adapter_parent")" == "actor" || "$(basename "$adapter_parent")" =~ ^global_step_ ]]; do
            adapter_parent="$(dirname "$adapter_parent")"
        done
        adapter_type="$(basename "$adapter_parent")"
        if algo_in_blacklist "$adapter_type"; then
            continue
        fi
        if [[ "${adapter_type,,}" == *opra* ]]; then
            continue
        fi
        if [[ "$adapter" =~ global_step_([0-9]+) ]]; then
            step_num="${BASH_REMATCH[1]}"
            cur="${MAX_STEP_BY_TYPE[$adapter_type]:-}"
            if [[ -z "$cur" || "$step_num" -gt "$cur" ]]; then
                MAX_STEP_BY_TYPE["$adapter_type"]="$step_num"
            fi
        fi
    done
    for t in "${!MAX_STEP_BY_TYPE[@]}"; do
        echo "[INFO] Non-OPRA max step for ${t}: ${MAX_STEP_BY_TYPE[$t]}"
    done
fi

# Group by adapter type (algorithm directory name)
# Key: "base_model::adapter_type", Value: list of global_step paths
declare -A ADAPTER_GROUPS
for adapter in "${ADAPTER_LIST[@]}"; do
    # Skip converted vLLM adapter dirs to avoid mis-grouping
    if [[ "$(basename "$adapter")" == *_vllm ]]; then
        echo "[INFO] Skipping converted adapter dir: $adapter"
        continue
    fi
    # adapter is like: /path/to/Qwen2.5-Math-1.5B_lora/global_step_100/actor
    # or: /path/to/Qwen2.5-Math-1.5B_lora/global_step_100
    
    # Get adapter type (algorithm) directory
    adapter_parent="$adapter"
    # Walk up to find the algorithm directory (contains global_step_*)
    while [[ "$(basename "$adapter_parent")" == "actor" || "$(basename "$adapter_parent")" =~ ^global_step_ ]]; do
        adapter_parent="$(dirname "$adapter_parent")"
    done
    
    adapter_type="$(basename "$adapter_parent")"
    if algo_in_blacklist "$adapter_type"; then
        echo "[INFO] Skipping $adapter (blacklisted algo: $adapter_type)"
        continue
    fi

    # OPRA-K-ABLATION: optional per-model k filter (based on adapter_type name)
    if [[ "${OPRA_K_FILTER_ACTIVE}" == "1" && "${adapter_type}" == *opra* ]]; then
        opra_base="${adapter_type%%_opra*}"
        opra_k=""
        if [[ "$adapter_type" =~ _k([0-9]+) ]]; then
            opra_k="${BASH_REMATCH[1]}"
        fi
        allowed_ks="${OPRA_K_FILTERS_MAP[$opra_base]:-}"
        if [[ -n "$allowed_ks" ]]; then
            if [[ -z "$opra_k" ]]; then
                echo "[INFO] Skipping $adapter (cannot parse k for ${opra_base})"
                continue
            fi
            if ! k_in_filter "$opra_k" "$allowed_ks"; then
                echo "[INFO] Skipping $adapter (k${opra_k} not in OPRA_K_FILTERS for ${opra_base})"
                continue
            fi
        fi
    fi
    
    # Extract step number for EVAL_STEPS filtering
    step_num=""
    if [[ "$adapter" =~ global_step_([0-9]+) ]]; then
        step_num="${BASH_REMATCH[1]}"
    fi
    is_opra=0
    if [[ "${adapter_type,,}" == *opra* ]]; then
        is_opra=1
    fi
    if [[ "${EVAL_NON_OPRA_MAX_STEP_ONLY}" == "1" && "$is_opra" -eq 0 ]]; then
        if [[ -n "$step_num" ]]; then
            max_step="${MAX_STEP_BY_TYPE[$adapter_type]:-}"
            if [[ -n "$max_step" && "$step_num" != "$max_step" ]]; then
                echo "[INFO] Skipping $adapter (non-OPRA step $step_num != max ${max_step})"
                continue
            fi
        fi
    else
        if [[ -n "$step_num" ]]; then
            if ! step_in_filter "$step_num" "$ALLOWED_STEPS"; then
                echo "[INFO] Skipping $adapter (step $step_num not in EVAL_STEPS)"
                continue
            fi
        fi
    fi

    # Skip unsupported PEFT types (vLLM LoRA does not support OFT)
    if [[ -f "${adapter}/adapter_config.json" ]]; then
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
            echo "[WARN] Skipping OFT adapter (unsupported by vLLM LoRA): $adapter"
            continue
        fi
    fi

    base_model=$(find_base_model "$adapter_type" || true)
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

check_group_complete() {
    local adapters="$1"
    LORA_ADAPTERS="${adapters}" \
    EVAL_DATASETS="${EVAL_DATASETS}" \
    EVAL_GROUP1_DATASETS="${EVAL_GROUP1_DATASETS}" \
    EVAL_GROUP2_DATASETS="${EVAL_GROUP2_DATASETS}" \
    NSAMP_G1="${NSAMP_G1:-}" \
    NSAMP_G2="${NSAMP_G2:-}" \
    MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-}" \
    EVAL_NUM_TEST_SAMPLE="${EVAL_NUM_TEST_SAMPLE:-}" \
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-}" \
    OUT_ROOT="${OUT_ROOT}" \
    ${PYTHON_BIN} - <<'PY'
import json
import os
import sys
from pathlib import Path

def _split(value):
    return [v.strip() for v in value.split(",") if v.strip()]

datasets = _split(os.environ.get("EVAL_DATASETS", ""))
group1 = set(_split(os.environ.get("EVAL_GROUP1_DATASETS", "")))
group2 = set(_split(os.environ.get("EVAL_GROUP2_DATASETS", "")))
adapters = [a.strip() for a in os.environ.get("LORA_ADAPTERS", "").split("|") if a.strip()]
out_root = Path(os.environ.get("OUT_ROOT", "."))

def group_idx(name: str) -> int:
    if name in group1:
        return 1
    if name in group2:
        return 2
    return 1

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

def _resolve_data_dir(out_root: Path) -> Path:
    env_dir = os.environ.get("EVAL_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    try:
        parts = out_root.resolve().parts
        if "eval_results" in parts:
            base = Path(*parts[:parts.index("eval_results")])
            candidate = base / "data"
            if candidate.exists():
                return candidate
    except Exception:
        pass
    return Path("./data")

def _count_dataset_samples(data_dir: Path, data_name: str, split: str = "test"):
    jsonl_path = data_dir / data_name / f"{split}.jsonl"
    if jsonl_path.exists():
        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return None
    json_path = data_dir / data_name / f"{split}.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return len(data)
        except Exception:
            return None
    return None

def _expected_samples(data_name: str, data_dir: Path) -> int | None:
    raw = os.environ.get("EVAL_NUM_TEST_SAMPLE", "")
    num = None
    if raw.strip():
        try:
            num = int(raw)
        except ValueError:
            num = None
    if num is not None and num <= 0:
        num = None
    ds_count = _count_dataset_samples(data_dir, data_name)
    if ds_count is None:
        return num
    if num is None:
        return ds_count
    return min(num, ds_count)

def _parse_int(val):
    try:
        return int(val)
    except Exception:
        return None

def _target_n_for_dataset(data_name: str) -> int | None:
    nsamp_g1 = _parse_int(os.environ.get("NSAMP_G1") or os.environ.get("MAX_SAMPLE_NUMS") or "")
    nsamp_g2 = _parse_int(os.environ.get("NSAMP_G2") or os.environ.get("MAX_SAMPLE_NUMS") or "")
    g = group_idx(data_name)
    if g == 1:
        return nsamp_g1
    if g == 2:
        return nsamp_g2
    return nsamp_g1 or nsamp_g2

def _iter_jsonl_files(out_dir: Path):
    files = sorted(out_dir.glob("*.jsonl"))
    if not files:
        return []
    non_part = [p for p in files if "_part" not in p.name]
    return non_part if non_part else files

def _count_jsonl_samples(out_dir: Path) -> int:
    files = _iter_jsonl_files(out_dir)
    if not files:
        return 0
    seen = set()
    count = 0
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and "idx" in obj:
                        idx = obj.get("idx")
                        if idx in seen:
                            continue
                        seen.add(idx)
                        count += 1
                    else:
                        count += 1
        except Exception:
            continue
    return count

def _min_sample_len(out_dir: Path) -> int:
    files = _iter_jsonl_files(out_dir)
    if not files:
        return 0
    by_idx = {}
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    idx_val = obj.get("idx")
                    if idx_val is None:
                        continue
                    sample_len = 0
                    for key in ("pred", "code", "score"):
                        val = obj.get(key)
                        if isinstance(val, list):
                            sample_len = len(val)
                            break
                    prev = by_idx.get(idx_val, 0)
                    if sample_len > prev:
                        by_idx[idx_val] = sample_len
        except Exception:
            continue
    if not by_idx:
        return 0
    return min(by_idx.values())

def _find_jsonl(metrics_path: Path):
    candidates = sorted(metrics_path.parent.glob("*.jsonl"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    matches = [p for p in candidates if metrics_path.name.startswith(p.stem + "_")]
    if matches:
        matches.sort(key=lambda p: len(p.stem), reverse=True)
        return matches[0]
    return None

def _load_score_lists(jsonl_path: Path):
    scores = []
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                score = obj.get("score")
                if isinstance(score, list):
                    scores.append([bool(x) for x in score])
    except Exception:
        return []
    return scores

def _ensure_metrics_std(metrics_path: Path) -> bool:
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    has_pass = isinstance(data.get("pass_at_k_percent"), dict)
    need_acc = "acc_std" not in data
    need_total = "total_acc_std" not in data
    need_pass = has_pass and ("pass_at_k_std" not in data)
    if not (need_acc or need_total or need_pass):
        return True
    jsonl_path = _find_jsonl(metrics_path)
    if not jsonl_path:
        return False
    score_mat = _load_score_lists(jsonl_path)
    if not score_mat:
        return False
    try:
        from evaluate import _compute_sample_std_fields
        pass_keys = list((data.get("pass_at_k_percent") or {}).keys())
        acc_std, total_std, pass_std = _compute_sample_std_fields(
            score_mat=score_mat,
            pass_at_k_keys=pass_keys,
            decimals=1,
        )
    except Exception:
        return False
    if acc_std is None and total_std is None and pass_std is None:
        return False
    if need_acc and acc_std is not None:
        data["acc_std"] = acc_std
    if need_total and total_std is not None:
        data["total_acc_std"] = total_std
    if need_pass and pass_std is not None:
        data["pass_at_k_std"] = pass_std
    metrics_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    return True

data_dir = _resolve_data_dir(out_root)

missing = []
total = 0
for adapter in adapters:
    safe_run_name, run_tag = adapter_meta(adapter)
    for data_name in datasets:
        total += 1
        out_dir = out_root / safe_run_name / run_tag / f"g{group_idx(data_name)}" / data_name
        expected = _expected_samples(data_name, data_dir)
        if expected is not None and expected > 0:
            have = _count_jsonl_samples(out_dir)
            if have < expected:
                missing.append(f"{out_dir} (jsonl {have}/{expected})")
                continue
        target_n = _target_n_for_dataset(data_name)
        if target_n is not None and target_n > 0:
            min_len = _min_sample_len(out_dir)
            if min_len < target_n:
                missing.append(f"{out_dir} (n={min_len}<{target_n})")
                continue
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

submit_qsub_job() {
    local job_name="$1"
    local base_model_path="$2"
    local adapters="$3"
    local n_gpus="$4"
    local jc_full="$5"
    local d_shm="$6"

    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log_file="/dev/null"

    local qsub_out
    qsub_out=$(qsub -N "$job_name" \
         -jc "$jc_full" \
         -ac "d=nvcr-cuda-12.4.1-ubuntu22.04,d_shm=${d_shm}" \
         -o "$log_file" \
         -v NUM_GPUS="${TP_NUM_GPUS}" \
         -v ALLOCATED_GPUS="${n_gpus}" \
         -v BASE_MODEL_PATH="${base_model_path}" \
         -v LORA_ADAPTERS="${adapters}" \
         -v OUT_ROOT="${OUT_ROOT}" \
         -v PROMPT_TYPE="${PROMPT_TYPE}" \
         -v MAX_TOKENS="${MAX_TOKENS}" \
         -v MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS}" \
         -v TEMP_G1="${TEMP_G1}" \
         -v TEMP_G2="${TEMP_G2}" \
        -v NSAMP_G1="${NSAMP_G1}" \
        -v NSAMP_G2="${NSAMP_G2}" \
        -v EVAL_NON_OPRA_MAX_STEP_ONLY="${EVAL_NON_OPRA_MAX_STEP_ONLY}" \
        -v EVAL_BASE_FT_ONLY="${EVAL_BASE_FT_ONLY}" \
        -v FULL_FT_ROOT="${FULL_FT_ROOT}" \
         -v FULL_FT_PREFIX="${FULL_FT_PREFIX}" \
         -v FULL_FT_OUT_ROOT="${FULL_FT_OUT_ROOT}" \
         -v FULL_FT_MODEL_FILTER="${FULL_FT_MODEL_FILTER}" \
         -v RUN_LORA_EVAL_SUBMITTED=1 \
         -V \
         "$SCRIPT_PATH")

    echo "[INFO] qsub output: ${qsub_out}" >&2
    local job_id
    job_id=$(echo "$qsub_out" | sed -n 's/.*Your job \([0-9]\+\).*/\1/p' | head -1)
    if [[ -z "$job_id" ]]; then
        job_id=$(echo "$qsub_out" | awk '{for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) {print $i; exit}}')
    fi
    echo "[INFO] Log file: $log_file" >&2
    echo "${job_id}|${log_file}"
}

mapfile -t GROUP_KEYS < <(printf '%s\n' "${!ADAPTER_GROUPS[@]}" | sort)

declare -A JOB_NAMES
declare -A GROUP_ADAPTERS
declare -A GROUP_BASE_MODEL
declare -A GROUP_ADAPTER_COUNT
declare -A RUNNING_JOBS

QSTAT_NAME_TOOL="${QSTAT_NAME_TOOL:-$HOME/tools/qstat_name.sh}"

refresh_running_jobs() {
    RUNNING_JOBS=()
    local names=""
    if [[ -x "$QSTAT_NAME_TOOL" ]]; then
        # qstat_name.sh prints a table; grab the last column (job name)
        names="$("$QSTAT_NAME_TOOL" 2>/dev/null | awk 'NR>1 {print $NF}' || true)"
    else
        names="$(qstat 2>/dev/null | awk 'NR>2 {print $3}')"
    fi
    while IFS= read -r name; do
        [[ -n "$name" ]] && RUNNING_JOBS["$name"]=1
    done <<< "$names"
}

job_name_running() {
    local name="$1"
    [[ -n "${RUNNING_JOBS[$name]:-}" ]]
}

for group_key in "${GROUP_KEYS[@]}"; do
    adapters="${ADAPTER_GROUPS[$group_key]}"
    adapter_count=$(echo "$adapters" | tr '|' '\n' | wc -l)

    base_key="${group_key%%::*}"
    adapter_type="${group_key##*::}"

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

    job_name="LORA_${PROJECT_NAME}_${EXP_NAMES}_${adapter_type}"
    job_name="${job_name//[^A-Za-z0-9_]/_}"
    job_name="${job_name:0:120}"

    GROUP_ADAPTERS["$group_key"]="$adapters"
    GROUP_BASE_MODEL["$group_key"]="$base_model_path"
    GROUP_ADAPTER_COUNT["$group_key"]="$adapter_count"
    JOB_NAMES["$group_key"]="$job_name"

    echo "=============================================="
    echo "Queued: $job_name"
    echo "Adapter Type: $adapter_type"
    echo "Base Model: $base_model_path"
    echo "Checkpoints: $adapter_count"
    echo "Datasets: $EVAL_DATASETS"
    echo "=============================================="
done

poll_interval="${QSTAT_POLL_INTERVAL:-60}"
refresh_running_jobs
echo "[INFO] Precheck: submitting unfinished groups without running jobs"
for group_key in "${GROUP_KEYS[@]}"; do
    job_name="${JOB_NAMES[$group_key]}"
    adapters="${GROUP_ADAPTERS[$group_key]}"
    base_model_path="${GROUP_BASE_MODEL[$group_key]}"
    if [[ -z "$job_name" || -z "$adapters" || -z "$base_model_path" ]]; then
        continue
    fi
    if check_group_complete "$adapters"; then
        echo "[INFO] ${job_name} complete (precheck)"
        continue
    fi
    if job_name_running "$job_name"; then
        echo "[INFO] ${job_name} already running (precheck)"
        continue
    fi
    echo "[INFO] Precheck submit for ${job_name}"
    if type -t select_resources_for_job >/dev/null 2>&1; then
        read -r jc_base n_gpus < <(select_resources_for_job "$PROJECT_NAME" "$job_name")
        jc_full="$(full_jclass_from_base "$jc_base")"
        d_shm="$(get_d_shm_for_jc "$jc_base")"
    else
        jc_full="gtn-container_g1.24h"
        n_gpus=1
        d_shm="64g"
    fi
    echo "[INFO] Using $n_gpus GPU(s), jc=$jc_full"
    submit_qsub_job "$job_name" "$base_model_path" "$adapters" "$n_gpus" "$jc_full" "$d_shm" >/dev/null
done

while true; do
    all_done=1
    refresh_running_jobs
    for group_key in "${GROUP_KEYS[@]}"; do
        job_name="${JOB_NAMES[$group_key]}"
        adapters="${GROUP_ADAPTERS[$group_key]}"
        base_model_path="${GROUP_BASE_MODEL[$group_key]}"
        if [[ -z "$job_name" || -z "$adapters" || -z "$base_model_path" ]]; then
            continue
        fi

        if check_group_complete "$adapters"; then
            echo "[INFO] ${job_name} complete"
            continue
        fi

        all_done=0
        if job_name_running "$job_name"; then
            continue
        fi

        echo "[INFO] Submitting job for ${job_name}"
        if type -t select_resources_for_job >/dev/null 2>&1; then
            read -r jc_base n_gpus < <(select_resources_for_job "$PROJECT_NAME" "$job_name")
            jc_full="$(full_jclass_from_base "$jc_base")"
            d_shm="$(get_d_shm_for_jc "$jc_base")"
        else
            jc_full="gtn-container_g1.24h"
            n_gpus=1
            d_shm="64g"
        fi

        echo "[INFO] Using $n_gpus GPU(s), jc=$jc_full"
        submit_qsub_job "$job_name" "$base_model_path" "$adapters" "$n_gpus" "$jc_full" "$d_shm" >/dev/null
    done

    if [[ "$all_done" -eq 1 ]]; then
        echo "[INFO] All groups complete"
        break
    fi

    sleep "$poll_interval"
done
