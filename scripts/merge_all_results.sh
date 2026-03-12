#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR=""
PROMPT_TYPE=""
OUT_JSON_DIR=""
OPRA_ONLY=0
INCREMENTAL=1
FAST_MODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --opra_only)
      OPRA_ONLY=1
      shift
      ;;
    --all_algos)
      OPRA_ONLY=0
      shift
      ;;
    --no_incremental)
      INCREMENTAL=0
      shift
      ;;
    --fast_mode)
      FAST_MODE=1
      shift
      ;;
    --no_fast_mode)
      FAST_MODE=0
      shift
      ;;
    --incremental)
      INCREMENTAL=1
      shift
      ;;
    -*)
      echo "[ERROR] Unknown option: $1" >&2
      exit 1
      ;;
    *)
      if [[ -z "${RESULTS_DIR}" ]]; then
        RESULTS_DIR="$1"
      elif [[ -z "${PROMPT_TYPE}" ]]; then
        PROMPT_TYPE="$1"
      elif [[ -z "${OUT_JSON_DIR}" ]]; then
        OUT_JSON_DIR="$1"
      else
        echo "[ERROR] Unexpected argument: $1" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

RESULTS_DIR="${RESULTS_DIR:-eval_results/OPRA-LoRA_think-boxed}"
# Normalize to avoid trailing-slash path prefix mismatches
RESULTS_DIR="${RESULTS_DIR%/}"
PROMPT_TYPE="${PROMPT_TYPE:-think-boxed}"

if [[ -z "${RESULTS_DIR}" ]]; then
  echo "Usage: $0 <results_dir> [prompt_type] [out_json_dir] [--opra_only|--all_algos] [--incremental|--no_incremental] [--fast_mode|--no_fast_mode]" >&2
  exit 1
fi

if [[ -z "${OUT_JSON_DIR}" ]]; then
  OUT_JSON_DIR="${RESULTS_DIR}/json"
fi

if [[ ! -d "${RESULTS_DIR}" ]]; then
  echo "[ERROR] results_dir not found: ${RESULTS_DIR}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/eval/bin/python3}"

should_merge_run() {
  local run_dir="$1"
  if [[ "${INCREMENTAL}" -eq 0 ]]; then
    return 0
  fi
  if find "${run_dir}" -type f -name '*_part*.jsonl' -print -quit | grep -q .; then
    return 0
  fi
  local needs_merge=0
  while IFS= read -r jsonl; do
    local metrics="${jsonl%.jsonl}_${PROMPT_TYPE}_metrics.json"
    if [[ ! -f "${metrics}" ]]; then
      needs_merge=1
      break
    fi
  done < <(find "${run_dir}" -type f -name '*.jsonl' ! -name '*_part*.jsonl')

  if [[ "${needs_merge}" -eq 1 ]]; then
    return 0
  fi
  return 1
}

mapfile -t RUN_DIRS < <(
  find "${RESULTS_DIR}" \
    -path "${RESULTS_DIR}/json" -prune -o \
    -type d \( -name g1 -o -name g2 \) -printf '%h\n' \
  | sort -u
)

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "[WARN] No run dirs found under ${RESULTS_DIR}" >&2
  exit 0
fi

declare -A MAX_STEP_BY_ALGO
if [[ "${OPRA_ONLY}" -eq 1 ]]; then
  for run_dir in "${RUN_DIRS[@]}"; do
    run_name="${run_dir#${RESULTS_DIR}/}"
    if [[ -z "${run_name}" || "${run_name}" == "${run_dir}" ]]; then
      continue
    fi
    algo_dir="${run_name%%/*}"
    if [[ "${algo_dir}" == *opra* ]]; then
      continue
    fi
    base_name="${run_name##*/}"
    if [[ "${base_name}" =~ __global_step_([0-9]+)$ ]]; then
      step="${BASH_REMATCH[1]}"
      prev="${MAX_STEP_BY_ALGO[$algo_dir]:-}"
      if [[ -z "${prev}" || "${step}" -gt "${prev}" ]]; then
        MAX_STEP_BY_ALGO["$algo_dir"]="${step}"
      fi
    fi
  done
fi

VALID_RUN_DIRS=()
RUN_NAMES=()
for run_dir in "${RUN_DIRS[@]}"; do
  run_name="${run_dir#${RESULTS_DIR}/}"
  if [[ -z "${run_name}" || "${run_name}" == "${run_dir}" ]]; then
    continue
  fi
  algo_dir="${run_name%%/*}"
  if [[ "${OPRA_ONLY}" -eq 1 && "${algo_dir}" != *opra* ]]; then
    base_name="${run_name##*/}"
    if [[ "${base_name}" =~ __global_step_([0-9]+)$ ]]; then
      step="${BASH_REMATCH[1]}"
      max_step="${MAX_STEP_BY_ALGO[$algo_dir]:-}"
      if [[ -n "${max_step}" && "${step}" != "${max_step}" ]]; then
        continue
      fi
    fi
  fi
  if ! should_merge_run "${run_dir}"; then
    continue
  fi
  VALID_RUN_DIRS+=("${run_dir}")
  RUN_NAMES+=("${run_name}")
done

total_runs=${#VALID_RUN_DIRS[@]}
current_run=0
progress_width=30
MAX_JOBS="${MAX_JOBS:-}"
if [[ -z "${MAX_JOBS}" ]]; then
  MAX_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi
if [[ "${MAX_JOBS}" -lt 1 ]]; then
  MAX_JOBS=1
fi

merge_log_dir="$(mktemp -d -t merge_results.XXXXXX)"

declare -A PID_TO_RUN
declare -A PID_TO_LOG

is_pid_finished() {
  local pid="$1"
  if ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  local state
  state="$(ps -o stat= -p "${pid}" 2>/dev/null | awk '{print $1}')"
  if [[ -n "${state}" && "${state}" == *Z* ]]; then
    return 0
  fi
  return 1
}

reap_one() {
  for pid in "${!PID_TO_RUN[@]}"; do
    if is_pid_finished "${pid}"; then
      wait "${pid}"
      status=$?
      running=$((running - 1))
      current_run=$((current_run + 1))
      render_progress "${current_run}" "${total_runs}" "merge"
      if [[ ${status} -ne 0 ]]; then
        err_run="${PID_TO_RUN[$pid]}"
        err_log="${PID_TO_LOG[$pid]}"
        printf "\n[ERROR] merge_results.py failed for %s. See %s\n" "${err_run}" "${err_log}" >&2
        exit 1
      fi
      rm -f "${PID_TO_LOG[$pid]}"
      unset PID_TO_RUN["$pid"]
      unset PID_TO_LOG["$pid"]
      return 0
    fi
  done
  return 1
}

render_progress() {
  local current="$1"
  local total="$2"
  local label="$3"
  local percent=$(( 100 * current / total ))
  local filled=$(( progress_width * current / total ))
  local empty=$(( progress_width - filled ))
  local bar
  bar="$(printf "%0.s#" $(seq 1 "$filled"))"
  bar+=$(printf "%0.s-" $(seq 1 "$empty"))
  printf "\r[%s] %3d%% (%d/%d) %s" "$bar" "$percent" "$current" "$total" "$label"
}

render_current_run() {
  local run_name="$1"
  printf "\n[RUN] %s\n" "$run_name"
}

running=0
for idx in "${!VALID_RUN_DIRS[@]}"; do
  run_dir="${VALID_RUN_DIRS[$idx]}"
  run_name="${RUN_NAMES[$idx]}"
  render_current_run "${run_name}"
  safe_name="${run_name//\//_}"
  log_path="${merge_log_dir}/${safe_name}.log"

  while (( running >= MAX_JOBS )); do
    if ! reap_one; then
      sleep 0.1
    fi
  done

  PASS_AT_KS="1,2,8,16,32,64,128,256,512,1024" \
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/merge_results.py" \
    --out_root "${RESULTS_DIR}" \
    --run_name "${run_name}" \
    --prompt_type "${PROMPT_TYPE}" \
    $( [[ "${FAST_MODE}" -eq 1 ]] && echo "--fast_mode" ) \
    > "${log_path}" 2>&1 &
  pid=$!
  PID_TO_RUN["$pid"]="${run_name}"
  PID_TO_LOG["$pid"]="${log_path}"
  running=$((running + 1))
done

while (( running > 0 )); do
  if ! reap_one; then
    sleep 0.1
  fi
done
if [[ ${total_runs} -gt 0 ]]; then
  printf "\n"
fi
rm -rf "${merge_log_dir}"

mkdir -p "${OUT_JSON_DIR}"

RESULTS_DIR="${RESULTS_DIR}" OUT_JSON_DIR="${OUT_JSON_DIR}" PROMPT_TYPE="${PROMPT_TYPE}" \
OPRA_ONLY="${OPRA_ONLY}" INCREMENTAL="${INCREMENTAL}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
import re
from pathlib import Path

results_dir = Path(os.environ.get("RESULTS_DIR", ".")).resolve()
out_json_dir = Path(os.environ.get("OUT_JSON_DIR", ".")).resolve()
prompt_type = os.environ.get("PROMPT_TYPE", "think-boxed")
opra_only = os.environ.get("OPRA_ONLY", "0") == "1"
incremental = os.environ.get("INCREMENTAL", "1") == "1"

desired_ks = [1, 2, 8, 16, 32, 64, 128, 256, 512, 1024]

def _filter_pass_at_k(pass_at_k):
    if not isinstance(pass_at_k, dict):
        return {}
    keys = []
    for k in pass_at_k.keys():
        if isinstance(k, int):
            keys.append(k)
        else:
            try:
                keys.append(int(str(k)))
            except Exception:
                continue
    if not keys:
        return {}
    max_k = max(keys)
    out = {}
    for k in desired_ks:
        if k <= max_k and str(k) in pass_at_k:
            out[str(k)] = pass_at_k[str(k)]
        elif k <= max_k and k in pass_at_k:
            out[str(k)] = pass_at_k[k]
    return out

metrics_files = []
for path in results_dir.rglob(f"*_{prompt_type}_metrics.json"):
    rel = path.relative_to(results_dir)
    if rel.parts and rel.parts[0] == "json":
        continue
    metrics_files.append(path)

max_step_by_algo = {}
step_re = re.compile(r"__global_step_(\d+)$")
if opra_only:
    for path in metrics_files:
        rel = path.relative_to(results_dir)
        if not rel.parts:
            continue
        algo_dir = rel.parts[0]
        if "opra" in algo_dir:
            continue
        if len(rel.parts) < 2:
            continue
        m = step_re.search(rel.parts[1])
        if not m:
            continue
        step = int(m.group(1))
        prev = max_step_by_algo.get(algo_dir)
        if prev is None or step > prev:
            max_step_by_algo[algo_dir] = step

filtered_metrics_files = []
if opra_only:
    for path in metrics_files:
        rel = path.relative_to(results_dir)
        if not rel.parts:
            continue
        algo_dir = rel.parts[0]
        if "opra" in algo_dir:
            filtered_metrics_files.append(path)
            continue
        if len(rel.parts) >= 2:
            m = step_re.search(rel.parts[1])
            if m:
                step = int(m.group(1))
                max_step = max_step_by_algo.get(algo_dir)
                if max_step is not None and step != max_step:
                    continue
        filtered_metrics_files.append(path)
    metrics_files = filtered_metrics_files
total_files = len(metrics_files)
bar_width = 30

def render_progress(current: int, total: int, label: str) -> None:
    if total <= 0:
        return
    percent = int(current * 100 / total)
    filled = int(bar_width * current / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(f"\\r[{bar}] {percent:3d}% ({current}/{total}) {label}", end="", flush=True)

for idx, mpath in enumerate(metrics_files, start=1):
    rel = mpath.relative_to(results_dir)
    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception:
        continue
    # Keep same relative structure and filename under json/
    out_path = out_json_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if incremental and out_path.exists():
        try:
            if out_path.stat().st_mtime >= mpath.stat().st_mtime:
                render_progress(idx, total_files, "export")
                continue
        except Exception:
            pass

    # Filter pass@k keys across related fields
    filtered = dict(data)
    pass_at_k = _filter_pass_at_k(data.get("pass_at_k_percent"))
    if pass_at_k:
        filtered["pass_at_k_percent"] = pass_at_k
        if isinstance(data.get("pass_at_k_valid_counts"), dict):
            filtered["pass_at_k_valid_counts"] = _filter_pass_at_k(data.get("pass_at_k_valid_counts"))
        if isinstance(data.get("pass_at_k_std"), dict):
            filtered["pass_at_k_std"] = _filter_pass_at_k(data.get("pass_at_k_std"))

    out_path.write_text(json.dumps(filtered, indent=4), encoding="utf-8")
    render_progress(idx, total_files, "export")

if total_files:
    print()
PY
