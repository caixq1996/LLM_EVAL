#!/usr/bin/env bash
# Monitor running qstat jobs and copy completed logs into completed/ when done.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LOG_ROOT="${EVAL_LOG_ROOT:-${LOG_ROOT:-${REPO_ROOT}}}"
LOG_DIR="${LOG_ROOT}/eval_log/eval_all/lora_jobs"
SHARD_DIR="${LOG_DIR}/shards"
COMPLETED_DIR="${LOG_DIR}/completed"
COMPLETED_RUNNING_DIR="${LOG_DIR}/completed_but_still_run"
POLL_SEC="${POLL_SEC:-30}"

mkdir -p "${COMPLETED_DIR}"
mkdir -p "${COMPLETED_RUNNING_DIR}"

log() { printf "[%(%F %T)T] %s\n" -1 "$*" >&2; }

get_running_jobs() {
  qstat -u "${USER}" 2>/dev/null \
    | awk 'NR>2 && $5=="r" {print $1 "|" $3}'
}

matches_job_name() {
  local file="$1"
  local job_name="$2"
  [[ "$file" == *"${job_name}"* ]]
}

copy_if_complete() {
  local file="$1"
  local base
  base="$(basename "$file")"
  local dest="${COMPLETED_DIR}/${base}"
  log "Checking: ${file}"
  if grep -q "All evaluations complete" "$file" 2>/dev/null; then
    if [[ ! -f "$dest" ]]; then
      cp -f "$file" "$dest"
      log "Copied: $file -> $dest"
    fi
    return 0
  fi
  return 1
}

copy_if_completed_but_running() {
  local file="$1"
  local base
  base="$(basename "$file")"
  local dest="${COMPLETED_RUNNING_DIR}/${base}"
  if [[ -f "$dest" ]]; then
    return 0
  fi
  cp -f "$file" "$dest"
  log "Copied (still running): $file -> $dest"
}

terminate_job() {
  local job_id="$1"
  [[ -n "${job_id}" ]] || return 0
  log "qdel ${job_id}"
  qdel "${job_id}" >/dev/null 2>&1 || true
}

collect_logs_for_job() {
  local job_id="$1"
  local -a logs=()

  if [[ -n "$job_id" ]]; then
    while IFS= read -r -d '' f; do
      logs+=("$f")
    done < <(find "${LOG_DIR}" -maxdepth 1 -name "*job${job_id}.log" -print0 2>/dev/null)
  fi

  if [[ -n "$job_id" && -d "${SHARD_DIR}" ]]; then
    while IFS= read -r -d '' f; do
      logs+=("$f")
    done < <(find "${SHARD_DIR}" -type f -name "*job${job_id}.log" -print0 2>/dev/null)
  fi

  printf '%s\n' "${logs[@]}"
}

log "Monitoring logs under: ${LOG_DIR} (poll=${POLL_SEC}s)"

while true; do
  mapfile -t running < <(get_running_jobs || true)
  if [[ ${#running[@]} -eq 0 ]]; then
    sleep "${POLL_SEC}"
    continue
  fi

  for entry in "${running[@]}"; do
    job_id="${entry%%|*}"
    job_name="${entry#*|}"
    [[ -n "${job_id}" ]] || continue

    log_complete=0
    while IFS= read -r f; do
      [[ -n "$f" ]] || continue
      if copy_if_complete "$f"; then
        log_complete=1
      fi
    done < <(collect_logs_for_job "$job_id")

    if [[ "${log_complete}" -eq 1 ]]; then
      terminate_job "${job_id}"
      continue
    fi

    qstat_out=""
    if qstat_out="$(qstat -j "${job_id}" 2>/dev/null)"; then
      check_out="$(python3 "${REPO_ROOT}/tools/check_eval_complete.py" --stdin <<< "${qstat_out}" 2>/dev/null || true)"
      if [[ "${check_out}" == "COMPLETE" ]]; then
        while IFS= read -r f; do
          [[ -n "$f" ]] || continue
          copy_if_completed_but_running "$f"
        done < <(collect_logs_for_job "$job_id")
        terminate_job "${job_id}"
      else
        if [[ "${check_out}" == INCOMPLETE* ]]; then
          log "Incomplete, ${check_out#INCOMPLETE }"
        else
          log "Incomplete, checker_failed"
        fi
      fi
    fi
  done

  sleep "${POLL_SEC}"
done
