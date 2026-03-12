#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/lib/run_eval_all_tmux_lib.sh"

get_d_shm_for_jc() {
  case "$1" in
    *_g1) echo "64g" ;;
    *_g4) echo "256g" ;;
    *_g8) echo "256g" ;;
    *) echo "256g" ;;
  esac
}

full_jclass_from_base() {
  echo "${1}.24h"
}

parse_qstat_with_time() {
  :
}

parse_gpu_free() {
  :
}

free_jobs_for_jc() {
  case "$1" in
    gtb-container_g1) echo "${MOCK_FREE_GTB_G1:-0}" ;;
    gtn-container_g1) echo "${MOCK_FREE_GTN_G1:-0}" ;;
    *) echo "0" ;;
  esac
}

base_rank() {
  case "$1" in
    gtb-container_g1) echo "10" ;;
    gtn-container_g1) echo "9" ;;
    *) echo "0" ;;
  esac
}

assert_eq() {
  local actual="$1"
  local expected="$2"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "assertion failed: actual='${actual}' expected='${expected}'" >&2
    exit 1
  fi
}

select_resources_for_job() {
  local project_name="$1"
  local job_name="$2"
  if [[ "${job_name}" == *GTB* ]]; then
    echo "gtb-container_g8 8"
  elif [[ "${job_name}" == *GPUB* ]]; then
    echo "gpub-container_g4 4"
  else
    echo "gtn-container_g4 4"
  fi
}

unset FORCE_G1_FAMILY
unset EVAL_JOB_SCALE
unset EVAL_G1_FAMILY_DEFAULT
export MOCK_FREE_GTB_G1="3"
export MOCK_FREE_GTN_G1="1"
read -r jc_base jc_full n_gpus d_shm < <(resolve_eval_resources "LLM_EVAL" "RUN_AUTO")
assert_eq "${jc_base}" "gtb-container_g1"
assert_eq "${jc_full}" "gtb-container_g1.24h"
assert_eq "${n_gpus}" "1"
assert_eq "${d_shm}" "64g"

export FORCE_G1_FAMILY="gtb"
read -r jc_base jc_full n_gpus d_shm < <(resolve_eval_resources "LLM_EVAL" "RUN_AUTO")
assert_eq "${jc_base}" "gtb-container_g1"
assert_eq "${jc_full}" "gtb-container_g1.24h"
assert_eq "${n_gpus}" "1"
assert_eq "${d_shm}" "64g"

export FORCE_G1_FAMILY="gtn|gtb"
read -r jc_base jc_full n_gpus d_shm < <(resolve_eval_resources "LLM_EVAL" "RUN_GTB")
assert_eq "${jc_base}" "gtb-container_g1"
assert_eq "${jc_full}" "gtb-container_g1.24h"
assert_eq "${n_gpus}" "1"
assert_eq "${d_shm}" "64g"

export FORCE_G1_FAMILY="gtn"
read -r jc_base jc_full n_gpus d_shm < <(resolve_eval_resources "LLM_EVAL" "RUN_AUTO")
assert_eq "${jc_base}" "gtn-container_g1"
assert_eq "${jc_full}" "gtn-container_g1.24h"
assert_eq "${n_gpus}" "1"
assert_eq "${d_shm}" "64g"

unset FORCE_G1_FAMILY
read -r jc_base jc_full n_gpus d_shm < <(resolve_eval_resources "LLM_EVAL" "RUN_GPUB")
assert_eq "${jc_base}" "gtb-container_g1"
assert_eq "${jc_full}" "gtb-container_g1.24h"
assert_eq "${n_gpus}" "1"
assert_eq "${d_shm}" "64g"

echo "ok"
