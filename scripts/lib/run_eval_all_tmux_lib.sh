#!/usr/bin/env bash

run_eval_is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

run_eval_default_tmux_session_name() {
  local exp_name="${1:-LLM_EVAL}"
  local base_only="${2:-false}"
  if run_eval_is_truthy "${base_only}"; then
    echo "${exp_name}_base_only"
  else
    echo "${exp_name}"
  fi
}

run_eval_base_job_name() {
  local project_name="$1"
  local exp_name="$2"
  local base_only="${3:-false}"
  local job_tag="${exp_name//[^A-Za-z0-9_]/_}"
  job_tag="${job_tag:0:60}"

  local suffix="BASE"
  if run_eval_is_truthy "${base_only}"; then
    suffix="BASEONLY"
  fi

  local job_name="EVAL_${project_name}_${job_tag}_${suffix}"
  job_name="${job_name//[^A-Za-z0-9_]/_}"
  job_name="${job_name:0:120}"
  echo "${job_name}"
}

normalize_force_g1_family() {
  local raw="${1:-}"
  raw="${raw//[[:space:]]/}"
  case "${raw}" in
    ""|auto|AUTO|any|ANY|gtn\|gtb|gtb\|gtn|gtn,gtb|gtb,gtn)
      echo "auto"
      ;;
    gtn|GTN)
      echo "gtn"
      ;;
    gtb|GTB)
      echo "gtb"
      ;;
    *)
      echo "auto"
      ;;
  esac
}

default_batch_g1_family() {
  local family="${EVAL_G1_FAMILY_DEFAULT:-auto}"
  case "${family}" in
    ""|auto|AUTO|gtn\|gtb|gtb\|gtn|gtn,gtb|gtb,gtn)
      echo "auto"
      ;;
    gtn|gtb)
      echo "${family}"
      ;;
    *)
      echo "auto"
      ;;
  esac
}

fallback_a100_g1_family() {
  local selected_base="$1"
  local default_family
  default_family="$(default_batch_g1_family)"

  if [[ "${selected_base}" == gtb-* ]]; then
    echo "gtb"
    return 0
  fi
  if [[ "${selected_base}" == gtn-* ]]; then
    echo "gtn"
    return 0
  fi

  if [[ "${default_family}" == "gtb" || "${default_family}" == "gtn" ]]; then
    echo "${default_family}"
  else
    echo "gtn"
  fi
}

eval_pick_a100_g1_base() {
  local selected_base="$1"
  local family_pref
  family_pref="$(normalize_force_g1_family "${2:-}")"

  if [[ "${family_pref}" == "gtn" || "${family_pref}" == "gtb" ]]; then
    echo "${family_pref}-container_g1 1"
    return 0
  fi

  if type -t parse_qstat_with_time >/dev/null 2>&1; then
    parse_qstat_with_time >/dev/null 2>&1 || true
  fi
  if type -t parse_gpu_free >/dev/null 2>&1; then
    parse_gpu_free >/dev/null 2>&1 || true
  fi

  if type -t free_jobs_for_jc >/dev/null 2>&1 && type -t base_rank >/dev/null 2>&1; then
    local best_jc=""
    local best_free=-1
    local best_br=-1
    local jc
    for jc in gtb-container_g1 gtn-container_g1; do
      local free_jobs
      free_jobs="$(free_jobs_for_jc "${jc}" 2>/dev/null || echo 0)"
      [[ "${free_jobs}" =~ ^-?[0-9]+$ ]] || free_jobs=0
      local br
      br="$(base_rank "${jc}" 2>/dev/null || echo 0)"
      [[ "${br}" =~ ^-?[0-9]+$ ]] || br=0
      if (( free_jobs > best_free )) || (( free_jobs == best_free && br > best_br )); then
        best_jc="${jc}"
        best_free="${free_jobs}"
        best_br="${br}"
      fi
    done
    if [[ -n "${best_jc}" && "${best_free}" -gt 0 ]]; then
      echo "${best_jc} 1"
      return 0
    fi
  fi

  local family
  family="$(fallback_a100_g1_family "${selected_base}")"
  echo "${family}-container_g1 1"
}

coerce_jc_base_to_g1() {
  local selected_base="$1"
  eval_pick_a100_g1_base "${selected_base}" "${2:-}"
}

eval_full_jclass_from_base() {
  local base="$1"
  case "$base" in
    gtn-container_g1) echo "gtn-container_g1.24h" ;;
    gtb-container_g1) echo "gtb-container_g1.24h" ;;
    gtn-container_g4) echo "gtn-container_g4.24h" ;;
    gtn-container_g8) echo "gtn-container_g8.24h" ;;
    gtb-container_g4) echo "gtb-container_g4.24h" ;;
    gtb-container_g8) echo "gtb-container_g8.24h" ;;
    gpub-container_g1) echo "gpub-container_g1.24h" ;;
    gpub-container_g4) echo "gpub-container_g4.24h" ;;
    gpub-container_g8) echo "gpub-container_g8.24h" ;;
    gpu-container_g1) echo "gpu-container_g1.24h" ;;
    gpu-container_g4) echo "gpu-container_g4.24h" ;;
    gpu-container_g8) echo "gpu-container_g8.24h" ;;
    gs-container_g1) echo "gs-container_g1.24h" ;;
    gs-container_g4) echo "gs-container_g4.24h" ;;
    gs-container_g8) echo "gs-container_g8.24h" ;;
    *) echo "$base" ;;
  esac
}

resolve_eval_resources() {
  local project_name="$1"
  local job_name="$2"
  local selected_base="gtn-container_g1"
  local selected_gpus="1"

  if type -t select_resources_for_job >/dev/null 2>&1; then
    read -r selected_base selected_gpus < <(select_resources_for_job "${project_name}" "${job_name}")
  fi

  local jc_base="${selected_base}"
  local n_gpus="${selected_gpus}"
  local job_scale="${EVAL_JOB_SCALE:-g1}"
  if [[ "${job_scale}" == "g1" ]]; then
    read -r jc_base n_gpus < <(coerce_jc_base_to_g1 "${selected_base}" "${FORCE_G1_FAMILY:-}")
  fi

  local jc_full
  if type -t full_jclass_from_base >/dev/null 2>&1; then
    jc_full="$(full_jclass_from_base "${jc_base}")"
    if [[ "${jc_full}" == "${jc_base}" ]]; then
      jc_full="$(eval_full_jclass_from_base "${jc_base}")"
    fi
  else
    jc_full="$(eval_full_jclass_from_base "${jc_base}")"
  fi
  local d_shm_val
  d_shm_val="$(get_d_shm_for_jc "${jc_base}")"

  echo "${jc_base} ${jc_full} ${n_gpus} ${d_shm_val}"
}
