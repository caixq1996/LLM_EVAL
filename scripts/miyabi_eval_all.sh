#!/bin/bash
#PBS -q regular-g
#PBS -l select=4
#PBS -l walltime=08:00:00
#PBS -W group_list=gq50
#PBS -N miyabi_eval_all
#PBS -j oe
#PBS -V

set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -------------------------------------------------------
# CLI flags (match run_eval_all.sh behavior)
# -------------------------------------------------------
KEEP_EXPORTED_HF="${KEEP_EXPORTED_HF:-false}"
RUN_EVAL_SUBMIT="${RUN_EVAL_SUBMIT:-0}"
RUN_EVAL_MULTI_SUBMIT="${RUN_EVAL_MULTI_SUBMIT:-false}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)
      RUN_EVAL_SUBMIT=1
      shift
      ;;
    --multi-submit)
      RUN_EVAL_MULTI_SUBMIT="true"
      RUN_EVAL_SUBMIT=1
      shift
      ;;
    --keep-exported)
      KEEP_EXPORTED_HF="true"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

if [[ $RUN_EVAL_MULTI_SUBMIT == "true" ]]; then
  RUN_EVAL_SUBMIT=1
fi

# -------------------------------------------------------
# Logging (align with run_eval_all.sh)
# -------------------------------------------------------
export TZ='JST-9'
_TS="$(date +%Y%m%d_%H%M%S)"

if [[ -n "${RUN_EVAL_SUBMITTED:-}" ]]; then
  _LOG_BASE="${EVAL_ROOT}/eval_log/eval_all/miyabi_qsub"
else
  _LOG_BASE="${EVAL_ROOT}/eval_log/eval_all/miyabi_main"
fi
mkdir -p "${_LOG_BASE}" "${EVAL_ROOT}/eval_log/eval_all/eval_gpus"

if [[ -n "${SUB_EXP_NAME:-}" ]]; then
  _EXP_TAG="${SUB_EXP_NAME}"
else
  _EXP_TAG="${EXP_NAMES:-default}"
fi
_MAIN_LOG="${_LOG_BASE}/miyabi_eval_all.${_TS}.${_EXP_TAG}.log"
echo "[INFO] Main script log: ${_MAIN_LOG}"
exec > >(tee -a "${_MAIN_LOG}") 2>&1

# -------------------------------------------------------
# Defaults / paths
# -------------------------------------------------------
if [[ -z "${WORK_HOME:-}" ]]; then
  echo "[WARN] WORK_HOME not set, fallback to /work/gq50/$USER"
  WORK_HOME="/work/gq50/$USER"
fi

PROJECT_NAME="${PROJECT_NAME:-OPRA}"
EXP_NAMES="${EXP_NAMES:-OPRA-LoRA}"
ROOT_DIR="${WORK_HOME}/project/${PROJECT_NAME}"
WORK_DIR="${WORK_DIR:-${EVAL_ROOT}}"

MODEL_ROOT="${MODEL_ROOT:-${ROOT_DIR}/checkpoints/${EXP_NAMES}}"
BASE_ROOT="${BASE_ROOT:-${WORK_HOME}/model}"
PROMPT_TYPE="${PROMPT_TYPE:-qwen25-math-cot}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-1024}"
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-false}"
TEMP_G1="${TEMP_G1:-0.6}"
TEMP_G2="${TEMP_G2:-0.0}"
NSAMP_G1="${NSAMP_G1:-${MAX_SAMPLE_NUMS}}"
NSAMP_G2="${NSAMP_G2:-${MAX_SAMPLE_NUMS}}"
EXPORT_ROOT="${EXPORT_ROOT:-${WORK_HOME}/export}"

export SPECIAL_ADAPTER_ALGORITHMS="${SPECIAL_ADAPTER_ALGORITHMS:-pissa:_pissa_base,qpissa:_qpissa_base}"
export EVAL_ONE_MODEL_TIMEOUT="${EVAL_ONE_MODEL_TIMEOUT:-21600}"

# OUT_ROOT depends on whether multi-submit passes SUB_EXP_NAME
if [[ -n "${SUB_EXP_NAME:-}" ]]; then
  OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/eval_results/${EXP_NAMES}_${PROMPT_TYPE}/${SUB_EXP_NAME}}"
else
  OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/eval_results/${EXP_NAMES}_${PROMPT_TYPE}}"
fi

NUM_GPUS_PER_NODE=1

# -------------------------------------------------------
# Multi-submit mode (PBS submit from login node)
# -------------------------------------------------------
if [[ -z "${PBS_JOBID:-}" && -z "${RUN_EVAL_SUBMITTED:-}" && "${RUN_EVAL_SUBMIT}" == "1" ]]; then
  if [[ ! -d "${MODEL_ROOT}" ]]; then
    echo "[ERROR] MODEL_ROOT not found: ${MODEL_ROOT}"
    exit 1
  fi

  MIYABI_SELECT_NODES="${MIYABI_SELECT_NODES:-4}"
  MIYABI_QUEUE="${MIYABI_QUEUE:-regular-g}"
  MIYABI_WALLTIME="${MIYABI_WALLTIME:-08:00:00}"
  MIYABI_GROUP="${MIYABI_GROUP:-gq50}"

  if [[ "${RUN_EVAL_MULTI_SUBMIT}" == "true" ]]; then
    SUBDIRS=()
    while IFS= read -r -d '' dir; do
      SUBDIRS+=("$dir")
    done < <(find "${MODEL_ROOT}" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)

    if [[ ${#SUBDIRS[@]} -eq 0 ]]; then
      echo "[WARN] No subdirectories found under ${MODEL_ROOT}, falling back to single submit."
    else
      echo "[INFO] Multi-submit mode: ${#SUBDIRS[@]} subdirectories under ${MODEL_ROOT}"
      for subdir in "${SUBDIRS[@]}"; do
        sub_name="$(basename "${subdir}")"
        job_tag="${sub_name//[^A-Za-z0-9_]/_}"
        job_tag="${job_tag:0:60}"
        job_name="EVAL_${PROJECT_NAME}_${job_tag}"
        job_name="${job_name//[^A-Za-z0-9_]/_}"
        job_name="${job_name:0:120}"

        echo "[INFO] Submitting: ${job_name} (MODEL_ROOT=${subdir}) nodes=${MIYABI_SELECT_NODES}"
        qsub -N "${job_name}" \
             -q "${MIYABI_QUEUE}" \
             -l "select=${MIYABI_SELECT_NODES}" \
             -l "walltime=${MIYABI_WALLTIME}" \
             -W "group_list=${MIYABI_GROUP}" \
             -v RUN_EVAL_SUBMITTED=1 \
             -v MODEL_ROOT="${subdir}" \
             -v EXP_NAMES="${EXP_NAMES}" \
             -v SUB_EXP_NAME="${sub_name}" \
             -v KEEP_EXPORTED_HF="${KEEP_EXPORTED_HF}" \
             -V \
             "$0"
      done
      echo "[INFO] Submitted ${#SUBDIRS[@]} jobs."
      exit 0
    fi
  fi

  # Single-submit mode (one PBS job for MODEL_ROOT)
  job_tag="${EXP_NAMES//[^A-Za-z0-9_]/_}"
  job_tag="${job_tag:0:60}"
  job_name="EVAL_${PROJECT_NAME}_${job_tag}"
  job_name="${job_name//[^A-Za-z0-9_]/_}"
  job_name="${job_name:0:120}"

  echo "[INFO] Submitting eval job: ${job_name} nodes=${MIYABI_SELECT_NODES}"
  qsub -N "${job_name}" \
       -q "${MIYABI_QUEUE}" \
       -l "select=${MIYABI_SELECT_NODES}" \
       -l "walltime=${MIYABI_WALLTIME}" \
       -W "group_list=${MIYABI_GROUP}" \
       -v RUN_EVAL_SUBMITTED=1 \
       -v KEEP_EXPORTED_HF="${KEEP_EXPORTED_HF}" \
       -V \
       "$0"
  exit 0
fi

# -------------------------------------------------------
# PBS job execution (single or multi-node)
# -------------------------------------------------------
cd "${PBS_O_WORKDIR:-$EVAL_ROOT}"
mkdir -p "${OUT_ROOT}"

NODE_LIST_FILE="${OUT_ROOT}/node_list.txt"
if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE}" ]]; then
  sort -u "${PBS_NODEFILE}" > "${NODE_LIST_FILE}"
else
  hostname > "${NODE_LIST_FILE}"
fi
NUM_NODES="$(wc -l < "${NODE_LIST_FILE}")"
echo "[INFO] Allocated node count: ${NUM_NODES}"

WRAPPER_SCRIPT="${OUT_ROOT}/pbs_worker_wrapper.sh"
cat << 'EOF_WRAPPER' > "${WRAPPER_SCRIPT}"
#!/bin/bash
export TZ="Asia/Tokyo"
export CUDA_HOME="/work/opt/local/aarch64/cores/cuda/12.8.1"
export CUDA_ROOT="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDA_LIB_PATH="${CUDA_HOME}/targets/sbsa-linux/lib"
EOF_WRAPPER

cat << EOF_WRAPPER >> "${WRAPPER_SCRIPT}"
export LD_LIBRARY_PATH="\${CUDA_LIB_PATH}:${WORK_HOME}/miniconda3/envs/eval/lib:\${LD_LIBRARY_PATH}"
export WORK_HOME="${WORK_HOME}"
export PROJECT_NAME="${PROJECT_NAME}"
export HF_HOME="${WORK_HOME}/cache/hf"
export CUDA_CACHE_PATH="${WORK_HOME}/cache/cuda"
export HF_HUB_OFFLINE=0
export RAY_TMPDIR=/tmp/ray
mkdir -p "\${RAY_TMPDIR}"

export PYTHONPATH="${ROOT_DIR}:${EVAL_ROOT}:\${PYTHONPATH}"
export TORCH_CPP_LOG_LEVEL=ERROR
export CUDA_VISIBLE_DEVICES="0"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export EVAL_ONE_MODEL_TIMEOUT="${EVAL_ONE_MODEL_TIMEOUT}"

if [[ -z "\${PASS_AT_KS:-}" ]]; then
  default_pass_ks=(1 8)
  pass_ks=()
  for k in "\${default_pass_ks[@]}"; do
    if (( k > 0 && k <= ${MAX_SAMPLE_NUMS} )) && [[ " \${pass_ks[*]} " != *" \$k "* ]]; then
      pass_ks+=("\$k")
    fi
  done
  PASS_AT_KS=\$(IFS=,; echo "\${pass_ks[*]}")
fi
export PASS_AT_KS

source "${WORK_HOME}/miniconda3/bin/activate" eval
export PATH="${WORK_HOME}/miniconda3/envs/eval/bin:\${PATH}"
PYTHON="${WORK_HOME}/miniconda3/envs/eval/bin/python"

NODE_LIST_FILE="${NODE_LIST_FILE}"
MY_HOSTNAME=\$(hostname)
if [ -f "\${NODE_LIST_FILE}" ]; then
  LINE_NUM=\$(grep -n "^\${MY_HOSTNAME}\$" "\${NODE_LIST_FILE}" | cut -d: -f1)
  if [ -z "\${LINE_NUM}" ]; then
    MY_RANK=0
    TOTAL_NODES=1
  else
    MY_RANK=\$((LINE_NUM - 1))
    TOTAL_NODES=\$(wc -l < "\${NODE_LIST_FILE}")
  fi
else
  MY_RANK=0
  TOTAL_NODES=1
fi

echo "[Wrapper] Host: \${MY_HOSTNAME}, Rank: \${MY_RANK}, Total: \${TOTAL_NODES}"

ARGS=(
  --model_root "${MODEL_ROOT}"
  --out_root "${OUT_ROOT}"
  --base_root "${BASE_ROOT}"
  --prompt_type "${PROMPT_TYPE}"
  --max_tokens_per_call "${MAX_TOKENS}"
  --nproc ${NUM_GPUS_PER_NODE}
  --use_vllm
  --pipeline_parallel_size 1
  --vllm_batch_size 0
  --temperature_g1 "${TEMP_G1}"
  --temperature_g2 "${TEMP_G2}"
  --n_sampling_g1 "${NSAMP_G1}"
  --n_sampling_g2 "${NSAMP_G2}"
  --shard_id "\${MY_RANK}"
  --num_shards "\${TOTAL_NODES}"
)

if [ "${SKIP_BASE_EVAL}" = "true" ]; then
  ARGS+=( --skip_base_eval )
fi

cd "${WORK_DIR}"
exec "\$PYTHON" -u "${WORK_DIR}/tools/run_qwen_eval_all_shared.py" "\${ARGS[@]}"
EOF_WRAPPER

chmod +x "${WRAPPER_SCRIPT}"

if [[ "${NUM_NODES}" -gt 1 ]]; then
  echo "[INFO] Multi-node mode (${NUM_NODES} nodes)."
  pbsdsh "${WRAPPER_SCRIPT}"
else
  echo "[INFO] Single-node mode."
  "${WRAPPER_SCRIPT}"
fi

EXIT_CODE=$?

# -------------------------------------------------------
# Merge results
# -------------------------------------------------------
if [[ "${EXIT_CODE}" -eq 0 ]]; then
  echo "[INFO] Evaluation finished. Merging results..."
  source "${WORK_HOME}/miniconda3/bin/activate" eval
  PYTHON="${WORK_HOME}/miniconda3/envs/eval/bin/python"

  if [[ "${SKIP_BASE_EVAL}" != "true" ]]; then
    shopt -s nullglob
    BASE_CANDIDATE_DIR="${OUT_ROOT}/base__$(basename "${BASE_ROOT}")"
    BASE_RUN_DIR=""
    if [[ -d "${BASE_CANDIDATE_DIR}" ]]; then
      BASE_RUN_DIR="${BASE_CANDIDATE_DIR}"
    else
      for d in "${OUT_ROOT}"/base__*; do
        if [[ -d "${d}" ]]; then
          BASE_RUN_DIR="${d}"
          break
        fi
      done
    fi
    shopt -u nullglob

    if [[ -n "${BASE_RUN_DIR}" ]]; then
      BASE_RUN_NAME="$(basename "${BASE_RUN_DIR}")"
      "${PYTHON}" "${WORK_DIR}/tools/merge_results.py" \
        --out_root "${OUT_ROOT}" \
        --run_name "${BASE_RUN_NAME}" \
        --prompt_type "${PROMPT_TYPE}"
    else
      echo "[WARN] No base__* run directory found under ${OUT_ROOT}, skip base merge."
    fi
  fi

  shopt -s nullglob
  RUN_DIRS=( "${OUT_ROOT}"/*__global_step_* )
  shopt -u nullglob
  if [[ "${#RUN_DIRS[@]}" -gt 0 ]]; then
    for RUN_DIR in "${RUN_DIRS[@]}"; do
      [[ -d "${RUN_DIR}" ]] || continue
      RUN_NAME="$(basename "${RUN_DIR}")"
      "${PYTHON}" "${WORK_DIR}/tools/merge_results.py" \
        --out_root "${OUT_ROOT}" \
        --run_name "${RUN_NAME}" \
        --prompt_type "${PROMPT_TYPE}"
    done
  else
    echo "[WARN] No *__global_step_* directories found under ${OUT_ROOT}, skip fine-tuned merges."
  fi
else
  echo "[ERROR] Evaluation failed with code ${EXIT_CODE}. Skip merge."
fi

# -------------------------------------------------------
# Cleanup exported HF models
# -------------------------------------------------------
if [[ "${KEEP_EXPORTED_HF}" != "true" ]]; then
  CLEANUP_TAG="${SUB_EXP_NAME:-${EXP_NAMES}}"
  echo "[INFO] Cleaning exported HF models in ${EXPORT_ROOT} (tag=${CLEANUP_TAG})"
  if [[ -d "${EXPORT_ROOT}" ]]; then
    find "${EXPORT_ROOT}" -maxdepth 1 -type d -name "*${CLEANUP_TAG}*" -exec rm -rf {} + 2>/dev/null || true
  fi
else
  echo "[INFO] KEEP_EXPORTED_HF=true, exported HF models retained."
fi

exit "${EXIT_CODE}"
