#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gtn-container_g8.24h
#$ -ac d=nvcr-cuda-12.4.1-ubuntu22.04,d_shm=256g
#$ -j y

set -x
set -e   # 如果希望有 worker 挂掉就整 job 失败，可以打开

PROJECT_NAME="VI-CURL"
EXP_NAMES="${EXP_NAMES:-VI-CURL_deepscaler_diff}"
MODEL_PATH="${MODEL_PATH:-checkpoints}" # giil | checkpoints

# 特殊 adapter 算法配置（需要特殊 base model 的算法）
# 格式: "algorithm1:suffix1,algorithm2:suffix2,..."
# 例如: "pissa:_pissa_base,qpissa:_qpissa_base"
export SPECIAL_ADAPTER_ALGORITHMS="${SPECIAL_ADAPTER_ALGORITHMS:-pissa:_pissa_base,qpissa:_qpissa_base}"

# -------------------------------------------------------
# Adaptive submit (use p90 scheduler on submit node)
#   - default: direct run
#   - submit: RUN_EVAL_SUBMIT=1 or --submit
# -------------------------------------------------------
# 解析命令行参数
KEEP_EXPORTED_HF="${KEEP_EXPORTED_HF:-false}"
RUN_EVAL_MULTI_SUBMIT="${RUN_EVAL_MULTI_SUBMIT:-false}"
MULTI_SUBMIT_BASE_ONCE="${MULTI_SUBMIT_BASE_ONCE:-true}"
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
    *)
      shift
      ;;
  esac
done

if [[ $RUN_EVAL_MULTI_SUBMIT == "true" ]]; then
  RUN_EVAL_SUBMIT=1
fi

# =======================================================
# 日志目录设置（在所有 qsub 逻辑之前）
# =======================================================
export TZ='JST-9'
_TS="$(date +%Y%m%d_%H%M%S)"

# 判断日志目录：直接运行 vs qsub 提交后
if [[ -n "${RUN_EVAL_SUBMITTED:-}" ]]; then
  _LOG_BASE="eval_log/eval_all/qsub_submit"
else
  _LOG_BASE="eval_log/eval_all/main"
fi
mkdir -p "${_LOG_BASE}" eval_log/eval_all/eval_gpus

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
  
  # 查找所有子目录
  SUBDIRS=()
  while IFS= read -r -d '' dir; do
    SUBDIRS+=("$dir")
  done < <(find "$_MODEL_ROOT" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
  
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
      echo "[INFO] Submitting base-only eval job: ${base_job_name} (MODEL_ROOT=${_MODEL_ROOT})"
      qsub -N "$base_job_name" \
           -jc "$base_jc_full" \
           -v NUM_GPUS="${base_n_gpus}" \
           -v MODEL_ROOT="${_MODEL_ROOT}" \
           -v EXP_NAMES="${EXP_NAMES}" \
           -v RUN_EVAL_SUBMITTED=1 \
           -v SKIP_STEP_EVAL=true \
           -v SKIP_BASE_EVAL=false \
           -v RUN_EVAL_MULTI_SUBMIT=false \
           -v KEEP_EXPORTED_HF="${KEEP_EXPORTED_HF}" \
           -V \
           "$0"
    else
      echo "[INFO] MULTI_SUBMIT_BASE_ONCE=false, skipping base-only job."
    fi
    
    for subdir in "${SUBDIRS[@]}"; do
      sub_name="$(basename "$subdir")"
      job_tag="${sub_name//[^A-Za-z0-9_]/_}"
      job_tag="${job_tag:0:60}"
      job_name="EVAL_${PROJECT_NAME}_${job_tag}"
      job_name="${job_name//[^A-Za-z0-9_]/_}"
      job_name="${job_name:0:120}"
      
      read -r jc_base n_gpus < <(select_resources_for_job "$PROJECT_NAME" "$job_name")
      jc_full="$(full_jclass_from_base "$jc_base")"
      
      echo "[INFO] Submitting: ${job_name} (MODEL_ROOT=${subdir})"
      qsub -N "$job_name" \
           -jc "$jc_full" \
           -v NUM_GPUS="${n_gpus}" \
           -v MODEL_ROOT="${subdir}" \
           -v EXP_NAMES="${EXP_NAMES}" \
           -v SUB_EXP_NAME="${sub_name}" \
           -v RUN_EVAL_SUBMITTED=1 \
           -v KEEP_EXPORTED_HF="${KEEP_EXPORTED_HF}" \
           -V \
           "$0"
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

  echo "[INFO] Submitting eval job: name=${job_name} jc=${jc_full} n_gpus=${n_gpus}"
  qsub -N "$job_name" \
       -jc "$jc_full" \
       -v NUM_GPUS="${n_gpus}" \
       -v RUN_EVAL_SUBMITTED=1 \
       -V \
       "$0"
  exit 0
fi

# 这里是 base 模型所在的「根目录」，里面有很多模型子目录
BASE_ROOT="${BASE_ROOT:-/hss/giil/caixq/model}"

PROMPT_TYPE="${PROMPT_TYPE:-think-boxed}"
MAX_TOKENS="${MAX_TOKENS:-3072}"

# 1. 自动探测 GPU 数量
if [[ -z "${NUM_GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --list-gpus | grep -c '^GPU')
    [[ "${NUM_GPUS}" -ge 1 ]] || NUM_GPUS=1
  else
    NUM_GPUS=1
  fi
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
MODEL_ROOT="${MODEL_ROOT:-/data/giil/caixq/ckpts/${EXP_NAMES}}"

MAX_SAMPLE_NUMS="${MAX_SAMPLE_NUMS:-8}"
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

# 导出目录（与 Python 代码中的 EXPORT_ROOT 一致）
EXPORT_ROOT="${EXPORT_ROOT:-${WORK_HOME:-/data/giil/caixq}/export}"

# Worker 超时设置（秒），防止单个 worker 卡死。默认 8 小时
WORKER_TIMEOUT="${WORKER_TIMEOUT:-28800}"

# 使用前面已设置的时间戳
TS="${_TS}"

echo "[INFO] Job started at ${TS}. Detected ${NUM_GPUS} GPUs."

TEMP_G1="${TEMP_G1:-0.6}"
TEMP_G2="${TEMP_G2:-0.0}"
NSAMP_G1="${NSAMP_G1:-${MAX_SAMPLE_NUMS}}"
NSAMP_G2="${NSAMP_G2:-${MAX_SAMPLE_NUMS}}"

export EVAL_ONE_MODEL_TIMEOUT="${EVAL_ONE_MODEL_TIMEOUT:-21600}"

# PASS@k 列表（受 MAX_SAMPLE_NUMS 限制）
if [[ -z "${PASS_AT_KS:-}" ]]; then
  # default_pass_ks=(1 8 16 32 64 128 256)
  default_pass_ks=(1 8)
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
    # 使用 _EXP_TAG 保持日志命名一致（multi-submit 时为算法名）
    LOG_FILE="eval_log/eval_all/eval_gpus/gpu_worker.${TS}.${_EXP_TAG}.rank_${i}.log"
    echo "[INFO] Starting Worker $i/$NUM_GPUS on GPU $i... Log: $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$i "$PYTHON_BIN" -u tools/run_qwen_eval_all_shared.py \
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
fi

echo "[INFO] All eval workers finished."

# =======================================================
# 3. 合并结果 (Merge Shards)
# =======================================================

echo "[INFO] Tasks completed. Starting merge process..."

cd "$PWD"  # 确保还在 repo 根目录（有 tools/merge_results.py）

# ---------- 3.1 合并 base__* 结果 ----------
if [ "$SKIP_BASE_EVAL" != "true" ]; then
    # 不再用 BASE_ROOT 的 basename，而是直接扫 OUT_ROOT 下的 base__*
    BASE_RUN_DIRS=$(ls -d "$OUT_ROOT"/base__* 2>/dev/null || true)

    if [ -n "$BASE_RUN_DIRS" ]; then
        for RUN_DIR in $BASE_RUN_DIRS; do
            if [ -d "$RUN_DIR" ]; then
                RUN_NAME=$(basename "$RUN_DIR")
                echo "[INFO] Merging results for base run: $RUN_NAME ..."
                "$PYTHON_BIN" tools/merge_results.py \
                  --out_root "$OUT_ROOT" \
                  --run_name "$RUN_NAME" \
                  --prompt_type "$PROMPT_TYPE"
            fi
        done
    else
        echo "[WARN] No base__* run directories found under $OUT_ROOT, skip base merge."
    fi
fi

# ---------- 3.2 合并各个 checkpoint 结果 ----------
# eval 脚本会在 $OUT_ROOT 下创建若干:
#   <run_name>__global_step_XXX
# 这里不再依赖 EXP_NAMES，直接扫所有 *__global_step_* 更稳
CKPT_RUN_DIRS=$(ls -d "$OUT_ROOT"/*__global_step_* 2>/dev/null || true)

if [ -n "$CKPT_RUN_DIRS" ]; then
    for RUN_DIR in $CKPT_RUN_DIRS; do
        if [ -d "$RUN_DIR" ]; then
            RUN_NAME=$(basename "$RUN_DIR")
            echo "[INFO] Merging results for checkpoint run: $RUN_NAME ..."
            "$PYTHON_BIN" tools/merge_results.py \
              --out_root "$OUT_ROOT" \
              --run_name "$RUN_NAME" \
              --prompt_type "$PROMPT_TYPE"
        fi
    done
else
    echo "[WARN] No *__global_step_* run directories found under $OUT_ROOT, nothing to merge."
fi

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
