#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

mkdir -p "${tmpdir}/model_root/run-a" "${tmpdir}/model_root/run-b" "${tmpdir}/out"

output="$(
  cd "${REPO_ROOT}" && \
  RUN_EVAL_MONITOR=1 \
  RUN_EVAL_PLAN_ONLY=1 \
  MODEL_ROOT="${tmpdir}/model_root" \
  OUT_ROOT="${tmpdir}/out" \
  EXP_NAMES="grpo_baselines" \
  PYTHON_BIN=/bin/true \
  SCHEDULER_TOOL=/nonexistent \
  bash ./scripts/run_eval_all_tmux.sh --base-only
)"

assert_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "${haystack}" != *"${needle}"* ]]; then
    echo "expected output to contain: ${needle}" >&2
    exit 1
  fi
}

assert_not_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "${haystack}" == *"${needle}"* ]]; then
    echo "expected output to not contain: ${needle}" >&2
    exit 1
  fi
}

assert_contains "${output}" "PLAN_TMUX_SESSION grpo_baselines_base_only"
assert_contains "${output}" "PLAN_JOB_COUNT 1"
assert_contains "${output}" "PLAN_JOB BASE_ONLY EVAL_LLM_EVAL_grpo_baselines_BASEONLY false true ${tmpdir}/out"
assert_not_contains "${output}" "run-a"
assert_not_contains "${output}" "run-b"

echo "ok"
