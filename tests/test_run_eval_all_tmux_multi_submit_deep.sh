#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

tmpdir="$(mktemp -d)"
session_name="test_multi_submit_deep_$$"
cleanup() {
  rm -rf "${tmpdir}"
}
trap cleanup EXIT

mkdir -p \
  "${tmpdir}/model/run_a/global_step_100" \
  "${tmpdir}/model/run_a/global_step_313" \
  "${tmpdir}/model/run_b/global_step_313" \
  "${tmpdir}/out" \
  "${tmpdir}/bin"

cat > "${tmpdir}/fake_scheduler.sh" <<'EOF'
select_resources_for_job() {
  echo "gtn-container_g1 1"
}

full_jclass_from_base() {
  echo "${1}.24h"
}
EOF

cat > "${tmpdir}/bin/qsub" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
capture_file="${QSUB_CAPTURE:?}"
printf 'ARGS:%s\n' "$*" >> "${capture_file}"
echo "Your job 999 (\"fake\") has been submitted"
EOF
chmod +x "${tmpdir}/bin/qsub"

cat > "${tmpdir}/bin/tmux" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
capture_file="${TMUX_CAPTURE:?}"
printf 'ARGS:%s\n' "$*" >> "${capture_file}"
exit 0
EOF
chmod +x "${tmpdir}/bin/tmux"

startup_output="$(
  cd "${REPO_ROOT}" && \
  PATH="${tmpdir}/bin:${PATH}" \
  QSUB_CAPTURE="${tmpdir}/qsub_calls.txt" \
  TMUX_CAPTURE="${tmpdir}/tmux_calls.txt" \
  MODEL_ROOT="${tmpdir}/model" \
  OUT_ROOT="${tmpdir}/out" \
  EXP_NAMES="grpo_baselines" \
  SCHEDULER_TOOL="${tmpdir}/fake_scheduler.sh" \
  TMUX_SESSION_NAME="${session_name}" \
  EVAL_STEPS="313" \
  bash ./scripts/run_eval_all_tmux.sh --multi-submit-deep
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

assert_contains "${startup_output}" "Started tmux session: ${session_name}"
assert_contains "${startup_output}" "Attach with: tmux attach -t ${session_name}"
assert_not_contains "${startup_output}" "Submitted 2 jobs."

tmux_calls="$(cat "${tmpdir}/tmux_calls.txt")"
assert_contains "${tmux_calls}" "new-session -d -s ${session_name}"
assert_contains "${tmux_calls}" "--multi-submit-deep"

if [[ -f "${tmpdir}/qsub_calls.txt" ]]; then
  echo "expected startup path to only create tmux monitor, but qsub was called" >&2
  cat "${tmpdir}/qsub_calls.txt" >&2
  exit 1
fi

plan_output="$(
  cd "${REPO_ROOT}" && \
  PATH="${tmpdir}/bin:${PATH}" \
  RUN_EVAL_MONITOR=1 \
  RUN_EVAL_PLAN_ONLY=1 \
  MODEL_ROOT="${tmpdir}/model" \
  OUT_ROOT="${tmpdir}/out" \
  EXP_NAMES="grpo_baselines" \
  SCHEDULER_TOOL="${tmpdir}/fake_scheduler.sh" \
  EVAL_STEPS="313" \
  bash ./scripts/run_eval_all_tmux.sh --multi-submit-deep
)"

assert_contains "${plan_output}" "PLAN_TMUX_SESSION grpo_baselines"
assert_contains "${plan_output}" "PLAN_JOB_COUNT 3"
assert_contains "${plan_output}" "PLAN_JOB BASE_ONLY EVAL_LLM_EVAL_grpo_baselines_BASE false true ${tmpdir}/out"
assert_contains "${plan_output}" "PLAN_JOB run_a/global_step_313 EVAL_LLM_EVAL_run_a_global_step_313 true false ${tmpdir}/out"
assert_contains "${plan_output}" "PLAN_JOB run_b/global_step_313 EVAL_LLM_EVAL_run_b_global_step_313 true false ${tmpdir}/out"
assert_not_contains "$(printf '%s\n' "${plan_output}" | grep '^PLAN_JOB ' || true)" "global_step_100"

echo "ok"
