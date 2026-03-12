#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmpdir}"
}
trap cleanup EXIT

mkdir -p \
  "${tmpdir}/model/run_a/global_step_313" \
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

cat > "${tmpdir}/bin/fake_python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
capture_file="${PY_CAPTURE:?}"
cat >> "${capture_file}"
printf '\n===PY_END===\n' >> "${capture_file}"
exit 0
EOF
chmod +x "${tmpdir}/bin/fake_python"

cat > "${tmpdir}/bin/fake_qstat_name" <<'EOF'
#!/usr/bin/env bash
echo "JOBID OWNER NAME"
EOF
chmod +x "${tmpdir}/bin/fake_qstat_name"

output="$(
  cd "${REPO_ROOT}" && \
  RUN_EVAL_MONITOR=1 \
  MODEL_ROOT="${tmpdir}/model" \
  OUT_ROOT="${tmpdir}/out" \
  EXP_NAMES="grpo_baselines" \
  PYTHON_BIN="${tmpdir}/bin/fake_python" \
  PY_CAPTURE="${tmpdir}/py_stdin.txt" \
  SCHEDULER_TOOL="${tmpdir}/fake_scheduler.sh" \
  QSTAT_NAME_TOOL="${tmpdir}/bin/fake_qstat_name" \
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

captured="$(cat "${tmpdir}/py_stdin.txt")"
assert_not_contains "${captured}" "finalize_run"
assert_contains "${captured}" "check_missing_by_group"
assert_contains "${output}" "All jobs complete"

echo "ok"
