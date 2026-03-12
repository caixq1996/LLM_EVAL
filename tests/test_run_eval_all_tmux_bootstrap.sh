#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

mkdir -p "${tmpdir}/job_scripts"
cp "${REPO_ROOT}/scripts/run_eval_all_tmux.sh" "${tmpdir}/job_scripts/18250999"

output="$(
  cd "${REPO_ROOT}" && \
  RUN_EVAL_SUBMITTED=1 \
  RUN_EVAL_BOOTSTRAP_ONLY=1 \
  bash "${tmpdir}/job_scripts/18250999"
)"

if [[ "${output}" != *"BOOTSTRAP_OK"* ]]; then
  echo "expected BOOTSTRAP_OK, got: ${output}" >&2
  exit 1
fi

echo "ok"
