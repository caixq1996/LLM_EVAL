# run_eval_all_tmux g1 Resume Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `scripts/run_eval_all_tmux.sh` default to `g1` jobs, correctly honor `--force-g1-family`, resume from existing shard outputs under `eval_results/grpo_baselines_think-boxed`, and only stop when final merged metrics with `pass@1024` exist.

**Architecture:** Keep the shell script as the orchestrator, but move fragile result-state logic into small Python helpers and a sourced shell library. Separate "raw shard outputs are enough to skip re-eval" from "final merged metrics are complete" so the monitor can resume, finalize, and only submit when compute is actually needed.

**Tech Stack:** Bash, Python 3, `pytest`, existing `tools/merge_results.py`, existing `tools/backfill_passk.py`

---

### Task 1: Add failing tests for result completeness and finalize behavior

**Files:**
- Create: `tests/test_eval_result_state.py`

**Step 1: Write the failing test**

Cover these cases with temporary directories:
- part-only shard outputs satisfy raw completeness but not final completeness
- finalization produces merged metrics with required `pass@k`
- raw outputs with too few samples per item do not satisfy `pass@1024`
- recursive finalization handles nested single-run output layout

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_eval_result_state.py -q`
Expected: FAIL because the helper APIs do not exist yet and current completeness logic is too weak.

### Task 2: Add failing tests for g1 job-class normalization

**Files:**
- Create: `tests/test_run_eval_all_tmux_lib.sh`

**Step 1: Write the failing test**

Source a new shell helper library and assert:
- default behavior coerces scheduler output to `*_g1`
- `FORCE_G1_FAMILY=gtb` forces `gtb-container_g1`
- `FORCE_G1_FAMILY='gtn|gtb'` preserves selected family while forcing `g1`

**Step 2: Run test to verify it fails**

Run: `bash tests/test_run_eval_all_tmux_lib.sh`
Expected: FAIL because the helper library does not exist yet.

### Task 3: Implement result-state helper and finalize flow

**Files:**
- Create: `tools/eval_result_state.py`
- Modify: `tools/run_qwen_eval_all_shared.py`

**Step 1: Write minimal implementation**

Implement helper functions to:
- load merged or part `.jsonl` outputs
- detect raw completeness vs final completeness
- require `pass@1024` using env-configured `EVAL_REQUIRED_PASS_K`
- support test overrides for expected dataset sizes
- finalize runs recursively via merge + backfill

**Step 2: Run targeted tests**

Run: `python3 -m pytest tests/test_eval_result_state.py -q`
Expected: PASS

### Task 4: Implement g1 scheduling helper and wire script to it

**Files:**
- Create: `scripts/lib/run_eval_all_tmux_lib.sh`
- Modify: `scripts/run_eval_all_tmux.sh`

**Step 1: Write minimal implementation**

Refactor submission sites to use one helper that:
- defaults to `g1`
- honors `FORCE_G1_FAMILY`
- treats `gtn|gtb` as "auto family, forced g1 scale"

Also replace the current shallow merge step with recursive finalization and pass `EVAL_REQUIRED_PASS_K=1024`.

**Step 2: Run shell regression test**

Run: `bash tests/test_run_eval_all_tmux_lib.sh`
Expected: PASS

### Task 5: Verify against the real `grpo_baselines` tree

**Files:**
- Modify: `scripts/run_eval_all_tmux.sh`
- Modify: `tools/eval_result_state.py`

**Step 1: Dry-run real checks**

Run targeted helper commands against:
- `/data/giil/caixq/ckpts/grpo_baselines`
- `/home/caixq/project/LLM_EVAL/eval_results/grpo_baselines_think-boxed`

Verify:
- missing final metrics are detected
- existing parts can be finalized without re-eval where appropriate
- truly incomplete datasets remain pending

**Step 2: Run full verification**

Run:
- `python3 -m pytest tests/test_eval_result_state.py -q`
- `bash tests/test_run_eval_all_tmux_lib.sh`
- targeted helper commands on real directories

Expected: all pass

### Task 6: Execute the real script

**Files:**
- Modify: `scripts/run_eval_all_tmux.sh`

**Step 1: Run the production command**

Run: `cd /home/caixq/project/LLM_EVAL && RUN_EVAL_MONITOR=1 scripts/run_eval_all_tmux.sh`

**Step 2: Confirm submission behavior**

Check fresh log output for:
- `g1` job classes
- base-only plus per-model jobs as needed
- resume/finalize behavior on existing shard results
