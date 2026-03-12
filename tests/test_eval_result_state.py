import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import merge_results
from tools.eval_result_state import check_missing_by_group, finalize_out_root


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _part_samples(num_items: int, score_len: int):
    rows = []
    for idx in range(num_items):
        rows.append(
            {
                "idx": idx,
                "question": f"q{idx}",
                "gt": "42",
                "pred": ["42"] * score_len,
                "score": [True] * score_len,
            }
        )
    return rows


def _pred_only_samples(num_items: int, pred_len: int):
    rows = _part_samples(num_items=num_items, score_len=pred_len)
    for row in rows:
        row.pop("score", None)
        row["report"] = ["ok"] * pred_len
    return rows


def _fake_evaluate_with_scores(expected_score_len: int):
    def _fake(data_name, prompt_type, samples=None, file_path=None, max_num_samples=None, execute=False):
        evaluated = []
        for sample in samples or []:
            updated = dict(sample)
            updated["score"] = [True] * expected_score_len
            evaluated.append(updated)
        result_json = {
            "num_samples": len(evaluated),
            "acc": 100.0,
            "total_acc": 100.0,
            "pass_at_k_percent": {str(expected_score_len): 100.0},
            "pass_at_k_valid_counts": {str(expected_score_len): len(evaluated)},
            "acc_std": 0.0,
            "total_acc_std": 0.0,
            "pass_at_k_std": {str(expected_score_len): 0.0},
        }
        return evaluated, result_json

    return _fake


def _fake_evaluate_assert_serial(expected_score_len: int):
    def _fake(data_name, prompt_type, samples=None, file_path=None, max_num_samples=None, execute=False):
        assert os.environ.get("EVAL_MP_WORKERS") == "1"
        assert os.environ.get("EVAL_THREAD_WORKERS") == "1"
        return _fake_evaluate_with_scores(expected_score_len)(
            data_name,
            prompt_type,
            samples=samples,
            file_path=file_path,
            max_num_samples=max_num_samples,
            execute=execute,
        )

    return _fake


@pytest.fixture(autouse=True)
def _eval_env(monkeypatch):
    monkeypatch.setenv("EVAL_GROUP1_DATASETS", "aime24x8")
    monkeypatch.setenv("EVAL_GROUP2_DATASETS", "")
    monkeypatch.setenv("EVAL_EXPECTED_SAMPLES_JSON", json.dumps({"aime24x8": 2}))
    monkeypatch.setenv("EVAL_REQUIRED_PASS_K", "8")
    monkeypatch.delenv("PASS_AT_KS", raising=False)


def test_part_only_outputs_are_eval_complete_but_not_final_complete(tmp_path: Path):
    out_root = tmp_path / "eval_results"
    ds_dir = out_root / "run_a" / "g1" / "aime24x8"
    _write_jsonl(ds_dir / "test_part0.jsonl", _part_samples(num_items=2, score_len=8))

    raw_missing = check_missing_by_group(out_root=out_root, run_name="run_a")
    final_missing = check_missing_by_group(out_root=out_root, run_name="run_a", final_required=True)

    assert raw_missing == {1: [], 2: []}
    assert final_missing == {1: ["aime24x8"], 2: []}


def test_finalize_out_root_merges_nested_run_and_backfills_required_pass_k(tmp_path: Path):
    out_root = tmp_path / "eval_results"
    ds_dir = out_root / "algo_x" / "global_step_313" / "Model__global_step_313" / "g1" / "aime24x8"
    _write_jsonl(ds_dir / "test_think-boxed_part0.jsonl", _part_samples(num_items=2, score_len=8))

    finalize_out_root(out_root=out_root, prompt_type="think-boxed")

    final_metrics = ds_dir / "test_think-boxed_think-boxed_metrics.json"
    assert final_metrics.exists()
    data = json.loads(final_metrics.read_text(encoding="utf-8"))
    assert data["num_samples"] == 2
    assert data["pass_at_k_percent"]["8"] == 100.0

    final_missing = check_missing_by_group(
        out_root=out_root,
        run_name="algo_x/global_step_313/Model__global_step_313",
        final_required=True,
    )
    assert final_missing == {1: [], 2: []}


def test_insufficient_sampling_depth_stays_incomplete(tmp_path: Path):
    out_root = tmp_path / "eval_results"
    ds_dir = out_root / "run_b" / "g1" / "aime24x8"
    _write_jsonl(ds_dir / "test_part0.jsonl", _part_samples(num_items=2, score_len=4))

    raw_missing = check_missing_by_group(out_root=out_root, run_name="run_b")

    assert raw_missing == {1: ["aime24x8"], 2: []}


def test_finalize_out_root_recovers_missing_scores_from_final_jsonl(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EVAL_GROUP1_DATASETS", "")
    monkeypatch.setenv("EVAL_GROUP2_DATASETS", "olympiadbench")
    monkeypatch.setenv("EVAL_EXPECTED_SAMPLES_JSON", json.dumps({"olympiadbench": 2}))
    monkeypatch.setenv("EVAL_REQUIRED_PASS_K", "8")
    monkeypatch.setattr(merge_results, "evaluate", _fake_evaluate_with_scores(expected_score_len=8))

    out_root = tmp_path / "eval_results"
    ds_dir = out_root / "base__ModelA" / "g2" / "olympiadbench"
    final_jsonl = ds_dir / "test_think-boxed_-1_seed0_t0.8_s0_e-1.jsonl"
    _write_jsonl(final_jsonl, _pred_only_samples(num_items=2, pred_len=8))

    finalize_out_root(out_root=out_root, prompt_type="think-boxed")

    metrics_file = ds_dir / "test_think-boxed_-1_seed0_t0.8_s0_e-1_think-boxed_metrics.json"
    assert metrics_file.exists()
    final_missing = check_missing_by_group(out_root=out_root, run_name="base__ModelA", final_required=True)
    assert final_missing == {1: [], 2: []}

    rows = [json.loads(line) for line in final_jsonl.read_text(encoding="utf-8").splitlines()]
    assert all("score" in row for row in rows)


def test_finalize_out_root_recovers_mixed_scores_from_part_jsonl(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EVAL_GROUP1_DATASETS", "")
    monkeypatch.setenv("EVAL_GROUP2_DATASETS", "minerva_math")
    monkeypatch.setenv("EVAL_EXPECTED_SAMPLES_JSON", json.dumps({"minerva_math": 2}))
    monkeypatch.setenv("EVAL_REQUIRED_PASS_K", "8")
    monkeypatch.setattr(merge_results, "evaluate", _fake_evaluate_with_scores(expected_score_len=8))

    out_root = tmp_path / "eval_results"
    ds_dir = out_root / "algo_x" / "global_step_313" / "Model__global_step_313" / "g2" / "minerva_math"
    part0_rows = _part_samples(num_items=1, score_len=8)
    part1_rows = _pred_only_samples(num_items=1, pred_len=8)
    part1_rows[0]["idx"] = 1
    _write_jsonl(ds_dir / "test_think-boxed_-1_seed0_t0.8_s0_e-1_part0.jsonl", part0_rows)
    _write_jsonl(ds_dir / "test_think-boxed_-1_seed0_t0.8_s0_e-1_part1.jsonl", part1_rows)

    finalize_out_root(out_root=out_root, prompt_type="think-boxed")

    final_jsonl = ds_dir / "test_think-boxed_-1_seed0_t0.8_s0_e-1.jsonl"
    metrics_file = ds_dir / "test_think-boxed_-1_seed0_t0.8_s0_e-1_think-boxed_metrics.json"
    assert final_jsonl.exists()
    assert metrics_file.exists()
    assert not list(ds_dir.glob("*_part*.jsonl"))

    final_missing = check_missing_by_group(
        out_root=out_root,
        run_name="algo_x/global_step_313/Model__global_step_313",
        final_required=True,
    )
    assert final_missing == {1: [], 2: []}

    rows = [json.loads(line) for line in final_jsonl.read_text(encoding="utf-8").splitlines()]
    assert all("score" in row for row in rows)


def test_finalize_out_root_forces_serial_score_recovery(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EVAL_GROUP1_DATASETS", "")
    monkeypatch.setenv("EVAL_GROUP2_DATASETS", "olympiadbench")
    monkeypatch.setenv("EVAL_EXPECTED_SAMPLES_JSON", json.dumps({"olympiadbench": 2}))
    monkeypatch.setenv("EVAL_REQUIRED_PASS_K", "8")
    monkeypatch.delenv("EVAL_MP_WORKERS", raising=False)
    monkeypatch.delenv("EVAL_THREAD_WORKERS", raising=False)
    monkeypatch.setattr(merge_results, "evaluate", _fake_evaluate_assert_serial(expected_score_len=8))

    out_root = tmp_path / "eval_results"
    ds_dir = out_root / "base__ModelB" / "g2" / "olympiadbench"
    final_jsonl = ds_dir / "test_think-boxed_-1_seed0_t0.8_s0_e-1.jsonl"
    _write_jsonl(final_jsonl, _pred_only_samples(num_items=2, pred_len=8))

    finalize_out_root(out_root=out_root, prompt_type="think-boxed")

    metrics_file = ds_dir / "test_think-boxed_-1_seed0_t0.8_s0_e-1_think-boxed_metrics.json"
    assert metrics_file.exists()
