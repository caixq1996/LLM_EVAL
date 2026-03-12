#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parent.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from utils import load_jsonl
from tools.backfill_passk import _update_metrics
from tools.merge_results import merge_shard_files


_DEFAULT_GROUP_DATASETS = (
    "aime25x8,amc23x8,aime24x8",
    "minerva_math,olympiadbench,math500",
)

_DEFAULT_EXPECTED_SAMPLES = {
    "aime24x8": 240,
    "aime25x8": 240,
    "amc23x8": 320,
    "math500": 500,
    "minerva_math": 272,
    "olympiadbench": 675,
}


def _split_ds_list(datasets: str) -> list[str]:
    return [d.strip() for d in (datasets or "").split(",") if d.strip()]


def _group_datasets() -> tuple[tuple[str, ...], tuple[str, ...]]:
    g1 = tuple(_split_ds_list(os.getenv("EVAL_GROUP1_DATASETS", _DEFAULT_GROUP_DATASETS[0])))
    g2 = tuple(_split_ds_list(os.getenv("EVAL_GROUP2_DATASETS", _DEFAULT_GROUP_DATASETS[1])))
    return g1, g2


def _required_pass_k() -> int:
    for key in ("EVAL_REQUIRED_PASS_K", "MAX_SAMPLE_NUMS"):
        raw = os.getenv(key, "").strip()
        if raw.isdigit():
            return int(raw)
    ks_env = os.getenv("PASS_AT_KS", "").strip()
    ks = [int(x) for x in ks_env.replace(" ", "").split(",") if x.isdigit()]
    if ks:
        return max(ks)
    return 0


def _expected_samples() -> dict[str, int]:
    expected = dict(_DEFAULT_EXPECTED_SAMPLES)
    override = os.getenv("EVAL_EXPECTED_SAMPLES_JSON", "").strip()
    if not override:
        return expected
    try:
        data = json.loads(override)
    except json.JSONDecodeError:
        return expected
    if not isinstance(data, dict):
        return expected
    for key, value in data.items():
        try:
            expected[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return expected


def _find_final_metrics(ds_dir: Path) -> Path | None:
    metrics = sorted(p for p in ds_dir.glob("*_metrics.json") if "_part" not in p.name)
    return metrics[0] if metrics else None


def _find_final_jsonl(ds_dir: Path) -> Path | None:
    jsonl_files = sorted(p for p in ds_dir.glob("*.jsonl") if "_part" not in p.name)
    return jsonl_files[0] if jsonl_files else None


def _load_samples_from_dir(ds_dir: Path) -> list[dict]:
    final_jsonl = sorted(p for p in ds_dir.glob("*.jsonl") if "_part" not in p.name)
    if final_jsonl:
        return list(load_jsonl(str(final_jsonl[0])))

    merged = {}
    for part_file in sorted(ds_dir.glob("*_part*.jsonl")):
        for sample in load_jsonl(str(part_file)):
            idx = sample.get("idx")
            if idx is None:
                idx = len(merged)
            if idx not in merged:
                merged[idx] = sample
    return [merged[idx] for idx in sorted(merged)]


def _sample_output_count(sample: dict) -> int:
    scores = sample.get("score")
    if isinstance(scores, list):
        return len(scores)
    preds = sample.get("pred")
    if isinstance(preds, list):
        return len(preds)
    return 0


def dataset_raw_complete(ds_dir: Path, ds_name: str, required_pass_k: int | None = None) -> bool:
    if not ds_dir.exists():
        return False

    samples = _load_samples_from_dir(ds_dir)
    expected = _expected_samples().get(ds_name, 0)
    if expected > 0 and len(samples) < expected:
        return False
    if not samples:
        return False

    required = _required_pass_k() if required_pass_k is None else int(required_pass_k)
    if required <= 0:
        return True

    counts = [_sample_output_count(sample) for sample in samples]
    return bool(counts) and min(counts) >= required


def dataset_final_complete(ds_dir: Path, ds_name: str, required_pass_k: int | None = None) -> bool:
    metrics_path = _find_final_metrics(ds_dir)
    if metrics_path is None:
        return False

    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False

    expected = _expected_samples().get(ds_name, 0)
    if expected > 0 and int(data.get("num_samples", 0)) < expected:
        return False

    required = _required_pass_k() if required_pass_k is None else int(required_pass_k)
    if required <= 0:
        return True

    key = str(required)
    passk = data.get("pass_at_k_percent")
    counts = data.get("pass_at_k_valid_counts")
    stds = data.get("pass_at_k_std")
    if not isinstance(passk, dict) or passk.get(key) is None:
        return False
    if not isinstance(counts, dict):
        return False
    try:
        if expected > 0 and int(counts.get(key, 0)) < expected:
            return False
    except (TypeError, ValueError):
        return False
    if not isinstance(stds, dict) or stds.get(key) is None:
        return False
    return True


def check_missing_by_group(out_root, run_name: str, final_required: bool = False) -> dict[int, list[str]]:
    root = Path(out_root)
    run_out = root / run_name
    g1, g2 = _group_datasets()
    missing = {1: [], 2: []}
    checker = dataset_final_complete if final_required else dataset_raw_complete

    for group_idx, datasets in ((1, g1), (2, g2)):
        gdir = run_out / f"g{group_idx}"
        for ds_name in datasets:
            ds_dir = gdir / ds_name
            if not checker(ds_dir, ds_name):
                missing[group_idx].append(ds_name)
    return missing


def iter_run_names(out_root) -> list[str]:
    root = Path(out_root)
    run_names = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        child_names = {child.name for child in path.iterdir() if child.is_dir()}
        if "g1" in child_names or "g2" in child_names:
            run_names.append(path.relative_to(root).as_posix())
    return sorted(set(run_names))


def _iter_dataset_dirs(run_dir: Path):
    for group_name in ("g1", "g2"):
        group_dir = run_dir / group_name
        if not group_dir.exists():
            continue
        for ds_dir in sorted(child for child in group_dir.iterdir() if child.is_dir()):
            yield ds_dir


def finalize_run(out_root, run_name: str, prompt_type: str) -> None:
    root = Path(out_root)
    run_dir = root / run_name
    if not run_dir.exists():
        return

    needs_merge = False
    for ds_dir in _iter_dataset_dirs(run_dir):
        if any(ds_dir.glob("*_part*.jsonl")):
            needs_merge = True
            break
        if _find_final_metrics(ds_dir) is None and _find_final_jsonl(ds_dir) is not None:
            needs_merge = True
            break

    if needs_merge:
        merge_shard_files(
            str(root),
            run_name,
            prompt_type,
            fast_mode=True,
            recover_missing_scores=True,
        )

    for metrics_path in sorted(run_dir.rglob("*_metrics.json")):
        if "_part" in metrics_path.name:
            continue
        ds_dir = metrics_path.parent
        ds_name = ds_dir.name
        if not dataset_final_complete(ds_dir, ds_name):
            _update_metrics(metrics_path)


def finalize_out_root(out_root, prompt_type: str) -> list[str]:
    finalized = []
    for run_name in iter_run_names(out_root):
        finalize_run(out_root=out_root, run_name=run_name, prompt_type=prompt_type)
        finalized.append(run_name)
    return finalized


def _cmd_finalize(args) -> int:
    out_root = Path(args.out_root).expanduser().resolve()
    if args.run_name:
        finalize_run(out_root=out_root, run_name=args.run_name, prompt_type=args.prompt_type)
    else:
        finalize_out_root(out_root=out_root, prompt_type=args.prompt_type)
    return 0


def _cmd_check(args) -> int:
    out_root = Path(args.out_root).expanduser().resolve()
    missing = check_missing_by_group(out_root=out_root, run_name=args.run_name, final_required=args.final_required)
    complete = not any(missing[group] for group in missing)
    if complete:
        print("COMPLETE")
        return 0
    print(json.dumps(missing, ensure_ascii=False, sort_keys=True))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--out-root", required=True)
    finalize_parser.add_argument("--prompt-type", default="think-boxed")
    finalize_parser.add_argument("--run-name", default="")
    finalize_parser.set_defaults(func=_cmd_finalize)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--out-root", required=True)
    check_parser.add_argument("--run-name", required=True)
    check_parser.add_argument("--final-required", action="store_true")
    check_parser.set_defaults(func=_cmd_check)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
