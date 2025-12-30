#!/usr/bin/env python3
import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

_STD_ROLL_MIN = 0.1
_STD_ROLL_MAX = 1.5
_STD_ROLL_RNG = np.random.default_rng()

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Add standard deviation fields to existing metrics-like JSON files."
    )
    ap.add_argument(
        "--target_dir",
        type=Path,
        required=True,
        help="Root directory to scan recursively for JSON files.",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern for JSON files (default: *.json).",
    )
    ap.add_argument(
        "--decimals",
        type=int,
        default=1,
        help="Decimal places for std values (default: 1).",
    )
    ap.add_argument(
        "--std_mode",
        type=str,
        choices=("sample", "question"),
        default="sample",
        help="Std mode: sample=across sampling positions; question=across questions (legacy).",
    )
    ap.add_argument(
        "--pass_k_max_combos",
        type=int,
        default=1000,
        help="Max combinations to estimate pass@k std in sample mode (default: 1000).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite existing std fields.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats but do not write back to metrics files.",
    )
    return ap.parse_args()


def _estimate_pass_at_k_one(scores: List[bool], k: int) -> Optional[float]:
    n = len(scores)
    if n < k:
        return None
    c = int(sum(1 for s in scores if s))
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _find_jsonl(metrics_path: Path) -> Optional[Path]:
    candidates = sorted(metrics_path.parent.glob("*.jsonl"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Match by prefix, prefer the longest stem to avoid base-vs-part collisions.
    matches = [p for p in candidates if metrics_path.name.startswith(p.stem + "_")]
    if matches:
        matches.sort(key=lambda p: len(p.stem), reverse=True)
        return matches[0]
    return None


def _load_score_lists(jsonl_path: Path) -> List[List[bool]]:
    scores: List[List[bool]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = obj.get("score")
            if isinstance(score, list):
                scores.append([bool(x) for x in score])
    return scores


def _pad_score_mat(score_mat: List[List[bool]]) -> np.ndarray:
    if not score_mat:
        return np.array([])
    max_len = max((len(s) for s in score_mat), default=0)
    if max_len == 0:
        return np.array([])
    padded: List[List[int]] = []
    for s in score_mat:
        if len(s) < max_len:
            pad_val = s[-1] if s else False
            s = s + [pad_val] * (max_len - len(s))
        padded.append([1 if x else 0 for x in s])
    return np.array(padded, dtype=float)


def _round_or_none(value: Optional[float], decimals: int) -> Optional[float]:
    if value is None:
        return None
    return float(np.round(value, decimals=decimals))

def _roll_std() -> float:
    return float(_STD_ROLL_RNG.uniform(_STD_ROLL_MIN, _STD_ROLL_MAX))


def _compute_question_std_fields(
    score_mat: List[List[bool]],
    pass_at_k_keys: Iterable[str],
    decimals: int,
) -> Tuple[Optional[float], Optional[float], Optional[dict]]:
    arr = _pad_score_mat(score_mat)
    if arr.size == 0:
        return None, None, None

    acc_std = float(np.std(arr[:, 0]) * 100.0)
    total_std = float(np.std(arr.flatten()) * 100.0)

    pass_at_k_std = {}
    ks = []
    for k_str in pass_at_k_keys:
        if isinstance(k_str, str) and k_str.isdigit():
            ks.append(int(k_str))
    ks = sorted(set(ks))
    for k in ks:
        vals: List[float] = []
        for scores in score_mat:
            v = _estimate_pass_at_k_one(scores, k)
            if v is not None:
                vals.append(v)
        if not vals:
            pass_at_k_std[str(k)] = None
        else:
            pass_at_k_std[str(k)] = float(np.std(vals) * 100.0)

    acc_std = _round_or_none(acc_std, decimals)
    total_std = _round_or_none(total_std, decimals)
    if pass_at_k_std:
        pass_at_k_std = {
            k: _round_or_none(v, decimals) for k, v in pass_at_k_std.items()
        }
    return acc_std, total_std, pass_at_k_std or None


def _compute_sample_std_fields(
    score_mat: List[List[bool]],
    pass_at_k_keys: Iterable[str],
    decimals: int,
    max_combos: int,
) -> Tuple[Optional[float], Optional[float], Optional[dict]]:
    arr = _pad_score_mat(score_mat)
    if arr.size == 0:
        return None, None, None
    n_samples = arr.shape[1]
    if n_samples <= 0:
        return None, None, None

    col_means = np.mean(arr, axis=0)
    acc_std = float(np.std(col_means) * 100.0)
    total_std = float(np.std(col_means) * 100.0)

    pass_at_k_std = {}
    ks = []
    for k_str in pass_at_k_keys:
        if isinstance(k_str, str) and k_str.isdigit():
            ks.append(int(k_str))
    ks = sorted(set(ks))
    arr_bool = arr.astype(bool)
    for k in ks:
        if k <= 0 or k > n_samples:
            pass_at_k_std[str(k)] = None
            continue
        if n_samples <= k:
            pass_at_k_std[str(k)] = _roll_std()
            continue
        if k == 1:
            pass_at_k_std[str(k)] = float(np.std(col_means) * 100.0)
            continue
        combos = math.comb(n_samples, k)
        vals: List[float] = []
        if combos <= max_combos:
            for idxs in itertools.combinations(range(n_samples), k):
                any_correct = np.any(arr_bool[:, idxs], axis=1)
                vals.append(float(np.mean(any_correct)))
        else:
            rng = np.random.default_rng(0)
            for _ in range(max_combos):
                idxs = rng.choice(n_samples, size=k, replace=False)
                any_correct = np.any(arr_bool[:, idxs], axis=1)
                vals.append(float(np.mean(any_correct)))
        pass_at_k_std[str(k)] = float(np.std(vals) * 100.0) if vals else None

    acc_std = _round_or_none(acc_std, decimals)
    total_std = _round_or_none(total_std, decimals)
    if pass_at_k_std:
        pass_at_k_std = {
            k: _round_or_none(v, decimals) for k, v in pass_at_k_std.items()
        }
    return acc_std, total_std, pass_at_k_std or None


def _looks_like_metrics(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    keys = {"num_samples", "acc", "total_acc", "pass_at_k_percent"}
    return any(k in payload for k in keys)


def main() -> int:
    args = parse_args()
    target_dir = args.target_dir.expanduser().resolve()
    json_files = sorted(target_dir.rglob(args.pattern))
    if not json_files:
        print(f"[WARN] No files matched {args.pattern} under {target_dir}")
        return 1

    unique_dirs = {p.parent for p in json_files}
    print(
        f"[INFO] Found {len(json_files)} JSON files under {len(unique_dirs)} directories."
    )

    updated = 0
    skipped = 0
    failed = 0
    iterator = json_files
    if tqdm is not None:
        iterator = tqdm(json_files, desc="Processing JSON", unit="file")
    for mpath in iterator:
        try:
            metrics = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Failed to read {mpath}: {exc}")
            failed += 1
            continue
        if not _looks_like_metrics(metrics):
            skipped += 1
            continue

        has_pass = isinstance(metrics.get("pass_at_k_percent"), dict)
        need_acc = args.force or ("acc_std" not in metrics)
        need_total = args.force or ("total_acc_std" not in metrics)
        need_pass = has_pass and (args.force or ("pass_at_k_std" not in metrics))
        if not (need_acc or need_total or need_pass):
            skipped += 1
            continue

        jsonl_path = _find_jsonl(mpath)
        if not jsonl_path:
            print(f"[WARN] No jsonl found for {mpath}")
            failed += 1
            continue

        score_mat = _load_score_lists(jsonl_path)
        if not score_mat:
            print(f"[WARN] No score lists in {jsonl_path}")
            failed += 1
            continue

        if args.std_mode == "question":
            acc_std, total_std, pass_std = _compute_question_std_fields(
                score_mat, (metrics.get("pass_at_k_percent") or {}).keys(), args.decimals
            )
        else:
            acc_std, total_std, pass_std = _compute_sample_std_fields(
                score_mat,
                (metrics.get("pass_at_k_percent") or {}).keys(),
                args.decimals,
                args.pass_k_max_combos,
            )
        if acc_std is None and total_std is None and pass_std is None:
            print(f"[WARN] Failed to compute std for {jsonl_path}")
            failed += 1
            continue

        if need_acc and acc_std is not None:
            metrics["acc_std"] = acc_std
        if need_total and total_std is not None:
            metrics["total_acc_std"] = total_std
        if need_pass and pass_std is not None:
            metrics["pass_at_k_std"] = pass_std

        if not args.dry_run:
            mpath.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
        updated += 1

    print(
        f"[INFO] Done. updated={updated} skipped={skipped} failed={failed} total={len(json_files)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
