#!/usr/bin/env python3
"""
Backfill pass@k metrics into *_metrics.json using nearby .jsonl samples.

Usage:
  python tools/backfill_passk.py --root /path/to/eval_results
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parent.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from utils import load_jsonl
from grader import math_equal
from tools.merge_results import _compute_sample_std_fields


def _compute_pass_at_k(score_mat: List[List[bool]], ks: Iterable[int]) -> Tuple[dict, dict]:
    def _estimate_pass_at_k_one(scores, k):
        n = len(scores)
        if n < k:
            return None
        c = int(sum(1 for s in scores if s))
        if c == 0:
            return 0.0
        if n - c < k:
            return 1.0
        from math import comb
        return 1.0 - (comb(n - c, k) / comb(n, k))

    results = {}
    counts = {}
    for k in ks:
        vals = []
        for scores in score_mat:
            v = _estimate_pass_at_k_one(scores, k)
            if v is not None:
                vals.append(v)
        if not vals:
            results[str(k)] = None
            counts[str(k)] = 0
        else:
            results[str(k)] = float(np.round(np.mean(vals) * 100.0, 1))
            counts[str(k)] = len(vals)
    return results, counts


def _target_ks(max_len: int) -> List[int]:
    base = [1, 2, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    return [k for k in base if k <= max_len]


def _load_samples_from_dir(ds_dir: Path) -> List[dict]:
    jsonl_files = sorted([p for p in ds_dir.glob("*.jsonl") if "_part" not in p.name])
    if jsonl_files:
        return list(load_jsonl(str(jsonl_files[0])))

    part_files = sorted(ds_dir.glob("*_part*.jsonl"))
    if not part_files:
        return []

    merged = {}
    for p in part_files:
        for sample in load_jsonl(str(p)):
            idx = sample.get("idx")
            if idx is None:
                idx = len(merged)
            if idx not in merged:
                merged[idx] = sample
    return [merged[k] for k in sorted(merged.keys())]


def _build_score_mat(samples: List[dict]) -> List[List[bool]]:
    score_mat = []
    for s in samples:
        scores = s.get("score")
        if isinstance(scores, list) and scores:
            score_mat.append([bool(x) for x in scores])
            continue
        preds = s.get("pred") or []
        gt = s.get("gt")
        if gt is None:
            continue
        pred_scores = []
        for pred in preds:
            try:
                pred_scores.append(bool(math_equal(pred, gt)))
            except Exception:
                pred_scores.append(False)
        if pred_scores:
            score_mat.append(pred_scores)
    return score_mat


def _needs_update(metrics: dict, ks: List[int]) -> bool:
    existing = metrics.get("pass_at_k_percent", {})
    existing_std = metrics.get("pass_at_k_std", {})
    existing_counts = metrics.get("pass_at_k_valid_counts", {})
    for k in ks:
        key = str(k)
        if key not in existing or existing.get(key) is None:
            return True
        if key not in existing_std:
            return True
        if key not in existing_counts:
            return True
    return False


def _update_metrics(metrics_path: Path, dry_run: bool = False) -> bool:
    ds_dir = metrics_path.parent
    samples = _load_samples_from_dir(ds_dir)
    if not samples:
        return False

    score_mat = _build_score_mat(samples)
    if not score_mat:
        return False

    max_len = max((len(s) for s in score_mat), default=0)
    if max_len <= 0:
        return False

    ks = _target_ks(max_len)
    if not ks:
        return False

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    if not _needs_update(metrics, ks):
        return False

    pass_at_k_percent, pass_at_k_valid_counts = _compute_pass_at_k(score_mat, ks)
    _, _, pass_at_k_std = _compute_sample_std_fields(
        score_mat=score_mat,
        pass_at_k_keys=[str(k) for k in ks],
        decimals=1
    )

    metrics.setdefault("pass_at_k_percent", {}).update(pass_at_k_percent)
    metrics.setdefault("pass_at_k_valid_counts", {}).update(pass_at_k_valid_counts)
    if pass_at_k_std:
        metrics.setdefault("pass_at_k_std", {}).update(pass_at_k_std)

    if dry_run:
        return True

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Root directory to scan")
    parser.add_argument("--dry-run", action="store_true", help="Report updates without writing")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        print(f"[ERROR] Root not found: {root}")
        sys.exit(1)

    updated = 0
    scanned = 0
    for metrics_path in root.rglob("*_metrics.json"):
        scanned += 1
        try:
            if _update_metrics(metrics_path, dry_run=args.dry_run):
                updated += 1
                print(f"[OK] Updated: {metrics_path}")
        except Exception as e:
            print(f"[WARN] Failed: {metrics_path} -> {e}")

    print(f"[DONE] scanned={scanned} updated={updated}")


if __name__ == "__main__":
    main()
