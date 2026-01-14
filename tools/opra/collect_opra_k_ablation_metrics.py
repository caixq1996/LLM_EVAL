#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _split_csv(value: str) -> List[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _parse_step_dir_name(name: str) -> Optional[Tuple[str, int]]:
    m = re.search(r"__global_step_(\d+)$", name)
    if not m:
        return None
    return name[: m.start()], int(m.group(1))


def _parse_k(run_name: str) -> Optional[int]:
    m = re.search(r"_k(\d+)", run_name, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_lora_rank(run_name: str) -> Optional[int]:
    m = re.search(r"(?:_r|_lora)(\d+)", run_name, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _get_pass_k(data: Dict, k: int) -> Optional[float]:
    for key in ("pass_at_k_percent", "pass_at_k", "pass@k", "pass@k_percent"):
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, dict):
            if k in value:
                return value[k]
            k_str = str(k)
            if k_str in value:
                return value[k_str]
    return None


def _metric_from_data(data: Dict, metric: str) -> Optional[float]:
    metric = (metric or "").lower()
    if metric.startswith("pass@"):
        try:
            k = int(metric.split("@", 1)[1])
        except Exception:
            return None
        val = _get_pass_k(data, k)
        if val is not None:
            return val
        return data.get("acc") or data.get("total_acc")
    if metric in ("acc", "accuracy"):
        return data.get("acc") or data.get("total_acc") or _get_pass_k(data, 1)
    if metric in ("total_acc", "total-acc"):
        return data.get("total_acc") or data.get("acc") or _get_pass_k(data, 1)
    return None


def _find_metrics_file(run_dir: Path, dataset: str, prompt_type: str) -> Optional[Path]:
    candidates = sorted(run_dir.glob(f"**/{dataset}/*metrics.json"))
    if prompt_type:
        candidates = [p for p in candidates if prompt_type in p.name]
    if not candidates:
        return None
    return candidates[-1]


def _load_metric(run_dir: Path, dataset: str, prompt_type: str, metric_key: str) -> Optional[float]:
    metrics_path = _find_metrics_file(run_dir, dataset, prompt_type)
    if metrics_path is None:
        return None
    try:
        data = json.loads(metrics_path.read_text())
    except Exception:
        return None
    return _metric_from_data(data, metric_key)


def _collect_runs(eval_root: Path, run_filter: str) -> Dict[str, Path]:
    run_map: Dict[str, Tuple[int, Path]] = {}
    for entry in eval_root.iterdir():
        if not entry.is_dir():
            continue
        parsed = _parse_step_dir_name(entry.name)
        if not parsed:
            continue
        run_name, step = parsed
        if run_name.startswith("base__"):
            continue
        if run_filter and run_filter not in run_name:
            continue
        prev = run_map.get(run_name)
        if prev is None or step > prev[0]:
            run_map[run_name] = (step, entry)
    return {name: path for name, (_, path) in run_map.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--prompt-type", type=str, default="")
    ap.add_argument("--performance-datasets", type=str, default="aime24x8,aime25x8,amc23x8")
    ap.add_argument("--knowledge-datasets", type=str, default="minerva_math,olympiadbench,math500")
    ap.add_argument("--performance-metric", type=str, default="pass@1")
    ap.add_argument("--knowledge-metric", type=str, default="acc")
    ap.add_argument("--run-filter", type=str, default="")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    eval_root = args.eval_root
    if not eval_root.exists():
        raise SystemExit(f"[ERROR] eval_root not found: {eval_root}")

    perf_datasets = _split_csv(args.performance_datasets)
    know_datasets = _split_csv(args.knowledge_datasets)
    runs = _collect_runs(eval_root, args.run_filter)

    rows: List[Dict] = []
    missing: Dict[str, List[str]] = {}
    for run_name, run_dir in sorted(runs.items()):
        k_val = _parse_k(run_name)
        if k_val is None:
            continue
        lora_rank = _parse_lora_rank(run_name)
        perf_vals = {}
        know_vals = {}
        for ds in perf_datasets:
            val = _load_metric(run_dir, ds, args.prompt_type, args.performance_metric)
            if val is None:
                missing.setdefault(run_name, []).append(ds)
            else:
                perf_vals[ds] = val
        for ds in know_datasets:
            val = _load_metric(run_dir, ds, args.prompt_type, args.knowledge_metric)
            if val is None:
                missing.setdefault(run_name, []).append(ds)
            else:
                know_vals[ds] = val
        if not perf_vals or not know_vals:
            continue
        perf_mean = sum(perf_vals.values()) / len(perf_vals)
        know_mean = sum(know_vals.values()) / len(know_vals)
        rows.append({
            "k": k_val,
            "lora_rank": lora_rank,
            "k_over_r": (k_val / lora_rank) if lora_rank else None,
            "run": run_name,
            "step": int(run_dir.name.split("__global_step_")[-1]),
            "performance": perf_mean,
            "knowledge": know_mean,
            "performance_by_dataset": perf_vals,
            "knowledge_by_dataset": know_vals,
        })

    rows.sort(key=lambda r: (r.get("lora_rank") or 0, r["k"], r["run"]))
    payload = {
        "eval_root": str(eval_root),
        "prompt_type": args.prompt_type,
        "performance_metric": args.performance_metric,
        "knowledge_metric": args.knowledge_metric,
        "performance_datasets": perf_datasets,
        "knowledge_datasets": know_datasets,
        "rows": rows,
        "missing": missing,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2))

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["k", "lora_rank", "k_over_r", "run", "step", "performance", "knowledge"])
            for row in rows:
                writer.writerow([
                    row["k"],
                    row.get("lora_rank"),
                    row.get("k_over_r"),
                    row["run"],
                    row["step"],
                    row["performance"],
                    row["knowledge"],
                ])

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
