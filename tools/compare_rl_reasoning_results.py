#!/usr/bin/env python3
"""Compare eval results across rl_reasoning_results and noisy-RLVR main_results.

Outputs:
  - CSV table with per-model mean/std per source (base vs grpo)
  - Bar chart comparing mean across sources
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import sys
from statistics import mean, pstdev

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required. Install it or run in an env that has it."
    ) from exc

DATASETS = [
    ("g1/aime24x8", "aime24x8"),
    ("g1/aime25x8", "aime25x8"),
    ("g1/amc23x8", "amc23x8"),
    ("g2/math500", "math500"),
    ("g2/minerva_math", "minerva_math"),
    ("g2/olympiadbench", "olympiadbench"),
]

STEP_RE = re.compile(r"__global_step_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-model base/grpo results across rl_reasoning_results and noisy-RLVR."
        )
    )
    parser.add_argument(
        "--rl-root",
        default="/home/caixq/project/LLM_EVAL/rl_reasoning_results",
        help="Root of rl_reasoning_results.",
    )
    parser.add_argument(
        "--noisy-root",
        default="/home/caixq/project/noisy-RLVR/eval_final_var/main_results",
        help="Root of noisy-RLVR main_results.",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/caixq/project/LLM_EVAL/rl_reasoning_results/_comparison",
        help="Output directory for tables and plots.",
    )
    parser.add_argument(
        "--metric",
        default="acc",
        help="Metric key: acc, total_acc, pass@1, pass@8, etc.",
    )
    parser.add_argument(
        "--pass-ks",
        default="",
        help="Comma-separated K values for pass@k outputs (e.g., 1,8).",
    )
    parser.add_argument(
        "--steps",
        choices=["best-all", "best-grpo", "keep-all"],
        default="best-grpo",
        help=(
            "How to handle multiple __global_step_ runs: best-all, best-grpo, or keep-all."
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Max directory depth to scan under each root.",
    )
    parser.add_argument(
        "--rl-include",
        action="append",
        default=["base__*", "*grpo*"],
        help="Glob pattern(s) to include from rl_reasoning_results.",
    )
    parser.add_argument(
        "--noisy-include",
        action="append",
        default=["*"],
        help="Glob pattern(s) to include from noisy-RLVR main_results.",
    )
    return parser.parse_args()


def matches_patterns(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def find_metrics_file(dataset_dir: str) -> str | None:
    try:
        candidates = [
            f
            for f in os.listdir(dataset_dir)
            if f.endswith("_metrics.json") and os.path.isfile(os.path.join(dataset_dir, f))
        ]
    except FileNotFoundError:
        return None
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(dataset_dir, candidates[0])


def get_metric_value(data: dict, metric: str) -> float | None:
    if metric.startswith("pass@"):
        try:
            k = metric.split("@", 1)[1]
        except IndexError:
            return None
        return data.get("pass_at_k_percent", {}).get(k)
    return data.get(metric)


def find_complete_runs(
    root: str,
    include_patterns: list[str],
    max_depth: int,
) -> list[dict]:
    runs = []
    root = os.path.abspath(root)
    root_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        name = os.path.basename(dirpath)
        if name.startswith("_"):
            continue
        if not matches_patterns(name, include_patterns):
            continue
        if not all(os.path.isdir(os.path.join(dirpath, d[0])) for d in DATASETS):
            continue
        run = {
            "path": dirpath,
            "name": name,
        }
        runs.append(run)
    return runs


def load_run_data(run: dict) -> dict | None:
    data_map = {}
    for rel_path, label in DATASETS:
        dataset_dir = os.path.join(run["path"], rel_path)
        metrics_file = find_metrics_file(dataset_dir)
        if not metrics_file:
            print(f"[WARN] Missing metrics file in {dataset_dir}", file=sys.stderr)
            return None
        try:
            with open(metrics_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Failed reading {metrics_file}: {exc}", file=sys.stderr)
            return None
        data_map[label] = data
    run["data"] = data_map
    return run


def compute_metric_stats(run: dict, metric: str) -> dict | None:
    values = {}
    for label, data in run["data"].items():
        val = get_metric_value(data, metric)
        if val is None:
            print(
                f"[WARN] Missing metric '{metric}' in {run['path']}",
                file=sys.stderr,
            )
            return None
        values[label] = float(val)
    avg = mean(values.values())
    avg_std = pstdev(values.values())
    run_copy = dict(run)
    run_copy["values"] = values
    run_copy["avg"] = avg
    run_copy["avg_std"] = avg_std
    return run_copy


def detect_pass_ks(source_runs: dict[str, list[dict]]) -> list[str]:
    per_source: dict[str, set[str]] = {}
    for source, runs in source_runs.items():
        source_ks: set[str] = set()
        for run in runs:
            run_keys: set[str] | None = None
            for data in run["data"].values():
                keys = {str(key) for key in data.get("pass_at_k_percent", {}).keys()}
                run_keys = keys if run_keys is None else run_keys & keys
            if run_keys:
                source_ks |= run_keys
        if source_ks:
            per_source[source] = source_ks

    if not per_source:
        return []

    common: set[str] | None = None
    for ks in per_source.values():
        common = ks if common is None else common & ks
    if not common:
        return []

    def sort_key(val: str) -> tuple[int, float | str]:
        try:
            return (0, float(val))
        except ValueError:
            return (1, val)

    return sorted(common, key=sort_key)


def clean_label(label: str) -> str:
    return label.replace("_nocurl", "").replace("_oracle", "")


def infer_models(roots: list[str]) -> list[str]:
    models = set()
    for root in roots:
        try:
            entries = os.listdir(root)
        except FileNotFoundError:
            continue
        for name in entries:
            if name.startswith("base__"):
                models.add(name.split("base__", 1)[1])
    return sorted(models, key=len, reverse=True)


def extract_model(name: str, models: list[str]) -> str | None:
    for model in models:
        if model in name:
            return model
    return None


def extract_algo(name: str, model: str | None) -> str:
    base = STEP_RE.sub("", name)
    if base.startswith("base__"):
        return "base"
    algo = base
    if model and model in base:
        algo = base.replace(model, "")
    algo = algo.strip("_")
    algo = algo.replace("_nocurl_", "_").replace("_oracle_", "_")
    algo = algo.replace("_nocurl", "").replace("_oracle", "")
    algo = algo.strip("_")
    return algo or base


def group_runs(runs: list[dict], steps_mode: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for run in runs:
        match = STEP_RE.search(run["name"])
        step = int(match.group(1)) if match else None
        base_name = STEP_RE.sub("", run["name"])
        run["step"] = step
        run["group_name"] = base_name
        groups.setdefault(base_name, []).append(run)

    selected = []
    for base_name, entries in groups.items():
        if len(entries) == 1:
            entry = entries[0]
            entry["label"] = clean_label(entry["name"])
            selected.append(entry)
            continue

        if steps_mode == "keep-all":
            for entry in entries:
                entry["label"] = clean_label(entry["name"])
                selected.append(entry)
            continue

        best = max(entries, key=lambda x: (x["avg"], x["step"] or -1))
        best["label"] = clean_label(base_name)
        selected.append(best)

    return selected


def categorize_run(name: str) -> str:
    base = STEP_RE.sub("", name).lower()
    if base.startswith("base__"):
        return "base"
    if "grpo" in base:
        return "grpo"
    return "other"


def select_best_per_category(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["model"], row["category"])
        grouped.setdefault(key, []).append(row)
    selected = []
    for (_, _), entries in grouped.items():
        best = max(entries, key=lambda x: (x["avg"], x["step"] or -1))
        selected.append(best)
    return selected


def compare_across_sources(rows: list[dict]) -> list[dict]:
    combined: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        key = (row["model"], row["category"])
        combined.setdefault(key, {})[row["source"]] = row
    output = []
    for (model, category), sources in sorted(combined.items()):
        if "rl_reasoning_results" not in sources or "noisy_RLVR" not in sources:
            continue
        output.append(
            {
                "model": model,
                "category": category,
                "rl": sources.get("rl_reasoning_results"),
                "noisy": sources.get("noisy_RLVR"),
            }
        )
    return output


def write_csv(rows: list[dict], metric: str, out_csv: str) -> None:
    fieldnames = [
        "model",
        "category",
        "rl_path",
        "rl_algo",
        "rl_mean",
        "rl_std",
        "rl_step",
        "noisy_path",
        "noisy_algo",
        "noisy_mean",
        "noisy_std",
        "noisy_step",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            rl = row["rl"]
            noisy = row["noisy"]
            writer.writerow(
                {
                    "model": row["model"],
                    "category": row["category"],
                    "rl_path": rl.get("path") if rl else "",
                    "rl_algo": rl.get("algo") if rl else "",
                    "rl_mean": f"{rl['avg']:.4f}" if rl else "",
                    "rl_std": f"{rl['avg_std']:.4f}" if rl else "",
                    "rl_step": rl.get("step") if rl else "",
                    "noisy_path": noisy.get("path") if noisy else "",
                    "noisy_algo": noisy.get("algo") if noisy else "",
                    "noisy_mean": f"{noisy['avg']:.4f}" if noisy else "",
                    "noisy_std": f"{noisy['avg_std']:.4f}" if noisy else "",
                    "noisy_step": noisy.get("step") if noisy else "",
                }
            )


def write_bar_chart(rows: list[dict], metric: str, out_png: str) -> None:
    labels = [f"{row['category']}:{row['model']}" for row in rows]
    rl_values = [row["rl"]["avg"] if row["rl"] else 0.0 for row in rows]
    noisy_values = [row["noisy"]["avg"] if row["noisy"] else 0.0 for row in rows]
    rl_err = [row["rl"]["avg_std"] if row["rl"] else 0.0 for row in rows]
    noisy_err = [row["noisy"]["avg_std"] if row["noisy"] else 0.0 for row in rows]

    x = list(range(len(labels)))
    width = 0.38

    fig_width = max(12, 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.bar(
        [i - width / 2 for i in x],
        rl_values,
        width=width,
        yerr=rl_err,
        capsize=3,
        color="#1f77b4",
        label="rl_reasoning_results",
    )
    ax.bar(
        [i + width / 2 for i in x],
        noisy_values,
        width=width,
        yerr=noisy_err,
        capsize=3,
        color="#ff7f0e",
        label="noisy_RLVR",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(f"{metric} (mean across 6 datasets)")
    ax.set_title("Per-model comparison across sources")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)


def main() -> int:
    args = parse_args()

    rl_runs = find_complete_runs(args.rl_root, args.rl_include, args.max_depth)
    noisy_runs = find_complete_runs(args.noisy_root, args.noisy_include, args.max_depth)
    models = infer_models([args.rl_root, args.noisy_root])

    source_runs = {}
    for source, runs in (
        ("rl_reasoning_results", rl_runs),
        ("noisy_RLVR", noisy_runs),
    ):
        loaded = []
        for run in runs:
            category = categorize_run(run["name"])
            if category not in {"base", "grpo"}:
                continue
            run_data = load_run_data(run)
            if run_data is None:
                continue
            model = extract_model(run_data["name"], models)
            run_data["model"] = model or "unknown"
            run_data["algo"] = extract_algo(run_data["name"], model)
            run_data["category"] = category
            run_data["source"] = source
            loaded.append(run_data)
        source_runs[source] = loaded

    metrics = []
    if args.metric.strip():
        metrics.append(args.metric)

    pass_k_list: list[str] = []
    if args.pass_ks:
        for item in args.pass_ks.split(","):
            token = item.strip()
            if token:
                pass_k_list.append(token)
    else:
        pass_k_list = detect_pass_ks(source_runs)

    for token in pass_k_list:
        metrics.append(f"pass@{token}")

    seen = set()
    metrics = [m for m in metrics if not (m in seen or seen.add(m))]

    wrote_any = False
    for metric in metrics:
        all_rows = []
        for source, loaded in source_runs.items():
            metric_rows = []
            for run in loaded:
                run_data = compute_metric_stats(run, metric)
                if run_data is None:
                    continue
                metric_rows.append(run_data)
            grouped = group_runs(metric_rows, args.steps)
            grouped = [
                row for row in grouped if row.get("category") in {"base", "grpo"}
            ]
            best_by_category = select_best_per_category(grouped)
            all_rows.extend(best_by_category)

        if not all_rows:
            print(f"[WARN] No runs found for metric '{metric}'.", file=sys.stderr)
            continue

        comparison_rows = compare_across_sources(all_rows)
        if not comparison_rows:
            print(
                f"[WARN] No comparable rows for metric '{metric}'.",
                file=sys.stderr,
            )
            continue

        os.makedirs(args.out_dir, exist_ok=True)
        metric_tag = metric.replace("@", "at")
        out_csv = os.path.join(args.out_dir, f"comparison_{metric_tag}.csv")
        out_png = os.path.join(args.out_dir, f"comparison_{metric_tag}.png")

        write_csv(comparison_rows, metric, out_csv)
        write_bar_chart(comparison_rows, metric, out_png)

        print(f"[OK] Wrote {out_csv}")
        print(f"[OK] Wrote {out_png}")
        wrote_any = True

    if not wrote_any:
        print("[ERROR] No outputs were generated.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
