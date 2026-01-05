#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the main VI-CuRL LaTeX table from eval outputs.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


STEP_RE = re.compile(r"__global_step_(\d+)$", re.IGNORECASE)

DATASET_ORDER: List[Tuple[str, Tuple[str, str]]] = [
    ("aime24x8", ("AIME", "2024")),
    ("aime25x8", ("AIME", "2025")),
    ("amc23x8", ("AMC", "2023")),
    ("math500", ("Math500", "")),
    ("minerva_math", ("Minerva", "MATH")),
    ("olympiadbench", ("Olympiad", "Bench")),
]

ALL_ROW_GROUPS: List[Tuple[str, List[Tuple[str, List[str]]]]] = [
    (
        "Oracle (w. Verifier)",
        [
            ("No Curriculum", ["ver_rule_grpo_nocurl"]),
            ("VCRL", ["ver_rule_grpo_vcrl"]),
            ("AdaRFT", ["ver_rule_grpo_adarft"]),
            ("VI-CuRL", ["ver_rule_grpo_curl"]),
        ],
    ),
    (
        "Majority Vote (w/o. Verifier)",
        [
            ("No Curriculum", ["vf_majority_vote_nocurl"]),
            ("VCRL", ["vf_majority_vote_vcrl"]),
            ("AdaRFT", ["vf_majority_vote_adarft"]),
            ("VI-CuRL", ["vf_majority_vote_curl"]),
        ],
    ),
    (
        "Entropy (w/o. Verifier)",
        [
            ("No Curriculum", ["vf_entropy_nocurl"]),
            ("VCRL", ["vf_entropy_vcrl"]),
            ("AdaRFT", ["vf_entropy_adarft"]),
            ("VI-CuRL", ["vf_entropy_curl"]),
        ],
    ),
]

ORACLE_GROUPS: List[Tuple[str, List[Tuple[str, List[str]]]]] = [
    (
        "Oracle (w. Verifier)",
        [
            ("No Curriculum", ["ver_rule_grpo_nocurl"]),
            ("VCRL", ["ver_rule_grpo_vcrl"]),
            ("AdaRFT", ["ver_rule_grpo_adarft"]),
            ("VI-CuRL", ["ver_rule_grpo_curl"]),
        ],
    ),
]

VERIFIER_FREE_GROUPS: List[Tuple[str, List[Tuple[str, List[str]]]]] = [
    (
        "Majority Vote (w/o. Verifier)",
        [
            ("No Curriculum", ["vf_majority_vote_nocurl"]),
            ("VI-CuRL", ["vf_majority_vote_curl"]),
        ],
    ),
    (
        "Entropy (w/o. Verifier)",
        [
            ("No Curriculum", ["vf_entropy_nocurl"]),
            ("VI-CuRL", ["vf_entropy_curl"]),
        ],
    ),
]
VI_CURL_KEYS = {
    "ver_rule_grpo_curl",
    "vf_majority_vote_curl",
    "vf_entropy_curl",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_target = repo_root / "VI-CURL" / "eval_results" / "VI-CURL_deepscaler_diff_think-boxed"
    ap = argparse.ArgumentParser(description="Build VI-CuRL main table from eval outputs")
    ap.add_argument(
        "--target_dir",
        type=Path,
        default=default_target,
        help="Root dir of eval outputs",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="pass@1",
        help="Metric to use (default: pass@1)",
    )
    ap.add_argument(
        "--step_policy",
        choices=["last", "best"],
        default="last",
        help="Use last global step or best value across steps",
    )
    ap.add_argument(
        "--vi_curl_best",
        action="store_true",
        default=False,
        help="Force VI-CuRL rows to use best result even if step_policy=last",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=[
            "Qwen2.5-Math-1.5B",
            "DeepSeek-R1-Distill-Qwen-1.5B",
            "Llama3.2-3B-Instruct",
            "Qwen2.5-Math-7B",
        ],
        help="Model list in display order",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=default_target / "main_results" / "main_results.tex",
        help="Write table to file (default: <target_dir>/main_results/main_results.tex)",
    )
    return ap.parse_args()


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _match_model(run_name: str, models: List[str]) -> Optional[str]:
    norm = _normalize_key(run_name)
    best = None
    best_len = -1
    for m in models:
        key = _normalize_key(m)
        if key and key in norm:
            if len(key) > best_len:
                best = m
                best_len = len(key)
    return best


def _infer_algo_key(run_name: str, algo_keys: List[str]) -> Optional[str]:
    name = run_name.lower()
    if name.startswith("base__"):
        return "base"
    for key in algo_keys:
        if key in name:
            return key
    return None


def _scan_run_dirs(target_dir: Path) -> List[Path]:
    run_dirs: List[Path] = []
    for p in target_dir.iterdir():
        if not p.is_dir():
            continue
        has_sub_runs = any(
            (sub / "g1").is_dir() or (sub / "g2").is_dir()
            for sub in p.iterdir()
            if sub.is_dir()
        )
        if has_sub_runs:
            for sub in p.iterdir():
                if sub.is_dir() and ((sub / "g1").is_dir() or (sub / "g2").is_dir()):
                    run_dirs.append(sub)
        elif (p / "g1").is_dir() or (p / "g2").is_dir():
            run_dirs.append(p)
    return run_dirs


def _parse_metric_key(metric: str) -> Tuple[str, str]:
    metric = metric.strip().lower()
    if metric.startswith("pass@"):
        k = metric.split("@", 1)[1]
        if not k.isdigit():
            raise ValueError(f"Invalid pass@k metric: {metric}")
        return ("pass@", k)
    raise ValueError(f"Unsupported metric: {metric}")


def _load_dataset_metric(
    ds_dir: Path, metric_key: Tuple[str, str]
) -> Tuple[Optional[float], Optional[float], int]:
    metric_type, k = metric_key
    mpaths = sorted(ds_dir.glob("*metrics.json"))
    if not mpaths:
        return None, None, 0

    values: List[float] = []
    stds: List[float] = []
    for mpath in mpaths:
        try:
            metrics = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        if metric_type == "pass@":
            pass_at = metrics.get("pass_at_k_percent") or {}
            pass_std = metrics.get("pass_at_k_std") or {}
            val = pass_at.get(str(k))
            if val is None:
                continue
            std = pass_std.get(str(k))
            values.append(float(val))
            stds.append(float(std) if std is not None else np.nan)

    if not values:
        return None, None, len(mpaths)

    if len(values) == 1:
        return values[0], stds[0], len(mpaths)

    return float(np.mean(values)), float(np.std(values)), len(mpaths)


def load_results(target_dir: Path, models: List[str], metric: str) -> pd.DataFrame:
    target_dir = target_dir.expanduser().resolve()
    if not target_dir.exists():
        raise FileNotFoundError(f"Target dir not found: {target_dir}")

    metric_key = _parse_metric_key(metric)
    algo_keys = sorted(
        {"base"} | {k for _, rows in ALL_ROW_GROUPS for _, ks in rows for k in ks},
        key=len,
        reverse=True,
    )

    rows: List[Dict[str, object]] = []
    dataset_keys = {k for k, _ in DATASET_ORDER}
    metrics_files = 0

    run_dirs = _scan_run_dirs(target_dir)
    iterator = run_dirs
    if tqdm is not None:
        iterator = tqdm(run_dirs, desc="Scanning runs", unit="run")

    for run_dir in iterator:
        run_name = run_dir.name
        step = -1
        m = STEP_RE.search(run_name)
        if m:
            step = int(m.group(1))
            run_name = STEP_RE.sub("", run_name)

        algo_key = _infer_algo_key(run_name, algo_keys)
        if algo_key is None:
            continue

        if run_name.lower().startswith("base__"):
            base_raw = run_name[6:]
            base = _match_model(base_raw, models)
        else:
            base = _match_model(run_name, models)

        if not base:
            continue

        for g in ("g1", "g2"):
            gdir = run_dir / g
            if not gdir.is_dir():
                continue
            for ds_dir in gdir.iterdir():
                if not ds_dir.is_dir():
                    continue
                dataset = ds_dir.name.lower()
                if dataset not in dataset_keys:
                    continue
                value, std, found = _load_dataset_metric(ds_dir, metric_key)
                metrics_files += found
                if tqdm is not None and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(files=metrics_files)
                if value is None:
                    continue
                rows.append(
                    {
                        "base": base,
                        "algo": algo_key,
                        "step": step,
                        "dataset": dataset,
                        "metric": metric.lower(),
                        "value": float(value),
                        "std": float(std) if std is not None else np.nan,
                    }
                )

    df = pd.DataFrame(rows)
    df.attrs["metrics_files"] = metrics_files
    df.attrs["run_dirs"] = len(run_dirs)
    return df


def reduce_steps(df: pd.DataFrame, step_policy: str, vi_curl_best: bool) -> pd.DataFrame:
    if df.empty:
        return df

    group_cols = ["base", "algo", "dataset", "metric", "step"]
    df_step = (
        df.groupby(group_cols, as_index=False)
        .agg({"value": "mean", "std": "mean"})
        .copy()
    )

    df_step["__use_best"] = step_policy == "best"
    if vi_curl_best:
        df_step.loc[df_step["algo"].isin(VI_CURL_KEYS), "__use_best"] = True

    key_cols = ["base", "algo", "dataset", "metric"]

    df_best = df_step[df_step["__use_best"]].copy()
    df_last = df_step[~df_step["__use_best"]].copy()

    picked: List[pd.DataFrame] = []
    if not df_best.empty:
        df_best = df_best.sort_values(
            key_cols + ["value", "step"],
            ascending=[True, True, True, True, False, False],
        )
        picked.append(df_best.drop_duplicates(subset=key_cols, keep="first"))

    if not df_last.empty:
        df_last = df_last.sort_values(
            key_cols + ["step", "value"],
            ascending=[True, True, True, True, False, False],
        )
        picked.append(df_last.drop_duplicates(subset=key_cols, keep="first"))

    out = pd.concat(picked, ignore_index=True) if picked else df_step.head(0)
    return out.drop(columns=["__use_best"], errors="ignore")


def _format_cell(value: Optional[float], std: Optional[float], bold: bool) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    if std is None or not np.isfinite(std):
        cell = f"{value:.1f}"
    else:
        cell = f"{value:.1f} $\\pm$ {std:.1f}"
    if bold:
        return f"\\textbf{{{cell}}}"
    return cell


def _collect_row(
    df: pd.DataFrame,
    base: str,
    algo_keys: Iterable[str],
    bold: bool,
) -> Tuple[List[str], Optional[float], Optional[float]]:
    sub = df[(df["base"] == base) & (df["algo"].isin(algo_keys))]
    values: List[Optional[float]] = []
    stds: List[Optional[float]] = []
    cells: List[str] = []

    for ds_key, _ in DATASET_ORDER:
        ds_sub = sub[sub["dataset"] == ds_key]
        if ds_sub.empty:
            cells.append("-")
            values.append(None)
            stds.append(None)
            continue
        val = float(ds_sub["value"].mean())
        std = float(ds_sub["std"].mean()) if ds_sub["std"].notna().any() else np.nan
        cells.append(_format_cell(val, std, bold))
        values.append(val)
        stds.append(std)

    vals_f = [v for v in values if v is not None and np.isfinite(v)]
    stds_f = [s for s in stds if s is not None and np.isfinite(s)]
    if not vals_f:
        return cells, None, None

    avg_val = float(np.mean(vals_f))
    avg_std = float(np.mean(stds_f)) if stds_f else np.nan
    return cells, avg_val, avg_std


def build_table(
    df: pd.DataFrame,
    models: List[str],
    metric: str,
    caption: str,
    label: str,
    row_groups: List[Tuple[str, List[Tuple[str, List[str]]]]],
) -> str:
    lines: List[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\resizebox{\\linewidth}{!}{")
    lines.append("    \\begin{tabular}{lrrrrrr|r}")
    lines.append("      \\toprule")
    lines.append(
        "      \\rowcolor{gray!15} \\textbf{Dataset} & \\textbf{AIME} & \\textbf{AIME} & \\textbf{AMC} & \\textbf{Math500} & \\textbf{Minerva} & \\textbf{Olympiad} & \\textbf{Average} \\\\")
    lines.append(
        "      \\rowcolor{gray!15} & \\textbf{2024} & \\textbf{2025} & \\textbf{2023} & & \\textbf{MATH} & \\textbf{Bench} & \\\\")
    lines.append("      \\midrule")

    for mi, model in enumerate(models):
        lines.append(
            f"      \\multicolumn{{8}}{{c}}{{\\cellcolor{{gray!5}}\\textbf{{{model}}}}} \\\\")
        lines.append("      \\midrule")

        cells, avg_val, avg_std = _collect_row(df, model, ["base"], bold=False)
        avg_cell = _format_cell(avg_val, avg_std, bold=False)
        lines.append(
            "      Base (No RL) & " + " & ".join(cells + [avg_cell]) + " \\\\")
        lines.append("      \\midrule")

        for gi, (group_title, rows) in enumerate(row_groups):
            lines.append(f"      \\multicolumn{{8}}{{l}}{{\\textit{{{group_title}}}}} \\\\")
            for row_label, algo_keys in rows:
                is_vi_curl = row_label.lower() == "vi-curl"
                cells, avg_val, avg_std = _collect_row(df, model, algo_keys, bold=is_vi_curl)
                avg_cell = _format_cell(avg_val, avg_std, bold=is_vi_curl)
                prefix = "      "
                if is_vi_curl:
                    prefix += "\\rowcolor{blue!10} "
                lines.append(
                    prefix + row_label + " & " + " & ".join(cells + [avg_cell]) + " \\\\")
            if not (mi == len(models) - 1 and gi == len(row_groups) - 1):
                lines.append("      \\midrule")

    lines.append("      \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("  }")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    df_raw = load_results(args.target_dir, args.models, args.metric)
    metrics_files = df_raw.attrs.get("metrics_files")
    run_dirs = df_raw.attrs.get("run_dirs")
    if metrics_files is not None:
        print(f"[INFO] Metrics files found: {metrics_files}")
    if run_dirs is not None:
        print(f"[INFO] Run directories scanned: {run_dirs}")
    df = reduce_steps(df_raw, args.step_policy, args.vi_curl_best)
    oracle_caption = (
        f"Comparison with \\textbf{{Oracle (w. Verifier)}} reward. Mean and standard deviation "
        f"\\textbf{{({args.metric})}} with 16 samples and 5 random seeds. We compare VI-CuRL against "
        "Curriculum baselines (VCRL, AdaRFT) and No Curriculum. \\colorbox{blue!10}{Blue} rows highlight our method."
    )
    vf_caption = (
        "Comparison in \\textbf{Verifier-Free} settings. We compare VI-CuRL against baselines using "
        "\\textbf{Majority Vote} and \\textbf{Entropy} as intrinsic reward signals. Note that VCRL and "
        "AdaRFT are excluded as they require ground-truth verifiers. \\colorbox{blue!10}{Blue} rows highlight our method."
    )
    table_oracle = build_table(
        df,
        args.models,
        args.metric,
        oracle_caption,
        "tab:main_results_oracle",
        ORACLE_GROUPS,
    )
    table_vf = build_table(
        df,
        args.models,
        args.metric,
        vf_caption,
        "tab:main_results_independent",
        VERIFIER_FREE_GROUPS,
    )
    table = table_oracle + "\n\n" + table_vf
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")
        print(f"[INFO] Wrote LaTeX table to: {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()
