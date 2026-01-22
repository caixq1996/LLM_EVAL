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
FORMAT_DECIMALS = 1

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
    default_rl_reasoning = repo_root / "LLM_EVAL" / "rl_reasoning_results"
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
        "--rl_reasoning_dir",
        type=Path,
        default=default_rl_reasoning,
        help="Root dir for base/ver_rule_grpo_nocurl results (default: LLM_EVAL/rl_reasoning_results)",
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


def _infer_algo_key(run_name: str, algo_keys: List[str], allow_nocurl_alias: bool = False) -> Optional[str]:
    name = run_name.lower()
    if name.startswith("base__"):
        return "base"
    for key in algo_keys:
        if key in name:
            return key
    if allow_nocurl_alias and "ver_rule_grpo" in name:
        return "ver_rule_grpo_nocurl"
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
) -> Tuple[Optional[float], Optional[float], int, List[str]]:
    metric_type, k = metric_key
    mpaths = sorted(ds_dir.glob("*metrics.json"))
    if not mpaths:
        return None, None, 0, []

    values: List[float] = []
    stds: List[float] = []
    used_paths: List[str] = []
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
            used_paths.append(str(mpath))

    if not values:
        return None, None, len(mpaths), []

    if len(values) == 1:
        return values[0], stds[0], len(mpaths), used_paths

    return float(np.mean(values)), float(np.std(values)), len(mpaths), used_paths


def _normalize_paths(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _merge_paths(values: Iterable[object]) -> List[str]:
    merged: List[str] = []
    for item in values:
        merged.extend(_normalize_paths(item))
    return sorted(set(merged))


def _load_results_dir(
    target_dir: Path,
    models: List[str],
    metric: str,
    algo_keys: List[str],
    only_algos: Optional[Iterable[str]] = None,
    allow_nocurl_alias: bool = False,
) -> Tuple[pd.DataFrame, int, int]:
    target_dir = target_dir.expanduser().resolve()
    if not target_dir.exists():
        raise FileNotFoundError(f"Target dir not found: {target_dir}")

    metric_key = _parse_metric_key(metric)
    dataset_keys = {k for k, _ in DATASET_ORDER}
    rows: List[Dict[str, object]] = []
    metrics_files = 0

    only_set = {a for a in only_algos} if only_algos else None

    run_dirs = _scan_run_dirs(target_dir)
    iterator = run_dirs
    if tqdm is not None:
        iterator = tqdm(run_dirs, desc=f"Scanning runs: {target_dir.name}", unit="run")

    for run_dir in iterator:
        run_name = run_dir.name
        step = -1
        m = STEP_RE.search(run_name)
        if m:
            step = int(m.group(1))
            run_name = STEP_RE.sub("", run_name)

        algo_key = _infer_algo_key(run_name, algo_keys, allow_nocurl_alias=allow_nocurl_alias)
        if algo_key is None:
            continue
        if only_set is not None and algo_key not in only_set:
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
                value, std, found, used_paths = _load_dataset_metric(ds_dir, metric_key)
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
                        "paths": used_paths,
                    }
                )

    df = pd.DataFrame(
        rows,
        columns=[
            "base",
            "algo",
            "step",
            "dataset",
            "metric",
            "value",
            "std",
            "paths",
        ],
    )
    return df, metrics_files, len(run_dirs)


def load_results(
    target_dir: Path,
    models: List[str],
    metric: str,
    rl_reasoning_dir: Optional[Path] = None,
) -> pd.DataFrame:
    algo_keys = sorted(
        {"base"} | {k for _, rows in ALL_ROW_GROUPS for _, ks in rows for k in ks},
        key=len,
        reverse=True,
    )

    df_main, metrics_main, runs_main = _load_results_dir(
        target_dir,
        models,
        metric,
        algo_keys,
    )

    override_algos = {"base", "ver_rule_grpo_nocurl"}
    df_override = pd.DataFrame()
    metrics_override = 0
    runs_override = 0
    if rl_reasoning_dir is not None:
        rl_reasoning_dir = rl_reasoning_dir.expanduser().resolve()
        if rl_reasoning_dir.exists():
            df_override, metrics_override, runs_override = _load_results_dir(
                rl_reasoning_dir,
                models,
                metric,
                algo_keys,
                only_algos=override_algos,
                allow_nocurl_alias=True,
            )
        else:
            print(f"[WARN] rl_reasoning_dir not found: {rl_reasoning_dir}")

    if not df_override.empty:
        override_pairs = df_override[["base", "algo"]].drop_duplicates()
        df_main = df_main.merge(
            override_pairs.assign(__drop__=1),
            on=["base", "algo"],
            how="left",
        )
        df_main = df_main[df_main["__drop__"].isna()].drop(columns=["__drop__"])
        df = pd.concat([df_main, df_override], ignore_index=True)
    else:
        df = df_main

    df.attrs["metrics_files"] = metrics_main + metrics_override
    df.attrs["run_dirs"] = runs_main + runs_override
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


def attach_paths(df_raw: pd.DataFrame, df_selected: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty or df_selected.empty:
        return df_selected
    cols = ["base", "algo", "dataset", "metric", "step"]
    raw_paths = df_raw[cols + ["paths"]].copy()
    raw_paths["paths"] = raw_paths["paths"].apply(_normalize_paths)
    grouped = (
        raw_paths.groupby(cols, as_index=False)["paths"]
        .agg(_merge_paths)
        .copy()
    )
    return df_selected.merge(grouped, on=cols, how="left")


def write_source_map(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    cols = ["base", "algo", "step", "dataset", "metric", "value", "std", "paths"]
    out_df = df[cols].copy()
    out_df["paths"] = out_df["paths"].apply(
        lambda items: ";".join(_normalize_paths(items))
    )
    out_df.sort_values(cols[:5], inplace=True)
    output_path.write_text(out_df.to_csv(index=False), encoding="utf-8")


def detect_pass_ks(target_dir: Path, rl_reasoning_dir: Optional[Path]) -> List[str]:
    ks: set[str] = set()
    roots = [target_dir]
    if rl_reasoning_dir is not None:
        roots.append(rl_reasoning_dir)
    for root in roots:
        if root is None or not root.exists():
            continue
        for run_dir in _scan_run_dirs(root):
            for g in ("g1", "g2"):
                gdir = run_dir / g
                if not gdir.is_dir():
                    continue
                for ds_dir in gdir.iterdir():
                    if not ds_dir.is_dir():
                        continue
                    for mpath in ds_dir.glob("*metrics.json"):
                        try:
                            metrics = json.loads(mpath.read_text(encoding="utf-8"))
                        except Exception:
                            continue
                        for key in (metrics.get("pass_at_k_percent") or {}).keys():
                            ks.add(str(key))

    def sort_key(val: str) -> Tuple[int, float | str]:
        try:
            return (0, float(val))
        except ValueError:
            return (1, val)

    return sorted(ks, key=sort_key)


def _format_cell(value: Optional[float], std: Optional[float], bold: bool) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    fmt = f".{FORMAT_DECIMALS}f"
    if std is None or not np.isfinite(std):
        cell = format(value, fmt)
    else:
        cell = f"{format(value, fmt)} $\\pm$ {format(std, fmt)}"
    if bold:
        return f"\\textbf{{{cell}}}"
    return cell


def _collect_row_raw(
    df: pd.DataFrame,
    base: str,
    algo_keys: Iterable[str],
) -> Tuple[List[Optional[float]], List[Optional[float]], Optional[float], Optional[float]]:
    sub = df[(df["base"] == base) & (df["algo"].isin(algo_keys))]
    values: List[Optional[float]] = []
    stds: List[Optional[float]] = []

    for ds_key, _ in DATASET_ORDER:
        ds_sub = sub[sub["dataset"] == ds_key]
        if ds_sub.empty:
            values.append(None)
            stds.append(None)
            continue
        val = float(ds_sub["value"].mean())
        std = float(ds_sub["std"].mean()) if ds_sub["std"].notna().any() else np.nan
        values.append(val)
        stds.append(std)

    vals_f = [v for v in values if v is not None and np.isfinite(v)]
    stds_f = [s for s in stds if s is not None and np.isfinite(s)]
    if not vals_f:
        return values, stds, None, None

    avg_val = float(np.mean(vals_f))
    avg_std = float(np.mean(stds_f)) if stds_f else np.nan
    return values, stds, avg_val, avg_std


def _is_max(val: Optional[float], max_val: Optional[float]) -> bool:
    if val is None or max_val is None:
        return False
    if not np.isfinite(val) or not np.isfinite(max_val):
        return False
    return bool(np.isclose(round(val, FORMAT_DECIMALS), round(max_val, FORMAT_DECIMALS), atol=1e-6))


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

        values, stds, avg_val, avg_std = _collect_row_raw(df, model, ["base"])
        cells = [_format_cell(v, s, bold=False) for v, s in zip(values, stds)]
        avg_cell = _format_cell(avg_val, avg_std, bold=False)
        lines.append(
            "      Base (No RL) & " + " & ".join(cells + [avg_cell]) + " \\\\")
        lines.append("      \\midrule")

        for gi, (group_title, rows) in enumerate(row_groups):
            lines.append(f"      \\multicolumn{{8}}{{l}}{{\\textit{{{group_title}}}}} \\\\")
            row_entries = []
            for row_label, algo_keys in rows:
                values, stds, avg_val, avg_std = _collect_row_raw(df, model, algo_keys)
                row_entries.append(
                    {
                        "label": row_label,
                        "values": values,
                        "stds": stds,
                        "avg_val": avg_val,
                        "avg_std": avg_std,
                    }
                )

            group_max_vals: List[Optional[float]] = []
            for idx in range(len(DATASET_ORDER)):
                col_vals = [
                    r["values"][idx]
                    for r in row_entries
                    if r["values"][idx] is not None and np.isfinite(r["values"][idx])
                ]
                group_max_vals.append(max(col_vals) if col_vals else None)
            avg_vals = [
                r["avg_val"] for r in row_entries if r["avg_val"] is not None and np.isfinite(r["avg_val"])
            ]
            group_max_avg = max(avg_vals) if avg_vals else None

            for row in row_entries:
                row_label = row["label"]
                is_vi_curl = row_label.lower() == "vi-curl"
                cells = [
                    _format_cell(val, std, _is_max(val, group_max_vals[idx]))
                    for idx, (val, std) in enumerate(zip(row["values"], row["stds"]))
                ]
                avg_cell = _format_cell(row["avg_val"], row["avg_std"], _is_max(row["avg_val"], group_max_avg))
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
    pass_ks = detect_pass_ks(args.target_dir, args.rl_reasoning_dir)
    metrics = [f"pass@{k}" for k in pass_ks]
    if not metrics and args.metric:
        metrics = [args.metric]

    output_base = args.output
    multiple = len(metrics) > 1

    for metric in metrics:
        df_raw = load_results(args.target_dir, args.models, metric, args.rl_reasoning_dir)
        metrics_files = df_raw.attrs.get("metrics_files")
        run_dirs = df_raw.attrs.get("run_dirs")
        if metrics_files is not None:
            print(f"[INFO] Metrics files found: {metrics_files}")
        if run_dirs is not None:
            print(f"[INFO] Run directories scanned: {run_dirs}")
        df = reduce_steps(df_raw, args.step_policy, args.vi_curl_best)
        df = attach_paths(df_raw, df)
        oracle_caption = (
            f"Comparison with \\textbf{{Oracle (w. Verifier)}} reward. Mean and standard deviation "
            f"\\textbf{{({metric})}} with 16 samples and 5 random seeds. We compare VI-CuRL against "
            "Curriculum baselines (VCRL, AdaRFT) and No Curriculum. \\colorbox{blue!10}{Blue} rows highlight our method."
        )
        vf_caption = (
            "Comparison in \\textbf{Verifier-Free} settings. Mean and standard deviation "
            f"\\textbf{{({metric})}} with 16 samples and 5 random seeds. We compare VI-CuRL against baselines using "
            "\\textbf{Majority Vote} and \\textbf{Entropy} as intrinsic reward signals. Note that VCRL and "
            "AdaRFT are excluded as they require ground-truth verifiers. \\colorbox{blue!10}{Blue} rows highlight our method."
        )
        table_oracle = build_table(
            df,
            args.models,
            metric,
            oracle_caption,
            f"tab:main_results_oracle_{metric}",
            ORACLE_GROUPS,
        )
        table_vf = build_table(
            df,
            args.models,
            metric,
            vf_caption,
            f"tab:main_results_independent_{metric}",
            VERIFIER_FREE_GROUPS,
        )
        table = table_oracle + "\n\n" + table_vf

        out_path = output_base
        if output_base and multiple:
            metric_tag = metric.replace("@", "at")
            out_path = output_base.with_name(f"{output_base.stem}_{metric_tag}{output_base.suffix}")

        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(table, encoding="utf-8")
            print(f"[INFO] Wrote LaTeX table to: {out_path}")
            metric_tag = metric.replace("@", "at")
            source_path = out_path.with_name(f"{out_path.stem}_sources_{metric_tag}.csv")
            write_source_map(df, source_path)
            print(f"[INFO] Wrote source map to: {source_path}")
        else:
            print(table)


if __name__ == "__main__":
    main()
