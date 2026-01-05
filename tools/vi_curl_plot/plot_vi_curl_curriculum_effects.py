#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot VI-CuRL curriculum effects from wandb output logs.

Given model and algorithm lists (as in examples/batch_curl_v6.sh), this script
auto-resolves VI-CURL run dirs and parses wandb output.log to plot:
  - Difficulty separation (kept vs dropped) + beta retention.
  - Confidence dynamics (conf_mean, tau).
  - Confidence vs difficulty correlation.

Outputs are saved under eval_log (default: eval_log/vi_curl/curriculum).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
except Exception:
    pass


_NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _split_csv(value: str) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _default_ckpt_dir() -> Path:
    env = os.environ.get("VI_CURL_CKPT_DIR") or os.environ.get("CHECKPOINT_DIR")
    candidates = [
        Path(env) if env else None,
        Path("/data/giil/caixq/ckpts"),
        Path.cwd().parent / "VI-CURL" / "checkpoints",
        Path.cwd().parent / "VI-CURL" / "outputs",
    ]
    for c in candidates:
        if c and c.exists():
            return c
    return Path.cwd()


def _compute_project_name(train_data: str) -> str:
    return f"VI-CURL_{train_data}"


def _compute_exp_name(exp: str, model: str, use_curl: bool) -> str:
    model_short = Path(model).name
    curl_tag = "curl" if use_curl else "nocurl"
    return f"{exp}_{curl_tag}_{model_short}"


def _find_latest_wandb_run_dir(wandb_dir: Path) -> Path:
    roots = [wandb_dir / "wandb", wandb_dir]
    for root in roots:
        lr = root / "latest-run"
        if lr.exists():
            return lr.resolve()

    run_dirs: List[Path] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for p in root.iterdir():
            if p.is_dir() and (p.name.startswith("run-") or p.name.startswith("offline-run-")):
                run_dirs.append(p)

    if not run_dirs:
        raise FileNotFoundError(f"No wandb run dirs found under: {wandb_dir}")
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0]


def _resolve_output_log(run_dir: Path) -> Path:
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        raise FileNotFoundError(f"WANDB_DIR not found: {wandb_dir}")
    run = _find_latest_wandb_run_dir(wandb_dir)
    output_log = run / "files" / "output.log"
    if not output_log.exists():
        raise FileNotFoundError(f"wandb output log not found: {output_log}")
    return output_log


def _parse_metrics_kv_line(line: str) -> Optional[Dict[str, Any]]:
    if " - " not in line or ":" not in line:
        return None
    out: Dict[str, Any] = {}
    for part in line.split(" - "):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        if k in {"training/global_step", "global_step", "step", "training/step"}:
            try:
                out[k] = int(float(v))
            except Exception:
                continue
            continue
        if not _NUM_RE.match(v):
            continue
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out or None


def _merge_by_step(records: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {}
    for r in records:
        step = None
        for k in ("training/global_step", "global_step", "step", "training/step"):
            if k in r:
                try:
                    step = int(r[k])
                    break
                except Exception:
                    continue
        if step is None:
            continue
        merged.setdefault(step, {}).update(r)
        merged[step]["step"] = int(step)
    return merged


def _read_log_df(log_path: Path, *, min_points: int = 2) -> pd.DataFrame:
    want_keys = {
        "vf_curl/diff_avg_kept",
        "vf_curl/diff_avg_dropped",
        "vf_curl/beta_actual",
        "vf_curl/beta_actual_x",
        "vf_curl/beta_target",
        "vf_curl/tau_x",
        "vf_curl/conf_mean",
        "vf_curl/conf_std",
        "vf_curl/corr_conf_diff",
    }
    records: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "vf_curl/" not in line:
                continue
            m = _parse_metrics_kv_line(line)
            if not m:
                continue
            if not any(k in m for k in want_keys):
                continue
            records.append(m)

    by_step = _merge_by_step(records)
    if not by_step:
        return pd.DataFrame()
    steps = sorted(by_step.keys())
    if len(steps) < min_points:
        return pd.DataFrame()

    rows = []
    for step in steps:
        r = by_step[step]
        rows.append(
            {
                "step": step,
                "diff_kept": r.get("vf_curl/diff_avg_kept", float("nan")),
                "diff_dropped": r.get("vf_curl/diff_avg_dropped", float("nan")),
                "beta_actual": r.get("vf_curl/beta_actual", r.get("vf_curl/beta_actual_x", float("nan"))),
                "beta_target": r.get("vf_curl/beta_target", float("nan")),
                "tau": r.get("vf_curl/tau_x", float("nan")),
                "conf_mean": r.get("vf_curl/conf_mean", float("nan")),
                "conf_std": r.get("vf_curl/conf_std", float("nan")),
                "corr_conf_diff": r.get("vf_curl/corr_conf_diff", float("nan")),
            }
        )
    df = pd.DataFrame(rows)
    return df


def _ema(values: Sequence[float], weight: float) -> List[float]:
    if not values:
        return []
    last = None
    out: List[float] = []
    for v in values:
        if last is None or math.isnan(float(v)):
            last = float(v)
        else:
            last = last * weight + (1.0 - weight) * float(v)
        out.append(last)
    return out


def _maybe_smooth(values: Sequence[float], *, weight: float, use_ema: bool) -> List[float]:
    raw = [float(v) for v in values]
    if not use_ema:
        return raw
    return _ema(raw, weight)


def _safe_title(text: str) -> str:
    return text.replace("/", "_")


@dataclass
class RunSpec:
    run_dir: Path
    run_name: str
    log_path: Path


def _resolve_runs(
    *,
    ckpt_dir: Path,
    train_data: str,
    project_name: Optional[str],
    models: Sequence[str],
    exps: Sequence[str],
    use_curl: str,
    run_dirs: Sequence[str],
    allow_missing: bool,
) -> List[RunSpec]:
    specs: List[RunSpec] = []
    if run_dirs:
        for rd in run_dirs:
            run_dir = Path(rd).expanduser().resolve()
            run_name = run_dir.name
            try:
                log_path = _resolve_output_log(run_dir)
            except Exception as exc:
                if allow_missing:
                    print(f"[WARN] Skip run_dir {run_dir}: {exc}")
                    continue
                raise
            specs.append(RunSpec(run_dir=run_dir, run_name=run_name, log_path=log_path))
        return specs

    project = project_name or _compute_project_name(train_data)
    curl_flags: List[bool] = []
    if use_curl == "both":
        curl_flags = [True, False]
    else:
        curl_flags = [_str2bool(use_curl)]

    for model in models:
        for exp in exps:
            for curl_flag in curl_flags:
                exp_name = _compute_exp_name(exp, model, curl_flag)
                run_dir = (ckpt_dir / project / exp_name).resolve()
                if not run_dir.exists():
                    if allow_missing:
                        print(f"[WARN] Run dir not found: {run_dir}")
                        continue
                    raise FileNotFoundError(f"Run dir not found: {run_dir}")
                try:
                    log_path = _resolve_output_log(run_dir)
                except Exception as exc:
                    if allow_missing:
                        print(f"[WARN] Missing output.log for {run_dir}: {exc}")
                        continue
                    raise
                specs.append(RunSpec(run_dir=run_dir, run_name=exp_name, log_path=log_path))
    return specs


def _plot_run(
    *,
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    smooth_weight: float,
    use_ema: bool,
    dpi: int,
) -> None:
    df = df.sort_values("step").reset_index(drop=True)
    steps = df["step"].to_numpy()

    diff_kept = _maybe_smooth(
        df["diff_kept"].fillna(method="ffill").tolist(),
        weight=smooth_weight,
        use_ema=use_ema,
    )
    diff_drop = _maybe_smooth(
        df["diff_dropped"].fillna(method="ffill").tolist(),
        weight=smooth_weight,
        use_ema=use_ema,
    )
    beta = df["beta_actual"].copy()
    if beta.isna().all():
        beta = df["beta_target"].copy()
    beta = beta.fillna(method="ffill").to_numpy()

    conf_mean = _maybe_smooth(
        df["conf_mean"].fillna(method="ffill").tolist(),
        weight=smooth_weight,
        use_ema=use_ema,
    )
    tau = df["tau"].replace([float("inf"), float("-inf")], float("nan")).fillna(method="ffill").to_numpy()
    corr = _maybe_smooth(
        df["corr_conf_diff"].fillna(method="ffill").tolist(),
        weight=smooth_weight,
        use_ema=use_ema,
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Panel 1: difficulty separation + beta
    ax = axes[0]
    ax.plot(steps, diff_drop, color="#e74c3c", linewidth=2.0, label="Dropped (hard)")
    ax.plot(steps, diff_kept, color="#2ecc71", linewidth=2.0, label="Kept (easy)")
    ax.set_ylabel("Difficulty")
    if np.nanmin(df[["diff_kept", "diff_dropped"]].to_numpy()) >= 0.0 and np.nanmax(
        df[["diff_kept", "diff_dropped"]].to_numpy()
    ) <= 1.0:
        ax.set_ylim(0.0, 1.05)
    ax2 = ax.twinx()
    ax2.plot(steps, beta, color="#3498db", linestyle="--", linewidth=1.6, label="Beta (retention)")
    ax2.set_ylabel("Beta")
    ax2.set_ylim(0.0, 1.05)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left")
    ax.set_title(title)

    # Panel 2: confidence dynamics
    ax = axes[1]
    ax.plot(steps, conf_mean, color="#2980b9", linewidth=2.0, label="Conf mean")
    ax.plot(steps, tau, color="#c0392b", linestyle="--", linewidth=1.8, label="Tau threshold")
    ax.set_ylabel("Confidence / Tau")
    ax.legend(loc="upper left")

    # Panel 3: correlation
    ax = axes[2]
    ax.plot(steps, corr, color="#8e44ad", linewidth=2.0, label="Conf-Diff corr")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.4)
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Training step")
    ax.legend(loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, default=None)
    ap.add_argument("--train_data", type=str, default="deepscaler_diff")
    ap.add_argument("--project_name", type=str, default=None)
    ap.add_argument("--models", type=str, default="")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--exps", type=str, default="")
    ap.add_argument("--exp", action="append", default=[])
    ap.add_argument("--use_curl", type=str, default="true", choices=["true", "false", "both"])
    ap.add_argument("--run_dir", action="append", default=[], help="Optional explicit run dir(s).")
    ap.add_argument("--out_dir", type=str, default="eval_log/vi_curl/curriculum")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--allow_missing", action="store_true", default=True)
    ap.add_argument("--strict", action="store_false", dest="allow_missing")
    ap.add_argument("--smooth_weight", type=float, default=0.8)
    ap.add_argument("--use_ema", action="store_true", default=True, help="Use EMA smoothing (default: True).")
    ap.add_argument("--no_ema", action="store_false", dest="use_ema", help="Disable EMA smoothing.")
    ap.add_argument("--min_points", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    models = _split_csv(args.models) + list(args.model)
    exps = _split_csv(args.exps) + list(args.exp)

    if not models and not args.run_dir:
        models = ["meta-llama/Llama-3.2-3B-Instruct"]
    if not exps and not args.run_dir:
        exps = ["vf_majority_vote", "vf_selfcert_logp", "vf_entropy", "ver_rule_grpo"]

    ckpt_dir = Path(args.ckpt_dir).expanduser() if args.ckpt_dir else _default_ckpt_dir()
    out_dir = Path(args.out_dir).expanduser().resolve()
    tag = args.tag.strip()

    specs = _resolve_runs(
        ckpt_dir=ckpt_dir,
        train_data=args.train_data,
        project_name=args.project_name,
        models=models,
        exps=exps,
        use_curl=args.use_curl,
        run_dirs=args.run_dir,
        allow_missing=bool(args.allow_missing),
    )
    if not specs:
        raise SystemExit("[ERROR] No valid runs found to plot.")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    for spec in specs:
        print(f"[INFO] Parsing: {spec.log_path}")
        df = _read_log_df(spec.log_path, min_points=args.min_points)
        if df.empty:
            print(f"[WARN] No vf_curl metrics found for {spec.run_name}; skip.")
            continue

        safe_name = _safe_title(spec.run_name)
        base = f"vi_curl_curriculum__{safe_name}"
        if tag:
            base = f"{base}__{tag}"
        csv_path = out_dir / f"{base}.csv"
        df.to_csv(csv_path, index=False)

        title = f"VI-CuRL Curriculum: {spec.run_name}"
        fig_path = out_dir / f"{base}.png"
        _plot_run(
            df=df,
            title=title,
            output_path=fig_path,
            smooth_weight=float(args.smooth_weight),
            use_ema=bool(args.use_ema),
            dpi=int(args.dpi),
        )
        print(f"[OK] Saved: {fig_path}")
        summary_rows.append(
            {
                "run_name": spec.run_name,
                "run_dir": str(spec.run_dir),
                "log_path": str(spec.log_path),
                "csv_path": str(csv_path),
                "fig_path": str(fig_path),
            }
        )

    if summary_rows:
        tag_suffix = tag or _hash_text(",".join(sorted(r["run_name"] for r in summary_rows)))
        summary_path = out_dir / f"vi_curl_curriculum_summary__{tag_suffix}.json"
        summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
        print(f"[OK] Summary: {summary_path}")


if __name__ == "__main__":
    main()
