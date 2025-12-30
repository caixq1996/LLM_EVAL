#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-ready figure generator for VI-CuRL / VI-CURL.

Produces three figures that align with the paper narrative:
  (1) Variance decomposition proxies vs step (σ_g,t^2, V_prob,t, and a combined proxy),
      using outputs from tools/vf_curl_grad_variance.py (merged JSON).
  (2) Curriculum schedule + confidence threshold dynamics (β_t, τ_t, conf mean/std),
      parsed from wandb-captured stdout logs (output.log).
  (3) Training stability + performance (e.g., Math500 val acc + grad_norm),
      comparing curl vs nocurl runs from output.log.

This script is intentionally standalone: it does NOT import vLLM/transformers code.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
except Exception:
    pass


_NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_latest_wandb_output_log(run_dir: Path) -> Path:
    # Typical layout (as seen in this repo):
    #   <run_dir>/wandb/wandb/latest-run -> run-.../
    #   <run_dir>/wandb/wandb/run-.../files/output.log
    wandb_dir = run_dir / "wandb"
    roots = [wandb_dir / "wandb", wandb_dir]
    for root in roots:
        lr = root / "latest-run"
        if lr.exists():
            p = lr.resolve() / "files" / "output.log"
            if p.exists():
                return p

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
    for rd in run_dirs:
        p = rd / "files" / "output.log"
        if p.exists():
            return p
    raise FileNotFoundError(f"wandb output.log not found under: {wandb_dir}")


def _parse_metrics_kv_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse wandb-captured stdout lines of the form:
      step:181 - key:value - key:value - ...
    Returns dict; non-numeric values are skipped.
    """
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

        # step can be int
        if k in {"step", "training/global_step", "training/step", "global_step"}:
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


def _merge_by_step(records: Iterable[Dict[str, Any]], *, step_keys: Sequence[str]) -> Dict[int, Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {}
    for r in records:
        step: Optional[int] = None
        for k in step_keys:
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


def _read_log_timeseries(
    log_path: Path,
    *,
    want_keys: Sequence[str],
    step_keys: Sequence[str] = ("training/global_step", "step", "global_step", "training/step"),
    line_filters: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      steps: [T]
      series: {key: [T] float, NaN if missing}
    """
    filters = list(line_filters) if line_filters else []
    if not filters:
        # Default: only keep lines that are likely to be metrics lines.
        filters = ["step:", "training/global_step:"]

    recs: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if filters and not any(s in line for s in filters):
                continue
            m = _parse_metrics_kv_line(line)
            if not m:
                continue
            if not any(k in m for k in want_keys) and not any(k in m for k in step_keys):
                continue
            recs.append(m)

    by_step = _merge_by_step(recs, step_keys=step_keys)
    steps_sorted = np.array(sorted(by_step.keys()), dtype=np.int64)

    series: Dict[str, np.ndarray] = {}
    for k in want_keys:
        series[k] = np.array([float(by_step[s].get(k, float("nan"))) for s in steps_sorted], dtype=np.float64)
    return steps_sorted, series


def _forward_fill(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    last = float("nan")
    for i in range(out.shape[0]):
        if math.isnan(float(out[i])):
            out[i] = last
        else:
            last = float(out[i])
    return out


def _ema(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    out = np.array(x, dtype=np.float64, copy=True)
    last = None
    for i in range(out.shape[0]):
        v = float(out[i])
        if math.isnan(v):
            continue
        if last is None:
            last = v
        else:
            last = alpha * v + (1.0 - alpha) * last
        out[i] = last
    return out


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return num / np.clip(den, 1e-12, None)


def _pct(r: np.ndarray) -> np.ndarray:
    return 100.0 * (r - 1.0)


def _savefig(fig: plt.Figure, out_base: Path, *, dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _load_vf_payload(vf_json: Path) -> Dict[str, Any]:
    payload = _read_json(vf_json)
    if not isinstance(payload, dict) or "rows" not in payload or "steps" not in payload:
        raise ValueError(f"Invalid vf_curl_grad_variance payload: {vf_json}")
    return payload


def plot_fig1_variance(
    *,
    vf_json: Path,
    out_path_base: Path,
    dpi: int,
) -> None:
    payload = _load_vf_payload(vf_json)
    rows: List[Dict[str, Any]] = list(payload["rows"])
    base_rows: List[Dict[str, Any]] = list(payload.get("baseline_rows") or rows)
    steps = np.array([int(s) for s in payload["steps"]], dtype=np.int64)

    def arr(key: str, src: Sequence[Dict[str, Any]]) -> np.ndarray:
        return np.array([float(r.get(key, float("nan"))) for r in src], dtype=np.float64)

    beta = arr("beta_target", rows)
    beta = np.where(np.isnan(beta), 1.0, beta)
    beta = np.clip(beta, 1e-6, 1.0)

    sigma = arr("sigma_kept", rows)
    vprob = arr("vprob_kept", rows)
    gbar_norm2 = arr("gbar_norm2_kept", rows)
    base_sigma_full = arr("sigma_full", base_rows)
    base_vprob_full = arr("vprob_full", base_rows)

    # Numerator comparisons (homogeneity of the selected kept set).
    sigma_r = _ratio(sigma, base_sigma_full)
    vprob_r = _ratio(vprob, base_vprob_full)

    # Variance decomposition proxy (first two terms + optional masking term):
    #   Var(\hat g_t) ≈ (sigma + V) / beta  +  ((1-beta)/beta) * ||E[g|w=1]||^2
    proxy_main = (sigma + vprob) / beta
    proxy_mask = np.zeros_like(proxy_main)
    if np.any(~np.isnan(gbar_norm2)):
        proxy_mask = ((1.0 - beta) / beta) * gbar_norm2
    proxy_total = proxy_main + proxy_mask

    base_total = base_sigma_full + base_vprob_full
    proxy_r = _ratio(proxy_total, base_total)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.plot(steps, _pct(sigma_r), marker="o", linewidth=1.8, label=r"$\Delta\%\,\sigma_{g,t}^2$ (kept)")
    ax.plot(steps, _pct(vprob_r), marker="o", linewidth=1.8, label=r"$\Delta\%\,V_{\mathrm{prob},t}$ (kept)")
    ax.plot(
        steps,
        _pct(proxy_r),
        marker="o",
        linewidth=2.0,
        color="tab:purple",
        label=r"$\Delta\%$ variance proxy",
    )
    ax.set_xlabel("training global_step")
    ax.set_ylabel(r"$\Delta\%$ vs baseline (full)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=3, fontsize=9)

    ax2 = ax.twinx()
    ax2.plot(steps, beta, color="tab:blue", linestyle="--", linewidth=1.5, alpha=0.5)
    ax2.set_ylabel(r"$\beta$")
    ax2.set_ylim(0.0, 1.05)

    title = str(payload.get("title") or "VI-CuRL variance proxies")
    ax.set_title(title)
    fig.tight_layout()
    _savefig(fig, out_path_base, dpi=dpi)


def plot_fig2_schedule(
    *,
    curl_run_dir: Path,
    out_path_base: Path,
    dpi: int,
) -> None:
    log_path = _find_latest_wandb_output_log(curl_run_dir)
    want = [
        "vf_curl/beta_target",
        "vf_curl/beta_actual_x",
        "vf_curl/beta_actual",
        "vf_curl/tau_x",
        "vf_curl/conf_mean",
        "vf_curl/conf_std",
        "vf_curl/samples_kept_x",
    ]
    filters = ["vf_curl/", "training/global_step:"]
    steps, s = _read_log_timeseries(log_path, want_keys=want, line_filters=filters)
    if steps.size == 0:
        raise RuntimeError(f"No vf_curl metrics found in log: {log_path}")

    beta_t = _forward_fill(np.where(np.isfinite(s["vf_curl/beta_target"]), s["vf_curl/beta_target"], np.nan))
    beta_a = _forward_fill(np.where(np.isfinite(s["vf_curl/beta_actual_x"]), s["vf_curl/beta_actual_x"], np.nan))
    if np.all(np.isnan(beta_a)):
        beta_a = _forward_fill(np.where(np.isfinite(s["vf_curl/beta_actual"]), s["vf_curl/beta_actual"], np.nan))

    tau = _forward_fill(np.where(np.isfinite(s["vf_curl/tau_x"]), s["vf_curl/tau_x"], np.nan))
    conf_m = _forward_fill(np.where(np.isfinite(s["vf_curl/conf_mean"]), s["vf_curl/conf_mean"], np.nan))
    conf_s = _forward_fill(np.where(np.isfinite(s["vf_curl/conf_std"]), s["vf_curl/conf_std"], np.nan))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

    ax0 = axes[0]
    ax0.plot(steps, beta_t, linewidth=2.0, label=r"$\beta_t$ (target)")
    if not np.all(np.isnan(beta_a)):
        ax0.plot(steps, beta_a, linewidth=2.0, linestyle="--", label=r"$\beta_t$ (actual)")
    ax0.set_ylabel(r"retention $\beta$")
    ax0.set_ylim(0.0, 1.05)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")

    ax1 = axes[1]
    if not np.all(np.isnan(conf_m)):
        ax1.plot(steps, conf_m, linewidth=2.0, label="conf mean")
    if not np.all(np.isnan(conf_s)) and not np.all(np.isnan(conf_m)):
        ax1.fill_between(steps, conf_m - conf_s, conf_m + conf_s, alpha=0.2, linewidth=0.0, label="conf ± std")
    if not np.all(np.isnan(tau)):
        ax1.plot(steps, tau, linewidth=2.0, linestyle="--", label=r"threshold $\tau_t$")
    ax1.set_xlabel("training global_step")
    ax1.set_ylabel("confidence / threshold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    fig.suptitle(f"Curriculum dynamics: {curl_run_dir.name}", y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    _savefig(fig, out_path_base, dpi=dpi)


def plot_fig3_stability_and_perf(
    *,
    curl_run_dir: Path,
    nocurl_run_dir: Path,
    dataset: str,
    out_path_base: Path,
    dpi: int,
) -> None:
    curl_log = _find_latest_wandb_output_log(curl_run_dir)
    nocurl_log = _find_latest_wandb_output_log(nocurl_run_dir)

    acc_key = f"val-core/{dataset}/acc/mean@1"

    # 1) Validation accuracy (sparse)
    steps_c_acc, s_c_acc = _read_log_timeseries(curl_log, want_keys=[acc_key], line_filters=[acc_key, "step:"])
    steps_n_acc, s_n_acc = _read_log_timeseries(nocurl_log, want_keys=[acc_key], line_filters=[acc_key, "step:"])
    acc_c = s_c_acc[acc_key]
    acc_n = s_n_acc[acc_key]

    # 2) Training stability proxy: grad norm (dense)
    steps_c_gn, s_c_gn = _read_log_timeseries(
        curl_log,
        want_keys=["actor/grad_norm"],
        line_filters=["actor/grad_norm:", "training/global_step:"],
    )
    steps_n_gn, s_n_gn = _read_log_timeseries(
        nocurl_log,
        want_keys=["actor/grad_norm"],
        line_filters=["actor/grad_norm:", "training/global_step:"],
    )
    gn_c = _ema(_forward_fill(s_c_gn["actor/grad_norm"]), alpha=0.05)
    gn_n = _ema(_forward_fill(s_n_gn["actor/grad_norm"]), alpha=0.05)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=False)

    ax0 = axes[0]
    if steps_c_acc.size:
        ax0.plot(steps_c_acc, 100.0 * acc_c, marker="o", linewidth=2.0, label=f"VI-CuRL (curl)  {curl_run_dir.name}")
    if steps_n_acc.size:
        ax0.plot(steps_n_acc, 100.0 * acc_n, marker="o", linewidth=2.0, linestyle="--", label=f"No-curriculum (nocurl)  {nocurl_run_dir.name}")
    ax0.set_ylabel(f"{dataset} val acc@1 (%)")
    ax0.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best", fontsize=9)

    ax1 = axes[1]
    if steps_c_gn.size:
        ax1.plot(steps_c_gn, gn_c, linewidth=2.0, label="curl (EMA grad_norm)")
    if steps_n_gn.size:
        ax1.plot(steps_n_gn, gn_n, linewidth=2.0, linestyle="--", label="nocurl (EMA grad_norm)")
    ax1.set_xlabel("training global_step")
    ax1.set_ylabel("actor/grad_norm (EMA)")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best")

    fig.suptitle("Stability + performance (same base model / reward)", y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    _savefig(fig, out_path_base, dpi=dpi)


def _resolve_vf_json(*, vf_out_dir: Path, run_name: str, tag: Optional[str]) -> Path:
    vf_out_dir = vf_out_dir.expanduser()
    if tag:
        p = vf_out_dir / f"vf_curl_grad_variance__{run_name}__{tag}.json"
        if not p.exists():
            raise FileNotFoundError(f"vf json not found: {p}")
        return p

    cands = sorted(
        [
            p
            for p in vf_out_dir.glob(f"vf_curl_grad_variance__{run_name}__*.json")
            if "__part" not in p.name and not p.name.endswith(".partial.json")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"No merged vf_curl_grad_variance json found under {vf_out_dir} for run={run_name}")
    return cands[0]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curl_run_dir", type=Path, required=True, help="CuRL run dir (contains wandb/)")
    ap.add_argument("--nocurl_run_dir", type=Path, required=True, help="No-curriculum baseline run dir (contains wandb/)")
    ap.add_argument("--vf_out_dir", type=Path, default=Path("project/LLM_EVAL/eval_log/vi_curl/grad_variance"))
    ap.add_argument("--vf_tag", type=str, default=None, help="Tag used by vf_curl_grad_variance (auto_xxx). If omitted, picks newest.")
    ap.add_argument("--vf_json", type=Path, default=None, help="Direct path to merged vf_curl_grad_variance__*.json (overrides --vf_out_dir/--vf_tag).")
    ap.add_argument("--dataset", type=str, default="math500", help="Dataset name for val-core/<dataset>/acc/mean@1")
    ap.add_argument("--save_dir", type=Path, default=None, help="Where to save figures (default: <vf_out_dir>/paper_figs/<run_name>)")
    ap.add_argument("--dpi", type=int, default=250)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    curl_run_dir = args.curl_run_dir.expanduser()
    nocurl_run_dir = args.nocurl_run_dir.expanduser()

    run_name = curl_run_dir.name

    if args.vf_json is not None:
        vf_json = args.vf_json.expanduser()
    else:
        vf_json = _resolve_vf_json(vf_out_dir=args.vf_out_dir, run_name=run_name, tag=args.vf_tag)

    if args.save_dir is None:
        # Default: split by vf json tag/stem to avoid overwriting figures across different
        # vf_curl_grad_variance runs (e.g., different beta_schedule_mode).
        prefix = f"vf_curl_grad_variance__{run_name}__"
        tag = None
        if vf_json.name.startswith(prefix) and vf_json.name.endswith(".json"):
            tag = vf_json.name[len(prefix) : -len(".json")]
        base_dir = args.vf_out_dir
        if base_dir.name in {"grad_variance", "vf_curl_grad_variance_all"}:
            base_dir = base_dir.parent
        save_dir = base_dir / "paper_figs" / run_name / (tag or vf_json.stem)
    else:
        save_dir = args.save_dir.expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] vf_json: {vf_json}")
    print(f"[INFO] save_dir: {save_dir}")

    plot_fig1_variance(
        vf_json=vf_json,
        out_path_base=save_dir / "fig1_variance_decomposition",
        dpi=int(args.dpi),
    )
    plot_fig2_schedule(
        curl_run_dir=curl_run_dir,
        out_path_base=save_dir / "fig2_curriculum_schedule",
        dpi=int(args.dpi),
    )
    plot_fig3_stability_and_perf(
        curl_run_dir=curl_run_dir,
        nocurl_run_dir=nocurl_run_dir,
        dataset=str(args.dataset),
        out_path_base=save_dir / "fig3_stability_performance",
        dpi=int(args.dpi),
    )

    print("[OK] Done.")
    print(f"[OK] Figure 1: {save_dir / 'fig1_variance_decomposition.png'}")
    print(f"[OK] Figure 2: {save_dir / 'fig2_curriculum_schedule.png'}")
    print(f"[OK] Figure 3: {save_dir / 'fig3_stability_performance.png'}")


if __name__ == "__main__":
    main()
