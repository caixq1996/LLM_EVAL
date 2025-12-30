#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot layerwise spectral leakage (eta) heatmaps for OPRA vs baselines.

Example:
  python project/LLM_EVAL/tools/plot_opra_layerwise_leakage.py \
    --base_model /path/to/base \
    --run opra=/path/to/opra_adapter \
    --run lora=/path/to/lora_adapter \
    --out_dir project/LLM_EVAL/eval_log/opra_layerwise
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM

from opra_spectral_utils import (
    compute_lora_delta,
    compute_weight_delta,
    get_base_weight,
    infer_layer_idx,
    infer_module_type,
    collect_run_entries,
    is_adapter_dir,
    is_hf_model_dir,
    iter_lora_modules,
    iter_weight_modules,
    list_run_dirs,
    projection_energy_ratio,
    resolve_checkpoint_root,
    resolve_base_model_path,
    resolve_run_dir,
    topk_svd,
)


def _parse_run_arg(run_arg: str) -> Tuple[str, str]:
    if "=" not in run_arg:
        raise ValueError("--run must be label=PATH")
    label, path = run_arg.split("=", 1)
    return label.strip(), path.strip()


def _load_peft_model(base_model: str, adapter_path: Path, *, dtype: torch.dtype, device: torch.device, device_map: Optional[str], trust_remote_code: bool) -> torch.nn.Module:
    try:
        from peft import PeftModel
    except Exception as exc:
        raise RuntimeError("peft is required for this script") from exc

    if device_map:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust_remote_code)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, trust_remote_code=trust_remote_code)
        model.to(device)
    model.eval()
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model


def _load_full_model(model_path: str, *, dtype: torch.dtype, device: torch.device, device_map: Optional[str], trust_remote_code: bool) -> torch.nn.Module:
    if device_map:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust_remote_code)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=trust_remote_code)
        model.to(device)
    model.eval()
    return model


def _dtype_from_name(name: str) -> torch.dtype:
    name = (name or "").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    return torch.float16


def _compute_layerwise_eta(
    model: torch.nn.Module,
    *,
    uv_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    principal_rank: int,
    use_lowrank: bool,
    module_filters: List[str],
    device: torch.device,
) -> List[Dict]:
    rows: List[Dict] = []
    for name, module in iter_lora_modules(model):
        if module_filters and not any(f in name for f in module_filters):
            continue
        layer = infer_layer_idx(name)
        module_type = infer_module_type(name)
        if layer is None:
            continue
        delta = compute_lora_delta(module, device=device, dtype=torch.float32)
        if delta is None or float(delta.pow(2).sum().item()) == 0.0:
            continue
        if name not in uv_cache:
            base_w = get_base_weight(module)
            u, v = topk_svd(base_w, principal_rank, device=device, use_lowrank=use_lowrank)
            uv_cache[name] = (u.detach().cpu() if u is not None else None, v.detach().cpu() if v is not None else None)
        u, v = uv_cache.get(name, (None, None))
        u = u.to(device) if u is not None else None
        v = v.to(device) if v is not None else None
        eta, den = projection_energy_ratio(delta, u, v)
        rows.append({
            "layer": layer,
            "module_type": module_type,
            "eta": eta,
            "delta_norm_sq": den,
        })
    return rows


def _compute_layerwise_eta_full(
    base_model: torch.nn.Module,
    tuned_model: torch.nn.Module,
    *,
    uv_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    principal_rank: int,
    use_lowrank: bool,
    module_filters: List[str],
    device: torch.device,
) -> List[Dict]:
    rows: List[Dict] = []
    tuned_modules = {name: module for name, module in iter_weight_modules(tuned_model)}
    for name, base_module in iter_weight_modules(base_model):
        if module_filters and not any(f in name for f in module_filters):
            continue
        tuned_module = tuned_modules.get(name)
        if tuned_module is None:
            continue
        layer = infer_layer_idx(name)
        module_type = infer_module_type(name)
        if layer is None:
            continue
        delta = compute_weight_delta(base_module, tuned_module, device=device, dtype=torch.float32)
        if delta is None or float(delta.pow(2).sum().item()) == 0.0:
            continue
        if name not in uv_cache:
            base_w = base_module.weight
            u, v = topk_svd(base_w, principal_rank, device=device, use_lowrank=use_lowrank)
            uv_cache[name] = (u.detach().cpu() if u is not None else None, v.detach().cpu() if v is not None else None)
        u, v = uv_cache.get(name, (None, None))
        u = u.to(device) if u is not None else None
        v = v.to(device) if v is not None else None
        eta, den = projection_energy_ratio(delta, u, v)
        rows.append({
            "layer": layer,
            "module_type": module_type,
            "eta": eta,
            "delta_norm_sq": den,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_root", type=str, default="project/OPRA/checkpoints/OPRA-LoRA")
    ap.add_argument("--base_model", type=str, default="", help="Base model path or HF id (auto-resolved if empty)")
    ap.add_argument("--run", action="append", required=True, help="label=adapter_path or run dir")
    ap.add_argument("--principal_rank", type=int, default=16)
    ap.add_argument("--module_filter", type=str, default="")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--device_map", type=str, default="")
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--use_lowrank", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true", default=False)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", type=str, default="Layerwise Spectral Leakage (eta)")
    ap.add_argument("--steps", type=str, default="", help="Comma-separated global steps to plot (default: all)")
    ap.add_argument("--step", type=int, default=-1, help="Pick a specific global_step (deprecated; use --steps)")
    args = ap.parse_args()

    checkpoint_root = resolve_checkpoint_root(args.checkpoint_root)
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_name(args.dtype)
    device_map = args.device_map.strip() or None
    module_filters = [s for s in args.module_filter.split(",") if s]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    uv_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    run_rows: List[Dict] = []

    run_labels: List[str] = []
    step_filter: Optional[List[int]] = None
    if args.steps.strip():
        step_filter = [int(s) for s in args.steps.split(",") if s.strip().isdigit()]
    elif args.step >= 0:
        step_filter = [args.step]

    run_step_maps: Dict[str, Dict[int, Tuple[str, str]]] = {}
    resolved_dirs: List[str] = []
    for run_arg in args.run:
        label, run_hint = _parse_run_arg(run_arg)
        resolved, entries = collect_run_entries(run_hint, checkpoint_root, step_filter=step_filter)
        if resolved is None:
            candidates = list_run_dirs(checkpoint_root)
            print(f"[WARN] Run not found: {run_hint}")
            if candidates:
                print(f"       Available runs under {args.checkpoint_root}: {', '.join(candidates[:8])}")
            continue
        if not entries:
            print(f"[WARN] No usable checkpoints found under {resolved}")
            continue
        run_labels.append(label)
        resolved_dirs.append(resolved)
        run_step_maps[label] = {step: (path, mode) for step, path, mode in entries}
        print(f"[INFO] Run {label}: {len(run_step_maps[label])} steps")

    if not run_step_maps:
        print("[ERROR] No runs to process")
        return

    all_steps = sorted({step for step_map in run_step_maps.values() for step in step_map.keys()})
    if not all_steps:
        print("[ERROR] No steps found to plot")
        return

    base_model_path = resolve_base_model_path(args.base_model.strip() or None, resolved_dirs[0], ["base_model", "/hss/giil/caixq/model", "/home/caixq/base_model"])
    need_base_ref = any(mode == "full" for step_map in run_step_maps.values() for _, mode in step_map.values())
    base_ref = None
    if need_base_ref:
        print(f"[INFO] Loading base model for full-weight comparison: {base_model_path}")
        base_ref = _load_full_model(base_model_path, dtype=dtype, device=device, device_map=device_map, trust_remote_code=args.trust_remote_code)

    print(f"[INFO] Plotting {len(all_steps)} steps to {out_dir}")
    for step in all_steps:
        step_rows: List[Dict] = []
        step_labels: List[str] = []
        for label in run_labels:
            entry = run_step_maps[label].get(step)
            if entry is None:
                print(f"[WARN] Missing step {step} for run {label}; skipping this run in step plot")
                continue
            path_str, mode = entry
            print(f"[INFO] Step {step}: loading {label} ({mode}) from {path_str}")
            if mode == "adapter":
                model = _load_peft_model(
                    base_model_path,
                    Path(path_str),
                    dtype=dtype,
                    device=device,
                    device_map=device_map,
                    trust_remote_code=args.trust_remote_code,
                )
                rows = _compute_layerwise_eta(
                    model,
                    uv_cache=uv_cache,
                    principal_rank=args.principal_rank,
                    use_lowrank=args.use_lowrank,
                    module_filters=module_filters,
                    device=device,
                )
                del model
            else:
                if base_ref is None:
                    print(f"[WARN] Base model not loaded; skipping full-model step {step} for {label}")
                    continue
                tuned_model = _load_full_model(path_str, dtype=dtype, device=device, device_map=device_map, trust_remote_code=args.trust_remote_code)
                rows = _compute_layerwise_eta_full(
                    base_ref,
                    tuned_model,
                    uv_cache=uv_cache,
                    principal_rank=args.principal_rank,
                    use_lowrank=args.use_lowrank,
                    module_filters=module_filters,
                    device=device,
                )
                del tuned_model

            for row in rows:
                row["run"] = label
                row["step"] = step
                step_rows.append(row)
                run_rows.append(row)
            step_labels.append(label)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not step_rows:
            print(f"[WARN] No data for step {step}; skipping plot")
            continue

        df_step = pd.DataFrame(step_rows)
        run_count = len(step_labels)
        fig, axes = plt.subplots(1, run_count, figsize=(6 * run_count, 6), squeeze=False)
        axes = axes[0]
        vmin, vmax = 0.0, 1.0

        for idx, label in enumerate(step_labels):
            sub = df_step[df_step["run"] == label].copy()
            sub["weight"] = sub["delta_norm_sq"].replace(0.0, 1.0)
            sub["weighted_eta"] = sub["eta"] * sub["weight"]
            grouped = sub.groupby(["layer", "module_type"], as_index=False).agg(
                weight_sum=("weight", "sum"),
                weighted_eta_sum=("weighted_eta", "sum"),
            )
            grouped["eta"] = grouped["weighted_eta_sum"] / (grouped["weight_sum"] + 1e-12)
            pivot = grouped.pivot_table(index="layer", columns="module_type", values="eta")
            pivot = pivot.sort_index()
            ax = axes[idx]
            sns.heatmap(pivot, ax=ax, vmin=vmin, vmax=vmax, cmap="viridis", cbar=idx == run_count - 1)
            ax.set_title(label)
            ax.set_xlabel("Module Type")
            ax.set_ylabel("Layer")

        fig.suptitle(f"{args.title} (step {step})")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = out_dir / f"layerwise_leakage_step_{step}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[INFO] Saved plot: {out_path}")

    if base_ref is not None:
        del base_ref

    if not run_rows:
        print("[ERROR] No layerwise leakage data computed")
        return

    df = pd.DataFrame(run_rows)
    df.to_csv(out_dir / "layerwise_leakage.csv", index=False)
    with (out_dir / "layerwise_leakage.jsonl").open("w", encoding="utf-8") as f:
        for row in run_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"[INFO] Wrote summary: {out_dir / 'layerwise_leakage.csv'}")


if __name__ == "__main__":
    main()
