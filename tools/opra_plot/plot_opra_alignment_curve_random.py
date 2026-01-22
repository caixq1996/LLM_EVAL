#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot spectral alignment (eta) across checkpoints with random baseline.

Example:
  python project/LLM_EVAL/tools/plot_opra_alignment_curve_random.py \
    --base_model /path/to/base \
    --run opra=/path/to/opra_run \
    --run lora=/path/to/lora_run \
    --out_dir project/LLM_EVAL/eval_log/opra_alignment_random
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

import sys

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from opra_spectral_utils import (
    load_peft_config,
    compute_lora_delta,
    compute_weight_delta,
    get_base_weight,
    infer_layer_idx,
    infer_module_type,
    is_adapter_dir,
    is_hf_model_dir,
    iter_lora_modules,
    iter_weight_modules,
    list_checkpoint_adapters,
    list_checkpoint_full_models,
    list_run_dirs,
    parse_step_from_path,
    projection_energy_ratio,
    resolve_checkpoint_root,
    resolve_base_model_path,
    resolve_run_dir,
    topk_svd,
)

from plot_config import setup_plot_style, add_font_size_args, get_font_sizes, create_legend


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
    config = load_peft_config(adapter_path)
    ignore_mismatched_sizes = bool(getattr(config, "_ignore_mismatched_sizes", False)) if config else False
    try:
        model = PeftModel.from_pretrained(model, str(adapter_path), config=config, is_trainable=False, ignore_mismatched_sizes=ignore_mismatched_sizes)
    except Exception as e:
        error_msg = str(e)
        if "size mismatch" in error_msg.lower() or "shape" in error_msg.lower():
            print(f"[WARN] Failed to load adapter from {adapter_path}: weight size mismatch (incompatible PEFT version)")
            print(f"       Skipping this adapter. Error: {error_msg[:200]}...")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return None
        raise
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


def _compute_alignment(
    model: torch.nn.Module,
    *,
    uv_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    principal_rank: int,
    use_lowrank: bool,
    module_filters: List[str],
    device: torch.device,
) -> Tuple[Optional[float], Optional[float], List[Dict]]:
    total_num = 0.0
    total_den = 0.0
    total_rand = 0.0
    rows: List[Dict] = []
    for name, module in iter_lora_modules(model):
        if module_filters and not any(f in name for f in module_filters):
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
        rand = torch.randn_like(delta)
        eta_rand, _ = projection_energy_ratio(rand, u, v)
        total_num += eta * den
        total_rand += eta_rand * den
        total_den += den
        rows.append({
            "module": name,
            "layer": infer_layer_idx(name),
            "module_type": infer_module_type(name),
            "eta": eta,
            "eta_random": eta_rand,
            "delta_norm_sq": den,
        })
    if total_den == 0.0:
        return None, None, rows
    return total_num / total_den, total_rand / total_den, rows


def _compute_alignment_full(
    base_model: torch.nn.Module,
    tuned_model: torch.nn.Module,
    *,
    uv_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    principal_rank: int,
    use_lowrank: bool,
    module_filters: List[str],
    device: torch.device,
) -> Tuple[Optional[float], Optional[float], List[Dict]]:
    total_num = 0.0
    total_den = 0.0
    total_rand = 0.0
    rows: List[Dict] = []

    tuned_modules = {name: module for name, module in iter_weight_modules(tuned_model)}
    for name, base_module in iter_weight_modules(base_model):
        if module_filters and not any(f in name for f in module_filters):
            continue
        tuned_module = tuned_modules.get(name)
        if tuned_module is None:
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
        rand = torch.randn_like(delta)
        eta_rand, _ = projection_energy_ratio(rand, u, v)
        total_num += eta * den
        total_rand += eta_rand * den
        total_den += den
        rows.append({
            "module": name,
            "layer": infer_layer_idx(name),
            "module_type": infer_module_type(name),
            "eta": eta,
            "eta_random": eta_rand,
            "delta_norm_sq": den,
        })
    if total_den == 0.0:
        return None, None, rows
    return total_num / total_den, total_rand / total_den, rows


def _dtype_from_name(name: str) -> torch.dtype:
    name = (name or "").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    return torch.float16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_root", type=str, default="project/OPRA/checkpoints/OPRA-LoRA")
    ap.add_argument("--base_model", type=str, default="", help="Base model path or HF id (auto-resolved if empty)")
    ap.add_argument("--run", action="append", required=True, help="label=PATH (adapter dir or run dir)")
    ap.add_argument("--principal_rank", type=int, default=16)
    ap.add_argument("--module_filter", type=str, default="", help="Comma-separated module name filters")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--device_map", type=str, default="", help="Device map for HF loading (e.g., auto)")
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--use_lowrank", action="store_true")
    ap.add_argument("--random_seed", type=int, default=1234)
    ap.add_argument("--trust_remote_code", action="store_true", default=False)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", type=str, default="Spectral Alignment Across Steps (Random Baseline)")
    ap.add_argument("--steps", type=str, default="", help="Comma-separated step numbers to keep (default: all)")
    ap.add_argument("--replot", action="store_true", help="Skip computation and replot from existing CSV data")
    add_font_size_args(ap)
    args = ap.parse_args()
    
    # Setup plot style with Times New Roman font
    setup_plot_style()
    font_sizes = get_font_sizes(args)

    checkpoint_root = resolve_checkpoint_root(args.checkpoint_root)
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_name(args.dtype)
    device_map = args.device_map.strip() or None
    module_filters = [s for s in args.module_filter.split(",") if s]
    wanted_steps = {int(s) for s in args.steps.split(",") if s.strip().isdigit()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Replot mode: load existing CSV and regenerate plots
    if args.replot:
        csv_path = out_dir / "alignment_curve_random.csv"
        if not csv_path.exists():
            print(f"[ERROR] Cannot replot: {csv_path} not found")
            return
        print(f"[INFO] Replot mode: loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        fig, ax = plt.subplots(figsize=(9, 5))
        for run, sub in df.groupby("run"):
            sub = sub.sort_values(by=["step"], na_position="last")
            fallback = pd.Series(np.arange(len(sub)), index=sub.index)
            x = sub["step"].fillna(fallback).to_numpy()
            y = sub["eta"].to_numpy()
            y_rand = sub["eta_random"].to_numpy()
            line = ax.plot(x, y, marker="o", linewidth=2, label=run)[0]
            ax.plot(x, y_rand, linestyle="--", linewidth=2, color=line.get_color(), label=f"{run} (random)")
        ax.set_xlabel("Global Step", fontsize=font_sizes['xlabel'], fontfamily=font_sizes['fontfamily'])
        ax.set_ylabel("Alignment Ratio (eta)", fontsize=font_sizes['ylabel'], fontfamily=font_sizes['fontfamily'])
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis='both', labelsize=font_sizes['tick'])
        create_legend(ax, font_sizes)
        fig.tight_layout()
        out_path = out_dir / "alignment_curve_random.png"
        fig.savefig(out_path, dpi=300)
        fig.savefig(out_dir / "alignment_curve_random.pdf", dpi=300)
        plt.close(fig)
        print(f"[INFO] Replot complete: {out_path} and .pdf")
        return

    print(f"[INFO] checkpoint_root={checkpoint_root}")
    print(f"[INFO] out_dir={out_dir}")

    all_rows: List[Dict] = []
    uv_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    run_entries: List[Tuple[str, List[Tuple[int, str, str]]]] = []
    run_dirs: List[str] = []
    for run_arg in args.run:
        label, run_hint = _parse_run_arg(run_arg)
        resolved = resolve_run_dir(run_hint, checkpoint_root)
        if resolved is None:
            candidates = list_run_dirs(checkpoint_root)
            print(f"[WARN] Run not found: {run_hint}")
            if candidates:
                print(f"       Available runs under {args.checkpoint_root}: {', '.join(candidates[:8])}")
            continue
        run_dirs.append(resolved)
        path = Path(resolved)
        entries: List[Tuple[int, str, str]] = []
        if is_adapter_dir(path):
            entries.append((parse_step_from_path(str(path)) or -1, str(path), "adapter"))
        elif is_hf_model_dir(path):
            adapter_dir = path.parent / "lora_adapter"
            if is_adapter_dir(adapter_dir):
                entries.append((parse_step_from_path(str(adapter_dir)) or -1, str(adapter_dir), "adapter"))
            elif is_adapter_dir(path.parent):
                entries.append((parse_step_from_path(str(path.parent)) or -1, str(path.parent), "adapter"))
            else:
                entries.append((parse_step_from_path(str(path)) or -1, str(path), "full"))
        elif path.name.startswith("global_step_"):
            step = parse_step_from_path(str(path)) or -1
            adapter_dir = path / "actor" / "lora_adapter"
            if is_adapter_dir(adapter_dir):
                entries.append((step, str(adapter_dir), "adapter"))
            else:
                actor_dir = path / "actor"
                if is_adapter_dir(actor_dir):
                    entries.append((step, str(actor_dir), "adapter"))
                else:
                    model_dir = path / "actor" / "huggingface"
                    if is_hf_model_dir(model_dir):
                        entries.append((step, str(model_dir), "full"))
        elif path.name == "actor":
            step = parse_step_from_path(str(path)) or -1
            adapter_dir = path / "lora_adapter"
            if is_adapter_dir(adapter_dir):
                entries.append((step, str(adapter_dir), "adapter"))
            elif is_adapter_dir(path):
                entries.append((step, str(path), "adapter"))
            else:
                model_dir = path / "huggingface"
                if is_hf_model_dir(model_dir):
                    entries.append((step, str(model_dir), "full"))
        else:
            adapters = list_checkpoint_adapters(str(path), only_latest=False)
            if adapters:
                entries.extend([(step, adapter, "adapter") for step, adapter in adapters])
            full_models = list_checkpoint_full_models(str(path), only_latest=False)
            if full_models:
                entries.extend([(step, model_dir, "full") for step, model_dir in full_models])
        if wanted_steps:
            entries = [e for e in entries if e[0] in wanted_steps]
        if not entries:
            print(f"[WARN] No usable checkpoints found under {path}")
            continue
        run_entries.append((label, entries))

    if not run_entries:
        print("[ERROR] No runs to process")
        return

    base_model = resolve_base_model_path(args.base_model.strip() or None, run_dirs[0], ["base_model", "/hss/giil/caixq/model", "/home/caixq/base_model"])
    print(f"[INFO] base_model={base_model}")

    need_base_model = any(mode == "full" for _, entries in run_entries for _, _, mode in entries)
    base_model_ref = None
    if need_base_model:
        print("[INFO] Loading base model for full-weight alignment")
        base_model_ref = _load_full_model(base_model, dtype=dtype, device=device, device_map=device_map, trust_remote_code=args.trust_remote_code)

    for label, entries in run_entries:
        entries = sorted(entries, key=lambda x: (x[0] is None, x[0]))
        print(f"[INFO] Run {label}: {len(entries)} checkpoints")
        for step, path_str, mode in entries:
            step_seed = args.random_seed + (step if step is not None else 0)
            torch.manual_seed(step_seed)
            print(f"[INFO] Loading {label} step {step} ({mode}) from {path_str}")
            if mode == "adapter":
                model = _load_peft_model(
                    base_model,
                    Path(path_str),
                    dtype=dtype,
                    device=device,
                    device_map=device_map,
                    trust_remote_code=args.trust_remote_code,
                )
                eta, eta_rand, rows = _compute_alignment(
                    model,
                    uv_cache=uv_cache,
                    principal_rank=args.principal_rank,
                    use_lowrank=args.use_lowrank,
                    module_filters=module_filters,
                    device=device,
                )
                all_rows.append({
                    "run": label,
                    "step": step,
                    "checkpoint_path": path_str,
                    "mode": mode,
                    "eta": eta,
                    "eta_random": eta_rand,
                    "num_modules": len(rows),
                })
                print(f"[INFO] {label} step {step}: eta={eta} eta_random={eta_rand}")
                del model
            else:
                if base_model_ref is None:
                    print(f"[WARN] Skipping full-model alignment for {path_str}: base model not loaded")
                    continue
                tuned_model = _load_full_model(path_str, dtype=dtype, device=device, device_map=device_map, trust_remote_code=args.trust_remote_code)
                eta, eta_rand, rows = _compute_alignment_full(
                    base_model_ref,
                    tuned_model,
                    uv_cache=uv_cache,
                    principal_rank=args.principal_rank,
                    use_lowrank=args.use_lowrank,
                    module_filters=module_filters,
                    device=device,
                )
                all_rows.append({
                    "run": label,
                    "step": step,
                    "checkpoint_path": path_str,
                    "mode": mode,
                    "eta": eta,
                    "eta_random": eta_rand,
                    "num_modules": len(rows),
                })
                print(f"[INFO] {label} step {step}: eta={eta} eta_random={eta_rand}")
                del tuned_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if base_model_ref is not None:
        del base_model_ref
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_rows:
        print("[ERROR] No alignment data computed")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "alignment_curve_random.csv", index=False)
    with (out_dir / "alignment_curve_random.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    fig, ax = plt.subplots(figsize=(9, 5))
    for run, sub in df.groupby("run"):
        sub = sub.sort_values(by=["step"], na_position="last")
        fallback = pd.Series(np.arange(len(sub)), index=sub.index)
        x = sub["step"].fillna(fallback).to_numpy()
        y = sub["eta"].to_numpy()
        y_rand = sub["eta_random"].to_numpy()
        line = ax.plot(x, y, marker="o", linewidth=2, label=run)[0]
        ax.plot(x, y_rand, linestyle="--", linewidth=2, color=line.get_color(), label=f"{run} (random)")
    ax.set_xlabel("Global Step", fontsize=font_sizes['xlabel'], fontfamily=font_sizes['fontfamily'])
    ax.set_ylabel("Alignment Ratio (eta)", fontsize=font_sizes['ylabel'], fontfamily=font_sizes['fontfamily'])
    # Title removed for publication
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis='both', labelsize=font_sizes['tick'])
    create_legend(ax, font_sizes)
    fig.tight_layout()
    out_path = out_dir / "alignment_curve_random.png"
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_dir / "alignment_curve_random.pdf", dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved plot: {out_path} and .pdf")


if __name__ == "__main__":
    main()
