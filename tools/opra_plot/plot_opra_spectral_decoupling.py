#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot principal vs residual energy scatter per step for OPRA vs LoRA.

Example:
  python project/LLM_EVAL/tools/plot_opra_spectral_decoupling.py \
    --run opra=Qwen2.5-math-1.5B_opra \
    --run lora=Qwen2.5-math-1.5B_vanilla \
    --prompt_file project/LLM_EVAL/data/gsm8k/test.jsonl \
    --out_dir project/LLM_EVAL/eval_log/opra_spectral_decoupling
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from opra_spectral_utils import (
    load_peft_config,
    collect_run_entries,
    list_run_dirs,
    resolve_checkpoint_root,
    resolve_base_model_path,
)


def _parse_run_arg(run_arg: str) -> Tuple[str, str]:
    if "=" not in run_arg:
        raise ValueError("--run must be label=PATH")
    label, path = run_arg.split("=", 1)
    return label.strip(), path.strip()


def _dtype_from_name(name: str) -> torch.dtype:
    name = (name or "").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    return torch.float16


def _load_prompts(prompt_file: Path, prompt_field: Optional[str], max_samples: int, seed: int) -> List[str]:
    if not prompt_file.exists():
        raise FileNotFoundError(prompt_file)
    if prompt_file.suffix == ".json":
        rows = json.loads(prompt_file.read_text(encoding="utf-8"))
    else:
        rows = []
        with prompt_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    prompts: List[str] = []
    for row in rows:
        if prompt_field and prompt_field in row:
            prompts.append(str(row[prompt_field]))
            continue
        for key in ("question", "problem", "prompt", "input"):
            if key in row:
                prompts.append(str(row[key]))
                break
    if not prompts:
        raise ValueError("No prompts found")
    if max_samples > 0 and len(prompts) > max_samples:
        random.seed(seed)
        prompts = random.sample(prompts, max_samples)
    return prompts


def _prepare_tokenizer(model_path: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.unk_token is not None:
            tok.pad_token = tok.unk_token
        else:
            raise ValueError("Tokenizer has no pad_token")
    tok.padding_side = "left"
    return tok


def _load_base_model(base_model: str, *, dtype: torch.dtype, device: torch.device, device_map: Optional[str], trust_remote_code: bool) -> torch.nn.Module:
    if device_map:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust_remote_code)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, trust_remote_code=trust_remote_code)
        model.to(device)
    model.eval()
    return model


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


def _collect_reprs(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    layers: List[int],
    *,
    batch_size: int,
    max_length: int,
    pool: str,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    reps = {layer: [] for layer in layers}
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden_states = outputs.hidden_states
        mask = inputs.get("attention_mask")
        for layer in layers:
            h = hidden_states[layer]
            if pool == "mean":
                mask_f = mask.unsqueeze(-1).to(h.dtype)
                pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            else:
                pooled = h[:, -1, :]
            reps[layer].append(pooled.detach().cpu().to(torch.float32))
    return {layer: torch.cat(chunks, dim=0) for layer, chunks in reps.items()}


def _compute_pca(x: torch.Tensor, k: int) -> torch.Tensor:
    # Replace NaN/Inf with 0 to prevent SVD convergence failure
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    x_centered = x - x.mean(dim=0, keepdim=True)
    x_centered = torch.where(torch.isfinite(x_centered), x_centered, torch.zeros_like(x_centered))
    _, _, vh = torch.linalg.svd(x_centered, full_matrices=False)
    return vh[:k, :].transpose(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_root", type=str, default="project/OPRA/checkpoints/OPRA-LoRA")
    ap.add_argument("--base_model", type=str, default="", help="Base model path or HF id (auto-resolved if empty)")
    ap.add_argument("--run", action="append", required=True, help="label=adapter_path or run dir")
    ap.add_argument("--prompt_file", required=True)
    ap.add_argument("--prompt_field", type=str, default="question")
    ap.add_argument("--num_samples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--principal_rank", type=int, default=16)
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "last"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--device_map", type=str, default="")
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--trust_remote_code", action="store_true", default=False)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", type=str, default="Spectral Decoupling Analysis")
    ap.add_argument("--steps", type=str, default="", help="Comma-separated global steps to plot (default: all)")
    ap.add_argument("--step", type=int, default=-1, help="Pick a specific global_step (deprecated; use --steps)")
    args = ap.parse_args()

    checkpoint_root = resolve_checkpoint_root(args.checkpoint_root)
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_name(args.dtype)
    device_map = args.device_map.strip() or None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_filter: Optional[List[int]] = None
    if args.steps.strip():
        step_filter = [int(s) for s in args.steps.split(",") if s.strip().isdigit()]
    elif args.step >= 0:
        step_filter = [args.step]

    run_specs: List[Tuple[str, Dict[int, Tuple[str, str]]]] = []
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
        run_specs.append((label, {step: (path, mode) for step, path, mode in entries}))
        resolved_dirs.append(resolved)
        print(f"[INFO] Run {label}: {len(entries)} steps")

    if not run_specs:
        print("[ERROR] No runs to process")
        return

    all_steps = sorted({step for _, step_map in run_specs for step in step_map.keys()})
    if not all_steps:
        print("[ERROR] No steps found to plot")
        return

    base_model = resolve_base_model_path(
        args.base_model.strip() or None,
        resolved_dirs[0],
        ["base_model", "/hss/giil/caixq/model", "/home/caixq/base_model"],
    )

    print(f"[INFO] checkpoint_root={checkpoint_root}")
    print(f"[INFO] base_model={base_model}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] prompt_file={args.prompt_file}")

    prompts = _load_prompts(Path(args.prompt_file), args.prompt_field or None, args.num_samples, args.seed)
    tokenizer = _prepare_tokenizer(base_model)

    print(f"[INFO] Loading base model for PCA")
    base_ref = _load_base_model(base_model, dtype=dtype, device=device, device_map=device_map, trust_remote_code=args.trust_remote_code)
    num_layers = getattr(base_ref.config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Cannot infer num_hidden_layers from model config")
    layers = list(range(1, num_layers + 1))

    print(f"[INFO] Collecting base representations ({len(prompts)} prompts, {len(layers)} layers)")
    base_reprs = _collect_reprs(
        base_ref,
        tokenizer,
        prompts,
        layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pool=args.pool,
        device=device,
    )
    base_pcs = {layer: _compute_pca(base_reprs[layer], args.principal_rank) for layer in layers}
    del base_ref
    if device.type == "cuda":
        torch.cuda.empty_cache()

    all_rows: List[Dict] = []
    print(f"[INFO] Plotting {len(all_steps)} steps to {out_dir}")
    for step in all_steps:
        step_rows: List[Dict] = []
        step_labels: List[str] = []
        print(f"[INFO] ===== Step {step} =====")
        for label, step_map in run_specs:
            entry = step_map.get(step)
            if entry is None:
                print(f"[WARN] Missing step {step} for run {label}; skipping")
                continue
            path_str, mode = entry
            print(f"[INFO] Loading {label} ({mode}) from {path_str}")
            if mode == "adapter":
                model = _load_peft_model(base_model, Path(path_str), dtype=dtype, device=device, device_map=device_map, trust_remote_code=args.trust_remote_code)
            else:
                model = _load_full_model(path_str, dtype=dtype, device=device, device_map=device_map, trust_remote_code=args.trust_remote_code)

            tuned_reprs = _collect_reprs(
                model,
                tokenizer,
                prompts,
                layers,
                batch_size=args.batch_size,
                max_length=args.max_length,
                pool=args.pool,
                device=device,
            )
            for layer in layers:
                delta = tuned_reprs[layer] - base_reprs[layer]
                pcs = base_pcs[layer]
                proj_p = delta @ pcs
                energy_principal = proj_p.pow(2).sum(dim=1).mean().item()
                energy_total = delta.pow(2).sum(dim=1).mean().item()
                energy_residual = max(0.0, energy_total - energy_principal)
                row = {
                    "run": label,
                    "layer": layer,
                    "energy_principal": energy_principal,
                    "energy_residual": energy_residual,
                    "step": step,
                }
                step_rows.append(row)
                all_rows.append(row)
            step_labels.append(label)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not step_rows:
            print(f"[WARN] No data for step {step}; skipping plot")
            continue

        df_step = pd.DataFrame(step_rows)
        fig, ax = plt.subplots(figsize=(8, 8))
        for run, sub in df_step.groupby("run"):
            ax.scatter(sub["energy_residual"], sub["energy_principal"], label=run, alpha=0.7, s=80)
            cx = sub["energy_residual"].mean()
            cy = sub["energy_principal"].mean()
            ax.scatter(cx, cy, marker="*", s=260, edgecolors="black", label=f"{run} (Mean)")
        ax.set_xlabel("Reasoning Adaptation (Residual Energy)")
        ax.set_ylabel("Knowledge Distortion (Principal Energy)")
        ax.set_title(f"{args.title} (step {step})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        out_path = out_dir / f"spectral_decoupling_step_{step}.png"
        fig.savefig(out_path, dpi=300)
        fig.savefig(out_dir / f"spectral_decoupling_step_{step}.pdf", dpi=300)
        plt.close(fig)
        print(f"[INFO] Saved plot: {out_path} and .pdf")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "spectral_decoupling.csv", index=False)
    with (out_dir / "spectral_decoupling.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"[INFO] Wrote summary: {out_dir / 'spectral_decoupling.csv'}")


if __name__ == "__main__":
    main()
