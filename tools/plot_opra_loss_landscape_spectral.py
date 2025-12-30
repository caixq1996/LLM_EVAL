#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot loss landscape along spectral directions (principal vs residual).

Example:
  python project/LLM_EVAL/tools/plot_opra_loss_landscape_spectral.py \
    --run opra=Qwen2.5-math-1.5B_opra \
    --run lora=Qwen2.5-math-1.5B_vanilla \
    --prompt_file project/LLM_EVAL/data/gsm8k/test.jsonl \
    --out_dir project/LLM_EVAL/eval_log/opra_loss_landscape_spectral
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from opra_spectral_utils import (
    collect_run_entries,
    get_base_weight,
    iter_lora_modules,
    list_run_dirs,
    resolve_checkpoint_root,
    resolve_base_model_path,
    topk_svd,
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


def _load_prompts(prompt_file: Path, prompt_field: Optional[str], answer_field: Optional[str], max_samples: int, seed: int) -> List[Tuple[str, Optional[str]]]:
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

    pairs: List[Tuple[str, Optional[str]]] = []
    for row in rows:
        prompt = None
        if prompt_field and prompt_field in row:
            prompt = str(row[prompt_field])
        else:
            for key in ("question", "problem", "prompt", "input"):
                if key in row:
                    prompt = str(row[key])
                    break
        if prompt is None:
            continue
        answer = None
        if answer_field and answer_field in row:
            answer = str(row[answer_field])
        pairs.append((prompt, answer))

    if not pairs:
        raise ValueError("No prompts found")
    if max_samples > 0 and len(pairs) > max_samples:
        random.seed(seed)
        pairs = random.sample(pairs, max_samples)
    return pairs


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


def _build_batches(
    tokenizer: AutoTokenizer,
    pairs: List[Tuple[str, Optional[str]]],
    *,
    max_length: int,
    batch_size: int,
    mask_prompt: bool,
) -> List[Dict[str, torch.Tensor]]:
    prompts = [p for p, _ in pairs]
    answers = [a for _, a in pairs]
    full_texts = []
    for prompt, answer in pairs:
        if answer:
            full_texts.append(f"{prompt}\n{answer}")
        else:
            full_texts.append(prompt)

    tokenized = tokenizer(full_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = tokenized["input_ids"].clone()

    if mask_prompt and any(a is not None for a in answers):
        prompt_tokens = tokenizer(prompts, padding=False, truncation=True, max_length=max_length)
        for i, ids in enumerate(prompt_tokens["input_ids"]):
            prompt_len = min(len(ids), labels.shape[1])
            labels[i, :prompt_len] = -100

    tokenized["labels"] = labels

    batches = []
    total = labels.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = {k: v[start:end] for k, v in tokenized.items()}
        batches.append(batch)
    return batches


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
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    _ensure_adalora_trainable_name(model)
    return model


def _load_full_model(model_path: str, *, dtype: torch.dtype, device: torch.device, device_map: Optional[str], trust_remote_code: bool) -> torch.nn.Module:
    if device_map:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust_remote_code)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=trust_remote_code)
        model.to(device)
    model.eval()
    model.config.use_cache = False
    return model


def _ensure_adalora_trainable_name(model: torch.nn.Module) -> None:
    base_model = getattr(model, "base_model", None)
    if base_model is None:
        return
    if base_model.__class__.__name__.lower() != "adaloramodel":
        return
    if hasattr(base_model, "trainable_adapter_name"):
        return
    adapter_name = getattr(model, "active_adapter", None)
    if isinstance(adapter_name, (list, tuple)):
        adapter_name = adapter_name[0] if adapter_name else None
    if not adapter_name:
        adapter_name = "default"
    base_model.trainable_adapter_name = adapter_name
    print(f"[INFO] AdaLora: set trainable_adapter_name={adapter_name} for loss evaluation")


def _collect_lora_param_entries(model: torch.nn.Module) -> List[Dict[str, object]]:
    name_map = {id(param): name for name, param in model.named_parameters()}
    entries: List[Dict[str, object]] = []
    for module_name, module in iter_lora_modules(model):
        base_weight = get_base_weight(module)
        if base_weight is None or not isinstance(base_weight, torch.Tensor) or base_weight.dim() != 2:
            continue
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if isinstance(lora_a, nn.ModuleDict) and isinstance(lora_b, nn.ModuleDict):
            keys = sorted(set(lora_a.keys()) & set(lora_b.keys()))
            for key in keys:
                a_mod = lora_a[key]
                b_mod = lora_b[key]
                if hasattr(a_mod, "weight"):
                    param = a_mod.weight
                    name = name_map.get(id(param), f"{module_name}.lora_A.{key}.weight")
                    entries.append({
                        "name": name,
                        "param": param,
                        "module": module_name,
                        "side": "A",
                        "base_weight": base_weight,
                    })
                if hasattr(b_mod, "weight"):
                    param = b_mod.weight
                    name = name_map.get(id(param), f"{module_name}.lora_B.{key}.weight")
                    entries.append({
                        "name": name,
                        "param": param,
                        "module": module_name,
                        "side": "B",
                        "base_weight": base_weight,
                    })
        else:
            if hasattr(lora_a, "weight"):
                param = lora_a.weight
                name = name_map.get(id(param), f"{module_name}.lora_A.weight")
                entries.append({
                    "name": name,
                    "param": param,
                    "module": module_name,
                    "side": "A",
                    "base_weight": base_weight,
                })
            if hasattr(lora_b, "weight"):
                param = lora_b.weight
                name = name_map.get(id(param), f"{module_name}.lora_B.weight")
                entries.append({
                    "name": name,
                    "param": param,
                    "module": module_name,
                    "side": "B",
                    "base_weight": base_weight,
                })
    return entries


def _get_principal_vectors(
    module_name: str,
    base_weight: torch.Tensor,
    *,
    principal_rank: int,
    device: torch.device,
    use_lowrank: bool,
    uv_cache: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    cached = uv_cache.get(module_name)
    if cached is not None:
        return cached
    if base_weight is None or base_weight.dim() != 2:
        uv_cache[module_name] = (None, None)
        return None, None
    u, v = topk_svd(base_weight, principal_rank, device=device, use_lowrank=use_lowrank)
    if u is None or v is None:
        uv_cache[module_name] = (None, None)
        return None, None
    u1 = u[:, 0].detach().cpu()
    v1 = v[:, 0].detach().cpu()
    uv_cache[module_name] = (u1, v1)
    return u1, v1


def _make_spectral_directions(
    entries: List[Dict[str, object]],
    *,
    principal_rank: int,
    seed: int,
    device: torch.device,
    use_lowrank: bool,
    uv_cache: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    unit_dirs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for entry in entries:
        name = str(entry["name"])
        param: torch.Tensor = entry["param"]  # type: ignore[assignment]
        module_name = str(entry["module"])
        side = str(entry["side"])
        base_weight: torch.Tensor = entry["base_weight"]  # type: ignore[assignment]
        shape = tuple(param.shape)
        u1, v1 = _get_principal_vectors(
            module_name,
            base_weight,
            principal_rank=principal_rank,
            device=device,
            use_lowrank=use_lowrank,
            uv_cache=uv_cache,
        )
        d1: torch.Tensor
        d2: torch.Tensor
        if u1 is None or v1 is None:
            d1 = torch.randn(shape, generator=gen, device="cpu", dtype=torch.float32)
            d2 = torch.randn(shape, generator=gen, device="cpu", dtype=torch.float32)
        else:
            u1f = u1.to(dtype=torch.float32)
            v1f = v1.to(dtype=torch.float32)
            if side == "A" and len(shape) == 2 and v1f.numel() == shape[1]:
                r = shape[0]
                rvec = torch.randn(r, generator=gen, device="cpu", dtype=torch.float32)
                d1 = torch.outer(rvec, v1f)
                rand = torch.randn(shape, generator=gen, device="cpu", dtype=torch.float32)
                proj = rand @ v1f
                d2 = rand - proj[:, None] * v1f[None, :]
            elif side == "B" and len(shape) == 2 and u1f.numel() == shape[0]:
                r = shape[1]
                cvec = torch.randn(r, generator=gen, device="cpu", dtype=torch.float32)
                d1 = torch.outer(u1f, cvec)
                rand = torch.randn(shape, generator=gen, device="cpu", dtype=torch.float32)
                proj = (u1f.unsqueeze(0) @ rand)
                d2 = rand - u1f.unsqueeze(1) * proj
            else:
                d1 = torch.randn(shape, generator=gen, device="cpu", dtype=torch.float32)
                d2 = torch.randn(shape, generator=gen, device="cpu", dtype=torch.float32)
        d1 = d1 / (d1.norm() + 1e-12)
        d2 = d2 / (d2.norm() + 1e-12)
        unit_dirs[name] = (d1, d2)
    return unit_dirs


def _apply_perturbation(
    params: Dict[str, torch.nn.Parameter],
    base_params: Dict[str, torch.Tensor],
    unit_dirs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    alpha: float,
    beta: float,
) -> None:
    with torch.no_grad():
        for name, param in params.items():
            base = base_params[name]
            d1_unit, d2_unit = unit_dirs[name]
            d1 = d1_unit.to(param.device, dtype=param.dtype)
            d2 = d2_unit.to(param.device, dtype=param.dtype)
            scale = base.norm().to(param.dtype)
            if scale.item() == 0.0:
                scale = torch.tensor(1.0, device=param.device, dtype=param.dtype)
            param.copy_(base + alpha * d1 * scale + beta * d2 * scale)


def _evaluate_loss(model: torch.nn.Module, batches: List[Dict[str, torch.Tensor]], device: torch.device) -> float:
    total_loss = 0.0
    total_tokens = 0
    for batch in batches:
        inputs = {k: v.to(device) for k, v in batch.items()}
        labels = inputs.get("labels")
        token_count = int((labels != -100).sum().item()) if labels is not None else int(labels.numel())
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
        if loss is None:
            continue
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count
    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_root", type=str, default="project/OPRA/checkpoints/OPRA-LoRA")
    ap.add_argument("--base_model", type=str, default="", help="Base model path or HF id (auto-resolved if empty)")
    ap.add_argument("--run", action="append", required=True, help="label=adapter_path or run dir")
    ap.add_argument("--prompt_file", required=True)
    ap.add_argument("--prompt_field", type=str, default="question")
    ap.add_argument("--answer_field", type=str, default="answer")
    ap.add_argument("--mask_prompt", action="store_true", default=False)
    ap.add_argument("--num_samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--direction_seed", type=int, default=1234)
    ap.add_argument("--grid_size", type=int, default=9)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--principal_rank", type=int, default=1)
    ap.add_argument("--use_lowrank", action="store_true", default=False)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--device_map", type=str, default="")
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--trust_remote_code", action="store_true", default=False)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", type=str, default="Spectral Loss Landscape (LoRA)")
    ap.add_argument("--steps", type=str, default="", help="Comma-separated global steps to plot (default: all)")
    ap.add_argument("--step", type=int, default=-1, help="Pick a specific global_step (deprecated; use --steps)")
    ap.add_argument("--shared_directions", action="store_true", default=True)
    ap.add_argument("--no_shared_directions", dest="shared_directions", action="store_false")
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
    pairs = _load_prompts(Path(args.prompt_file), args.prompt_field or None, args.answer_field or None, args.num_samples, args.seed)
    tokenizer = _prepare_tokenizer(base_model)
    batches = _build_batches(tokenizer, pairs, max_length=args.max_length, batch_size=args.batch_size, mask_prompt=args.mask_prompt)
    print(f"[INFO] Prepared {len(pairs)} samples in {len(batches)} batches")

    alphas = np.linspace(-args.radius, args.radius, args.grid_size)
    betas = np.linspace(-args.radius, args.radius, args.grid_size)
    X, Y = np.meshgrid(alphas, betas)

    unit_dirs: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
    uv_cache: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {}

    for step in all_steps:
        print(f"[INFO] ===== Step {step} =====")
        run_losses: Dict[str, np.ndarray] = {}

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

            entries = _collect_lora_param_entries(model)
            if not entries:
                print(f"[WARN] No LoRA params found for {label}; skipping")
                del model
                continue

            params = {entry["name"]: entry["param"] for entry in entries}  # type: ignore[dict-item]

            if unit_dirs is None or not args.shared_directions:
                unit_dirs = _make_spectral_directions(
                    entries,
                    principal_rank=args.principal_rank,
                    seed=args.direction_seed,
                    device=device,
                    use_lowrank=args.use_lowrank,
                    uv_cache=uv_cache,
                )
            else:
                missing = [entry for entry in entries if entry["name"] not in unit_dirs]
                if missing:
                    extra = _make_spectral_directions(
                        missing,
                        principal_rank=args.principal_rank,
                        seed=args.direction_seed + 1,
                        device=device,
                        use_lowrank=args.use_lowrank,
                        uv_cache=uv_cache,
                    )
                    unit_dirs.update(extra)

            base_params = {name: p.detach().clone() for name, p in params.items()}

            loss_grid = np.zeros((args.grid_size, args.grid_size), dtype=np.float32)
            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):
                    _apply_perturbation(params, base_params, unit_dirs, float(alpha), float(beta))
                    loss = _evaluate_loss(model, batches, device)
                    loss_grid[j, i] = loss
                print(f"[INFO] {label} step {step}: row {i + 1}/{args.grid_size}")

            _apply_perturbation(params, base_params, unit_dirs, 0.0, 0.0)
            run_losses[label] = loss_grid
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not run_losses:
            print(f"[WARN] No loss grid for step {step}")
            continue

        npz_path = out_dir / f"spectral_loss_landscape_step_{step}.npz"
        np.savez(npz_path, alphas=alphas, betas=betas, **{f"loss_{k}": v for k, v in run_losses.items()})
        print(f"[INFO] Saved grid: {npz_path}")

        labels = list(run_losses.keys())
        fig2d, axes2d = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5), squeeze=False)
        axes2d = axes2d[0]
        for idx, label in enumerate(labels):
            ax = axes2d[idx]
            cs = ax.contourf(X, Y, run_losses[label], levels=30, cmap="viridis")
            ax.set_title(label)
            ax.set_xlabel("alpha (principal)")
            ax.set_ylabel("beta (residual)")
            fig2d.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        fig2d.suptitle(f"{args.title} 2D (step {step})")
        fig2d.tight_layout(rect=[0, 0, 1, 0.95])
        out_2d = out_dir / f"spectral_loss_landscape_2d_step_{step}.png"
        fig2d.savefig(out_2d, dpi=300)
        plt.close(fig2d)
        print(f"[INFO] Saved plot: {out_2d}")

        fig3d = plt.figure(figsize=(6 * len(labels), 5))
        for idx, label in enumerate(labels, start=1):
            ax = fig3d.add_subplot(1, len(labels), idx, projection="3d")
            ax.plot_surface(X, Y, run_losses[label], cmap="viridis", linewidth=0, antialiased=True)
            ax.set_title(label)
            ax.set_xlabel("alpha (principal)")
            ax.set_ylabel("beta (residual)")
            ax.set_zlabel("loss")
        fig3d.suptitle(f"{args.title} 3D (step {step})")
        fig3d.tight_layout(rect=[0, 0, 1, 0.95])
        out_3d = out_dir / f"spectral_loss_landscape_3d_step_{step}.png"
        fig3d.savefig(out_3d, dpi=300)
        plt.close(fig3d)
        print(f"[INFO] Saved plot: {out_3d}")


if __name__ == "__main__":
    main()
