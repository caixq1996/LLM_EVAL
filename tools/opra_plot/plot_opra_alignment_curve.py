#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot spectral alignment (eta) across checkpoints for OPRA vs baselines.

Example:
  python project/LLM_EVAL/tools/plot_opra_alignment_curve.py \
    --base_model /path/to/base \
    --run opra=/path/to/opra_run \
    --run lora=/path/to/lora_run \
    --out_dir project/LLM_EVAL/eval_log/opra_alignment
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
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

_WANDB_RUN_RE = re.compile(r"^run-")

# Algorithm name mapping for legend display
ALGO_NAME_MAP = {
    "vanilla": "LoRA",
    "opra": "OPRA",
    "opra_opra": "OPRA-OPRA",
    "adalora": "AdaLoRA",
    "dora": "DoRA",
    "rslora": "RSLoRA",
    "pissa": "PiSSA",
    "qpissa": "QPiSSA",
    "qlora": "QLoRA",
    "olora": "OLoRA",
    "oft": "OFT",
}


def get_algo_display_name(run_name: str) -> str:
    """Extract algorithm suffix and return display name for legend."""
    # Extract suffix after last underscore (e.g., 'Qwen2.5-math-1.5B_vanilla' -> 'vanilla')
    parts = run_name.rsplit("_", 1)
    if len(parts) == 2:
        suffix = parts[1]
        return ALGO_NAME_MAP.get(suffix, suffix)
    return run_name


def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _load_wandb_config(run_dir: Path) -> Optional[dict]:
    config_path = run_dir / "files" / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
    except Exception as exc:
        print(f"[WARN] PyYAML unavailable; cannot read {config_path}: {exc}")
        return None
    try:
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Failed to parse {config_path}: {exc}")
        return None


def _extract_experiment_name(cfg: dict) -> Optional[str]:
    trainer = cfg.get("trainer")
    if isinstance(trainer, dict):
        trainer_val = trainer.get("value", trainer)
        if isinstance(trainer_val, dict):
            exp = trainer_val.get("experiment_name")
            if isinstance(exp, dict):
                exp = exp.get("value")
            return exp
    return None


def _wandb_started_at(run_dir: Path) -> Optional[float]:
    meta_path = run_dir / "files" / "wandb-metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Failed to read {meta_path}: {exc}")
        return None
    started_at = meta.get("startedAt")
    if not started_at:
        return None
    try:
        ts = started_at.replace("Z", "+00:00")
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return None


def _index_wandb_runs(wandb_root: Path) -> Dict[str, List[Tuple[float, Path]]]:
    index: Dict[str, List[Tuple[float, Path]]] = {}
    if not wandb_root.exists():
        return index
    for run_dir in wandb_root.iterdir():
        if not run_dir.is_dir() or not _WANDB_RUN_RE.match(run_dir.name):
            continue
        cfg = _load_wandb_config(run_dir)
        if not isinstance(cfg, dict):
            continue
        exp_name = _extract_experiment_name(cfg)
        if not exp_name:
            continue
        started_at = _wandb_started_at(run_dir) or run_dir.stat().st_mtime
        index.setdefault(exp_name, []).append((started_at, run_dir))
    return index


def _select_wandb_run(wandb_index: Dict[str, List[Tuple[float, Path]]], label: str) -> Optional[Path]:
    if label in wandb_index:
        return max(wandb_index[label], key=lambda x: x[0])[1]
    norm_label = _norm_key(label)
    candidates: List[Tuple[int, str]] = []
    for key in wandb_index:
        norm_key = _norm_key(key)
        if norm_label == norm_key or norm_label in norm_key or norm_key in norm_label:
            candidates.append((len(norm_key), key))
    if not candidates:
        return None
    _, best_key = max(candidates, key=lambda x: x[0])
    return max(wandb_index[best_key], key=lambda x: x[0])[1]


def _parse_metric_value(line: str, metric_key: str) -> Optional[float]:
    pattern = rf"{re.escape(metric_key)}:([+-]?[0-9]*\\.?[0-9]+(?:[eE][+-]?[0-9]+)?)"
    match = re.search(pattern, line)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _discover_eta_keys(
    log_path: Path, *, prefer_grad: Optional[str], prefer_param: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    if not log_path.exists():
        return (None, None)
    key_counts: Dict[str, int] = {}
    pattern = re.compile(r"([A-Za-z0-9_./-]*eta[A-Za-z0-9_./-]*)\\s*:")
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "eta" not in line:
                continue
            for match in pattern.finditer(line):
                key = match.group(1)
                key_counts[key] = key_counts.get(key, 0) + 1
    if not key_counts:
        return (None, None)

    def score(key: str) -> Tuple[int, int]:
        key_l = key.lower()
        base = 0
        if prefer_param and key == prefer_param:
            base += 100
        if prefer_grad and key == prefer_grad:
            base += 100
        if key_l.startswith("actor/"):
            base += 10
        if "grad" in key_l:
            base += 5
        if any(tok in key_l for tok in ("delta", "param", "w_eta", "dw")):
            base += 4
        return (base, key_counts.get(key, 0))

    keys = list(key_counts.keys())
    grad_candidates = [k for k in keys if "grad" in k.lower()]
    param_candidates = [k for k in keys if any(tok in k.lower() for tok in ("delta", "param", "w_eta", "dw"))]

    grad_key = max(grad_candidates, key=score) if grad_candidates else None
    param_key = max(param_candidates, key=score) if param_candidates else None
    if param_key is None:
        non_grad = [k for k in keys if k != grad_key]
        if non_grad:
            param_key = max(non_grad, key=score)
        elif grad_key:
            param_key = grad_key
    return (grad_key, param_key)


def _collect_eta_from_wandb(
    run_dir: Path, *, grad_key: Optional[str], param_key: Optional[str]
) -> List[Dict[str, Optional[float]]]:
    log_path = run_dir / "files" / "output.log"
    if not log_path.exists():
        print(f"[WARN] WandB output log missing: {log_path}")
        return []
    def _parse_rows(active_grad: Optional[str], active_param: Optional[str]) -> List[Dict[str, Optional[float]]]:
        rows: Dict[int, Dict[str, Optional[float]]] = {}
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "step:" not in line or "actor/" not in line:
                    continue
                if line.lstrip().startswith("wandb:"):
                    continue
                step_match = re.search(r"\\bstep:(\\d+)\\b", line)
                if not step_match:
                    continue
                try:
                    step = int(step_match.group(1))
                except Exception:
                    continue
                grad_eta = _parse_metric_value(line, active_grad) if active_grad else None
                param_eta = _parse_metric_value(line, active_param) if active_param else None
                if grad_eta is None and param_eta is None:
                    continue
                entry = rows.setdefault(step, {"step": step, "eta_grad": None, "eta_param": None})
                if grad_eta is not None:
                    entry["eta_grad"] = grad_eta
                if param_eta is not None:
                    entry["eta_param"] = param_eta
        return [rows[k] for k in sorted(rows.keys())]

    rows = _parse_rows(grad_key, param_key)
    if rows:
        return rows
    alt_grad, alt_param = _discover_eta_keys(log_path, prefer_grad=grad_key, prefer_param=param_key)
    if alt_grad or alt_param:
        if alt_grad != grad_key or alt_param != param_key:
            print(f"[INFO] Auto-detected eta keys in {log_path.name}: grad={alt_grad} param={alt_param}")
        return _parse_rows(alt_grad, alt_param)
    return []

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
                import torch
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
) -> Tuple[Optional[float], List[Dict]]:
    total_num = 0.0
    total_den = 0.0
    rows: List[Dict] = []
    for name, module in iter_lora_modules(model):
        if module_filters and not any(f in name for f in module_filters):
            continue
        delta = compute_lora_delta(module, device=device, dtype=torch.float32)
        if delta is None or float(delta.pow(2).sum().item()) == 0.0:
            continue
        u, v = uv_cache.get(name, (None, None))
        if u is None or v is None:
            base_w = get_base_weight(module)
            u, v = topk_svd(base_w, principal_rank, device=device, use_lowrank=use_lowrank)
            uv_cache[name] = (u, v)
        u = u.to(device) if u is not None else None
        v = v.to(device) if v is not None else None
        eta, den = projection_energy_ratio(delta, u, v)
        total_num += eta * den
        total_den += den
        rows.append({
            "module": name,
            "layer": infer_layer_idx(name),
            "module_type": infer_module_type(name),
            "eta": eta,
            "delta_norm_sq": den,
        })
    if total_den == 0.0:
        return None, rows
    return total_num / total_den, rows


def _compute_alignment_full(
    base_model: torch.nn.Module,
    tuned_model: torch.nn.Module,
    *,
    uv_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    principal_rank: int,
    use_lowrank: bool,
    module_filters: List[str],
    device: torch.device,
) -> Tuple[Optional[float], List[Dict]]:
    total_num = 0.0
    total_den = 0.0
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
        total_num += eta * den
        total_den += den
        rows.append({
            "module": name,
            "layer": infer_layer_idx(name),
            "module_type": infer_module_type(name),
            "eta": eta,
            "delta_norm_sq": den,
        })
    if total_den == 0.0:
        return None, rows
    return total_num / total_den, rows


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
    ap.add_argument("--trust_remote_code", action="store_true", default=False)
    ap.add_argument(
        "--eta_source",
        type=str,
        default=os.getenv("ETA_SOURCE", "both"),
        choices=["checkpoint", "compute", "recompute", "wandb", "both"],
    )
    ap.add_argument("--wandb_root", type=str, default=os.getenv("WANDB_ROOT", "project/OPRA/wandb"))
    ap.add_argument("--wandb_metric_grad", type=str, default=os.getenv("WANDB_METRIC_GRAD", "actor/grad_eta"))
    ap.add_argument("--wandb_metric_param", type=str, default=os.getenv("WANDB_METRIC_PARAM", "actor/delta_w_eta"))
    ap.add_argument("--wandb_plot_both", action="store_true", default=os.getenv("WANDB_PLOT_BOTH", "1") != "0")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", type=str, default="Spectral Alignment Across Steps")
    ap.add_argument("--steps", type=str, default="", help="Comma-separated step numbers to keep (default: all)")
    args = ap.parse_args()

    checkpoint_root = resolve_checkpoint_root(args.checkpoint_root)
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_name(args.dtype)
    device_map = args.device_map.strip() or None
    module_filters = [s for s in args.module_filter.split(",") if s]
    wanted_steps = {int(s) for s in args.steps.split(",") if s.strip().isdigit()}
    eta_source = (args.eta_source or "both").strip().lower()
    if eta_source in ("checkpoint", "compute", "recompute"):
        eta_source = "compute"
    elif eta_source not in ("wandb", "both"):
        print(f"[ERROR] Unsupported eta_source: {args.eta_source}")
        return
    use_wandb = eta_source in ("wandb", "both")
    use_compute = eta_source in ("compute", "both")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] checkpoint_root={checkpoint_root}")
    print(f"[INFO] out_dir={out_dir}")

    compute_rows: List[Dict] = []
    wandb_rows: List[Dict] = []
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

    if use_wandb:
        wandb_root = Path(args.wandb_root)
        wandb_index = _index_wandb_runs(wandb_root)
        if not wandb_index:
            msg = f"No WandB runs found under {wandb_root}"
            if not use_compute:
                print(f"[ERROR] {msg}")
                return
            print(f"[WARN] {msg}; skipping WandB eta.")
        else:
            print(
                f"[INFO] WandB eta source: grad={args.wandb_metric_grad} "
                f"param={args.wandb_metric_param} root={wandb_root}"
            )
            for label, _ in run_entries:
                run_dir = _select_wandb_run(wandb_index, label)
                if run_dir is None:
                    print(f"[WARN] No WandB run matched label '{label}' under {wandb_root}")
                    continue
                rows = _collect_eta_from_wandb(
                    run_dir,
                    grad_key=args.wandb_metric_grad,
                    param_key=args.wandb_metric_param,
                )
                if wanted_steps:
                    rows = [r for r in rows if r["step"] in wanted_steps]
                if not rows:
                    print(f"[WARN] No eta metrics found in WandB log for {label}: {run_dir}")
                    continue
                has_grad = any(r.get("eta_grad") is not None for r in rows)
                if not has_grad:
                    print(f"[INFO] {label}: no grad eta logged (expected for OFT); plotting param only.")
                for row in rows:
                    eta_param = row.get("eta_param")
                    eta_grad = row.get("eta_grad")
                    wandb_rows.append({
                        "run": label,
                        "step": row["step"],
                        "eta": eta_param if eta_param is not None else eta_grad,
                        "eta_param": eta_param,
                        "eta_grad": eta_grad,
                        "source": "wandb",
                        "wandb_run": str(run_dir),
                    })
            if not wandb_rows:
                msg = "No WandB eta data collected"
                if not use_compute:
                    print(f"[ERROR] {msg}")
                    return
                print(f"[WARN] {msg}; skipping WandB plot.")

    if use_compute:
        base_model = resolve_base_model_path(
            args.base_model.strip() or None,
            run_dirs[0],
            ["base_model", "/hss/giil/caixq/model", "/home/caixq/base_model"],
        )
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
                    if model is None:
                        print(f"[WARN] Skipping {label} step {step}: adapter failed to load")
                        continue
                    eta, rows = _compute_alignment(
                        model,
                        uv_cache=uv_cache,
                        principal_rank=args.principal_rank,
                        use_lowrank=args.use_lowrank,
                        module_filters=module_filters,
                        device=device,
                    )
                    compute_rows.append({
                        "run": label,
                        "step": step,
                        "checkpoint_path": path_str,
                        "mode": mode,
                        "eta": eta,
                        "num_modules": len(rows),
                        "source": "compute",
                    })
                    print(f"[INFO] {label} step {step}: eta={eta}")
                    del model
                else:
                    if base_model_ref is None:
                        print(f"[WARN] Skipping full-model alignment for {path_str}: base model not loaded")
                        continue
                    tuned_model = _load_full_model(path_str, dtype=dtype, device=device, device_map=device_map, trust_remote_code=args.trust_remote_code)
                    eta, rows = _compute_alignment_full(
                        base_model_ref,
                        tuned_model,
                        uv_cache=uv_cache,
                        principal_rank=args.principal_rank,
                        use_lowrank=args.use_lowrank,
                        module_filters=module_filters,
                        device=device,
                    )
                    compute_rows.append({
                        "run": label,
                        "step": step,
                        "checkpoint_path": path_str,
                        "mode": mode,
                        "eta": eta,
                        "num_modules": len(rows),
                        "source": "compute",
                    })
                    print(f"[INFO] {label} step {step}: eta={eta}")
                    del tuned_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if base_model_ref is not None:
            del base_model_ref
            if device.type == "cuda":
                torch.cuda.empty_cache()

    combined_rows: List[Dict] = []
    if use_compute:
        combined_rows.extend(compute_rows)
    if use_wandb:
        combined_rows.extend(wandb_rows)
    if not combined_rows:
        print("[ERROR] No alignment data collected")
        return

    df = pd.DataFrame(combined_rows)
    df.to_csv(out_dir / "alignment_curve.csv", index=False)
    with (out_dir / "alignment_curve.jsonl").open("w", encoding="utf-8") as f:
        for row in combined_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    if use_compute and compute_rows:
        df_compute = pd.DataFrame(compute_rows)
        for run, sub in df_compute.groupby("run"):
            sub = sub.sort_values(by=["step"], na_position="last")
            fallback = pd.Series(np.arange(len(sub)), index=sub.index)
            x = sub["step"].fillna(fallback).to_numpy()
            y = sub["eta"].to_numpy()
            display_name = get_algo_display_name(run)
            label = f"{display_name} (recompute)" if use_wandb else display_name
            ax.plot(x, y, marker="o", linewidth=2, label=label)
            plotted = True
    if use_wandb and wandb_rows:
        df_wandb = pd.DataFrame(wandb_rows)
        for run, sub in df_wandb.groupby("run"):
            sub = sub.sort_values(by=["step"])
            x = sub["step"].to_numpy()
            display_name = get_algo_display_name(run)
            plotted_any = False
            if args.wandb_plot_both:
                if sub["eta_param"].notna().any():
                    ax.plot(x, sub["eta_param"].to_numpy(), marker="s", linewidth=1.5, linestyle="--", label=f"{display_name} (δW)")
                    plotted_any = True
                if sub["eta_grad"].notna().any():
                    ax.plot(x, sub["eta_grad"].to_numpy(), marker="^", linewidth=1.2, linestyle=":", label=f"{display_name} (grad)")
                    plotted_any = True
            else:
                y = sub["eta"].to_numpy()
                label = f"{display_name} (wandb)" if use_compute else display_name
                ax.plot(x, y, marker="s", linewidth=1.5, linestyle="--", label=label)
                plotted_any = True
            if plotted_any:
                plotted = True

    if not plotted:
        print("[ERROR] No data available for plotting")
        return
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Alignment Ratio (eta)")
    ax.set_title(args.title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "alignment_curve.png", dpi=300)
    fig.savefig(out_dir / "alignment_curve.pdf", dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved plot: {out_dir / 'alignment_curve.png'} and .pdf (source={eta_source})")


if __name__ == "__main__":
    main()
