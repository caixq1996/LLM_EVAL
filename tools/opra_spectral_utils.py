from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

_LAYER_RE = re.compile(r"(?:layers|layer)\.(\d+)")
_STEP_RE = re.compile(r"(?:global_step_|step_|checkpoint-)(\d+)")
_KNOWN_VARIANTS = {
    "opra",
    "vanilla",
    "adalora",
    "dora",
    "pissa",
    "qlora",
    "oft",
}


def infer_layer_idx(name: str) -> Optional[int]:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def infer_module_type(name: str) -> str:
    return name.split(".")[-1]


def iter_lora_modules(model: nn.Module):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            yield name, module


def iter_weight_modules(model: nn.Module):
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.dim() == 2:
            yield name, module


def get_base_weight(module: nn.Module) -> torch.Tensor:
    if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
        return module.base_layer.weight
    if hasattr(module, "weight"):
        return module.weight
    raise ValueError("Cannot find base weight for LoRA module")


def _get_adapter_items(module: nn.Module) -> List[Tuple[str, object, object]]:
    lora_a = getattr(module, "lora_A", None)
    lora_b = getattr(module, "lora_B", None)
    if isinstance(lora_a, (nn.ModuleDict, nn.ParameterDict)) and isinstance(lora_b, (nn.ModuleDict, nn.ParameterDict)):
        names = sorted(set(lora_a.keys()) & set(lora_b.keys()))
        return [(name, lora_a[name], lora_b[name]) for name in names]
    return [("default", lora_a, lora_b)]


def _resolve_adapter_tensor(adapter_obj: object) -> Optional[torch.Tensor]:
    if adapter_obj is None:
        return None
    if isinstance(adapter_obj, nn.ParameterDict):
        if "default" in adapter_obj:
            adapter_obj = adapter_obj["default"]
        else:
            for key in adapter_obj.keys():
                adapter_obj = adapter_obj[key]
                break
            else:
                return None
    if hasattr(adapter_obj, "weight"):
        return adapter_obj.weight
    if isinstance(adapter_obj, (torch.Tensor, nn.Parameter)):
        return adapter_obj
    return None


def _get_scaling(module: nn.Module, adapter_name: str) -> float:
    scaling = getattr(module, "scaling", None)
    if isinstance(scaling, dict):
        return float(scaling.get(adapter_name, 1.0))
    if scaling is None:
        return 1.0
    return float(scaling)


def compute_lora_delta(module: nn.Module, *, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    delta = None
    for adapter_name, a_lin, b_lin in _get_adapter_items(module):
        update = None
        if hasattr(module, "get_delta_weight"):
            try:
                update = module.get_delta_weight(adapter_name)
            except Exception:
                update = None
        if update is None:
            a_weight = _resolve_adapter_tensor(a_lin)
            b_weight = _resolve_adapter_tensor(b_lin)
            if a_weight is None or b_weight is None:
                continue
            a_weight = a_weight.detach().to(device=device, dtype=dtype)
            b_weight = b_weight.detach().to(device=device, dtype=dtype)
            update = b_weight @ a_weight
            update = update * _get_scaling(module, adapter_name)
        elif not isinstance(update, torch.Tensor):
            continue
        else:
            update = update.detach().to(device=device, dtype=dtype)
        delta = update if delta is None else (delta + update)
    return delta


def compute_weight_delta(
    base_module: nn.Module,
    tuned_module: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    base_w = getattr(base_module, "weight", None)
    tuned_w = getattr(tuned_module, "weight", None)
    if base_w is None or tuned_w is None:
        return None
    if base_w.shape != tuned_w.shape or base_w.dim() != 2:
        return None
    base_w = base_w.detach().to(device=device, dtype=dtype)
    tuned_w = tuned_w.detach().to(device=device, dtype=dtype)
    return tuned_w - base_w


def topk_svd(weight: torch.Tensor, k: int, *, device: torch.device, use_lowrank: bool, niter: int = 2) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if k <= 0:
        return None, None
    w = weight.detach().to(device=device, dtype=torch.float32)
    m, n = w.shape
    k = min(k, m, n)
    if k <= 0:
        return None, None
    if use_lowrank and min(m, n) > 1:
        q = min(k, min(m, n) - 1)
        u, _, v = torch.svd_lowrank(w, q=q, niter=niter)
        return u[:, :k], v[:, :k]
    u, _, vh = torch.linalg.svd(w, full_matrices=False)
    return u[:, :k], vh[:k, :].T


def projection_energy_ratio(delta: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> Tuple[float, float]:
    if delta is None:
        return 0.0, 0.0
    if u is None or v is None:
        return 0.0, float(delta.pow(2).sum().item())
    delta = delta.to(dtype=torch.float32)
    u = u.to(device=delta.device, dtype=delta.dtype)
    v = v.to(device=delta.device, dtype=delta.dtype)
    u_t_delta = u.transpose(0, 1) @ delta
    delta_v = delta @ v
    u_t_delta_v = u_t_delta @ v
    proj = (u @ u_t_delta) + (delta_v @ v.transpose(0, 1)) - (u @ u_t_delta_v @ v.transpose(0, 1))
    num = float(proj.pow(2).sum().item())
    den = float(delta.pow(2).sum().item())
    return (num / (den + 1e-12)), den


def parse_step_from_path(path: str | None) -> Optional[int]:
    if not path:
        return None
    match = _STEP_RE.search(str(path))
    return int(match.group(1)) if match else None


def is_adapter_dir(path) -> bool:
    path = _as_path(path)
    if not path.exists() or not path.is_dir():
        return False
    if (path / "adapter_config.json").exists():
        return True
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        if (path / name).exists():
            return True
    return False


def is_hf_model_dir(path) -> bool:
    path = _as_path(path)
    if not path.exists() or not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    if (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists():
        return True
    if list(path.glob("model-*.safetensors")):
        return True
    return False


def split_run_name(run_name: str) -> Tuple[str, Optional[str]]:
    parts = run_name.split("_")
    if len(parts) >= 2 and parts[-1].lower() in _KNOWN_VARIANTS:
        return "_".join(parts[:-1]), parts[-1].lower()
    return run_name, None


def resolve_run_dir(run_hint: str, checkpoint_root) -> Optional[str]:
    if not run_hint:
        return None
    run_path = _as_path(run_hint)
    if run_path.exists():
        return str(run_path)
    root = _as_path(checkpoint_root)
    candidate = root / run_hint
    if candidate.exists():
        return str(candidate)
    return None


def list_run_dirs(checkpoint_root: str) -> List[str]:
    root = _as_path(checkpoint_root)
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def resolve_checkpoint_root(checkpoint_root: str) -> str:
    root = _as_path(checkpoint_root)
    if root.exists():
        return str(root)
    repo_root = _as_path(__file__).resolve().parents[2]
    candidate = repo_root / "OPRA" / "checkpoints" / "OPRA-LoRA"
    if candidate.exists():
        return str(candidate)
    return str(root)


def resolve_base_model_path(base_model: Optional[str], run_dir: str, search_roots: Iterable[str]) -> str:
    if base_model:
        return base_model
    run_path = _as_path(run_dir)
    while True:
        name = run_path.name
        if name in {"lora_adapter", "huggingface", "actor"} or _STEP_RE.search(name):
            if run_path.parent == run_path:
                break
            run_path = run_path.parent
            continue
        break
    base_name, _ = split_run_name(run_path.name)
    for root in search_roots:
        cand = _as_path(root) / base_name
        if cand.exists():
            return str(cand)
    return base_name


def list_checkpoint_adapters(run_dir: str, *, only_latest: bool = False) -> List[Tuple[int, str]]:
    run_path = _as_path(run_dir)
    step_dirs = sorted([p for p in run_path.glob("global_step_*") if p.is_dir()], key=lambda p: parse_step_from_path(str(p)) or -1)
    if only_latest and step_dirs:
        step_dirs = [step_dirs[-1]]
    entries = []
    for step_dir in step_dirs:
        adapter_dir = step_dir / "actor" / "lora_adapter"
        if is_adapter_dir(adapter_dir):
            step = parse_step_from_path(str(step_dir))
            entries.append((step or -1, str(adapter_dir)))
    return entries


def list_checkpoint_full_models(run_dir: str, *, only_latest: bool = False) -> List[Tuple[int, str]]:
    run_path = _as_path(run_dir)
    step_dirs = sorted([p for p in run_path.glob("global_step_*") if p.is_dir()], key=lambda p: parse_step_from_path(str(p)) or -1)
    if only_latest and step_dirs:
        step_dirs = [step_dirs[-1]]
    entries = []
    for step_dir in step_dirs:
        model_dir = step_dir / "actor" / "huggingface"
        if is_hf_model_dir(model_dir):
            step = parse_step_from_path(str(step_dir))
            entries.append((step or -1, str(model_dir)))
    return entries


def collect_run_entries(
    run_hint: str,
    checkpoint_root: str,
    *,
    step_filter: Optional[Iterable[int]] = None,
) -> Tuple[Optional[str], List[Tuple[int, str, str]]]:
    resolved = resolve_run_dir(run_hint, checkpoint_root)
    if resolved is None:
        return None, []
    path = _as_path(resolved)
    entries: List[Tuple[int, str, str]] = []

    if is_adapter_dir(path):
        entries.append((parse_step_from_path(str(path)) or -1, str(path), "adapter"))
    elif is_hf_model_dir(path):
        adapter_dir = path.parent / "lora_adapter"
        if is_adapter_dir(adapter_dir):
            entries.append((parse_step_from_path(str(adapter_dir)) or -1, str(adapter_dir), "adapter"))
        else:
            entries.append((parse_step_from_path(str(path)) or -1, str(path), "full"))
    elif path.name.startswith("global_step_"):
        step = parse_step_from_path(str(path)) or -1
        adapter_dir = path / "actor" / "lora_adapter"
        if is_adapter_dir(adapter_dir):
            entries.append((step, str(adapter_dir), "adapter"))
        else:
            model_dir = path / "actor" / "huggingface"
            if is_hf_model_dir(model_dir):
                entries.append((step, str(model_dir), "full"))
    elif path.name == "actor":
        step = parse_step_from_path(str(path)) or -1
        adapter_dir = path / "lora_adapter"
        if is_adapter_dir(adapter_dir):
            entries.append((step, str(adapter_dir), "adapter"))
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

    if step_filter:
        step_filter_set = set(step_filter)
        entries = [e for e in entries if e[0] in step_filter_set]

    return str(path), entries


def _as_path(path_like) -> "Path":
    from pathlib import Path
    return path_like if isinstance(path_like, Path) else Path(str(path_like))
