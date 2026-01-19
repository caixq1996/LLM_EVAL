from __future__ import annotations

import re
import os
import json
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
    "qpissa",
    "qlora",
    "olora",
    "rslora",
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
        elif hasattr(module, "oft_R"):
            yield name, module


def iter_weight_modules(model: nn.Module):
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.dim() == 2:
            yield name, module


def _adapter_max_rank(adapter_path: "Path") -> Optional[int]:
    from pathlib import Path
    adapter_path = adapter_path if isinstance(adapter_path, Path) else Path(str(adapter_path))
    candidates = [
        adapter_path / "adapter_model.safetensors",
        adapter_path / "lora_adapter" / "adapter_model.safetensors",
    ]
    safetensors_path = next((p for p in candidates if p.exists()), None)
    if safetensors_path is None:
        return None
    try:
        from safetensors import safe_open
    except Exception:
        return None
    max_rank = 0
    try:
        with safe_open(str(safetensors_path), framework="pt") as f:
            for key in f.keys():
                if ".lora_A." in key or key.endswith(".lora_A") or ".lora_E." in key or key.endswith(".lora_E"):
                    shape = f.get_tensor(key).shape
                    if shape:
                        max_rank = max(max_rank, int(shape[0]))
    except Exception as exc:
        print(f"[WARN] Failed to inspect {safetensors_path}: {exc}")
        return None
    return max_rank if max_rank > 0 else None


def _adapter_rank_range(adapter_path: "Path") -> Tuple[Optional[int], Optional[int]]:
    try:
        from safetensors import safe_open
    except Exception:
        return (None, None)
    safetensors_path = adapter_path / "adapter_model.safetensors"
    if not safetensors_path.exists():
        return (None, None)
    min_rank = None
    max_rank = 0
    try:
        with safe_open(str(safetensors_path), framework="pt") as f:
            for key in f.keys():
                if ".lora_A." in key or key.endswith(".lora_A") or ".lora_E." in key or key.endswith(".lora_E"):
                    shape = f.get_tensor(key).shape
                    if shape:
                        rank = int(shape[0])
                        max_rank = max(max_rank, rank)
                        min_rank = rank if min_rank is None else min(min_rank, rank)
    except Exception as exc:
        print(f"[WARN] Failed to inspect {safetensors_path}: {exc}")
        return (None, None)
    if max_rank <= 0:
        return (None, None)
    return (min_rank, max_rank)


def _load_raw_adapter_config(adapter_path: "Path") -> Optional[dict]:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Failed to read {config_path}: {exc}")
        return None


def load_peft_config(adapter_path) -> Optional["PeftConfig"]:
    try:
        from peft import PeftConfig
    except Exception:
        return None
    raw_cfg = _load_raw_adapter_config(adapter_path)
    raw_rank_pattern_zero = False
    raw_init_r = None
    raw_r = None
    if isinstance(raw_cfg, dict):
        raw_init_r = raw_cfg.get("init_r")
        raw_r = raw_cfg.get("r")
        raw_rank_pattern = raw_cfg.get("rank_pattern")
        if isinstance(raw_rank_pattern, dict) and raw_rank_pattern:
            try:
                raw_rank_pattern_zero = all(sum(v) == 0 for v in raw_rank_pattern.values())
            except Exception:
                raw_rank_pattern_zero = False
    try:
        config = PeftConfig.from_pretrained(str(adapter_path))
    except Exception as exc:
        print(f"[WARN] Failed to load adapter config from {adapter_path}: {exc}")
        return None
    peft_type = getattr(config, "peft_type", "")
    peft_type_name = None
    if hasattr(peft_type, "value"):
        peft_type_name = peft_type.value
    elif hasattr(peft_type, "name"):
        peft_type_name = peft_type.name
    else:
        peft_type_name = str(peft_type)
    if str(peft_type_name).upper() == "ADALORA":
        init_r = getattr(config, "init_r", None)
        r = getattr(config, "r", None)
        if init_r in (None, 0) and r not in (None, 0):
            config.init_r = r
            init_r = r
        if r in (None, 0) and init_r not in (None, 0):
            config.r = init_r
        rank_pattern = getattr(config, "rank_pattern", None)
        if raw_rank_pattern_zero:
            config.rank_pattern = None
            setattr(config, "_ignore_mismatched_sizes", True)
            print("[INFO] AdaLora: raw rank_pattern all-zero; drop rank_pattern for load")
        elif rank_pattern:
            rank_sums = []
            for v in rank_pattern.values():
                try:
                    rank_sums.append(sum(v) if isinstance(v, (list, tuple)) else int(v))
                except Exception:
                    continue
            rank_pattern_max = max(rank_sums) if rank_sums else None
            rank_pattern_min = min(rank_sums) if rank_sums else None
            if rank_pattern_max == 0:
                config.rank_pattern = None
                setattr(config, "_ignore_mismatched_sizes", True)
                print("[INFO] AdaLora: rank_pattern all-zero; ignore rank_pattern for load")
            else:
                adapter_min_rank, adapter_max_rank = _adapter_rank_range(adapter_path)
                if rank_pattern_min == 0 and adapter_max_rank:
                    config.rank_pattern = None
                    setattr(config, "_ignore_mismatched_sizes", True)
                    print("[INFO] AdaLora: rank_pattern has zero entries; ignore for load")
                elif adapter_max_rank is not None and rank_pattern_max is not None and adapter_max_rank > rank_pattern_max:
                    config.rank_pattern = None
                    setattr(config, "_ignore_mismatched_sizes", True)
                    print(f"[INFO] AdaLora: ignoring rank_pattern for load (rank_pattern_max={rank_pattern_max}, adapter_max={adapter_max_rank})")
                elif adapter_min_rank is not None and adapter_max_rank is not None:
                    if adapter_min_rank == adapter_max_rank and (
                        rank_pattern_min != adapter_min_rank or rank_pattern_max != adapter_max_rank
                    ):
                        config.rank_pattern = None
                        setattr(config, "_ignore_mismatched_sizes", True)
                        print(
                            "[INFO] AdaLora: adapter weights use full rank; drop rank_pattern for load "
                            f"(adapter_rank={adapter_max_rank}, rank_pattern_range={rank_pattern_min}-{rank_pattern_max})"
                        )
        if raw_rank_pattern_zero:
            final_r = raw_init_r if isinstance(raw_init_r, int) and raw_init_r > 0 else raw_r
            if isinstance(final_r, int) and final_r > 0:
                config.init_r = final_r
                config.r = final_r

    # Handle OFT adapter weight format mismatch between PEFT versions
    if str(peft_type_name).upper() == "OFT":
        setattr(config, "_ignore_mismatched_sizes", True)
        print("[INFO] OFT: setting ignore_mismatched_sizes=True due to potential weight format differences")
    return config


def get_base_weight(module: nn.Module) -> torch.Tensor:
    if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
        return module.base_layer.weight
    if hasattr(module, "weight"):
        return module.weight
    raise ValueError("Cannot find base weight for LoRA module")


def _get_adapter_items(module: nn.Module) -> List[Tuple[str, object, object]]:
    lora_a = getattr(module, "lora_A", None)
    lora_b = getattr(module, "lora_B", None)
    oft_r = getattr(module, "oft_R", None)

    if isinstance(lora_a, (nn.ModuleDict, nn.ParameterDict)) and isinstance(lora_b, (nn.ModuleDict, nn.ParameterDict)):
        names = sorted(set(lora_a.keys()) & set(lora_b.keys()))
        return [(name, lora_a[name], lora_b[name]) for name in names]
    
    # Handle OFT adapters
    if isinstance(oft_r, (nn.ModuleDict, nn.ParameterDict)):
        names = sorted(oft_r.keys())
        # Return oft_R[name] as first item, None as second
        # compute_lora_delta checks get_delta_weight first, checking a_lin/b_lin only as fallback
        return [(name, oft_r[name], None) for name in names]
    
    if oft_r is not None:
        return [("default", oft_r, None)]

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
    # Get AdaLoRA-specific attributes (lora_E for SVD-based importance, ranknum for rank normalization)
    lora_e = getattr(module, "lora_E", None)
    ranknum = getattr(module, "ranknum", None)
    
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
            
            # Handle AdaLoRA: apply lora_E scaling to A matrix
            if lora_e is not None:
                e_tensor = None
                if isinstance(lora_e, (nn.ModuleDict, nn.ParameterDict)):
                    if adapter_name in lora_e:
                        e_tensor = _resolve_adapter_tensor(lora_e[adapter_name])
                elif adapter_name == "default":
                    e_tensor = _resolve_adapter_tensor(lora_e)
                if e_tensor is not None:
                    e = e_tensor.detach().to(device=device, dtype=dtype)
                    if e.dim() == 1:
                        a_weight = a_weight * e.unsqueeze(-1)
                    else:
                        a_weight = a_weight * e
            
            # Handle AdaLoRA: apply ranknum normalization to A matrix
            if ranknum is not None:
                r_tensor = None
                if isinstance(ranknum, (nn.ModuleDict, nn.ParameterDict)):
                    if adapter_name in ranknum:
                        r_tensor = _resolve_adapter_tensor(ranknum[adapter_name])
                elif adapter_name == "default":
                    r_tensor = _resolve_adapter_tensor(ranknum)
                if r_tensor is not None:
                    r_val = r_tensor.detach().to(device=device, dtype=dtype)
                    a_weight = a_weight / (r_val + 1e-5)
            
            update = b_weight @ a_weight
            update = update * _get_scaling(module, adapter_name)
        elif not isinstance(update, torch.Tensor):
            continue
        else:
            update = update.detach().to(device=device, dtype=dtype)
            # Fix for OFT: peft may return the rotation matrix (hidden_size x hidden_size)
            # instead of the projected delta weight (out_features x in_features)
            # If so, we need to apply it to the base weight: delta_W = W @ (R - I) or similar
            # Since get_delta_weight returns the delta, assume it returns W @ (R - I) if shapes match.
            # If mismatch, assume it returned (R - I) and we need W @ update.
            base_w = get_base_weight(module)
            if base_w is not None:
                base_w_shape = base_w.shape
                if update.shape != base_w_shape:
                    # Check if base_W @ update yields correct shape
                    # Case: K_proj (256, 1536) vs Update (1536, 1536) -> (256, 1536)
                    if update.shape[0] == base_w_shape[1] and update.shape[1] == base_w_shape[1]: 
                         # Likely R applied to input
                         base_w_dev = base_w.detach().to(device=device, dtype=dtype)
                         update = base_w_dev @ update
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
    run_name = run_path.name
    adapter_suffix = get_adapter_base_model_suffix(run_name)
    for root in search_roots:
        base_root = _as_path(root)
        if not base_root.exists():
            continue
        found = find_base_model_dir(base_root, run_name, adapter_suffix=adapter_suffix)
        if found is not None:
            return str(found)
    base_name, _ = split_run_name(run_name)
    for root in search_roots:
        cand = _as_path(root) / base_name
        if cand.exists():
            return str(cand)
    return base_name


def get_adapter_base_model_suffix(run_name: str) -> Optional[str]:
    special_algos = os.environ.get("SPECIAL_ADAPTER_ALGORITHMS", "pissa:_pissa_base,qpissa:_qpissa_base")
    if not special_algos:
        return None
    algo_map = {}
    for pair in special_algos.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        algo, suffix = pair.split(":", 1)
        algo = algo.strip().lower()
        suffix = suffix.strip()
        if algo and suffix:
            algo_map[algo] = suffix
    if not algo_map:
        return None
    run_key = f"_{run_name.lower()}_"
    for algo, suffix in sorted(algo_map.items(), key=lambda x: len(x[0]), reverse=True):
        if f"_{algo}_" in run_key:
            return suffix
    return None


def find_base_model_dir(base_root, run_name: str, adapter_suffix: Optional[str] = None):
    base_root = _as_path(base_root)
    if not base_root.exists():
        return None
    run_key = _norm(run_name)
    best = None
    if adapter_suffix:
        for d in base_root.iterdir():
            if not d.is_dir() or not d.name.endswith(adapter_suffix):
                continue
            base_name = d.name[: -len(adapter_suffix)]
            key = _norm(base_name)
            if key and (key in run_key or run_key in key):
                if best is None or len(key) > len(_norm(best.name.replace(adapter_suffix, ""))):
                    best = d
        if best is not None:
            return best
    for d in base_root.iterdir():
        if not d.is_dir():
            continue
        key = _norm(d.name)
        if not key:
            continue
        if key in run_key or run_key in key:
            if best is None or len(key) > len(_norm(best.name)):
                best = d
    return best


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
            continue
        actor_dir = step_dir / "actor"
        if is_adapter_dir(actor_dir):
            step = parse_step_from_path(str(step_dir))
            entries.append((step or -1, str(actor_dir)))
            continue
        if is_adapter_dir(step_dir):
            step = parse_step_from_path(str(step_dir))
            entries.append((step or -1, str(step_dir)))
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

    if step_filter:
        step_filter_set = set(step_filter)
        entries = [e for e in entries if e[0] in step_filter_set]

    return str(path), entries


def _as_path(path_like) -> "Path":
    from pathlib import Path
    return path_like if isinstance(path_like, Path) else Path(str(path_like))


def _norm(s: str) -> str:
    return re.sub("[^a-z0-9]+", "", s.lower())
