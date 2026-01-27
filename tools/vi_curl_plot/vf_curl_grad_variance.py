#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VF-CuRL small-dim gradient variance analysis (checkpoint sweep).

Goal (approximate):
For a fixed prompt subset {x_i} and multiple training checkpoints, estimate:
  - action variance proxy:   Var_{y|x}( g(x,y;θ) )
  - problem mean gradient:   ḡ(x) = E_{y|x}[ g(x,y;θ) ]
Then, for curriculum phase/step t (keep mask w_t(x)=1):
  - σ_{g,t}²  ≈ E_{x|w_t=1} Var_{y|x}(g)
  - V_prob,t  ≈ Var_{x|w_t=1}( ḡ(x) )

This script implements a cheap proxy for g(x,y;θ) using the *logit-gradient* of
the token-level NLL (cross-entropy) and then projects it to a small dimension.
Concretely, at each generated token, we approximate dL/dlogits = softmax - one_hot
with a sparse top-k approximation and CountSketch into `proj_dim`.

Notes:
  - Default uses vLLM (`--use_vllm`, default: True). Since checkpoints are FSDP shards,
    each step is exported to a temporary HF directory under `--hf_export_root` and
    deleted after use unless `--keep_hf_export` is set.
  - Transformers backend (`--no_use_vllm`) uses a custom token loop and computes
    CountSketch + confidence on-the-fly, without storing per-token full-vocab logits.
    Memory is dominated by KV-cache (and scales with batch/sequence length), so
    `--gen_batch_size` can usually be set higher than 1 (still OOM-splittable).
  - If `--baseline_run_dir` is provided, ratios use the baseline run at the same step.
    Otherwise, ratios use the same run with "full data" (all prompts) as baseline.

Example:
  python project/LLM_EVAL/tools/vi_curl_plot/vf_curl_grad_variance.py \\
    --run_dir /data/giil/caixq/ckpts/VI-CURL_deepscaler/ver_rule_grpo_curl_Qwen2.5-math-1.5B \\
    --baseline_run_dir /data/giil/caixq/ckpts/VI-CURL_deepscaler/ver_rule_grpo_nocurl_Qwen2.5-math-1.5B \\
    --prompt_file /path/to/prompts.jsonl --prompt_field prompt --num_prompts 512 --subset_mode random --seed 42 \\
    --k_rollouts 4 --max_new_tokens 64 --temperature 1.0 --top_p 1.0 \\
    --proj_dim 256 --topk 32 --confidence_metric neg_entropy \\
    --start_beta 0.2 --end_beta 1.0 --schedule_steps 8000 \\
    --batch_var_seeds 0,1,2,3 --batch_var_num_prompts 128 --batch_var_k 1 \\
    --out_dir project/LLM_EVAL/eval_log/vi_curl/grad_variance
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import concurrent.futures
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

try:
    from accelerate import init_empty_weights
except Exception:  # pragma: no cover
    init_empty_weights = None

try:
    from torch.distributed.tensor import DTensor
    from torch.distributed._tensor import Shard
except Exception:  # pragma: no cover
    from torch.distributed._tensor import DTensor, Shard

try:
    from vllm.distributed.parallel_state import destroy_model_parallel
except Exception:  # pragma: no cover
    try:
        from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
    except Exception:  # pragma: no cover
        destroy_model_parallel = None

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover
    LLM = None
    SamplingParams = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Import shared plot configuration
try:
    from plot_config import setup_plot_style, add_font_size_args, get_font_sizes, get_default_font_sizes
    HAS_PLOT_CONFIG = True
except ImportError:
    try:
        from tools.vi_curl_plot.plot_config import setup_plot_style, add_font_size_args, get_font_sizes, get_default_font_sizes  # type: ignore
        HAS_PLOT_CONFIG = True
    except ImportError:
        HAS_PLOT_CONFIG = False

        def setup_plot_style() -> None:
            pass

        def add_font_size_args(parser: argparse.ArgumentParser) -> None:
            return None

        def get_font_sizes(args: argparse.Namespace, script_name: str | None = None) -> Dict[str, Any]:
            return {
                "xlabel": 14,
                "ylabel": 14,
                "legend": 12,
                "tick": 12,
                "xtick": 12,
                "ytick": 12,
                "colorbar": 12,
            }

        def get_default_font_sizes(script_name: str | None = None) -> Dict[str, int]:
            return {
                "xlabel": 14,
                "ylabel": 14,
                "legend": 12,
                "tick": 12,
                "xtick": 12,
                "ytick": 12,
                "colorbar": 12,
            }


FONT_SIZES: Dict[str, Any] = {}



_STEP_RE = re.compile(r"global_step_(\d+)")
_NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True)
class AnalysisConfig:
    run_dir: str
    baseline_run_dir: Optional[str]
    prompt_file: str
    prompt_field: str
    prompt_type: str
    num_prompts: int
    num_prompt_ratio: float
    subset_mode: str
    seed: int
    max_prompt_len: int
    tokenizer_dir: Optional[str]
    k_rollouts: int
    max_new_tokens: int
    temperature: float
    top_p: float
    sampling_mode: str
    prob_mode: str
    sanitize_logits: bool
    proj_dim: int
    topk: int
    confidence_metric: str
    start_beta: float
    end_beta: float
    schedule_steps: int
    beta_source: str
    beta_log_path: Optional[str]
    beta_log_root: Optional[str]
    use_log_tau: bool
    gen_batch_size: int
    steps: Optional[str]
    batch_var_seeds: Optional[str]
    batch_var_num_prompts: int
    batch_var_k: int
    compute_passk: bool
    data_name: str
    answer_field: str
    passk_ks: str


def _sha1_json(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _tag_with_beta_mode(tag: str, *, beta_mode: str) -> str:
    """Make tag mode-specific to avoid overwriting outputs across beta schedules."""
    beta_mode = str(beta_mode)
    if beta_mode == "train":
        return tag
    suffix = f"__{beta_mode}"
    if tag.endswith(suffix):
        return tag
    return f"{tag}{suffix}"


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _step_num_from_dir(p: Path) -> int:
    m = _STEP_RE.search(p.name)
    return int(m.group(1)) if m else -1


def list_step_dirs(run_dir: Path) -> List[Path]:
    step_dirs = [p for p in run_dir.glob("global_step_*") if p.is_dir()]
    step_dirs = [p for p in step_dirs if _step_num_from_dir(p) >= 0]
    return sorted(step_dirs, key=_step_num_from_dir)


def parse_steps_arg(steps: Optional[str], available_steps: Sequence[int]) -> List[int]:
    if not steps:
        return list(available_steps)
    s = steps.strip()
    if not s:
        return list(available_steps)
    # formats:
    #   "20,40,60"
    #   "20-300" (inclusive)
    #   "20-300:20" (inclusive, stride)
    if "," in s:
        wanted = []
        for part in s.split(","):
            part = part.strip()
            if part.isdigit():
                wanted.append(int(part))
        return [x for x in available_steps if x in set(wanted)]
    if "-" in s:
        if ":" in s:
            r, stride_s = s.split(":", 1)
            stride = int(stride_s)
        else:
            r, stride = s, 1
        lo_s, hi_s = r.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        wanted = set(range(lo, hi + 1, stride))
        return [x for x in available_steps if x in wanted]
    if s.isdigit():
        v = int(s)
        return [x for x in available_steps if x == v]
    raise ValueError(f"Unrecognized --steps format: {steps}")


def compute_beta_target(step: int, start_beta: float, end_beta: float, schedule_steps: int) -> float:
    progress = min(float(step) / max(int(schedule_steps), 1), 1.0)
    return float(start_beta + (end_beta - start_beta) * progress)

def compute_beta_target_rescaled(step: int, start_beta: float, end_beta: float, ref_steps: Sequence[int]) -> float:
    """Rescale beta across a finite reference step set.

    Maps the earliest ref step to start_beta and the latest ref step to end_beta.
    Intermediate steps are spaced uniformly by checkpoint index (not by numeric step size).
    """
    if not ref_steps:
        return float(start_beta)
    steps_sorted = sorted({int(s) for s in ref_steps})
    if not steps_sorted:
        return float(start_beta)
    if len(steps_sorted) == 1:
        return float(end_beta)
    try:
        idx = steps_sorted.index(int(step))
        progress = float(idx) / float(len(steps_sorted) - 1)
    except ValueError:
        lo, hi = int(steps_sorted[0]), int(steps_sorted[-1])
        if hi <= lo:
            progress = 1.0
        else:
            progress = (float(step) - float(lo)) / (float(hi) - float(lo))
            progress = float(max(0.0, min(1.0, progress)))
    return float(start_beta + (end_beta - start_beta) * progress)


def _find_latest_wandb_output_log(run_dir: Path) -> Path:
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


def _find_wandb_run_dir_by_id(wandb_root: Path, run_id: str) -> Optional[Path]:
    if not run_id:
        return None
    run_id = run_id.strip()
    if not run_id:
        return None
    for p in wandb_root.iterdir():
        if not p.is_dir():
            continue
        if p.name.endswith(f"-{run_id}") and (p.name.startswith("run-") or p.name.startswith("offline-run-")):
            return p
    return None


def _find_wandb_output_log_by_name(wandb_root: Path, run_name: str, *, max_lines: int = 2000) -> Optional[Path]:
    if not run_name:
        return None
    candidates = [p for p in wandb_root.iterdir() if p.is_dir() and (p.name.startswith("run-") or p.name.startswith("offline-run-"))]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in candidates:
        log_path = run_dir / "files" / "output.log"
        if not log_path.exists():
            continue
        try:
            with log_path.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if run_name in line:
                        return log_path
                    if i >= max_lines:
                        break
        except Exception:
            continue
    return None


def _resolve_beta_log_path(
    *,
    run_dir: Path,
    beta_log_path: Optional[Path],
    beta_log_root: Optional[Path],
) -> Path:
    if beta_log_path is not None:
        return beta_log_path

    if beta_log_root is not None:
        wandb_root = beta_log_root.expanduser().resolve()
        if wandb_root.exists():
            run_id_path = run_dir / "wandb_run_id.txt"
            run_id = ""
            if run_id_path.exists():
                try:
                    run_id = run_id_path.read_text(encoding="utf-8").strip()
                except Exception:
                    run_id = ""
            run_dir_by_id = _find_wandb_run_dir_by_id(wandb_root, run_id)
            if run_dir_by_id is not None:
                log_path = run_dir_by_id / "files" / "output.log"
                if log_path.exists():
                    return log_path

            log_path = _find_wandb_output_log_by_name(wandb_root, run_dir.name)
            if log_path is not None and log_path.exists():
                return log_path

    return _find_latest_wandb_output_log(run_dir)


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


def _load_beta_from_log(log_path: Path) -> Dict[int, Dict[str, float]]:
    beta_by_step: Dict[int, Dict[str, float]] = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "vf_curl/beta_target" not in line and "vf_curl/beta_actual" not in line and "vf_curl/beta_actual_x" not in line:
                continue
            data = _parse_metrics_kv_line(line)
            if not data:
                continue
            step = (
                data.get("training/global_step")
                or data.get("global_step")
                or data.get("step")
                or data.get("training/step")
            )
            if step is None:
                continue
            try:
                step_i = int(step)
            except Exception:
                continue
            if step_i <= 0:
                continue
            slot = beta_by_step.setdefault(step_i, {})
            if "vf_curl/beta_target" in data:
                slot["beta_target"] = float(data["vf_curl/beta_target"])
            if "vf_curl/beta_actual" in data:
                slot["beta_actual"] = float(data["vf_curl/beta_actual"])
            if "vf_curl/beta_actual_x" in data and "beta_actual" not in slot:
                slot["beta_actual"] = float(data["vf_curl/beta_actual_x"])
            if "vf_curl/tau_x" in data:
                slot["tau"] = float(data["vf_curl/tau_x"])
    return beta_by_step


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_jsonl(path: Path) -> List[Any]:
    items: List[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_prompts(prompt_file: Path, prompt_field: str) -> List[str]:
    if not prompt_file.exists():
        raise FileNotFoundError(f"prompt_file not found: {prompt_file}")

    ext = prompt_file.suffix.lower()
    if ext == ".txt":
        prompts = [ln.strip() for ln in prompt_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return prompts
    if ext == ".jsonl":
        items = _load_jsonl(prompt_file)
    elif ext == ".json":
        items = _read_json(prompt_file)
    elif ext == ".parquet":
        try:
            import pandas as pd
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Reading parquet requires pandas (and pyarrow).") from e
        df = pd.read_parquet(prompt_file)
        cols = list(df.columns)
        field = prompt_field if prompt_field in cols else None
        if field is None:
            # common fallbacks
            for c in ("prompt", "question", "input", "instruction", "text"):
                if c in cols:
                    field = c
                    break
        if field is None:
            raise KeyError(f"Cannot find prompt field in parquet. Have columns={cols}")
        return [str(x) for x in df[field].tolist()]
    else:
        raise ValueError(f"Unsupported prompt_file extension: {prompt_file}")

    prompts: List[str] = []
    fallbacks = [prompt_field, "prompt", "question", "input", "instruction", "text", "problem"]
    if isinstance(items, dict):
        # allow {"data": [...]} style
        if "data" in items and isinstance(items["data"], list):
            items = items["data"]
        else:
            raise ValueError(f"Unsupported JSON object root (expected list): keys={list(items.keys())[:10]}")

    if not isinstance(items, list):
        raise ValueError("prompt_file must contain a list (or jsonl lines).")

    for it in items:
        if isinstance(it, str):
            prompts.append(it)
            continue
        if isinstance(it, dict):
            val = None
            for k in fallbacks:
                if k in it and it[k] is not None:
                    val = it[k]
                    break
            if val is None:
                continue
            prompts.append(str(val))
            continue
    if not prompts:
        raise ValueError(f"No prompts extracted from {prompt_file}")
    return prompts


def load_examples(prompt_file: Path) -> List[Dict[str, Any]]:
    if not prompt_file.exists():
        raise FileNotFoundError(f"prompt_file not found: {prompt_file}")
    ext = prompt_file.suffix.lower()
    if ext == ".jsonl":
        items = _load_jsonl(prompt_file)
    elif ext == ".json":
        items = _read_json(prompt_file)
    else:
        raise ValueError(f"Unsupported prompt_file extension for examples: {prompt_file}")

    if isinstance(items, dict):
        if "data" in items and isinstance(items["data"], list):
            items = items["data"]
        else:
            raise ValueError(f"Unsupported JSON object root (expected list): keys={list(items.keys())[:10]}")
    if not isinstance(items, list):
        raise ValueError("prompt_file must contain a list (or jsonl lines).")
    return [it for it in items if isinstance(it, dict)]


def build_prompts_and_answers(
    examples: Sequence[Dict[str, Any]],
    *,
    data_name: str,
    prompt_type: str,
    prompt_field: str,
    answer_field: str,
) -> Tuple[List[str], List[str]]:
    eval_root = Path(__file__).resolve().parents[2]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from parser import parse_ground_truth, parse_question  # type: ignore
    from utils import construct_prompt  # type: ignore

    args = SimpleNamespace()
    args.prompt_type = str(prompt_type)
    args.num_shots = 0
    args.adapt_few_shot = False

    prompts: List[str] = []
    answers: List[str] = []
    for ex in examples:
        if data_name != "custom":
            question = parse_question(ex, data_name)
            _, gt = parse_ground_truth(ex, data_name)
        else:
            question = str(ex.get(prompt_field, ""))
            gt = str(ex.get(answer_field, ""))
        sample = {"question": question, "gt_ans": gt}
        prompt = construct_prompt(sample, data_name if data_name != "custom" else "math500", args)
        prompts.append(prompt)
        answers.append(str(gt))
    return prompts, answers


def _safe_format(tpl: str, **kwargs) -> str:
    s = tpl.replace("{", "{{").replace("}", "}}")
    for k in kwargs.keys():
        s = s.replace("{{" + k + "}}", "{" + k + "}")
    return s.format(**kwargs)


def apply_prompt_template(raw_prompts: Sequence[str], prompt_type: str) -> List[str]:
    """
    Apply LLM_EVAL prompt template (same naming as run_qwen_eval_all.sge).

    This is a minimal "zero-shot" variant: only wraps each prompt with the
    prompt_type input template (no demos/few-shot).
    """
    if not prompt_type or prompt_type.lower() in ("none", "raw"):
        return list(raw_prompts)

    # Ensure we can import LLM_EVAL/utils.py when running from anywhere
    eval_root = Path(__file__).resolve().parents[2]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))

    try:
        from utils import PROMPT_TEMPLATES as _PROMPT_TEMPLATES  # type: ignore
        from utils import _safe_format as _eval_safe_format  # type: ignore
        safe_format = _eval_safe_format
    except Exception:
        _PROMPT_TEMPLATES = None
        safe_format = _safe_format

    if _PROMPT_TEMPLATES is None or prompt_type not in _PROMPT_TEMPLATES:
        if _PROMPT_TEMPLATES is None:
            raise RuntimeError("Failed to import LLM_EVAL.utils.PROMPT_TEMPLATES; cannot use --prompt_type.")
        raise ValueError(f"Unknown prompt_type={prompt_type}. Available: {sorted(_PROMPT_TEMPLATES.keys())[:30]} ...")

    input_template = _PROMPT_TEMPLATES[prompt_type][0]
    return [safe_format(input_template, input=p) for p in raw_prompts]


def select_prompt_subset_indices(total: int, num_prompts: int, seed: int, subset_mode: str) -> List[int]:
    if num_prompts <= 0 or num_prompts >= total:
        return list(range(total))
    if subset_mode == "first":
        return list(range(num_prompts))
    if subset_mode == "random":
        rng = random.Random(seed)
        idxs = list(range(total))
        rng.shuffle(idxs)
        idxs = idxs[:num_prompts]
        return sorted(idxs)
    raise ValueError(f"Unknown subset_mode: {subset_mode} (use 'random' or 'first')")


def select_prompt_subset(prompts: Sequence[str], num_prompts: int, seed: int, subset_mode: str) -> List[str]:
    prompts = list(prompts)
    idxs = select_prompt_subset_indices(len(prompts), num_prompts, seed, subset_mode)
    return [prompts[i] for i in idxs]


def _resolve_tokenizer_dir(
    *,
    run_dir: Path,
    step_dirs: Sequence[Path],
    tokenizer_dir: Optional[Path],
    base_model_dir: Optional[Path],
) -> Optional[Path]:
    candidates: List[Path] = []
    if tokenizer_dir is not None:
        candidates.append(tokenizer_dir)
    if base_model_dir is not None:
        candidates.append(base_model_dir)
    candidates.append(run_dir / "actor" / "huggingface")
    for step_dir in step_dirs:
        candidates.append(step_dir / "actor" / "huggingface")
    for cand in candidates:
        if _has_hf_weights(cand):
            return cand
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _load_tokenizer(tokenizer_dir: Path, *, trust_remote_code: bool) -> Any:
    return AutoTokenizer.from_pretrained(
        str(tokenizer_dir), trust_remote_code=trust_remote_code, use_fast=True, padding_side="left"
    )


def _filter_prompts_by_length(
    prompts: Sequence[str],
    *,
    max_prompt_len: int,
    tokenizer_dir: Path,
    trust_remote_code: bool,
    answers: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[int], Optional[List[str]]]:
    tokenizer = _load_tokenizer(tokenizer_dir, trust_remote_code=trust_remote_code)
    lengths: List[int] = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        lengths.append(int(len(ids)))
    keep_mask = [int(l) <= int(max_prompt_len) for l in lengths]
    kept_prompts = [p for p, keep in zip(prompts, keep_mask) if keep]
    kept_lengths = [l for l, keep in zip(lengths, keep_mask) if keep]
    kept_answers: Optional[List[str]] = None
    if answers is not None:
        kept_answers = [a for a, keep in zip(answers, keep_mask) if keep]
    return kept_prompts, kept_lengths, kept_answers


def _rank_id(p: Path) -> int:
    m = re.search(r"rank_(\d+)", p.name)
    return int(m.group(1)) if m else -1


def _world_size_from_name(p: Path) -> int:
    m = re.search(r"world_size_(\d+)", p.name)
    return int(m.group(1)) if m else -1


def _load_one_shard(path: Path) -> Dict[str, Any]:
    sd = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if not isinstance(sd, dict):
        raise RuntimeError(f"Unexpected shard format: {path} type={type(sd)}")
    return sd


def _assemble_param(key: str, shards: List[Any]) -> torch.Tensor:
    v0 = shards[0]
    if isinstance(v0, DTensor):
        placements = tuple(getattr(v0, "placements", ()))
        shard_dim = None
        for pl in placements:
            if isinstance(pl, Shard):
                shard_dim = int(getattr(pl, "dim", 0))
                break
        if shard_dim is None:
            return v0.to_local().contiguous()
        locals_: List[torch.Tensor] = []
        for v in shards:
            if isinstance(v, DTensor):
                locals_.append(v.to_local())
            else:
                locals_.append(v)
        try:
            return torch.cat(locals_, dim=shard_dim).contiguous()
        except Exception as e:
            raise RuntimeError(
                f"Concatenate DTensor shards failed on key={key}, dim={shard_dim}, shapes={[tuple(t.shape) for t in locals_]}"
            ) from e

    # non-DTensor case: either replicated (same shape), or sharded along dim=0 (common legacy)
    if all(isinstance(v, torch.Tensor) for v in shards):
        shapes = [tuple(v.shape) for v in shards]
        if len(set(shapes)) == 1:
            return shards[0].contiguous()
        for dim in (0, 1):
            try:
                return torch.cat(shards, dim=dim).contiguous()
            except Exception:
                continue
    # fallback: keep rank0
    if isinstance(v0, torch.Tensor):
        return v0.contiguous()
    raise RuntimeError(f"Unsupported param type for key={key}: {type(v0)}")


def load_fsdp_full_state_dict(actor_dir: Path, *, shard_load_workers: int = 1) -> Tuple[Dict[str, torch.Tensor], int]:
    fsdp_cfg_path = actor_dir / "fsdp_config.json"
    world_size = None
    if fsdp_cfg_path.exists():
        cfg = _read_json(fsdp_cfg_path)
        world_size = int(cfg.get("world_size", 0)) or None

    shard_candidates = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"), key=_rank_id)
    if not shard_candidates:
        raise FileNotFoundError(f"No model shards found under {actor_dir}")

    shards_by_ws: Dict[int, List[Path]] = {}
    for f in shard_candidates:
        ws = _world_size_from_name(f)
        if ws > 0:
            shards_by_ws.setdefault(ws, []).append(f)

    if world_size is None:
        # prefer a detected world size > 1, else fallback to any
        ws_choices = sorted([ws for ws in shards_by_ws.keys() if ws > 1], reverse=True)
        if not ws_choices:
            ws_choices = sorted([ws for ws in shards_by_ws.keys() if ws > 0], reverse=True)
        if not ws_choices:
            raise RuntimeError(f"Unable to infer world_size from shard names under {actor_dir}")
        world_size = ws_choices[0]

    shard_files = sorted(shards_by_ws.get(world_size, []), key=_rank_id)
    if len(shard_files) != world_size:
        raise RuntimeError(f"world_size={world_size} but found {len(shard_files)} shards under {actor_dir}")

    load_workers = int(shard_load_workers) if shard_load_workers is not None else 1
    if load_workers <= 1 or len(shard_files) <= 1:
        shard_state_dicts = [_load_one_shard(p) for p in shard_files]
    else:
        from concurrent.futures import ThreadPoolExecutor

        max_workers = min(load_workers, len(shard_files))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            shard_state_dicts = list(ex.map(_load_one_shard, shard_files))
    keys = list(shard_state_dicts[0].keys())
    full_sd: Dict[str, torch.Tensor] = {}
    for k in keys:
        full_sd[k] = _assemble_param(k, [sd[k] for sd in shard_state_dicts])
    return full_sd, world_size


def load_model_and_tokenizer_from_step(
    step_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool = True,
    shard_load_workers: int = 1,
) -> Tuple[torch.nn.Module, Any, int]:
    actor_dir = step_dir / "actor"
    hf_dir = actor_dir / "huggingface"
    if not hf_dir.exists():
        raise FileNotFoundError(f"Missing actor/huggingface in {step_dir}")

    config = AutoConfig.from_pretrained(str(hf_dir), trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=trust_remote_code, use_fast=True, padding_side="left")

    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has no pad/eos/unk token to use as pad.")

    state_dict, world_size = load_fsdp_full_state_dict(actor_dir, shard_load_workers=shard_load_workers)

    empty_ctx = contextlib.nullcontext()
    if init_empty_weights is not None:
        empty_ctx = init_empty_weights()
    with empty_ctx:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    if hasattr(model, "to_empty") and init_empty_weights is not None:
        model.to_empty(device="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing or unexpected:
        # This is common across transformer versions; warn but continue.
        print(f"[WARN] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing (first 10):", missing[:10])
        if unexpected:
            print("  unexpected (first 10):", unexpected[:10])

    del state_dict

    model.eval()
    model.to(device=device, dtype=dtype)
    return model, tokenizer, world_size


def build_vocab_sketch_maps(
    vocab_size: int,
    proj_dim: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) & 0xFFFFFFFF)
    bucket_map = torch.randint(0, int(proj_dim), (int(vocab_size),), generator=gen, dtype=torch.int64)
    sign_map = torch.randint(0, 2, (int(vocab_size),), generator=gen, dtype=torch.int8)
    sign_map = (sign_map * 2 - 1).to(torch.float32)
    return bucket_map.to(device), sign_map.to(device)


def _has_hf_weights(hf_dir: Path) -> bool:
    if not hf_dir.exists():
        return False
    # Common HF weight filenames.
    if any(hf_dir.glob("model*.safetensors")):
        return True
    if any(hf_dir.glob("pytorch_model*.bin")):
        return True
    if any(hf_dir.glob("model*.bin")):
        return True
    return False


def _export_step_to_hf_dir(
    step_dir: Path,
    out_dir: Path,
    *,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> None:
    """
    Export one FSDP-sharded step checkpoint to a HF-pretrained directory.

    This is used by the vLLM backend (vLLM cannot directly load the FSDP shards).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, world_size = load_model_and_tokenizer_from_step(
        step_dir, device=torch.device("cpu"), dtype=dtype, trust_remote_code=trust_remote_code
    )
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    # Preserve chat template if present.
    src_jinja = step_dir / "actor" / "huggingface" / "chat_template.jinja"
    if src_jinja.exists():
        try:
            shutil.copy2(src_jinja, out_dir / "chat_template.jinja")
        except Exception:
            pass
    meta = {"source": str(step_dir), "world_size": int(world_size)}
    try:
        (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    del model, tokenizer
    gc.collect()


def _require_vllm():
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("vLLM backend requested but `vllm` is not available in this environment.") from e
    return LLM, SamplingParams


def _vllm_dtype_from_arg(dtype_arg: str) -> str:
    s = str(dtype_arg).lower()
    if s in ("bf16", "bfloat16"):
        return "bfloat16"
    if s in ("fp16", "float16", "half"):
        return "float16"
    if s in ("fp32", "float32"):
        return "float32"
    return "auto"


def _logprob_value(x: Any) -> float:
    # vLLM uses vllm.sequence.Logprob with `.logprob`.
    try:
        return float(getattr(x, "logprob"))
    except Exception:
        return float(x)


def _sketch_and_conf_from_vllm_completion(
    *,
    token_ids: Sequence[int],
    logprobs: Optional[List[Dict[int, Any]]],
    cumulative_logprob: Optional[float],
    bucket_map: np.ndarray,  # [V] int64
    sign_map: np.ndarray,  # [V] float32
    proj_dim: int,
    topk: int,
    confidence_metric: str,
    prob_mode: str,
    eos_token_id: Optional[int],
    vocab_size: int,
) -> Tuple[np.ndarray, float]:
    """
    Compute CountSketch(logit-grad) proxy + confidence from one vLLM completion.

    Returns:
      g:    [proj_dim] float32
      conf: float      (avg_logp or -entropy)
    """
    if logprobs is None:
        raise RuntimeError("vLLM returned no logprobs; set SamplingParams(logprobs=K).")

    g = np.zeros((int(proj_dim),), dtype=np.float32)
    token_count = 0
    ent_sum = 0.0
    logp_sum = 0.0

    k_top = int(min(max(int(topk), 1), int(vocab_size)))
    log_other = math.log(max(int(vocab_size) - k_top, 1))

    for t, y in enumerate(token_ids):
        if t >= len(logprobs):
            break
        lp_dict = logprobs[t] or {}
        y = int(y)
        token_count += 1

        if lp_dict:
            ids = np.fromiter(lp_dict.keys(), dtype=np.int64, count=len(lp_dict))
            vals = list(lp_dict.values())
            lps = np.fromiter((_logprob_value(v) for v in vals), dtype=np.float32, count=len(vals))
            probs = np.exp(lps).astype(np.float32)  # probabilities under vLLM's returned distribution

            if prob_mode == "topk":
                s = float(probs.sum())
                if s > 0.0 and math.isfinite(s):
                    probs = probs / s

            # CountSketch accumulate p_top
            buckets = bucket_map[ids]
            signs = sign_map[ids]
            np.add.at(g, buckets, signs * probs)

            # Confidence helpers
            if confidence_metric == "neg_entropy":
                ent_top = -float((probs * np.log(np.clip(probs, 1e-12, None))).sum())
                if prob_mode == "full":
                    p_sum = float(min(probs.sum(), 1.0))
                    p_rest = float(max(1.0 - p_sum, 0.0))
                    ent_rest = -p_rest * (math.log(max(p_rest, 1e-12)) - log_other)
                    ent_sum += ent_top + ent_rest
                else:
                    ent_sum += ent_top

            if cumulative_logprob is None and confidence_metric == "avg_logp":
                if y in lp_dict:
                    logp_sum += float(_logprob_value(lp_dict[y]))
                else:
                    logp_sum += -math.log(max(int(vocab_size), 1))

            match = (ids == y)
            in_top = bool(match.any())
            p_y = float(probs[match].sum()) if in_top else 0.0
        else:
            in_top = False
            p_y = 0.0
            if cumulative_logprob is None and confidence_metric == "avg_logp":
                logp_sum += -math.log(max(int(vocab_size), 1))

        if y < 0 or y >= int(bucket_map.shape[0]):
            raise RuntimeError(f"Token id out of bucket_map range: y={y}, vocab_size={bucket_map.shape[0]}")

        b_y = int(bucket_map[y])
        s_y = float(sign_map[y])

        # Add p_y at y if y not in topk; always subtract 1 at y.
        if not in_top:
            g[b_y] += s_y * float(p_y)
        g[b_y] += s_y * (-1.0)

        if eos_token_id is not None and y == int(eos_token_id):
            break

    denom = float(max(token_count, 1))
    g = g / denom

    if confidence_metric == "avg_logp":
        if cumulative_logprob is not None:
            conf = float(cumulative_logprob) / denom
        else:
            conf = float(logp_sum) / denom
    elif confidence_metric == "neg_entropy":
        conf = -float(ent_sum) / denom
    else:
        raise ValueError(f"Unknown confidence_metric: {confidence_metric}")

    return g, conf


def infer_actor_world_size(actor_dir: Path) -> int:
    fsdp_cfg_path = actor_dir / "fsdp_config.json"
    if fsdp_cfg_path.exists():
        try:
            cfg = _read_json(fsdp_cfg_path)
            ws = int(cfg.get("world_size", 0) or 0)
            if ws > 0:
                return ws
        except Exception:
            pass
    shard_candidates = list(actor_dir.glob("model_world_size_*_rank_*.pt"))
    ws_values = [int(_world_size_from_name(p)) for p in shard_candidates]
    ws_values = [ws for ws in ws_values if ws > 0]
    if not ws_values:
        return 1
    ws_choices = sorted({ws for ws in ws_values if ws > 1}, reverse=True)
    return int(ws_choices[0] if ws_choices else max(ws_values))


def has_hf_weights(model_dir: Path) -> bool:
    if not model_dir or not model_dir.exists():
        return False
    if list(model_dir.glob("*.safetensors")):
        return True
    if list(model_dir.glob("pytorch_model*.bin")):
        return True
    if (model_dir / "pytorch_model.bin.index.json").exists():
        return True
    return False


def is_lora_checkpoint(step_dir: Path) -> bool:
    """
    Check if a checkpoint contains a LoRA adapter (vs full weights).
    Used for Multi-LoRA batching optimization.
    """
    lora_dir = step_dir / "actor" / "lora_adapter"
    if not lora_dir.exists():
        return False
    # Check for adapter_config.json (PEFT standard)
    config_file = lora_dir / "adapter_config.json"
    return config_file.exists()


def resolve_hf_model_dir_for_step(
    step_dir: Path,
    *,
    hf_export_root: Optional[Path],
    base_model_dir: Optional[Path],
    export_hf_if_needed: bool,
) -> Path:
    actor_dir = step_dir / "actor"
    actor_hf = actor_dir / "huggingface"
    if has_hf_weights(actor_hf):
        return actor_hf

    if hf_export_root is None:
        raise FileNotFoundError(
            f"No HF weights under {actor_hf}, and --hf_export_root not provided for vLLM mode."
        )

    export_dir = hf_export_root / step_dir.parent.name / step_dir.name
    if has_hf_weights(export_dir):
        return export_dir
    if not export_hf_if_needed:
        raise FileNotFoundError(
            f"Missing exported HF weights for step={step_dir}. "
            f"Expected {export_dir}. Set --export_hf_if_needed to auto-export."
        )

    # Export via existing LLM_EVAL helper.
    base_dir = base_model_dir or actor_hf
    lora_dir = actor_dir / "lora_adapter"
    if lora_dir.exists() and not has_hf_weights(base_dir):
        raise FileNotFoundError(
            f"LoRA adapter detected at {lora_dir} but base_model_dir has no HF weights: {base_dir}. "
            f"Pass --base_model_dir pointing to the original base model."
        )

    # Import lazily to keep transformers-only runs fast.
    from tools.export_fsdp_dtensor_to_hf import export_one_step_to_hf  # type: ignore

    return export_one_step_to_hf(step_dir, base_dir, hf_export_root)


class AsyncExporter:
    """
    Pipeline prefetching: export the next checkpoint's HF weights asynchronously
    while the current checkpoint is being evaluated on GPU.
    """
    
    def __init__(self, max_workers: int = 1):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._pending_future: Optional[concurrent.futures.Future] = None
        self._pending_step_dir: Optional[Path] = None
        self._enabled = os.environ.get("VLLM_ENABLE_PIPELINE_PREFETCH", "true").lower() in ("1", "true", "yes")
    
    def prefetch(
        self,
        step_dir: Path,
        hf_export_root: Path,
        base_model_dir: Optional[Path],
        export_hf_if_needed: bool = True,
    ) -> None:
        """Start async export for the given step_dir."""
        if not self._enabled:
            return
        # Wait for any previous prefetch to complete first
        if self._pending_future is not None:
            try:
                self._pending_future.result(timeout=1)
            except Exception:
                pass
        self._pending_step_dir = step_dir
        self._pending_future = self._executor.submit(
            resolve_hf_model_dir_for_step,
            step_dir,
            hf_export_root=hf_export_root,
            base_model_dir=base_model_dir,
            export_hf_if_needed=export_hf_if_needed,
        )
    
    def get(self, step_dir: Path, timeout: float = 300) -> Optional[Path]:
        """
        Get the prefetched result if it matches step_dir.
        Returns None if no prefetch was done for this step.
        """
        if not self._enabled:
            return None
        if self._pending_step_dir != step_dir or self._pending_future is None:
            return None
        try:
            result = self._pending_future.result(timeout=timeout)
            self._pending_future = None
            self._pending_step_dir = None
            return result
        except Exception as e:
            print(f"[WARN] Prefetch failed for {step_dir}: {e}")
            self._pending_future = None
            self._pending_step_dir = None
            return None
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        if self._pending_future is not None:
            try:
                self._pending_future.result(timeout=60)
            except Exception:
                pass
        self._executor.shutdown(wait=wait)


def _vllm_dtype_from_torch(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    return "auto"


def _extract_logprob_value(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (float, int)):
        return float(v)
    # vLLM may return Logprob objects with .logprob
    lp = getattr(v, "logprob", None)
    if lp is not None:
        try:
            return float(lp)
        except Exception:
            return None
    return None


def vllm_generate_rollouts_grad_sketch_and_conf(
    llm: Any,
    prompts: Sequence[str],
    *,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    sampling_mode: str,
    topk: int,
    proj_dim: int,
    bucket_map: np.ndarray,
    sign_map: np.ndarray,
    confidence_metric: str,
    eos_token_id: Optional[int],
    vocab_size: int,
    seed: Optional[int],
    return_token_ids: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[List[int]]]]:
    if SamplingParams is None:
        raise RuntimeError("vLLM is not available. Install vllm or run without --use_vllm.")

    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1")

    k_top = int(min(max(int(topk), 1), int(vocab_size)))
    # vLLM enforces a per-engine `max_logprobs` limit (default: 20).
    # Clamp to avoid ValueError and keep sampling/sketch consistent.
    try:
        engine = getattr(llm, "llm_engine", None)
        model_cfg = None
        if engine is not None:
            if hasattr(engine, "get_model_config"):
                model_cfg = engine.get_model_config()
            else:
                model_cfg = getattr(engine, "model_config", None)
        max_lp = int(getattr(model_cfg, "max_logprobs", 0) or 0) if model_cfg is not None else 0
        if max_lp > 0 and k_top > max_lp:
            print(f"[WARN] vLLM max_logprobs={max_lp} < requested --topk={k_top}; clamping to {max_lp}.")
            k_top = int(max_lp)
    except Exception:
        pass
    log_other = math.log(max(int(vocab_size) - k_top, 1))

    # vLLM-native multi-sampling via n=k
    sp = SamplingParams(
        n=int(k),
        max_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(k_top) if str(sampling_mode) == "topk" else -1,
        logprobs=int(k_top),
    )
    if seed is not None:
        try:
            sp.seed = int(seed)
        except Exception:
            pass

    req_outs = llm.generate(list(prompts), sp)

    n_prompts = len(prompts)
    n_seq = n_prompts * int(k)
    g_out = np.zeros((n_seq, int(proj_dim)), dtype=np.float32)
    conf_out = np.zeros((n_seq,), dtype=np.float32)
    token_ids_out: Optional[List[List[int]]] = [] if return_token_ids else None

    out_idx = 0
    eps = 1e-8

    for req in req_outs:
        outs = getattr(req, "outputs", None) or []
        if len(outs) != int(k):
            # Some vLLM versions may return fewer outputs on errors; be explicit.
            raise RuntimeError(f"vLLM returned {len(outs)} outputs, expected n={k}.")
        for o in outs:
            token_ids = list(getattr(o, "token_ids", []) or [])
            logprobs_list = list(getattr(o, "logprobs", []) or [])
            if token_ids_out is not None:
                token_ids_out.append(token_ids)
            if len(token_ids) != len(logprobs_list):
                raise RuntimeError(f"vLLM token_ids/logprobs length mismatch: {len(token_ids)} vs {len(logprobs_list)}")

            g_vec = np.zeros((int(proj_dim),), dtype=np.float32)
            token_count = 0
            logp_sum = 0.0
            ent_sum = 0.0

            for y_t, lp_dict in zip(token_ids, logprobs_list):
                if eos_token_id is not None and int(y_t) == int(eos_token_id):
                    # Include EOS token (matches the transformers path).
                    pass

                if not isinstance(lp_dict, dict) or not lp_dict:
                    # If logprobs are missing, we cannot compute the sketch/confidence.
                    raise RuntimeError("vLLM did not return per-token logprobs; cannot compute sketch.")

                # Use all provided logprobs entries (typically top-k + the sampled token).
                ids: List[int] = []
                lps: List[float] = []
                lp_y = None
                y_i = int(y_t)

                for tid, v in lp_dict.items():
                    try:
                        tid_i = int(tid)
                    except Exception:
                        continue
                    lp = _extract_logprob_value(v)
                    if lp is None:
                        continue
                    ids.append(tid_i)
                    lps.append(lp)
                    if tid_i == y_i:
                        lp_y = lp

                if not ids:
                    raise RuntimeError("vLLM logprobs dict contained no usable entries.")

                probs = np.exp(np.array(lps, dtype=np.float32))
                buckets = bucket_map[np.array(ids, dtype=np.int64)]
                signs = sign_map[np.array(ids, dtype=np.int64)]
                np.add.at(g_vec, buckets, signs * probs)

                # subtract 1 at y
                by = int(bucket_map[y_i])
                sy = float(sign_map[y_i])
                g_vec[by] += sy * (-1.0)

                if confidence_metric == "avg_logp":
                    if lp_y is None:
                        lp_y = float(math.log(eps))
                    logp_sum += float(lp_y)
                else:
                    p_sum = float(np.clip(probs.sum(), 0.0, 1.0))
                    ent_top = float(-(probs * np.log(np.clip(probs, eps, None))).sum())
                    p_rest = max(1.0 - p_sum, 0.0)
                    ent_rest = -p_rest * (math.log(max(p_rest, eps)) - log_other) if p_rest > 0 else 0.0
                    ent_sum += ent_top + ent_rest

                token_count += 1

                if eos_token_id is not None and y_i == int(eos_token_id):
                    break

            token_count = max(int(token_count), 1)
            g_out[out_idx] = g_vec / float(token_count)
            if confidence_metric == "avg_logp":
                conf_out[out_idx] = float(logp_sum / float(token_count))
            else:
                conf_out[out_idx] = float(-(ent_sum / float(token_count)))
            out_idx += 1

    if out_idx != n_seq:
        raise RuntimeError(f"Internal error: produced {out_idx} sequences, expected {n_seq}.")
    if token_ids_out is not None and len(token_ids_out) != n_seq:
        raise RuntimeError(f"Internal error: produced {len(token_ids_out)} token lists, expected {n_seq}.")

    return torch.from_numpy(g_out), torch.from_numpy(conf_out), token_ids_out


@torch.no_grad()
def generate_rollouts_with_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    sanitize_logits: bool = True,
) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
    """
    Robust rollout generation with per-step logits.

    We avoid `model.generate(... do_sample=True ...)` here: if a checkpoint produces
    NaN/Inf logits, Transformers' sampling path can trigger a CUDA device-side
    assert in `torch.multinomial`, which then poisons the whole process.

    This custom loop explicitly sanitizes logits/probs and falls back to argmax
    on invalid rows, so evaluation can keep going even for partially broken steps.
    """

    tokenized = tokenizer(list(prompts), padding=True, return_tensors="pt", add_special_tokens=True)
    input_ids = tokenized.input_ids.to(model.device)
    attention_mask = tokenized.attention_mask.to(model.device)
    token_type_ids = getattr(tokenized, "token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(model.device)

    input_len = int(input_ids.shape[1])
    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1")

    # Expand to [B*k, ...] like HF generate(num_return_sequences=k)
    if k != 1:
        input_ids = input_ids.repeat_interleave(k, dim=0)
        attention_mask = attention_mask.repeat_interleave(k, dim=0)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.repeat_interleave(k, dim=0)

    do_sample = bool(temperature and float(temperature) > 0.0)
    eos_token_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None

    sequences = input_ids
    scores: List[torch.Tensor] = []
    active = torch.ones((int(sequences.shape[0]),), device=sequences.device, dtype=torch.bool)

    model_kwargs: Dict[str, Any] = {"attention_mask": attention_mask, "use_cache": True, "past_key_values": None}
    if token_type_ids is not None:
        model_kwargs["token_type_ids"] = token_type_ids

    # Transformers caching API changed across versions (e.g. `cache_position`).
    # We'll supply `cache_position` if supported; otherwise, silently drop it.
    model_kwargs["cache_position"] = torch.arange(sequences.shape[1], device=sequences.device, dtype=torch.long)

    def _apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
        p = float(p)
        if not (0.0 < p < 1.0):
            return probs
        # Nucleus filtering: expensive for large vocab; only used if p<1.
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum > p
        mask[..., 0] = False  # keep at least one
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        filtered = torch.zeros_like(probs)
        filtered.scatter_(1, sorted_idx, sorted_probs)
        return filtered

    for _t in range(int(max_new_tokens)):
        try:
            model_inputs = model.prepare_inputs_for_generation(sequences, **model_kwargs)
        except TypeError as e:
            # Older transformers may not accept `cache_position` (or other kwargs).
            if "cache_position" in model_kwargs:
                mk = dict(model_kwargs)
                mk.pop("cache_position", None)
                model_inputs = model.prepare_inputs_for_generation(sequences, **mk)
                model_kwargs = mk
            else:
                raise e
        out = model(**model_inputs, return_dict=True)
        logits = out.logits[:, -1, :]  # [N, V]

        if sanitize_logits and torch.is_floating_point(logits):
            finfo = torch.finfo(logits.dtype)
            logits = torch.nan_to_num(logits, nan=finfo.min, posinf=finfo.max, neginf=finfo.min)

        scores.append(logits)

        logits_f = logits.float()
        if do_sample:
            temp = float(temperature)
            if temp != 1.0:
                logits_f = logits_f / max(temp, 1e-6)

            # Default for inactive sequences: keep emitting EOS (if available) to stabilize shapes.
            next_tokens = torch.zeros((int(logits_f.shape[0]),), device=logits_f.device, dtype=torch.long)
            if eos_token_id is not None:
                next_tokens.fill_(int(eos_token_id))

            if active.any():
                logits_a = logits_f[active]
                probs = torch.softmax(logits_a, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                probs = torch.clamp(probs, min=0.0)
                if top_p and float(top_p) < 1.0:
                    probs = _apply_top_p(probs, float(top_p))

                row_sum = probs.sum(dim=-1, keepdim=True)
                valid = (row_sum.squeeze(-1) > 0) & torch.isfinite(row_sum.squeeze(-1))

                sampled = torch.empty((int(probs.shape[0]),), device=probs.device, dtype=torch.long)
                if valid.any():
                    probs_v = probs[valid] / row_sum[valid]
                    sampled[valid] = torch.multinomial(probs_v, num_samples=1).squeeze(1)
                if (~valid).any():
                    sampled[~valid] = torch.argmax(logits_a[~valid], dim=-1)

                next_tokens[active] = sampled
        else:
            next_tokens = torch.argmax(logits_f, dim=-1).to(torch.long)
            if eos_token_id is not None:
                next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, int(eos_token_id)))

        sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)

        # Update cache + masks for the next step (GenerationMixin-compatible).
        model_kwargs["past_key_values"] = getattr(out, "past_key_values", None)
        if "attention_mask" in model_kwargs and model_kwargs["attention_mask"] is not None:
            am = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat([am, am.new_ones((am.shape[0], 1))], dim=-1)
        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            tti = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([tti, tti[:, -1].unsqueeze(-1)], dim=-1)
        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            cp = model_kwargs["cache_position"]
            # `use_cache=True` path: keep only last position and increment.
            model_kwargs["cache_position"] = cp[-1:] + 1

        if eos_token_id is not None:
            ended_now = (next_tokens == int(eos_token_id)) & active
            active = active & (~ended_now)
            if not active.any():
                break

    gen_tokens = sequences[:, input_len:]
    return gen_tokens, scores, input_len


def _safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def _parse_ks(ks: str) -> List[int]:
    out: List[int] = []
    for part in (ks or "").split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            v = int(part)
            if v > 0:
                out.append(v)
    return sorted(set(out))


def _decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(list(token_ids), skip_special_tokens=True)
    except Exception:
        try:
            return tokenizer.decode(list(token_ids))
        except Exception:
            return ""


@torch.no_grad()
def generate_rollouts_grad_sketch_and_conf(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    *,
    bucket_map: torch.Tensor,  # [V]
    sign_map: torch.Tensor,  # [V]
    topk: int,
    proj_dim: int,
    confidence_metric: str,
    eos_token_id: Optional[int],
    vocab_size: int,
    sanitize_logits: bool = True,
    sampling_mode: str = "topk",
    prob_mode: str = "topk",
    return_gen_tokens: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Fast path: generate + compute logit-grad CountSketch + confidence *on the fly*.

    This avoids storing per-token full-vocab logits (huge) and avoids an extra
    pass over generated tokens, enabling larger `gen_batch_size`.

    sampling_mode:
      - "topk": sample from the same `topk` subset used for the sketch (fast)
      - "full": sample from full softmax over vocab (slow)

    prob_mode:
      - "topk": normalize probabilities within the `topk` subset (fast, approximate)
      - "full": use logsumexp over the full vocab for probabilities/logp (slow, accurate)
    """

    tokenized = tokenizer(list(prompts), padding=True, return_tensors="pt", add_special_tokens=True)
    input_ids = tokenized.input_ids.to(model.device)
    attention_mask = tokenized.attention_mask.to(model.device)
    token_type_ids = getattr(tokenized, "token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(model.device)

    input_len = int(input_ids.shape[1])
    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1")

    if k != 1:
        input_ids = input_ids.repeat_interleave(k, dim=0)
        attention_mask = attention_mask.repeat_interleave(k, dim=0)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.repeat_interleave(k, dim=0)

    do_sample = bool(temperature and float(temperature) > 0.0)
    sampling_mode = str(sampling_mode)
    prob_mode = str(prob_mode)

    k_top = int(min(max(int(topk), 1), int(vocab_size)))
    log_other = math.log(max(int(vocab_size) - k_top, 1))

    n = int(input_ids.shape[0])
    g = torch.zeros((n, int(proj_dim)), device=model.device, dtype=torch.float32)
    token_count = torch.zeros((n,), device=model.device, dtype=torch.float32)
    logp_sum = torch.zeros((n,), device=model.device, dtype=torch.float32)
    ent_sum = torch.zeros((n,), device=model.device, dtype=torch.float32)
    active = torch.ones((n,), device=model.device, dtype=torch.bool)

    sequences = input_ids
    model_kwargs: Dict[str, Any] = {"attention_mask": attention_mask, "use_cache": True, "past_key_values": None}
    if token_type_ids is not None:
        model_kwargs["token_type_ids"] = token_type_ids
    model_kwargs["cache_position"] = torch.arange(sequences.shape[1], device=sequences.device, dtype=torch.long)

    def _apply_top_p_in_topk(p_top_norm: torch.Tensor, p: float) -> torch.Tensor:
        p = float(p)
        if not (0.0 < p < 1.0):
            return p_top_norm
        # topk logits already sorted in descending order from torch.topk.
        cum = torch.cumsum(p_top_norm, dim=-1)
        mask = cum > p
        mask[..., 0] = False
        out = p_top_norm.masked_fill(mask, 0.0)
        row_sum = out.sum(dim=-1, keepdim=True)
        return out / torch.clamp(row_sum, min=1e-12)

    for _t in range(int(max_new_tokens)):
        try:
            model_inputs = model.prepare_inputs_for_generation(sequences, **model_kwargs)
        except TypeError as e:
            if "cache_position" in model_kwargs:
                mk = dict(model_kwargs)
                mk.pop("cache_position", None)
                model_inputs = model.prepare_inputs_for_generation(sequences, **mk)
                model_kwargs = mk
            else:
                raise e

        out = model(**model_inputs, return_dict=True)
        logits = out.logits[:, -1, :]  # [N, V]

        if sanitize_logits and torch.is_floating_point(logits):
            finfo = torch.finfo(logits.dtype)
            logits = torch.nan_to_num(logits, nan=finfo.min, posinf=finfo.max, neginf=finfo.min)

        logits_f = logits.float()

        # top-k logits used for both sketch and (fast) sampling
        topv, topi = torch.topk(logits_f, k_top, dim=-1)

        active_f = active.to(torch.float32)
        token_count += active_f

        if prob_mode == "full":
            logZ = torch.logsumexp(logits_f, dim=-1)  # [N]
            p_top = torch.exp(topv - logZ.unsqueeze(-1)) * active_f.unsqueeze(-1)  # [N, K]
        elif prob_mode == "topk":
            p_top = torch.softmax(topv, dim=-1) * active_f.unsqueeze(-1)  # [N, K]
            logZ = None
        else:
            raise ValueError(f"Unknown prob_mode: {prob_mode} (use 'topk' or 'full')")

        # CountSketch accumulate p_top at their bucket
        b_top = bucket_map[topi]  # [N, K]
        s_top = sign_map[topi]  # [N, K]
        g.scatter_add_(1, b_top, s_top * p_top)

        # Choose next tokens
        if not do_sample:
            next_tokens = topi[:, 0].to(torch.long)
            if eos_token_id is not None:
                next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, int(eos_token_id)))
        else:
            if sampling_mode == "topk":
                p_sample = p_top
                row_sum = p_sample.sum(dim=-1, keepdim=True)
                valid = (row_sum.squeeze(-1) > 0) & torch.isfinite(row_sum.squeeze(-1))

                # Renormalize on valid rows; fallback to argmax otherwise.
                p_norm = torch.zeros_like(p_sample)
                if valid.any():
                    p_norm[valid] = p_sample[valid] / torch.clamp(row_sum[valid], min=1e-12)
                    if top_p and float(top_p) < 1.0:
                        p_norm[valid] = _apply_top_p_in_topk(p_norm[valid], float(top_p))

                next_tokens = topi[:, 0].to(torch.long)
                if active.any():
                    idx = torch.zeros((n,), device=topi.device, dtype=torch.long)
                    active_valid = active & valid
                    if active_valid.any():
                        idx_a = torch.multinomial(p_norm[active_valid], num_samples=1).squeeze(1)
                        idx[active_valid] = idx_a
                        next_tokens[active_valid] = topi[active_valid].gather(1, idx_a.unsqueeze(1)).squeeze(1)
            elif sampling_mode == "full":
                # Slow path: full softmax sampling (kept for accuracy/debug).
                temp = float(temperature)
                logits_s = logits_f / max(temp, 1e-6)
                next_tokens = topi[:, 0].to(torch.long)
                if eos_token_id is not None:
                    next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, int(eos_token_id)))
                if active.any():
                    logits_a = logits_s[active]
                    probs = torch.softmax(logits_a, dim=-1)
                    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                    probs = torch.clamp(probs, min=0.0)
                    if top_p and float(top_p) < 1.0:
                        # nucleus on full vocab (expensive)
                        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                        cum = torch.cumsum(sorted_probs, dim=-1)
                        mask = cum > float(top_p)
                        mask[..., 0] = False
                        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                        probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
                    row_sum = probs.sum(dim=-1, keepdim=True)
                    valid = (row_sum.squeeze(-1) > 0) & torch.isfinite(row_sum.squeeze(-1))
                    if valid.any():
                        probs_v = probs[valid] / torch.clamp(row_sum[valid], min=1e-12)
                        sampled = torch.multinomial(probs_v, num_samples=1).squeeze(1)
                        # map back
                        idx_all = torch.nonzero(active, as_tuple=False).squeeze(1)
                        idx_valid = idx_all[valid]
                        next_tokens[idx_valid] = sampled
                    # invalid rows: keep argmax (topi[:,0]) already set
            else:
                raise ValueError(f"Unknown sampling_mode: {sampling_mode} (use 'topk' or 'full')")

        y_t = next_tokens

        # logp / entropy confidence
        if confidence_metric == "avg_logp":
            if prob_mode == "full":
                assert logZ is not None
                y_logit = logits_f.gather(1, y_t.unsqueeze(-1)).squeeze(-1)
                logp = (y_logit - logZ) * active_f
            else:
                # Approx: log prob under top-k normalized distribution.
                match = (topi == y_t.unsqueeze(-1)).to(torch.float32)
                p_y = (p_top * match).sum(dim=-1)  # includes active_f
                logp = _safe_log(p_y) * active_f
            logp_sum += logp
        elif confidence_metric == "neg_entropy":
            if prob_mode == "full":
                p_sum = torch.clamp(p_top.sum(dim=-1), max=1.0)  # [N]
                ent_top = -(p_top * _safe_log(p_top)).sum(dim=-1)  # [N]
                p_rest = torch.clamp(1.0 - p_sum, min=0.0)
                ent_rest = -p_rest * (_safe_log(p_rest) - log_other)
                ent_sum += (ent_top + ent_rest) * active_f
            else:
                ent_top = -(p_top * _safe_log(p_top)).sum(dim=-1)
                ent_sum += ent_top * active_f
        else:
            raise ValueError(f"Unknown confidence_metric: {confidence_metric}")

        # add p_y at y if y not in topk; always subtract 1 at y
        in_top = (topi == y_t.unsqueeze(-1)).any(dim=-1)  # [N]
        b_y = bucket_map[y_t]  # [N]
        s_y = sign_map[y_t]  # [N]

        if prob_mode == "full":
            assert logZ is not None
            y_logit = logits_f.gather(1, y_t.unsqueeze(-1)).squeeze(-1)
            p_y = torch.exp(y_logit - logZ) * active_f
        else:
            match = (topi == y_t.unsqueeze(-1)).to(torch.float32)
            p_y = (p_top * match).sum(dim=-1)  # includes active_f

        add_mask = (~in_top).to(torch.float32) * p_y  # [N]
        g.scatter_add_(1, b_y.unsqueeze(-1), (s_y * add_mask).unsqueeze(-1))
        g.scatter_add_(1, b_y.unsqueeze(-1), (s_y * (-1.0 * active_f)).unsqueeze(-1))

        # advance sequences + cache
        sequences = torch.cat([sequences, y_t.unsqueeze(1)], dim=1)
        model_kwargs["past_key_values"] = getattr(out, "past_key_values", None)
        if "attention_mask" in model_kwargs and model_kwargs["attention_mask"] is not None:
            am = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat([am, am.new_ones((am.shape[0], 1))], dim=-1)
        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            tti = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([tti, tti[:, -1].unsqueeze(-1)], dim=-1)
        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            cp = model_kwargs["cache_position"]
            model_kwargs["cache_position"] = cp[-1:] + 1

        if eos_token_id is not None:
            ended_now = (y_t == int(eos_token_id)) & active
            active = active & (~ended_now)
            if not active.any():
                break

    token_count = torch.clamp(token_count, min=1.0)
    g = g / token_count.unsqueeze(-1)

    if confidence_metric == "avg_logp":
        conf = logp_sum / token_count
    else:
        conf = -(ent_sum / token_count)
    gen_tokens = None
    if return_gen_tokens:
        gen_tokens = sequences[:, input_len:].detach().cpu()

    return g, conf, gen_tokens


@torch.no_grad()
def compute_logit_grad_sketch_and_conf(
    gen_tokens: torch.Tensor,  # [N, T]
    scores: List[torch.Tensor],  # len T, each [N, V]
    *,
    bucket_map: torch.Tensor,  # [V]
    sign_map: torch.Tensor,  # [V]
    topk: int,
    proj_dim: int,
    confidence_metric: str,
    eos_token_id: Optional[int],
    vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      g:    [N, proj_dim]  float32
      conf: [N]           float32 (avg_logp or -entropy approx)
    """
    device = scores[0].device if scores else gen_tokens.device
    n, t_max = gen_tokens.shape
    g = torch.zeros((n, int(proj_dim)), device=device, dtype=torch.float32)

    token_count = torch.zeros((n,), device=device, dtype=torch.float32)
    logp_sum = torch.zeros((n,), device=device, dtype=torch.float32)
    ent_sum = torch.zeros((n,), device=device, dtype=torch.float32)

    active = torch.ones((n,), device=device, dtype=torch.bool)
    k = int(min(topk, vocab_size))
    log_other = math.log(max(int(vocab_size) - k, 1))

    for step_idx, score_t in enumerate(scores):
        if step_idx >= t_max:
            break
        # score_t: [N, V], gen_tokens[:, step_idx]: [N]
        y_t = gen_tokens[:, step_idx]

        # include this token if still active
        active_f = active.to(torch.float32)
        token_count += active_f

        logits = score_t.float()
        # Be robust to broken checkpoints / numeric issues.
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        logZ = torch.logsumexp(logits, dim=-1)  # [N]
        y_logit = logits.gather(1, y_t.unsqueeze(-1)).squeeze(-1)  # [N]
        logp = y_logit - logZ
        logp_sum += logp * active_f

        # top-k probs under the *full* softmax via exp(logit - logZ)
        topv, topi = torch.topk(logits, k, dim=-1)
        p_top = torch.exp(topv - logZ.unsqueeze(-1)) * active_f.unsqueeze(-1)  # [N, k]

        # CountSketch accumulate p_top at their bucket
        b_top = bucket_map[topi]  # [N, k]
        s_top = sign_map[topi]  # [N, k]
        g.scatter_add_(1, b_top, s_top * p_top)

        # add p_y at y if y not in topk; always subtract 1 at y
        in_top = (topi == y_t.unsqueeze(-1)).any(dim=-1)  # [N]
        p_y = torch.exp(y_logit - logZ) * active_f  # [N]
        b_y = bucket_map[y_t]  # [N]
        s_y = sign_map[y_t]  # [N]

        add_mask = (~in_top).to(torch.float32) * p_y  # [N]
        g.scatter_add_(1, b_y.unsqueeze(-1), (s_y * add_mask).unsqueeze(-1))
        g.scatter_add_(1, b_y.unsqueeze(-1), (s_y * (-1.0 * active_f)).unsqueeze(-1))

        if confidence_metric == "neg_entropy":
            # approximate entropy with top-k + uniform remainder mass
            p_sum = torch.clamp(p_top.sum(dim=-1), max=1.0)  # [N]
            ent_top = -(p_top * _safe_log(p_top)).sum(dim=-1)  # [N]
            p_rest = torch.clamp(1.0 - p_sum, min=0.0)
            # uniform over remaining vocab
            ent_rest = -p_rest * (_safe_log(p_rest) - log_other)
            ent_sum += (ent_top + ent_rest) * active_f

        # update active after consuming eos
        if eos_token_id is not None:
            ended_now = (y_t == int(eos_token_id)) & active
            active = active & (~ended_now)

    token_count = torch.clamp(token_count, min=1.0)
    g = g / token_count.unsqueeze(-1)

    if confidence_metric == "avg_logp":
        conf = logp_sum / token_count
    elif confidence_metric == "neg_entropy":
        conf = -(ent_sum / token_count)
    else:
        raise ValueError(f"Unknown confidence_metric: {confidence_metric}")

    return g, conf


def trace_var_from_vectors(x: np.ndarray) -> float:
    """Trace of covariance: E||x||^2 - ||E[x]||^2."""
    if x.size == 0:
        return float("nan")
    mean = x.mean(axis=0)
    mean_norm2 = float(np.dot(mean, mean))
    e_norm2 = float(np.mean(np.sum(x * x, axis=1)))
    return max(0.0, e_norm2 - mean_norm2)


def trace_var_from_vectors_torch(x: torch.Tensor) -> torch.Tensor:
    """Torch version of trace(var): E||x||^2 - ||E[x]||^2."""
    if x.numel() == 0:
        return torch.tensor(float("nan"), device=x.device, dtype=torch.float32)
    x_f = x.to(torch.float32)
    mean = x_f.mean(dim=0)
    mean_norm2 = (mean * mean).sum()
    e_norm2 = (x_f * x_f).sum(dim=1).mean()
    return torch.clamp(e_norm2 - mean_norm2, min=0.0)


def compute_step_metrics(
    *,
    step_dir: Path,
    prompts: Sequence[str],
    answers: Optional[Sequence[str]],
    data_name: str,
    prompt_type: str,
    device: torch.device,
    dtype: torch.dtype,
    k_rollouts: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    sampling_mode: str,
    prob_mode: str,
    proj_dim: int,
    topk: int,
    confidence_metric: str,
    beta_target: Optional[float],
    tau_override: Optional[float],
    sketch_seed: int,
    gen_batch_size: int,
    sanitize_logits: bool,
    progress_desc: str,
    trust_remote_code: bool = True,
    shard_load_workers: int = 1,
    empty_cache_each_batch: bool = False,
    inference_mode: bool = True,
    use_vllm: bool = False,
    hf_export_root: Optional[Path] = None,
    base_model_dir: Optional[Path] = None,
    export_hf_if_needed: bool = False,
    keep_hf_export: bool = False,
    vllm_tensor_parallel_size: int = 1,
    vllm_pipeline_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_num_seqs: int = 0,
    batch_var_seeds: Sequence[int] = (),
    batch_var_prompts: Optional[Sequence[str]] = None,
    batch_var_k: int = 1,
    batch_var_sketch_seed: Optional[int] = None,
    compute_passk: bool = False,
    passk_ks: Sequence[int] = (),
) -> Dict[str, Any]:
    step_num = int(_step_num_from_dir(step_dir))
    actor_dir = step_dir / "actor"

    llm = None
    model = None
    tokenizer = None
    world_size = infer_actor_world_size(actor_dir)
    tmp_export_root: Optional[Path] = None
    out: Dict[str, Any] = {}

    try:
        if use_vllm:
            if LLM is None:
                raise RuntimeError("vLLM is not available. Install vllm or run without --use_vllm.")
            if hf_export_root is None:
                raise FileNotFoundError("--hf_export_root must be set when using vLLM and weights are not present in the step dir.")
            if not bool(keep_hf_export):
                hf_export_root.mkdir(parents=True, exist_ok=True)
                tmp_export_root = Path(
                    tempfile.mkdtemp(prefix=f"vf_curl_grad_variance_step{step_num}_", dir=str(hf_export_root))
                )
                hf_root_for_step = tmp_export_root
            else:
                hf_root_for_step = hf_export_root
            hf_dir = resolve_hf_model_dir_for_step(
                step_dir,
                hf_export_root=hf_root_for_step,
                base_model_dir=base_model_dir,
                export_hf_if_needed=export_hf_if_needed,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(hf_dir), trust_remote_code=trust_remote_code, use_fast=True, padding_side="left"
            )
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif tokenizer.unk_token_id is not None:
                    tokenizer.pad_token = tokenizer.unk_token

            cfg = AutoConfig.from_pretrained(str(hf_dir), trust_remote_code=trust_remote_code)
            tok_size = int(len(tokenizer))
            cfg_size = int(getattr(cfg, "vocab_size", 0) or 0)
            vocab_size = max(tok_size, cfg_size, 1)
            eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None

            llm_kwargs: Dict[str, Any] = {
                "model": str(hf_dir),
                "trust_remote_code": bool(trust_remote_code),
                "dtype": _vllm_dtype_from_torch(dtype),
                "tensor_parallel_size": int(vllm_tensor_parallel_size),
                "pipeline_parallel_size": int(vllm_pipeline_parallel_size),
                "gpu_memory_utilization": float(vllm_gpu_memory_utilization),
            }
            # Enable prefix caching for faster inference on repeated prompts
            if os.environ.get("VLLM_ENABLE_PREFIX_CACHING", "true").lower() in ("1", "true", "yes"):
                llm_kwargs["enable_prefix_caching"] = True
            # Allow requesting logprobs up to --topk (vLLM default max_logprobs is 20).
            try:
                llm_kwargs["max_logprobs"] = int(max(20, int(topk)))
            except Exception:
                llm_kwargs["max_logprobs"] = 20
            if int(vllm_max_num_seqs) > 0:
                llm_kwargs["max_num_seqs"] = int(vllm_max_num_seqs)
            llm = LLM(**llm_kwargs)

            bucket_map_t, sign_map_t = build_vocab_sketch_maps(vocab_size, proj_dim, sketch_seed, device=torch.device("cpu"))
            bucket_map_np = bucket_map_t.cpu().numpy()
            sign_map_np = sign_map_t.cpu().numpy()
            compute_device = torch.device("cpu")
        else:
            model, tokenizer, world_size = load_model_and_tokenizer_from_step(
                step_dir,
                device=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                shard_load_workers=shard_load_workers,
            )
            tok_size = int(len(tokenizer))
            cfg_size = int(getattr(model.config, "vocab_size", 0) or 0)
            vocab_size = max(tok_size, cfg_size, 1)
            bucket_map_t, sign_map_t = build_vocab_sketch_maps(vocab_size, proj_dim, sketch_seed, device=model.device)
            bucket_map_np = None
            sign_map_np = None
            eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
            compute_device = model.device

        assert tokenizer is not None

        total = len(prompts)
        passk_enabled = bool(compute_passk)
        answers_list: Optional[List[str]] = None
        score_mat: Optional[List[List[bool]]] = None
        passk_ks_list = [int(k) for k in passk_ks if int(k) > 0] if passk_enabled else []
        passk_compute = None
        passk_eval = None
        if passk_enabled:
            if answers is None:
                raise ValueError("--compute_passk requires answers to be provided.")
            answers_list = list(answers)
            if len(answers_list) != total:
                raise ValueError(f"answers length mismatch: {len(answers_list)} vs prompts {total}")
            eval_root = Path(__file__).resolve().parents[2]
            if str(eval_root) not in sys.path:
                sys.path.insert(0, str(eval_root))
            from evaluate import _compute_pass_at_k  # type: ignore
            from grader import math_equal  # type: ignore
            from parser import run_execute  # type: ignore
            from python_executor import PythonExecutor  # type: ignore

            executor = PythonExecutor(get_answer_from_stdout=True)

            def _eval_one(text: str, gt: str) -> bool:
                try:
                    pred, _ = run_execute(executor, text, prompt_type, data_name, execute=False)
                    return bool(math_equal(pred, gt))
                except Exception:
                    return False

            passk_eval = _eval_one
            passk_compute = _compute_pass_at_k
            score_mat = [[] for _ in range(total)]
        g_means_t = torch.empty((total, int(proj_dim)), device=compute_device, dtype=torch.float32)
        var_traces_t = torch.empty((total,), device=compute_device, dtype=torch.float32)
        confs_t = torch.empty((total,), device=compute_device, dtype=torch.float32)

        infer_ctx = torch.inference_mode() if bool(inference_mode) else torch.no_grad()

        def _run_one(
            batch_ps: Sequence[str], *, k: int, seed: Optional[int]
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
            if use_vllm:
                assert llm is not None
                assert bucket_map_np is not None and sign_map_np is not None
                return vllm_generate_rollouts_grad_sketch_and_conf(
                    llm,
                    batch_ps,
                    k=int(k),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    sampling_mode=sampling_mode,
                    topk=topk,
                    proj_dim=proj_dim,
                    bucket_map=bucket_map_np,
                    sign_map=sign_map_np,
                    confidence_metric=confidence_metric,
                    eos_token_id=eos_id,
                    vocab_size=vocab_size,
                    seed=seed,
                    return_token_ids=bool(passk_enabled),
                )
            assert model is not None
            if seed is not None:
                _set_all_seeds(int(seed))
            return generate_rollouts_grad_sketch_and_conf(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_ps,
                k=int(k),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                bucket_map=bucket_map_t,
                sign_map=sign_map_t,
                topk=topk,
                proj_dim=proj_dim,
                confidence_metric=confidence_metric,
                eos_token_id=eos_id,
                vocab_size=vocab_size,
                sanitize_logits=sanitize_logits,
                sampling_mode=sampling_mode,
                prob_mode=prob_mode,
                return_gen_tokens=bool(passk_enabled),
            )

        with infer_ctx:
            for start in range(0, total, int(gen_batch_size)):
                batch_prompts = prompts[start : start + int(gen_batch_size)]
                seed = sketch_seed * 1000003 + step_num * 1009 + start
                token_payload = None
                try:
                    g_seq, conf_seq, token_payload = _run_one(batch_prompts, k=int(k_rollouts), seed=int(seed))
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and len(batch_prompts) > 1:
                        print(f"[WARN] OOM at gen_batch_size={len(batch_prompts)}; splitting batch. err={e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        mid = len(batch_prompts) // 2
                        g1, c1, t1 = _run_one(batch_prompts[:mid], k=int(k_rollouts), seed=int(seed))
                        g2, c2, t2 = _run_one(batch_prompts[mid:], k=int(k_rollouts), seed=int(seed) + 1)
                        g_seq = torch.cat([g1, g2], dim=0)
                        conf_seq = torch.cat([c1, c2], dim=0)
                        if passk_enabled:
                            if use_vllm:
                                token_payload = (t1 or []) + (t2 or [])
                            else:
                                token_payload = torch.cat([t1, t2], dim=0) if t1 is not None and t2 is not None else None
                    else:
                        raise

                b = len(batch_prompts)
                g_seq = g_seq.view(b, int(k_rollouts), int(proj_dim))
                conf_seq = conf_seq.view(b, int(k_rollouts))

                g_mean = g_seq.mean(dim=1)  # [B, D]
                e_norm2 = g_seq.pow(2).sum(dim=-1).mean(dim=1)  # [B]
                mean_norm2 = g_mean.pow(2).sum(dim=-1)  # [B]
                var_trace = torch.clamp(e_norm2 - mean_norm2, min=0.0)
                conf_prompt = conf_seq.mean(dim=1)  # [B]

                if passk_enabled and score_mat is not None and answers_list is not None and passk_eval is not None:
                    if token_payload is None:
                        raise RuntimeError("Pass@k enabled but generation did not return token IDs.")
                    if use_vllm:
                        token_lists = token_payload
                    else:
                        token_lists = token_payload.tolist() if hasattr(token_payload, "tolist") else token_payload
                    if token_lists is None or len(token_lists) != int(b) * int(k_rollouts):
                        raise RuntimeError(f"Token list size mismatch: {0 if token_lists is None else len(token_lists)} vs {int(b) * int(k_rollouts)}")
                    for i in range(int(b)):
                        gt = answers_list[start + i]
                        scores_i: List[bool] = []
                        base = i * int(k_rollouts)
                        for j in range(int(k_rollouts)):
                            token_ids = token_lists[base + j] if token_lists is not None else []
                            text = _decode_tokens(tokenizer, token_ids)
                            scores_i.append(bool(passk_eval(text, gt)))
                        score_mat[start + i] = scores_i

                g_means_t[start : start + b] = g_mean
                var_traces_t[start : start + b] = var_trace
                confs_t[start : start + b] = conf_prompt

                del g_seq, conf_seq, g_mean, e_norm2, mean_norm2, var_trace, conf_prompt
                if bool(empty_cache_each_batch) and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if (start // int(gen_batch_size)) % 10 == 0:
                    print(f"[{progress_desc}] prompts {start}/{total}")

        # Aggregate on CPU once (fast and keeps results close to the original numpy path).
        g_means_np = g_means_t.detach().cpu().numpy().astype(np.float64, copy=False)
        var_traces_np = var_traces_t.detach().cpu().numpy().astype(np.float64, copy=False)
        confs_np = confs_t.detach().cpu().numpy().astype(np.float64, copy=False)

        full_sigma = float(np.mean(var_traces_np)) if var_traces_np.size else float("nan")
        full_vprob = float(trace_var_from_vectors(g_means_np)) if g_means_np.size else float("nan")
        full_gbar_norm2 = float(np.dot(g_means_np.mean(axis=0), g_means_np.mean(axis=0))) if g_means_np.size else float("nan")

        mask_keep_np = np.ones((int(total),), dtype=bool)
        tau = float("nan")
        kept = int(mask_keep_np.sum())
        if confs_np.size:
            if tau_override is not None:
                tau = float(tau_override)
                if not np.isfinite(tau):
                    if math.isinf(tau):
                        mask_keep_np = np.ones((int(total),), dtype=bool) if tau < 0 else np.zeros((int(total),), dtype=bool)
                    else:
                        mask_keep_np = np.zeros((int(total),), dtype=bool)
                else:
                    mask_keep_np = confs_np >= tau
            elif beta_target is not None:
                q = 1.0 - float(beta_target)
                tau = float(np.quantile(confs_np, q))
                mask_keep_np = confs_np >= tau
            kept = int(mask_keep_np.sum())

        kept_sigma = float(np.mean(var_traces_np[mask_keep_np])) if kept > 0 else float("nan")
        kept_vprob = float(trace_var_from_vectors(g_means_np[mask_keep_np])) if kept > 0 else float("nan")
        kept_gbar_norm2 = (
            float(np.dot(g_means_np[mask_keep_np].mean(axis=0), g_means_np[mask_keep_np].mean(axis=0))) if kept > 0 else float("nan")
        )

        # --- Bias metrics: curriculum vs full dataset gradient difference ---
        # gbar_kept = mean gradient over kept prompts
        # gbar_full = mean gradient over all prompts
        # Bias measures how much the curriculum-selected gradient differs from the full gradient
        gbar_full_vec = g_means_np.mean(axis=0) if g_means_np.size else np.zeros(proj_dim)
        gbar_kept_vec = g_means_np[mask_keep_np].mean(axis=0) if kept > 0 else np.zeros(proj_dim)
        
        # L2 norm of gradient difference (absolute bias)
        bias_l2 = float(np.linalg.norm(gbar_kept_vec - gbar_full_vec))
        
        # Relative bias: ||gbar_kept - gbar_full|| / ||gbar_full||
        gbar_full_norm = float(np.linalg.norm(gbar_full_vec))
        bias_relative = bias_l2 / (gbar_full_norm + 1e-10) if gbar_full_norm > 1e-10 else float("nan")
        
        # Cosine similarity: measures directional alignment (1 = identical direction)
        gbar_kept_norm = float(np.linalg.norm(gbar_kept_vec))
        if gbar_kept_norm > 1e-10 and gbar_full_norm > 1e-10:
            cosine_similarity = float(np.dot(gbar_kept_vec, gbar_full_vec) / (gbar_kept_norm * gbar_full_norm))
        else:
            cosine_similarity = float("nan")
        
        # Variance ratios for convenience
        sigma_ratio = kept_sigma / full_sigma if full_sigma > 1e-10 else float("nan")
        vprob_ratio = kept_vprob / full_vprob if full_vprob > 1e-10 else float("nan")

        beta_actual = float(kept) / float(total) if total > 0 else float("nan")
        out = {
            "world_size": int(world_size),
            "vocab_size": int(vocab_size),
            "num_prompts": int(total),
            "beta_target": None if beta_target is None else float(beta_target),
            "tau": tau,
            "kept": int(kept),
            "dropped": int(total - kept),
            "beta_actual": float(beta_actual),
            # Variance metrics
            "sigma_kept": float(kept_sigma),
            "vprob_kept": float(kept_vprob),
            "gbar_norm2_kept": float(kept_gbar_norm2),
            "sigma_full": float(full_sigma),
            "vprob_full": float(full_vprob),
            "gbar_norm2_full": float(full_gbar_norm2),
            "sigma_ratio": float(sigma_ratio),
            "vprob_ratio": float(vprob_ratio),
            # Bias metrics
            "bias_l2": float(bias_l2),
            "bias_relative": float(bias_relative),
            "cosine_similarity": float(cosine_similarity),
            # Raw gradient vectors (projections, for post-hoc analysis)
            "gbar_kept_vec": gbar_kept_vec.tolist(),
            "gbar_full_vec": gbar_full_vec.tolist(),
        }

        if passk_enabled and score_mat is not None and passk_compute is not None:
            if not passk_ks_list:
                passk_ks_list = [1]
            kept_scores = [s for s, keep in zip(score_mat, mask_keep_np) if bool(keep)]
            drop_scores = [s for s, keep in zip(score_mat, mask_keep_np) if not bool(keep)]
            pass_kept, counts_kept = passk_compute(kept_scores, passk_ks_list)
            pass_drop, counts_drop = passk_compute(drop_scores, passk_ks_list)
            pass_full, counts_full = passk_compute(score_mat, passk_ks_list)
            out.update(
                {
                    "passk_ks": [int(k) for k in passk_ks_list],
                    "pass_at_k_kept": pass_kept,
                    "pass_at_k_dropped": pass_drop,
                    "pass_at_k_full": pass_full,
                    "valid_counts_kept": counts_kept,
                    "valid_counts_dropped": counts_drop,
                    "valid_counts_full": counts_full,
                }
            )

        if batch_var_seeds:
            bv_prompts = list(batch_var_prompts) if batch_var_prompts is not None else list(prompts)
            bv_seed = int(batch_var_sketch_seed) if batch_var_sketch_seed is not None else int(sketch_seed) + 999
            if use_vllm:
                bv_bucket_map_t, bv_sign_map_t = build_vocab_sketch_maps(vocab_size, proj_dim, bv_seed, device=torch.device("cpu"))
                bv_bucket_map_np = bv_bucket_map_t.cpu().numpy()
                bv_sign_map_np = bv_sign_map_t.cpu().numpy()
            else:
                assert model is not None
                bv_bucket_map_t, bv_sign_map_t = build_vocab_sketch_maps(vocab_size, proj_dim, bv_seed, device=model.device)
                bv_bucket_map_np = None
                bv_sign_map_np = None

            def _run_one_bv(
                batch_ps: Sequence[str], *, k: int, seed: Optional[int]
            ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
                if use_vllm:
                    assert llm is not None
                    assert bv_bucket_map_np is not None and bv_sign_map_np is not None
                    return vllm_generate_rollouts_grad_sketch_and_conf(
                        llm,
                        batch_ps,
                        k=int(k),
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        sampling_mode=sampling_mode,
                        topk=topk,
                        proj_dim=proj_dim,
                        bucket_map=bv_bucket_map_np,
                        sign_map=bv_sign_map_np,
                        confidence_metric=confidence_metric,
                        eos_token_id=eos_id,
                        vocab_size=vocab_size,
                        seed=seed,
                        return_token_ids=False,
                    )
                assert model is not None
                if seed is not None:
                    _set_all_seeds(int(seed))
                return generate_rollouts_grad_sketch_and_conf(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch_ps,
                    k=int(k),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    bucket_map=bv_bucket_map_t,
                    sign_map=bv_sign_map_t,
                    topk=topk,
                    proj_dim=proj_dim,
                    confidence_metric=confidence_metric,
                    eos_token_id=eos_id,
                    vocab_size=vocab_size,
                    sanitize_logits=sanitize_logits,
                    sampling_mode=sampling_mode,
                    prob_mode=prob_mode,
                    return_gen_tokens=False,
                )

            norms: List[float] = []
            for s in batch_var_seeds:
                if not use_vllm:
                    _set_all_seeds(int(s))
                g_sum = torch.zeros((int(proj_dim),), device=compute_device, dtype=torch.float32)
                n_seq_total = 0
                for start in range(0, len(bv_prompts), int(gen_batch_size)):
                    batch_ps = bv_prompts[start : start + int(gen_batch_size)]
                    seed = None
                    if use_vllm:
                        seed = int(s) * 1000003 + step_num * 1009 + start
                    g_seq, _, _ = _run_one_bv(batch_ps, k=int(batch_var_k), seed=seed)
                    g_sum += g_seq.sum(dim=0)
                    n_seq_total += int(g_seq.shape[0])
                    del g_seq
                    if bool(empty_cache_each_batch) and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                g_batch = g_sum / max(int(n_seq_total), 1)
                norms.append(float(torch.linalg.norm(g_batch).item()))

            var = float(np.var(np.array(norms, dtype=np.float64), ddof=1)) if len(norms) >= 2 else 0.0
            out["batch_grad_norm_var"] = float(var)
            out["batch_grad_norm_var_detail"] = {
                "seeds": list(map(int, batch_var_seeds)),
                "l2_norms": norms,
                "var": float(var),
            }
            del bv_bucket_map_t, bv_sign_map_t

        del g_means_t, var_traces_t, confs_t, bucket_map_t, sign_map_t
        return out
    finally:
        if llm is not None:
            try:
                if hasattr(llm, "clear_cache"):
                    llm.clear_cache()
            except Exception:
                pass
            try:
                del llm
            except Exception:
                pass
            if destroy_model_parallel is not None:
                try:
                    destroy_model_parallel()
                except Exception:
                    pass
        if model is not None:
            try:
                del model
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if tmp_export_root is not None and not bool(keep_hf_export):
            try:
                shutil.rmtree(tmp_export_root, ignore_errors=True)
            except Exception:
                pass


def compute_batch_grad_norm_var(
    *,
    step_dir: Path,
    prompts: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    seeds: Sequence[int],
    batch_k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    sampling_mode: str,
    prob_mode: str,
    proj_dim: int,
    topk: int,
    confidence_metric: str,
    sketch_seed: int,
    gen_batch_size: int,
    sanitize_logits: bool,
) -> Dict[str, Any]:
    if not seeds:
        return {"seeds": [], "l2_norms": [], "var": float("nan")}

    model, tokenizer, world_size = load_model_and_tokenizer_from_step(step_dir, device=device, dtype=dtype)
    tok_size = int(len(tokenizer))
    cfg_size = int(getattr(model.config, "vocab_size", 0) or 0)
    vocab_size = max(tok_size, cfg_size, 1)
    bucket_map, sign_map = build_vocab_sketch_maps(vocab_size, proj_dim, sketch_seed, device=model.device)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None

    norms: List[float] = []
    for s in seeds:
        _set_all_seeds(int(s))
        # compute g over all prompts, average across sequences
        g_sum = torch.zeros((int(proj_dim),), device=model.device, dtype=torch.float32)
        n_seq_total = 0
        total = len(prompts)
        for start in range(0, total, int(gen_batch_size)):
            batch_prompts = prompts[start : start + int(gen_batch_size)]
            def _run_one(batch_ps: Sequence[str]) -> torch.Tensor:
                g_seq, _, _ = generate_rollouts_grad_sketch_and_conf(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch_ps,
                    k=batch_k,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    bucket_map=bucket_map,
                    sign_map=sign_map,
                    topk=topk,
                    proj_dim=proj_dim,
                    confidence_metric=confidence_metric,
                    eos_token_id=eos_id,
                    vocab_size=vocab_size,
                    sanitize_logits=sanitize_logits,
                    sampling_mode=sampling_mode,
                    prob_mode=prob_mode,
                    return_gen_tokens=False,
                )
                return g_seq

            try:
                g_seq = _run_one(batch_prompts)
            except RuntimeError as e:
                if torch.cuda.is_available() and "out of memory" in str(e).lower() and len(batch_prompts) > 1:
                    print(f"[WARN] OOM at gen_batch_size={len(batch_prompts)}; splitting batch. err={e}")
                    torch.cuda.empty_cache()
                    mid = len(batch_prompts) // 2
                    g1 = _run_one(batch_prompts[:mid])
                    g2 = _run_one(batch_prompts[mid:])
                    g_seq = torch.cat([g1, g2], dim=0)
                else:
                    raise
            g_sum += g_seq.sum(dim=0)
            n_seq_total += int(g_seq.shape[0])
            del g_seq
        g_batch = g_sum / max(n_seq_total, 1)
        norms.append(float(torch.linalg.norm(g_batch).item()))

    var = float(np.var(np.array(norms, dtype=np.float64), ddof=1)) if len(norms) >= 2 else 0.0

    del bucket_map, sign_map, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"seeds": list(map(int, seeds)), "l2_norms": norms, "var": var}


def plot_results(
    steps: Sequence[int],
    rows: Sequence[Dict[str, Any]],
    baseline_rows: Optional[Sequence[Dict[str, Any]]],
    out_path: Path,
    title: str,
    baseline_full_only: bool = False,
    font_sizes: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    setup_plot_style()
    if font_sizes is None:
        font_sizes = FONT_SIZES or get_default_font_sizes("vf_curl_grad_variance.py")
    plot_label_size = int(font_sizes.get("xlabel", 14))
    plot_tick_size = int(font_sizes.get("tick", 12))
    plot_legend_size = int(font_sizes.get("legend", 12))
    plot_xlabel = "training steps"
    xs = np.array(list(steps), dtype=np.int64)

    def _arr(key: str, src: Sequence[Dict[str, Any]]) -> np.ndarray:
        return np.array([float(r.get(key, float("nan"))) for r in src], dtype=np.float64)

    sigma_kept = _arr("sigma_kept", rows)
    vprob_kept = _arr("vprob_kept", rows)
    gbar_norm2_kept = _arr("gbar_norm2_kept", rows) if any("gbar_norm2_kept" in r for r in rows) else None
    batch_var = _arr("batch_grad_norm_var", rows) if any("batch_grad_norm_var" in r for r in rows) else None
    tau = _arr("tau", rows) if any("tau" in r for r in rows) else None
    kept = _arr("kept", rows) if any("kept" in r for r in rows) else None
    num_prompts = _arr("num_prompts", rows) if any("num_prompts" in r for r in rows) else None

    betas = np.array(
        [float(r.get("beta_target", 1.0) if r.get("beta_target", None) is not None else 1.0) for r in rows],
        dtype=np.float64,
    )
    betas = np.clip(betas, 1e-6, 1.0)
    beta_actual = None
    if kept is not None and num_prompts is not None:
        beta_actual = kept / np.clip(num_prompts, 1.0, None)
        beta_actual = np.clip(beta_actual, 0.0, 1.0)

    def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return num / np.clip(den, 1e-12, None)

    def _pct(r: np.ndarray) -> np.ndarray:
        return 100.0 * (r - 1.0)

    if baseline_rows is not None:
        # When a baseline run is provided, vf_curl_grad_variance.py computes baseline metrics
        # using the *same* beta_target at each step. This allows a fair "kept-vs-kept"
        # comparison (same retention rate), while we can also still compare against the
        # baseline's "full data" metrics (w=1, beta=1).
        base_sigma_kept = _arr("sigma_kept", baseline_rows)
        base_vprob_kept = _arr("vprob_kept", baseline_rows)
        base_gbar_norm2_kept = _arr("gbar_norm2_kept", baseline_rows) if gbar_norm2_kept is not None else None

        base_sigma_full = _arr("sigma_full", baseline_rows)
        base_vprob_full = _arr("vprob_full", baseline_rows)
        base_batch_var = _arr("batch_grad_norm_var", baseline_rows) if batch_var is not None else None
        denom_desc = "baseline kept@β (solid) + baseline full (dashed)"
    else:
        base_sigma_kept = None
        base_vprob_kept = None
        base_gbar_norm2_kept = None
        base_sigma_full = _arr("sigma_full", rows)
        base_vprob_full = _arr("vprob_full", rows)
        base_batch_var = _arr("batch_grad_norm_var", rows) if batch_var is not None else None
        denom_desc = "same-run full"

    # Numerator comparisons (homogeneity within the kept set).
    # - vs *_full: compares the "easy" kept subset to the baseline full distribution
    # - vs *_kept: fair curl-vs-nocurl under the same beta_target (same retention)
    sigma_num_ratio_full = _ratio(sigma_kept, base_sigma_full)
    vprob_num_ratio_full = _ratio(vprob_kept, base_vprob_full)
    sigma_num_ratio_kept = _ratio(sigma_kept, base_sigma_kept) if base_sigma_kept is not None else None
    vprob_num_ratio_kept = _ratio(vprob_kept, base_vprob_kept) if base_vprob_kept is not None else None

    # Estimator variance proxy (matches VI-CuRL's 1/beta normalization):
    #   Var(\hat g_t) ≈ (sigma_kept + V_prob_kept) / beta  +  ((1-beta)/beta) * ||E[g|w=1]||^2
    # Here we use gbar_norm2_kept (if present) as a cheap proxy for ||∇J_t||^2.
    proxy_main = (sigma_kept + vprob_kept) / betas
    proxy_mask = None
    if gbar_norm2_kept is not None:
        proxy_mask = ((1.0 - betas) / betas) * gbar_norm2_kept
    proxy_total = proxy_main + (proxy_mask if proxy_mask is not None else 0.0)
    base_total_full = base_sigma_full + base_vprob_full
    proxy_ratio_full = _ratio(proxy_total, base_total_full)
    proxy_ratio_kept = None
    if base_sigma_kept is not None and base_vprob_kept is not None:
        base_proxy_main = (base_sigma_kept + base_vprob_kept) / betas
        base_proxy_mask = None
        if base_gbar_norm2_kept is not None:
            base_proxy_mask = ((1.0 - betas) / betas) * base_gbar_norm2_kept
        base_proxy_total = base_proxy_main + (base_proxy_mask if base_proxy_mask is not None else 0.0)
        proxy_ratio_kept = _ratio(proxy_total, base_proxy_total)

    batch_ratio = None
    if baseline_rows is not None and batch_var is not None and base_batch_var is not None:
        batch_ratio = _ratio(batch_var, base_batch_var)

    out_path = Path(out_path)
    out_base = out_path.with_suffix("") if out_path.suffix else out_path
    out_base.parent.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    def _save(fig: plt.Figure, suffix: str) -> None:
        out_file = out_base.parent / f"{out_base.name}__{suffix}.pdf"
        fig.tight_layout()
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_file)

    full_only = bool(baseline_full_only)
    if full_only:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xs, sigma_num_ratio_full, marker="o", linewidth=1.6, color="tab:blue", label=r"$\sigma_{g,t}^2$ (action)")
        ax.plot(xs, vprob_num_ratio_full, marker="o", linewidth=1.6, color="tab:orange", label=r"$V_{\mathrm{prob},t}$ (problem)")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel(r"Variance Ratio (curl / baseline)", fontsize=plot_label_size)
        ax.set_xlabel(plot_xlabel, fontsize=plot_label_size)
        ax.tick_params(axis="both", labelsize=plot_tick_size)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.legend(loc="best", fontsize=plot_legend_size)
        ax.grid(True, alpha=0.3)
        _save(fig, "sigma_vprob")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        if sigma_num_ratio_kept is not None:
            ax.plot(xs, sigma_num_ratio_kept, marker="o", linewidth=1.6, color="tab:blue", label="vs baseline kept@β")
            ax.plot(
                xs,
                sigma_num_ratio_full,
                marker="o",
                linewidth=1.2,
                linestyle="--",
                alpha=0.55,
                color="tab:blue",
                label="vs baseline full",
            )
            ax.legend(loc="best", fontsize=plot_legend_size)
        else:
            ax.plot(xs, sigma_num_ratio_full, marker="o", linewidth=1.5)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel(r"$\sigma_{g,t}^2$ Ratio (curl / baseline)", fontsize=plot_label_size)
        ax.set_xlabel(plot_xlabel, fontsize=plot_label_size)
        ax.tick_params(axis="both", labelsize=plot_tick_size)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(True, alpha=0.3)
        _save(fig, "sigma")

        fig, ax = plt.subplots(figsize=(10, 4))
        if vprob_num_ratio_kept is not None:
            ax.plot(xs, vprob_num_ratio_kept, marker="o", linewidth=1.6, color="tab:orange", label="vs baseline kept@β")
            ax.plot(xs, vprob_num_ratio_full, marker="o", linewidth=1.2, linestyle="--", alpha=0.55, color="tab:orange", label="vs baseline full")
            ax.legend(loc="best", fontsize=plot_legend_size)
        else:
            ax.plot(xs, vprob_num_ratio_full, marker="o", linewidth=1.5, color="tab:orange")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel(r"$V_{\mathrm{prob},t}$ Ratio (curl / baseline)", fontsize=plot_label_size)
        ax.set_xlabel(plot_xlabel, fontsize=plot_label_size)
        ax.tick_params(axis="both", labelsize=plot_tick_size)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(True, alpha=0.3)
        _save(fig, "vprob")

    fig, ax = plt.subplots(figsize=(10, 4))
    if proxy_ratio_kept is not None and not full_only:
        ax.plot(xs, proxy_ratio_kept, marker="o", linewidth=2.0, color="tab:purple", label="vs baseline kept@β")
        ax.plot(xs, proxy_ratio_full, marker="o", linewidth=1.4, linestyle="--", alpha=0.55, color="tab:purple", label="vs baseline full")
        ax.legend(loc="best", fontsize=plot_legend_size)
    else:
        label = "vs baseline full" if full_only and baseline_rows is not None else None
        ax.plot(xs, proxy_ratio_full, marker="o", linewidth=1.8, color="tab:purple", label=label)
        if label:
            ax.legend(loc="best", fontsize=plot_legend_size)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel(r"Variance Proxy Ratio (curl / baseline)", fontsize=plot_label_size)
    ax.set_xlabel(plot_xlabel, fontsize=plot_label_size)
    ax.tick_params(axis="both", labelsize=plot_tick_size)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, alpha=0.3)
    _save(fig, "proxy")

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(xs, betas, linewidth=1.8, color="tab:blue", label=r"$\beta$ target")
    if beta_actual is not None and np.any(np.isfinite(beta_actual)):
        ax.plot(xs, beta_actual, linewidth=1.4, linestyle="--", color="tab:blue", alpha=0.7, label=r"$\beta$ actual (eval)")
    ax.set_ylabel(r"$\beta$", fontsize=plot_label_size)
    ax.set_xlabel(plot_xlabel, fontsize=plot_label_size)
    ax.tick_params(axis="both", labelsize=plot_tick_size)
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, alpha=0.3)

    if tau is not None and np.any(np.isfinite(tau)):
        ax_tau = ax.twinx()
        ax_tau.plot(xs, tau, linewidth=1.8, color="tab:red", label=r"$\tau$ (threshold)")
        ax_tau.set_ylabel(r"$\tau$", fontsize=plot_label_size)
        ax_tau.tick_params(axis="y", labelsize=plot_tick_size)
        ax_tau.yaxis.set_major_locator(MaxNLocator(nbins=6))
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_tau.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=plot_legend_size)
    else:
        ax.legend(loc="best", fontsize=plot_legend_size)
    _save(fig, "schedule")

    if batch_var is not None:
        fig, ax = plt.subplots(figsize=(10, 3.6))
        if batch_ratio is not None:
            ax.plot(xs, batch_ratio, marker="o", linewidth=1.5, color="tab:green")
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            ax.set_ylabel(r"Batch $\|g\|$ Var Ratio (curl / baseline)", fontsize=plot_label_size)
        else:
            ax.plot(xs, batch_var, marker="o", linewidth=1.5, color="tab:green")
            ax.set_ylabel(r"Var$_{\mathrm{seed}}(\|g_{\mathrm{batch}}\|^2)$", fontsize=plot_label_size)
        ax.set_xlabel(plot_xlabel, fontsize=plot_label_size)
        ax.tick_params(axis="both", labelsize=plot_tick_size)
        ax.grid(True, alpha=0.3)
        _save(fig, "batch_var")

    return saved


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True, help="Run dir containing global_step_* checkpoints")
    ap.add_argument("--baseline_run_dir", type=Path, default=None, help="Baseline run dir (no curriculum) for ratios")

    ap.add_argument("--prompt_file", type=Path, required=True, help="Prompt source: .jsonl/.json/.txt/.parquet")
    ap.add_argument("--prompt_field", type=str, default="prompt", help="Field name for JSON/Parquet prompts")
    ap.add_argument("--prompt_type", type=str, default="raw",
                    help="LLM_EVAL prompt template name (e.g. think-boxed). Use 'raw' to disable.")
    ap.add_argument("--compute_passk", action="store_true", default=False,
                    help="Also compute pass@k on kept/dropped prompts using the same rollouts.")
    ap.add_argument("--data_name", type=str, default="custom",
                    help="Dataset name for pass@k parsing (use 'custom' to read prompt/answer fields).")
    ap.add_argument("--answer_field", type=str, default="answer",
                    help="Answer field name for --data_name=custom.")
    ap.add_argument("--passk_ks", type=str, default="1,8",
                    help="Comma-separated k list for pass@k (e.g., '1,4,8').")
    ap.add_argument("--num_prompts", type=int, default=512)
    ap.add_argument("--num_prompt_ratio", type=float, default=0.0)
    ap.add_argument("--subset_mode", type=str, default="random", choices=["random", "first"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_prompt_len", type=int, default=0, help="Filter prompts longer than this token length.")
    ap.add_argument("--tokenizer_dir", type=Path, default=None, help="Tokenizer path for prompt length filtering.")

    ap.add_argument("--k_rollouts", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--sampling_mode", type=str, default="topk", choices=["topk", "full"],
                    help="How to sample next tokens. 'topk' samples from the same --topk subset (fast). "
                         "'full' samples from full softmax (slow, closer to true sampling).")
    ap.add_argument("--prob_mode", type=str, default="topk", choices=["topk", "full"],
                    help="How to compute probabilities/logp for sketch/confidence. 'topk' normalizes within --topk "
                         "(fast). 'full' uses logsumexp over full vocab (slow, more accurate).")
    ap.add_argument("--gen_batch_size", type=int, default=1, help="Prompts per generate() call")
    ap.add_argument("--shard_load_workers", type=int, default=4,
                    help="Threads for loading FSDP shard .pt files (transformers engine only).")
    ap.add_argument("--empty_cache_each_batch", action="store_true", default=False,
                    help="Call torch.cuda.empty_cache() after each prompt batch (slower; only if needed for VRAM).")
    ap.add_argument("--inference_mode", action="store_true", default=True,
                    help="Use torch.inference_mode() for faster eval (default: True).")
    ap.add_argument("--no_inference_mode", action="store_false", dest="inference_mode")
    ap.add_argument("--enable_tf32", action="store_true", default=True,
                    help="Enable TF32 matmul on CUDA for speed (default: True).")
    ap.add_argument("--no_enable_tf32", action="store_false", dest="enable_tf32")
    ap.add_argument("--sanitize_logits", action="store_true", default=True,
                    help="Sanitize NaN/Inf logits during sampling to avoid multinomial crashes.")
    ap.add_argument("--no_sanitize_logits", action="store_false", dest="sanitize_logits")

    ap.add_argument("--proj_dim", type=int, default=256, help="Sketch dimension for gradient vectors")
    ap.add_argument("--topk", type=int, default=32, help="Top-k logits used to approximate softmax gradient")
    # Match VI-CURL training defaults (examples/curl_v6.sh): confidence_metric=neg_entropy unless overridden.
    ap.add_argument("--confidence_metric", type=str, default="neg_entropy", choices=["avg_logp", "neg_entropy"])

    ap.add_argument("--start_beta", type=float, default=0.2)
    ap.add_argument("--end_beta", type=float, default=1.0)
    ap.add_argument("--schedule_steps", type=int, default=8000)
    ap.add_argument(
        "--beta_schedule_mode",
        type=str,
        default="train",
        choices=["train", "rescaled"],
        help="How to map checkpoint step -> beta_target. "
             "'train': beta_target = start_beta + (end_beta-start_beta)*min(step/schedule_steps, 1). "
             "'rescaled': ignore schedule_steps; linearly map the first/last reference checkpoints "
             "to start_beta/end_beta and space intermediate checkpoints uniformly by checkpoint index "
             "(use --beta_ref_steps to define the reference set; defaults to all available steps for sharded runs).",
    )
    ap.add_argument(
        "--beta_ref_steps",
        type=str,
        default=None,
        help="Reference step list for --beta_schedule_mode=rescaled. Same format as --steps (e.g. '20-313:20' or '20,40,60'). "
             "If omitted, uses all available steps under --run_dir for sharded runs, otherwise uses the selected --steps.",
    )
    ap.add_argument("--beta_source", type=str, default="compute", choices=["compute", "log"])
    ap.add_argument("--beta_log_path", type=Path, default=None, help="Optional wandb output.log path; default uses latest-run.")
    ap.add_argument("--beta_log_root", type=Path, default=None, help="Optional wandb root (e.g., project/VI-CURL/wandb).")
    ap.add_argument("--use_log_tau", action="store_true", default=True)
    ap.add_argument("--no_use_log_tau", action="store_false", dest="use_log_tau")

    ap.add_argument("--steps", type=str, default=None, help="Subset of steps: '20,40' or '20-300:20'")

    ap.add_argument("--batch_var_seeds", type=str, default=None, help="Comma-separated seeds for batch ||g|| variance")
    ap.add_argument("--batch_var_num_prompts", type=int, default=128)
    ap.add_argument("--batch_var_k", type=int, default=1)

    ap.add_argument("--out_dir", type=Path, default=Path("project/LLM_EVAL/eval_log/vi_curl/grad_variance"))
    ap.add_argument("--tag", type=str, default=None,
                    help="Optional output tag (for multi-process partial runs).")
    ap.add_argument("--part_id", type=int, default=0,
                    help="Part id for partial runs (0-indexed). Used together with --num_parts.")
    ap.add_argument("--num_parts", type=int, default=1,
                    help="Total parts for partial runs. If >1, output filename includes __part<id>.")
    ap.add_argument("--no_plot", action="store_true", default=False,
                    help="Skip plotting (useful for partial runs; merge later).")
    ap.add_argument("--plot_baseline_full_only", action="store_true", default=False,
                    help="Only plot the vs-baseline-full curve (solid line) when baseline is available.")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Resume from a partial progress file if present (default: True).")
    ap.add_argument("--no_resume", action="store_false", dest="resume")

    ap.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    ap.add_argument("--use_vllm", action="store_true", default=True,
                    help="Use vLLM for faster generation (default: True).")
    ap.add_argument("--no_use_vllm", action="store_false", dest="use_vllm",
                    help="Disable vLLM and use transformers engine.")
    default_export_root = os.getenv("VLLM_HF_EXPORT_ROOT")
    if default_export_root:
        hf_export_root_default = Path(default_export_root)
    else:
        shm = Path("/dev/shm")
        if shm.exists() and os.access(str(shm), os.W_OK):
            hf_export_root_default = shm / "vf_curl_grad_variance_vllm_export"
        else:
            hf_export_root_default = Path(os.getenv("WORK_HOME", "/data/giil/caixq")) / "export"
    ap.add_argument(
        "--hf_export_root",
        type=Path,
        default=hf_export_root_default,
        help="Parent dir for temporary exported HF step weights in vLLM mode (default: /dev/shm if available).",
    )
    ap.add_argument("--base_model_dir", type=Path, default=None,
                    help="Base HF model dir (required when step has a LoRA adapter).")
    ap.add_argument("--export_hf_if_needed", action="store_true", default=True,
                    help="Auto-export step weights to HF under --hf_export_root if missing (default: True).")
    ap.add_argument("--no_export_hf_if_needed", action="store_false", dest="export_hf_if_needed")
    ap.add_argument(
        "--keep_hf_export",
        action="store_true",
        default=False,
        help="Keep exported HF weights under --hf_export_root (default: export to a temp dir and delete after each step).",
    )
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    ap.add_argument("--vllm_pipeline_parallel_size", type=int, default=1)
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--vllm_max_num_seqs", type=int, default=0,
                    help="vLLM max_num_seqs (0=auto). Set >= gen_batch_size*k_rollouts for best throughput.")
    add_font_size_args(ap)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    setup_plot_style()
    font_sizes = get_font_sizes(args, "vf_curl_grad_variance.py")
    global FONT_SIZES
    FONT_SIZES = font_sizes
    if args.num_parts < 1:
        raise ValueError("--num_parts must be >= 1")
    if args.part_id < 0 or args.part_id >= args.num_parts:
        raise ValueError(f"--part_id must be in [0, {args.num_parts})")

    if str(args.sampling_mode) == "full" and str(args.prob_mode) != "full":
        print("[WARN] --sampling_mode=full forces --prob_mode=full")
        args.prob_mode = "full"
    if str(args.confidence_metric) == "neg_entropy" and str(args.prob_mode) != "full":
        print("[WARN] --confidence_metric=neg_entropy is most meaningful with --prob_mode=full (topk mode underestimates entropy).")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device=cuda but CUDA not available; falling back to cpu")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if bool(args.use_vllm) and device.type != "cuda":
        raise ValueError("--use_vllm requires --device=cuda")

    if device.type == "cuda" and torch.cuda.is_available() and bool(args.enable_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    answers: Optional[List[str]] = None
    if bool(args.compute_passk):
        examples = load_examples(args.prompt_file)
        if not examples:
            raise ValueError(f"No examples loaded from {args.prompt_file}")
        prompts_all, answers = build_prompts_and_answers(
            examples,
            data_name=str(args.data_name),
            prompt_type=str(args.prompt_type),
            prompt_field=str(args.prompt_field),
            answer_field=str(args.answer_field),
        )
        prompts_all_raw = None
    else:
        prompts_all_raw = load_prompts(args.prompt_file, args.prompt_field)
        prompts_all = apply_prompt_template(prompts_all_raw, args.prompt_type)

    step_dirs = list_step_dirs(args.run_dir)
    if not step_dirs:
        raise FileNotFoundError(f"No global_step_* dirs under {args.run_dir}")

    if int(args.max_prompt_len) > 0:
        tok_dir = _resolve_tokenizer_dir(
            run_dir=args.run_dir,
            step_dirs=step_dirs,
            tokenizer_dir=args.tokenizer_dir,
            base_model_dir=args.base_model_dir,
        )
        if tok_dir is None:
            raise FileNotFoundError("max_prompt_len set but tokenizer_dir/base_model_dir not found.")
        before_count = len(prompts_all)
        prompts_all, lengths, answers = _filter_prompts_by_length(
            prompts_all,
            max_prompt_len=int(args.max_prompt_len),
            tokenizer_dir=tok_dir,
            trust_remote_code=bool(args.trust_remote_code),
            answers=answers,
        )
        if not prompts_all:
            raise ValueError(f"All prompts exceed max_prompt_len={args.max_prompt_len}.")
        print(
            f"[INFO] Prompt length filter: max_len={args.max_prompt_len} kept={len(prompts_all)} "
            f"dropped={before_count - len(prompts_all)}"
        )
        if args.tokenizer_dir is None:
            args.tokenizer_dir = tok_dir

    num_prompt_ratio = float(args.num_prompt_ratio)
    if num_prompt_ratio > 1.0:
        raise ValueError(f"--num_prompt_ratio must be in (0, 1], got {num_prompt_ratio}")
    num_prompts_eff = int(args.num_prompts)
    if num_prompt_ratio > 0.0:
        num_prompts_eff = int(round(len(prompts_all) * num_prompt_ratio))
        num_prompts_eff = max(1, min(num_prompts_eff, len(prompts_all)))
        print(f"[INFO] Prompt ratio: {num_prompt_ratio:.4f} -> num_prompts={num_prompts_eff}")

    idxs = select_prompt_subset_indices(len(prompts_all), num_prompts_eff, args.seed, args.subset_mode)
    prompts = [prompts_all[i] for i in idxs]
    if answers is not None:
        answers = [answers[i] for i in idxs]
    print(f"Loaded prompts: {len(prompts_all)} -> using {len(prompts)} (subset_mode={args.subset_mode})")
    available_steps = [_step_num_from_dir(p) for p in step_dirs]
    selected_steps = parse_steps_arg(args.steps, available_steps)
    step_dirs = [p for p in step_dirs if _step_num_from_dir(p) in set(selected_steps)]
    steps = [_step_num_from_dir(p) for p in step_dirs]
    print(f"Steps: {steps}")
    beta_mode = str(args.beta_schedule_mode)
    beta_ref_steps: List[int] = []
    if beta_mode == "rescaled":
        if args.beta_ref_steps:
            beta_ref_steps = parse_steps_arg(str(args.beta_ref_steps), available_steps)
        else:
            # In sharded mode each worker only sees a subset of --steps, so default to all available steps
            # under --run_dir to keep beta_target consistent across parts.
            beta_ref_steps = list(available_steps) if int(args.num_parts) > 1 else list(selected_steps)
        if not beta_ref_steps:
            beta_ref_steps = list(selected_steps) if selected_steps else list(available_steps)
    elif beta_mode != "train":
        raise ValueError(f"Unknown --beta_schedule_mode: {beta_mode}")

    beta_by_step: Dict[int, float] = {}
    if beta_mode == "train":
        beta_by_step = {
            int(step): compute_beta_target(int(step), float(args.start_beta), float(args.end_beta), int(args.schedule_steps))
            for step in steps
        }
    else:
        beta_by_step = {
            int(step): compute_beta_target_rescaled(int(step), float(args.start_beta), float(args.end_beta), beta_ref_steps)
            for step in steps
        }

    if steps:
        step_min = int(min(steps))
        step_max = int(max(steps))
        beta_min = float(beta_by_step.get(step_min, float("nan")))
        beta_max = float(beta_by_step.get(step_max, float("nan")))
        if beta_mode == "train":
            print(
                f"[INFO] beta_target schedule_mode=train: start_beta={float(args.start_beta):.4f} "
                f"end_beta={float(args.end_beta):.4f} schedule_steps={int(args.schedule_steps)}"
            )
            print(f"[INFO] selected_steps range: {step_min}..{step_max} => beta_target {beta_min:.4f}..{beta_max:.4f}")
            if int(args.schedule_steps) > 0:
                progress_max = float(step_max) / float(int(args.schedule_steps))
                if progress_max < 0.2 and abs(beta_max - beta_min) < 0.05:
                    print(
                        f"[INFO] Note: max_step/schedule_steps={progress_max:.4f}; "
                        "beta_target will stay close to start_beta for these checkpoints (expected)."
                    )
        else:
            ref_min = int(min(beta_ref_steps)) if beta_ref_steps else step_min
            ref_max = int(max(beta_ref_steps)) if beta_ref_steps else step_max
            print(
                f"[INFO] beta_target schedule_mode=rescaled: start_beta={float(args.start_beta):.4f} "
                f"end_beta={float(args.end_beta):.4f} ref_steps={len(beta_ref_steps)} ({ref_min}..{ref_max})"
            )
            print(f"[INFO] selected_steps range: {step_min}..{step_max} => beta_target {beta_min:.4f}..{beta_max:.4f}")

    beta_log: Dict[int, Dict[str, float]] = {}
    if str(args.beta_source) == "log":
        log_path = _resolve_beta_log_path(
            run_dir=args.run_dir,
            beta_log_path=args.beta_log_path,
            beta_log_root=args.beta_log_root,
        )
        log_path = log_path.expanduser().resolve()
        if not log_path.exists():
            raise FileNotFoundError(f"beta_source=log but log file not found: {log_path}")
        beta_log = _load_beta_from_log(log_path)
        print(f"[INFO] Loaded beta from log: {log_path} (steps={len(beta_log)})")

    baseline_step_dirs: Optional[List[Path]] = None
    baseline_steps: Optional[List[int]] = None
    if args.baseline_run_dir is not None:
        baseline_step_dirs = list_step_dirs(args.baseline_run_dir)
        baseline_steps = [_step_num_from_dir(p) for p in baseline_step_dirs]
        baseline_step_dirs = [p for p in baseline_step_dirs if _step_num_from_dir(p) in set(steps)]
        if len(baseline_step_dirs) != len(step_dirs):
            print("[WARN] baseline_run_dir missing some steps; ratios will use same-run full baseline for missing ones")

    cfg = AnalysisConfig(
        run_dir=str(args.run_dir),
        baseline_run_dir=str(args.baseline_run_dir) if args.baseline_run_dir is not None else None,
        prompt_file=str(args.prompt_file),
        prompt_field=str(args.prompt_field),
        prompt_type=str(args.prompt_type),
        num_prompts=int(args.num_prompts),
        num_prompt_ratio=float(args.num_prompt_ratio),
        subset_mode=str(args.subset_mode),
        seed=int(args.seed),
        max_prompt_len=int(args.max_prompt_len),
        tokenizer_dir=str(args.tokenizer_dir) if args.tokenizer_dir else None,
        k_rollouts=int(args.k_rollouts),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        sampling_mode=str(args.sampling_mode),
        prob_mode=str(args.prob_mode),
        sanitize_logits=bool(args.sanitize_logits),
        proj_dim=int(args.proj_dim),
        topk=int(args.topk),
        confidence_metric=str(args.confidence_metric),
        start_beta=float(args.start_beta),
        end_beta=float(args.end_beta),
        schedule_steps=int(args.schedule_steps),
        beta_source=str(args.beta_source),
        beta_log_path=str(args.beta_log_path) if args.beta_log_path else None,
        beta_log_root=str(args.beta_log_root) if args.beta_log_root else None,
        use_log_tau=bool(args.use_log_tau),
        gen_batch_size=int(args.gen_batch_size),
        steps=str(args.steps) if args.steps else None,
        batch_var_seeds=str(args.batch_var_seeds) if args.batch_var_seeds else None,
        batch_var_num_prompts=int(args.batch_var_num_prompts),
        batch_var_k=int(args.batch_var_k),
        compute_passk=bool(args.compute_passk),
        data_name=str(args.data_name),
        answer_field=str(args.answer_field),
        passk_ks=str(args.passk_ks),
    )
    analysis_hash_obj = asdict(cfg)
    # Baseline policy matters for ratio semantics:
    # - "kept_by_beta": compare kept-vs-kept against baseline run at the same beta_target
    # - "same_run_full": compare kept-vs-full within the same run (no baseline provided)
    analysis_hash_obj["baseline_policy"] = "kept_by_beta" if cfg.baseline_run_dir else "same_run_full"
    if bool(args.use_vllm):
        analysis_hash_obj["engine"] = "vllm"
    beta_source = str(analysis_hash_obj.get("beta_source") or "compute").strip().lower()
    beta_log_path = str(analysis_hash_obj.get("beta_log_path") or "").strip()
    beta_log_root = str(analysis_hash_obj.get("beta_log_root") or "").strip()
    if beta_source in ("compute", "") and beta_log_path == "" and beta_log_root == "":
        analysis_hash_obj.pop("beta_source", None)
        analysis_hash_obj.pop("beta_log_path", None)
        analysis_hash_obj.pop("beta_log_root", None)
        analysis_hash_obj.pop("use_log_tau", None)
    if int(analysis_hash_obj.get("max_prompt_len") or 0) <= 0 and not analysis_hash_obj.get("tokenizer_dir"):
        analysis_hash_obj.pop("max_prompt_len", None)
        analysis_hash_obj.pop("tokenizer_dir", None)
    if not bool(cfg.compute_passk):
        analysis_hash_obj.pop("compute_passk", None)
        analysis_hash_obj.pop("data_name", None)
        analysis_hash_obj.pop("answer_field", None)
        analysis_hash_obj.pop("passk_ks", None)
    # Keep backward-compatible cache keys for the default ("train") mapping:
    # only include these keys when explicitly using rescaled mode.
    if str(args.beta_schedule_mode) != "train":
        analysis_hash_obj["beta_schedule_mode"] = str(args.beta_schedule_mode)
        analysis_hash_obj["beta_ref_steps"] = beta_ref_steps
    analysis_id = _sha1_json(analysis_hash_obj)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_key = Path(cfg.run_dir).name
    tag = None
    if args.tag:
        tag = _tag_with_beta_mode(str(args.tag), beta_mode=beta_mode)
        if tag != str(args.tag):
            print(f"[INFO] tag auto-suffixed for beta_schedule_mode: {args.tag} -> {tag}")
        if int(args.num_parts) > 1:
            base = f"vf_curl_grad_variance__{run_key}__{tag}__part{int(args.part_id)}"
        else:
            base = f"vf_curl_grad_variance__{run_key}__{tag}"
    else:
        base = f"vf_curl_grad_variance__{run_key}__{analysis_id}"
    json_path = out_dir / f"{base}.json"
    fig_base = out_dir / f"{base}"
    progress_path = out_dir / f"{base}.partial.json"

    if json_path.exists():
        try:
            payload = _read_json(json_path)
            if str(payload.get("analysis_id", "")) == str(analysis_id):
                print(f"[CACHE] Loading {json_path}")
                if not args.no_plot:
                    rows = payload["rows"]
                    base_rows = payload.get("baseline_rows")
                    saved = plot_results(
                        steps=payload["steps"],
                        rows=rows,
                        baseline_rows=base_rows,
                        out_path=fig_base,
                        title=payload.get("title", ""),
                        baseline_full_only=bool(args.plot_baseline_full_only),
                    )
                    for p in saved:
                        print(f"[OK] Plot saved to {p}")
                return
            print(
                f"[CACHE] Found {json_path} but analysis_id mismatch; recomputing.\n"
                f"  cached={payload.get('analysis_id')}\n"
                f"  current={analysis_id}"
            )
        except Exception as e:
            print(f"[WARN] Failed to read cache {json_path}: {e}; recomputing.")

    rows: List[Dict[str, Any]] = []
    baseline_by_step: Dict[int, Dict[str, Any]] = {}
    if bool(args.resume) and progress_path.exists():
        try:
            partial = _read_json(progress_path)
            if partial.get("analysis_id") == analysis_id:
                rows = list(partial.get("rows") or [])
                raw_b = partial.get("baseline_by_step") or {}
                if isinstance(raw_b, dict):
                    baseline_by_step = {int(k): v for k, v in raw_b.items()}
                done = sorted({int(r.get("step", -1)) for r in rows if isinstance(r, dict) and "step" in r and int(r.get("step", -1)) >= 0})
                print(f"[RESUME] Loaded {len(rows)} computed steps from {progress_path}")
                if done:
                    print(f"[RESUME] Done steps (first 10): {done[:10]}")
            else:
                print(f"[WARN] Found {progress_path} but analysis_id mismatch; ignoring.")
        except Exception as e:
            print(f"[WARN] Failed to load resume file {progress_path}: {e}; starting fresh.")

    base_seeds: List[int] = []
    if args.batch_var_seeds:
        base_seeds = [int(x) for x in args.batch_var_seeds.split(",") if x.strip()]
    passk_ks = _parse_ks(str(args.passk_ks)) if bool(args.compute_passk) else []
    if bool(args.compute_passk) and not passk_ks:
        passk_ks = [1]

    # batch-var prompts subset (deterministic)
    batch_var_prompts = prompts
    if args.batch_var_num_prompts > 0 and args.batch_var_num_prompts < len(prompts):
        batch_var_prompts = select_prompt_subset(prompts, args.batch_var_num_prompts, args.seed + 12345, "random")

    done_steps = {int(r["step"]) for r in rows if isinstance(r, dict) and "step" in r}

    # Initialize async exporter for pipeline prefetching (exports next checkpoint while current one is running)
    async_exporter: Optional[AsyncExporter] = None
    if bool(args.use_vllm) and args.hf_export_root is not None:
        async_exporter = AsyncExporter(max_workers=1)

    start_time = time.time()
    for step_idx, step_dir in enumerate(step_dirs):
        # Pipeline prefetch: start exporting next checkpoint if available
        if async_exporter is not None and step_idx + 1 < len(step_dirs):
            next_step_dir = step_dirs[step_idx + 1]
            async_exporter.prefetch(
                next_step_dir,
                hf_export_root=args.hf_export_root,
                base_model_dir=args.base_model_dir,
                export_hf_if_needed=bool(args.export_hf_if_needed),
            )

        step = _step_num_from_dir(step_dir)
        beta = float(beta_by_step.get(int(step), float("nan")))
        if not np.isfinite(beta):
            beta = compute_beta_target(step, cfg.start_beta, cfg.end_beta, cfg.schedule_steps)
        beta_target_log = None
        beta_actual_log = None
        tau_log = None
        tau_override = None
        if str(args.beta_source) == "log":
            slot = beta_log.get(int(step), {})
            if "beta_target" in slot:
                beta_target_log = float(slot["beta_target"])
                beta = beta_target_log
            if "beta_actual" in slot:
                beta_actual_log = float(slot["beta_actual"])
            if "tau" in slot:
                tau_log = float(slot["tau"])
            if bool(args.use_log_tau) and tau_log is not None:
                tau_override = float(tau_log)
        print(f"\n=== Step {step} (beta_target={beta:.4f}) ===")

        if int(step) not in done_steps:
            m = compute_step_metrics(
                step_dir=step_dir,
                prompts=prompts,
                answers=answers,
                data_name=str(args.data_name),
                prompt_type=str(args.prompt_type),
                device=device,
                dtype=dtype,
                k_rollouts=cfg.k_rollouts,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                sampling_mode=cfg.sampling_mode,
                prob_mode=cfg.prob_mode,
                proj_dim=cfg.proj_dim,
                topk=cfg.topk,
                confidence_metric=cfg.confidence_metric,
                beta_target=beta,
                tau_override=tau_override,
                sketch_seed=cfg.seed,
                gen_batch_size=cfg.gen_batch_size,
                sanitize_logits=bool(args.sanitize_logits),
                progress_desc=f"{Path(cfg.run_dir).name}@{step}",
                trust_remote_code=bool(args.trust_remote_code),
                shard_load_workers=int(args.shard_load_workers),
                empty_cache_each_batch=bool(args.empty_cache_each_batch),
                inference_mode=bool(args.inference_mode),
                use_vllm=bool(args.use_vllm),
                hf_export_root=args.hf_export_root,
                base_model_dir=args.base_model_dir,
                export_hf_if_needed=bool(args.export_hf_if_needed),
                keep_hf_export=bool(args.keep_hf_export),
                vllm_tensor_parallel_size=int(args.vllm_tensor_parallel_size),
                vllm_pipeline_parallel_size=int(args.vllm_pipeline_parallel_size),
                vllm_gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
                vllm_max_num_seqs=int(args.vllm_max_num_seqs),
                batch_var_seeds=base_seeds,
                batch_var_prompts=batch_var_prompts,
                batch_var_k=cfg.batch_var_k,
                compute_passk=bool(cfg.compute_passk),
                passk_ks=passk_ks,
            )

            m["step"] = int(step)
            if str(args.beta_source) == "log":
                m["beta_target_log"] = beta_target_log
                m["beta_actual_log"] = beta_actual_log
                m["tau_log"] = tau_log
            rows.append(m)
            done_steps.add(int(step))

            _atomic_write_json(
                progress_path,
                {
                    "analysis_id": analysis_id,
                    "engine": "vllm" if bool(args.use_vllm) else "transformers",
                    "tag": args.tag,
                    "part_id": int(args.part_id),
                    "num_parts": int(args.num_parts),
                    "config": asdict(cfg),
                    "steps": steps,
                    "rows": rows,
                    "baseline_by_step": baseline_by_step,
                },
            )
            print(f"[RESUME] Saved progress: {progress_path} (rows={len(rows)})")
        else:
            print(f"[RESUME] Skip step={step} (already computed)")

        if baseline_step_dirs is not None:
            base_step_dir = next((p for p in baseline_step_dirs if _step_num_from_dir(p) == step), None)
            if base_step_dir is not None and int(step) not in baseline_by_step:
                b = compute_step_metrics(
                    step_dir=base_step_dir,
                    prompts=prompts,
                    answers=None,
                    data_name=str(args.data_name),
                    prompt_type=str(args.prompt_type),
                    device=device,
                    dtype=dtype,
                    k_rollouts=cfg.k_rollouts,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    sampling_mode=cfg.sampling_mode,
                    prob_mode=cfg.prob_mode,
                    proj_dim=cfg.proj_dim,
                    topk=cfg.topk,
                    confidence_metric=cfg.confidence_metric,
                    beta_target=beta,  # baseline: use the same retention rate for a fair kept-vs-kept comparison
                    tau_override=None,
                    sketch_seed=cfg.seed,
                    gen_batch_size=cfg.gen_batch_size,
                    sanitize_logits=bool(args.sanitize_logits),
                    progress_desc=f"{Path(cfg.baseline_run_dir).name}@{step}",
                    trust_remote_code=bool(args.trust_remote_code),
                    shard_load_workers=int(args.shard_load_workers),
                    empty_cache_each_batch=bool(args.empty_cache_each_batch),
                    inference_mode=bool(args.inference_mode),
                    use_vllm=bool(args.use_vllm),
                    hf_export_root=args.hf_export_root,
                    base_model_dir=args.base_model_dir,
                    export_hf_if_needed=bool(args.export_hf_if_needed),
                    keep_hf_export=bool(args.keep_hf_export),
                    vllm_tensor_parallel_size=int(args.vllm_tensor_parallel_size),
                    vllm_pipeline_parallel_size=int(args.vllm_pipeline_parallel_size),
                    vllm_gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
                    vllm_max_num_seqs=int(args.vllm_max_num_seqs),
                    batch_var_seeds=base_seeds,
                    batch_var_prompts=batch_var_prompts,
                    batch_var_k=cfg.batch_var_k,
                    compute_passk=False,
                    passk_ks=(),
                )
                b["step"] = int(step)
                baseline_by_step[int(step)] = b
                _atomic_write_json(
                    progress_path,
                    {
                        "analysis_id": analysis_id,
                        "engine": "vllm" if bool(args.use_vllm) else "transformers",
                        "tag": args.tag,
                        "part_id": int(args.part_id),
                        "num_parts": int(args.num_parts),
                        "config": asdict(cfg),
                        "steps": steps,
                        "rows": rows,
                        "baseline_by_step": baseline_by_step,
                    },
                )
                print(f"[RESUME] Saved progress: {progress_path} (rows={len(rows)}, baseline={len(baseline_by_step)})")

    elapsed = time.time() - start_time
    print(f"\n[Done] computed {len(rows)} steps in {elapsed/60:.1f} min")

    # Shutdown async exporter
    if async_exporter is not None:
        async_exporter.shutdown(wait=True)

    # Ensure deterministic order.
    rows = sorted(rows, key=lambda r: int(r.get("step", -1)))

    title = Path(cfg.run_dir).name
    if cfg.baseline_run_dir:
        title += f" vs {Path(cfg.baseline_run_dir).name}"

    baseline_rows: Optional[List[Dict[str, Any]]] = None
    if cfg.baseline_run_dir:
        baseline_rows = []
        row_by_step: Dict[int, Dict[str, Any]] = {int(r.get("step", -1)): r for r in rows if isinstance(r, dict) and "step" in r}
        for i, step in enumerate(steps):
            b = baseline_by_step.get(int(step))
            # If missing, fallback to same-run "full data" baseline for that step.
            if b is not None:
                baseline_rows.append(b)
                continue
            r = row_by_step.get(int(step), rows[i])
            # Convert same-run full metrics into a "baseline_kept" surrogate so plotting
            # logic can consistently use sigma_kept/vprob_kept denominators.
            rr = dict(r)
            rr["beta_target"] = float(rr.get("beta_target")) if rr.get("beta_target") is not None else float("nan")
            rr["tau"] = float("nan")
            rr["kept"] = int(rr.get("num_prompts", rr.get("kept", 0)))
            rr["sigma_kept"] = float(rr.get("sigma_full", float("nan")))
            rr["vprob_kept"] = float(rr.get("vprob_full", float("nan")))
            rr["gbar_norm2_kept"] = float(rr.get("gbar_norm2_full", float("nan")))
            baseline_rows.append(rr)

    cfg_payload = asdict(cfg)
    cfg_payload["beta_schedule_mode"] = beta_mode
    cfg_payload["beta_ref_steps"] = beta_ref_steps if beta_mode == "rescaled" else None

    payload: Dict[str, Any] = {
        "analysis_id": analysis_id,
        "engine": "vllm" if bool(args.use_vllm) else "transformers",
        "tag": tag if args.tag else None,
        "part_id": int(args.part_id),
        "num_parts": int(args.num_parts),
        "config": cfg_payload,
        "title": title,
        "steps": steps,
        "rows": rows,
        "baseline_rows": baseline_rows,
    }
    _atomic_write_json(json_path, payload)
    if progress_path.exists():
        try:
            progress_path.unlink()
        except Exception:
            pass

    if not args.no_plot:
        saved = plot_results(
            steps=steps,
            rows=rows,
            baseline_rows=baseline_rows,
            out_path=fig_base,
            title=title,
            baseline_full_only=bool(args.plot_baseline_full_only),
        )
        for p in saved:
            print(f"[OK] Saved plot:    {p}")
    print(f"[OK] Saved metrics: {json_path}")

    if bool(cfg.compute_passk) and int(args.num_parts) <= 1:
        has_passk = any(isinstance(r, dict) and "pass_at_k_kept" in r for r in rows)
        if has_passk:
            passk_tag = str(tag) if tag is not None else str(analysis_id)
            passk_json = out_dir / f"vi_curl_passk__{run_key}__{passk_tag}.json"
            passk_base = out_dir / f"vi_curl_passk__{run_key}__{passk_tag}"
            passk_payload = {
                "run_name": run_key,
                "tag": passk_tag,
                "num_parts": int(args.num_parts),
                "config": cfg_payload,
                "steps": steps,
                "rows": rows,
            }
            _atomic_write_json(passk_json, passk_payload)

            ks = _parse_ks(str(cfg.passk_ks))
            if not ks:
                sample = rows[0].get("pass_at_k_kept", {}) if rows else {}
                if isinstance(sample, dict):
                    for k in sample.keys():
                        if str(k).isdigit():
                            ks.append(int(k))
            ks = sorted(set(ks)) or [1]

            if not args.no_plot:
                eval_root = Path(__file__).resolve().parents[2]
                if str(eval_root) not in sys.path:
                    sys.path.insert(0, str(eval_root))
                from tools.vi_curl_plot.vi_curl_passk_kept_dropped import _plot_passk  # type: ignore

                saved = _plot_passk(steps=steps, rows=rows, ks=ks, out_path=passk_base, title=f"{run_key} kept vs dropped pass@k")
                for p in saved:
                    print(f"[OK] Saved passk plot: {p}")
            print(f"[OK] Saved passk json: {passk_json}")


if __name__ == "__main__":
    main()
