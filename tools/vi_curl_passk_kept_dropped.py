#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute pass@k on curriculum kept/dropped subsets across checkpoints.

Workflow:
  - Load prompts + ground truth from a JSONL dataset (default: math500).
  - For each checkpoint step, generate k rollouts and compute:
      * per-prompt confidence (avg_logp or neg_entropy)
      * per-prompt correctness for pass@k
  - Use beta schedule to split kept vs dropped prompts (by confidence quantile).
  - Report pass@k on kept vs dropped and render plots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
except Exception:
    pass

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from evaluate import _compute_pass_at_k  # type: ignore
from grader import math_equal  # type: ignore
from parser import parse_ground_truth, parse_question, run_execute  # type: ignore
from python_executor import PythonExecutor  # type: ignore
from utils import construct_prompt, load_jsonl  # type: ignore

try:
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:
    LLM = None
    SamplingParams = None

# Avoid noisy tokenizers fork warnings (vLLM uses multiprocessing).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from tools.export_fsdp_dtensor_to_hf import export_one_step_to_hf  # type: ignore
except Exception:
    export_one_step_to_hf = None


_STEP_RE = re.compile(r"global_step_(\d+)$")
_NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _step_num_from_dir(p: Path) -> int:
    m = _STEP_RE.match(p.name)
    return int(m.group(1)) if m else -1


def list_step_dirs(run_dir: Path) -> List[Path]:
    return sorted([p for p in run_dir.glob("global_step_*") if p.is_dir()], key=_step_num_from_dir)


def parse_steps_spec(steps: str, available: Sequence[int]) -> List[int]:
    s = (steps or "auto").strip().lower()
    if s in {"auto", "all", "*"}:
        return list(available)
    if "," in s:
        wanted: List[int] = []
        for part in s.split(","):
            part = part.strip()
            if part.isdigit():
                wanted.append(int(part))
        return [x for x in available if x in set(wanted)]
    if "-" in s:
        if ":" in s:
            r, stride_s = s.split(":", 1)
            stride = int(stride_s)
        else:
            r, stride = s, 1
        lo_s, hi_s = r.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        wanted = set(range(lo, hi + 1, stride))
        return [x for x in available if x in wanted]
    if s.isdigit():
        v = int(s)
        return [x for x in available if x == v]
    raise ValueError(f"Unrecognized --steps format: {steps}")


def compute_beta_target(step: int, start_beta: float, end_beta: float, schedule_steps: int) -> float:
    progress = min(float(step) / max(int(schedule_steps), 1), 1.0)
    return float(start_beta + (end_beta - start_beta) * progress)


def compute_beta_target_rescaled(step: int, start_beta: float, end_beta: float, ref_steps: Sequence[int]) -> float:
    if not ref_steps:
        return float(start_beta)
    steps_sorted = sorted({int(s) for s in ref_steps})
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


def _has_hf_weights(hf_dir: Path) -> bool:
    if not hf_dir.exists():
        return False
    if any(hf_dir.glob("model*.safetensors")):
        return True
    if any(hf_dir.glob("pytorch_model*.bin")):
        return True
    if (hf_dir / "pytorch_model.bin.index.json").exists():
        return True
    return False


def _maybe_export_hf(
    step_dir: Path,
    *,
    hf_export_root: Path,
    base_model_dir: Optional[Path],
    export_hf_if_needed: bool,
) -> Tuple[Path, bool]:
    hf_dir = step_dir / "actor" / "huggingface"
    if _has_hf_weights(hf_dir):
        return hf_dir, False
    if not export_hf_if_needed:
        raise FileNotFoundError(f"No HF weights under {hf_dir} and export disabled.")
    if export_one_step_to_hf is None:
        raise RuntimeError("export_one_step_to_hf is unavailable.")
    base_dir = base_model_dir or hf_dir
    export_root = hf_export_root.expanduser().resolve()
    out_dir = export_one_step_to_hf(step_dir, base_dir, export_root)
    return Path(out_dir), True


def _load_examples(prompt_file: Path) -> List[Dict[str, Any]]:
    return list(load_jsonl(prompt_file))


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


def _load_tokenizer(tokenizer_dir: Path):
    if AutoTokenizer is None:
        raise RuntimeError("transformers is required for prompt length filtering.")
    return AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True, trust_remote_code=True)


def _select_subset(
    examples: List[Dict[str, Any]],
    *,
    num_prompts: int,
    subset_mode: str,
    seed: int,
) -> List[Dict[str, Any]]:
    if num_prompts <= 0 or num_prompts >= len(examples):
        return list(examples)
    if subset_mode == "first":
        return list(examples[:num_prompts])
    rng = random.Random(int(seed))
    return [examples[i] for i in rng.sample(range(len(examples)), num_prompts)]


def _build_prompts(
    examples: List[Dict[str, Any]],
    *,
    data_name: str,
    prompt_type: str,
    prompt_field: str,
    answer_field: str,
) -> Tuple[List[str], List[str]]:
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


def _filter_prompts_by_length(
    prompts: Sequence[str],
    answers: Sequence[str],
    *,
    max_prompt_len: int,
    tokenizer_dir: Path,
) -> Tuple[List[str], List[str], List[int]]:
    tokenizer = _load_tokenizer(tokenizer_dir)
    lengths: List[int] = []
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        lengths.append(int(len(ids)))
    keep_mask = [int(l) <= int(max_prompt_len) for l in lengths]
    kept_prompts = [p for p, keep in zip(prompts, keep_mask) if keep]
    kept_answers = [a for a, keep in zip(answers, keep_mask) if keep]
    kept_lengths = [l for l, keep in zip(lengths, keep_mask) if keep]
    return kept_prompts, kept_answers, kept_lengths


def _parse_max_logprobs(err_msg: str) -> Optional[int]:
    m = re.search(r"max allowed:\s*(\d+)", err_msg)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _logprob_value(x: Any) -> float:
    try:
        return float(getattr(x, "logprob"))
    except Exception:
        return float(x)


def _confidence_from_vllm_output(
    *,
    token_ids: Sequence[int],
    logprobs: Sequence[Dict[int, Any]],
    cumulative_logprob: Optional[float],
    confidence_metric: str,
    eos_id: Optional[int],
    vocab_size: int,
    entropy_topk: int,
) -> float:
    if confidence_metric == "avg_logp" and cumulative_logprob is not None:
        token_count = 0
        for y in token_ids:
            token_count += 1
            if eos_id is not None and int(y) == int(eos_id):
                break
        token_count = max(int(token_count), 1)
        return float(cumulative_logprob) / float(token_count)

    token_count = 0
    logp_sum = 0.0
    ent_sum = 0.0
    for t, y in enumerate(token_ids):
        if t >= len(logprobs):
            break
        lp_dict = logprobs[t] or {}
        y_i = int(y)
        token_count += 1
        if confidence_metric == "avg_logp":
            if y_i in lp_dict:
                logp_sum += _logprob_value(lp_dict[y_i])
            else:
                logp_sum += float("nan")
        else:
            if lp_dict:
                k_top = int(min(int(entropy_topk), int(vocab_size), len(lp_dict)))
                log_other = math.log(max(int(vocab_size) - k_top, 1))
                vals = list(lp_dict.values())
                lps = np.fromiter((_logprob_value(v) for v in vals), dtype=np.float64, count=len(vals))
                probs = np.exp(lps)
                p_sum = float(min(float(np.sum(probs)), 1.0))
                ent_top = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
                p_rest = float(max(1.0 - p_sum, 0.0))
                ent_rest = -p_rest * (math.log(max(p_rest, 1e-12)) - log_other) if p_rest > 0 else 0.0
                ent_sum += ent_top + float(ent_rest)
        if eos_id is not None and y_i == int(eos_id):
            break
    token_count = max(int(token_count), 1)
    if confidence_metric == "avg_logp":
        return float(logp_sum / float(token_count))
    return float(-(ent_sum / float(token_count)))


def _generate_with_vllm(
    *,
    model_dir: Path,
    prompts: Sequence[str],
    answers: Sequence[str],
    data_name: str,
    prompt_type: str,
    k_rollouts: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    confidence_metric: str,
    entropy_topk: int,
    gen_batch_size: int,
    do_sample: bool,
    seed: Optional[int],
    vllm_tensor_parallel_size: int,
    vllm_pipeline_parallel_size: int,
    vllm_gpu_memory_utilization: float,
    vllm_max_num_seqs: int,
) -> Tuple[np.ndarray, List[List[bool]]]:
    if LLM is None or SamplingParams is None:
        raise RuntimeError("vLLM backend requested but vLLM is not available.")

    k_eff = int(k_rollouts) if do_sample else 1
    if not do_sample and int(k_rollouts) != 1:
        print(f"[WARN] do_sample=False: forcing k_rollouts {k_rollouts} -> 1 for greedy generation.")

    llm_kwargs: Dict[str, Any] = {
        "model": str(model_dir),
        "trust_remote_code": True,
        "tensor_parallel_size": int(vllm_tensor_parallel_size),
        "pipeline_parallel_size": int(vllm_pipeline_parallel_size),
        "gpu_memory_utilization": float(vllm_gpu_memory_utilization),
    }
    if int(vllm_max_num_seqs) > 0:
        llm_kwargs["max_num_seqs"] = int(vllm_max_num_seqs)

    confidence_metric = str(confidence_metric).strip().lower()
    if confidence_metric not in {"avg_logp", "neg_entropy"}:
        raise ValueError(f"Unknown confidence_metric: {confidence_metric}")

    requested_logprobs = 1 if confidence_metric == "avg_logp" else int(max(int(entropy_topk), 1))

    def _make_params(logprobs_k: int) -> Any:
        return SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=float(temperature) if do_sample else 0.0,
            top_p=float(top_p) if do_sample else 1.0,
            n=int(k_eff),
            logprobs=int(logprobs_k),
            seed=int(seed) if seed is not None else None,
        )

    llm = None
    conf_list: List[float] = []
    score_mat: List[List[bool]] = []
    try:
        llm = LLM(**llm_kwargs)
        tok = llm.get_tokenizer()
        eos_id = getattr(tok, "eos_token_id", None)
        vocab_size = int(getattr(tok, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            try:
                vocab_size = int(len(tok))
            except Exception:
                vocab_size = 1

        params = _make_params(requested_logprobs)
        executor = PythonExecutor(get_answer_from_stdout=True)

        total = len(prompts)
        bs = int(gen_batch_size) if int(gen_batch_size) > 0 else total
        num_batches = int(math.ceil(total / float(max(bs, 1))))
        processed = 0
        start_time = time.time()
        total_in_toks = 0
        total_out_toks = 0
        pbar = tqdm(total=total, desc="Processed prompts", dynamic_ncols=True) if tqdm is not None else None
        for start in range(0, total, bs):
            end = min(total, start + bs)
            batch_idx = int(start // max(bs, 1)) + 1
            batch_prompts = list(prompts[start:end])
            batch_answers = answers[start:end]
            batch_start = time.time()
            try:
                outs = llm.generate(batch_prompts, params, use_tqdm=False)
            except ValueError as e:
                max_lp = _parse_max_logprobs(str(e))
                cur_lp = int(getattr(params, "logprobs", requested_logprobs) or requested_logprobs)
                if max_lp is not None and cur_lp > max_lp:
                    print(f"[WARN] vLLM logprobs cap: requested={cur_lp} > max_allowed={max_lp}; using {max_lp}.")
                    params = _make_params(int(max_lp))
                    outs = llm.generate(batch_prompts, params, use_tqdm=False)
                else:
                    raise
            for req, gt in zip(outs, batch_answers):
                outputs = getattr(req, "outputs", None) or []
                n_out = len(outputs)
                prompt_ids = getattr(req, "prompt_token_ids", None)
                if prompt_ids is not None:
                    total_in_toks += len(prompt_ids) * max(n_out, 1)
                for out in outputs:
                    token_ids = getattr(out, "token_ids", None)
                    if token_ids is not None:
                        total_out_toks += len(token_ids)
                per_seq_conf: List[float] = []
                per_seq_scores: List[bool] = []
                for out in outputs:
                    token_ids = list(getattr(out, "token_ids", []) or [])
                    logprobs = list(getattr(out, "logprobs", []) or [])
                    cumulative_logprob = getattr(out, "cumulative_logprob", None)
                    conf_val = _confidence_from_vllm_output(
                        token_ids=token_ids,
                        logprobs=logprobs,
                        cumulative_logprob=cumulative_logprob,
                        confidence_metric=confidence_metric,
                        eos_id=eos_id,
                        vocab_size=vocab_size,
                        entropy_topk=entropy_topk,
                    )
                    text = str(getattr(out, "text", ""))
                    pred, _ = run_execute(executor, text, prompt_type, data_name, execute=False)
                    per_seq_scores.append(bool(math_equal(pred, gt)))
                    per_seq_conf.append(conf_val)
                if not per_seq_conf:
                    per_seq_conf = [float("nan")]
                conf_list.append(float(np.nanmean(np.array(per_seq_conf, dtype=np.float64))))
                score_mat.append(per_seq_scores if per_seq_scores else [False])
            processed += len(batch_prompts)
            batch_elapsed = time.time() - batch_start
            elapsed = time.time() - start_time
            eta = (elapsed / processed) * (total - processed) if processed > 0 else 0.0
            in_spd = total_in_toks / elapsed if elapsed > 0 else 0.0
            out_spd = total_out_toks / elapsed if elapsed > 0 else 0.0
            if pbar is not None:
                pbar.update(len(batch_prompts))
                pbar.set_postfix_str(
                    f"batch {batch_idx}/{num_batches} last {batch_elapsed:.1f}s eta {eta/60:.1f}m "
                    f"in {in_spd:.1f} tok/s out {out_spd:.1f} tok/s"
                )
            elif batch_idx % 10 == 0 or batch_idx == num_batches:
                print(
                    f"[gen] batch {batch_idx}/{num_batches} prompts {processed}/{total} "
                    f"last {batch_elapsed:.1f}s eta {eta/60:.1f}m "
                    f"in {in_spd:.1f} tok/s out {out_spd:.1f} tok/s"
                )
        if pbar is not None:
            pbar.close()
    finally:
        try:
            import gc
            import torch

            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel  # type: ignore
        except Exception:
            try:
                from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel  # type: ignore
            except Exception:
                destroy_model_parallel = None
        if destroy_model_parallel is not None:
            try:
                destroy_model_parallel()
            except Exception:
                pass

    return np.array(conf_list, dtype=np.float64), score_mat


def _compute_keep_mask(conf: np.ndarray, beta_target: float) -> Tuple[np.ndarray, float]:
    if conf.size < 2 or beta_target >= 1.0:
        return np.ones(conf.shape, dtype=bool), float("-inf")
    if beta_target <= 0.0:
        return np.zeros(conf.shape, dtype=bool), float("inf")
    q = 1.0 - float(beta_target)
    tau = float(np.quantile(conf, q))
    keep = conf >= tau
    return keep, tau


def _compute_keep_mask_from_tau(conf: np.ndarray, tau: float) -> np.ndarray:
    if conf.size == 0:
        return np.zeros(conf.shape, dtype=bool)
    if math.isinf(tau):
        return np.zeros(conf.shape, dtype=bool) if tau > 0 else np.ones(conf.shape, dtype=bool)
    if not np.isfinite(tau):
        return np.zeros(conf.shape, dtype=bool)
    return conf >= float(tau)


def _plot_passk(
    *,
    steps: Sequence[int],
    rows: Sequence[Dict[str, Any]],
    ks: Sequence[int],
    out_path: Path,
    title: str,
) -> None:
    def _coerce_float(value: Any) -> float:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _pick_value(row: Dict[str, Any], key: str, fallback_key: Optional[str] = None) -> float:
        if key in row and row.get(key) is not None:
            return _coerce_float(row.get(key))
        if fallback_key:
            return _coerce_float(row.get(fallback_key))
        return float("nan")

    steps_arr = np.array(list(steps), dtype=np.int64)
    n_rows = len(ks) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, max(3 * n_rows, 6)), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for idx, k in enumerate(ks):
        ax = axes[idx]
        kept = []
        dropped = []
        for r in rows:
            kept_val = r["pass_at_k_kept"].get(str(k))
            drop_val = r["pass_at_k_dropped"].get(str(k))
            kept.append(_coerce_float(kept_val))
            dropped.append(_coerce_float(drop_val))
        ax.plot(steps_arr, kept, marker="o", label="Kept")
        ax.plot(steps_arr, dropped, marker="o", label="Dropped")
        ax.set_ylabel(f"Pass@{k} (%)")
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[-1]
    use_beta_log = any(r.get("beta_target_log") is not None for r in rows)
    use_actual_log = any(r.get("beta_actual_log") is not None for r in rows)
    if use_beta_log:
        beta_t = [_pick_value(r, "beta_target_log", "beta_target") for r in rows]
    else:
        beta_t = [_pick_value(r, "beta_target") for r in rows]
    if use_actual_log:
        beta_a = [_pick_value(r, "beta_actual_log", "beta_actual") for r in rows]
    else:
        beta_a = [_pick_value(r, "beta_actual") for r in rows]
    ax.plot(steps_arr, beta_t, color="#e67e22", label="beta_target (log)" if use_beta_log else "beta_target")
    ax.plot(steps_arr, beta_a, color="#3498db", label="beta_actual (log)" if use_actual_log else "beta_actual", linestyle="--")
    ax.set_ylabel("Beta")
    ax.set_xlabel("Training step")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _default_tag(obj: Dict[str, Any]) -> str:
    obj = dict(obj)
    if str(obj.get("beta_source") or "compute") in {"compute", ""} and not obj.get("beta_log_path"):
        obj.pop("beta_source", None)
        obj.pop("beta_log_path", None)
        obj.pop("beta_log_root", None)
        obj.pop("use_log_tau", None)
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"auto_{hashlib.sha1(payload).hexdigest()[:12]}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--prompt_file", type=Path, default=Path("data/math500/test.jsonl"))
    ap.add_argument("--prompt_field", type=str, default="problem")
    ap.add_argument("--answer_field", type=str, default="answer")
    ap.add_argument("--data_name", type=str, default="math500")
    ap.add_argument("--prompt_type", type=str, default="think-boxed")
    ap.add_argument("--num_prompts", type=int, default=512)
    ap.add_argument("--num_prompt_ratio", type=float, default=0.0)
    ap.add_argument("--subset_mode", type=str, default="random", choices=["random", "first"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_rollouts", type=int, default=4)
    ap.add_argument("--target_ks", type=str, default="1,4")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--gen_batch_size", type=int, default=4)
    ap.add_argument("--confidence_metric", type=str, default="neg_entropy", choices=["avg_logp", "neg_entropy"])
    ap.add_argument("--entropy_topk", type=int, default=32)
    ap.add_argument("--max_prompt_len", type=int, default=0, help="Filter prompts longer than this token length.")
    ap.add_argument("--tokenizer_dir", type=Path, default=None, help="Tokenizer path for prompt length filtering.")
    ap.add_argument("--start_beta", type=float, default=0.2)
    ap.add_argument("--end_beta", type=float, default=1.0)
    ap.add_argument("--schedule_steps", type=int, default=8000)
    ap.add_argument("--beta_schedule_mode", type=str, default="train", choices=["train", "rescaled"])
    ap.add_argument("--beta_ref_steps", type=str, default="")
    ap.add_argument("--beta_source", type=str, default="compute", choices=["compute", "log"])
    ap.add_argument("--beta_log_path", type=Path, default=None, help="Optional wandb output.log path; default uses latest-run.")
    ap.add_argument("--beta_log_root", type=Path, default=None, help="Optional wandb root (e.g., project/VI-CURL/wandb).")
    ap.add_argument("--use_log_tau", action="store_true", default=True)
    ap.add_argument("--no_use_log_tau", action="store_false", dest="use_log_tau")
    ap.add_argument("--steps", type=str, default="auto")
    ap.add_argument("--use_vllm", action="store_true", default=True)
    ap.add_argument("--no_use_vllm", action="store_false", dest="use_vllm")
    ap.add_argument("--hf_export_root", type=Path, default=Path("/dev/shm/vi_curl_passk_export"))
    ap.add_argument("--base_model_dir", type=Path, default=None)
    ap.add_argument("--export_hf_if_needed", action="store_true", default=True)
    ap.add_argument("--keep_hf_export", action="store_true", default=False)
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    ap.add_argument("--vllm_pipeline_parallel_size", type=int, default=1)
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--vllm_max_num_seqs", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=Path("eval_log/vi_curl/curriculum_passk"))
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--part_id", type=int, default=0)
    ap.add_argument("--num_parts", type=int, default=1)
    ap.add_argument(
        "--force_part_shard",
        action="store_true",
        default=False,
        help="Force part sharding even if --steps is explicitly provided.",
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", action="store_false", dest="resume")
    ap.add_argument("--no_plot", action="store_true", default=False)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    prompt_file = args.prompt_file.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    if not prompt_file.exists():
        raise FileNotFoundError(f"prompt_file not found: {prompt_file}")

    step_dirs = list_step_dirs(run_dir)
    if not step_dirs:
        raise FileNotFoundError(f"No global_step_* dirs under {run_dir}")
    available_steps = [_step_num_from_dir(p) for p in step_dirs]
    selected_steps = parse_steps_spec(args.steps, available_steps)
    steps = [s for s in selected_steps if s in set(available_steps)]
    step_dirs = [p for p in step_dirs if _step_num_from_dir(p) in set(steps)]

    if args.num_parts > 1:
        steps_arg = str(args.steps or "").strip().lower()
        has_explicit_steps = bool(steps_arg) and steps_arg not in {"auto", "all", "*"}
        if bool(args.force_part_shard) or not has_explicit_steps:
            steps = [s for i, s in enumerate(steps) if i % int(args.num_parts) == int(args.part_id)]
            step_dirs = [p for p in step_dirs if _step_num_from_dir(p) in set(steps)]

    if not steps:
        print("[WARN] No steps selected.")
        return

    beta_ref_steps: List[int] = []
    if args.beta_schedule_mode == "rescaled":
        if args.beta_ref_steps:
            beta_ref_steps = parse_steps_spec(args.beta_ref_steps, available_steps)
        else:
            beta_ref_steps = list(available_steps)

    beta_log: Dict[int, Dict[str, float]] = {}
    if args.beta_source == "log":
        log_path = _resolve_beta_log_path(
            run_dir=run_dir,
            beta_log_path=args.beta_log_path,
            beta_log_root=args.beta_log_root,
        )
        log_path = log_path.expanduser().resolve()
        if not log_path.exists():
            raise FileNotFoundError(f"beta_source=log but log file not found: {log_path}")
        beta_log = _load_beta_from_log(log_path)
        print(f"[INFO] Loaded beta from log: {log_path} (steps={len(beta_log)})")

    beta_by_step: Dict[int, float] = {}
    if args.beta_schedule_mode == "train":
        beta_by_step = {
            int(step): compute_beta_target(int(step), float(args.start_beta), float(args.end_beta), int(args.schedule_steps))
            for step in steps
        }
    else:
        beta_by_step = {
            int(step): compute_beta_target_rescaled(int(step), float(args.start_beta), float(args.end_beta), beta_ref_steps)
            for step in steps
        }

    examples_all = _load_examples(prompt_file)
    prompts_all, answers_all = _build_prompts(
        examples_all,
        data_name=str(args.data_name),
        prompt_type=str(args.prompt_type),
        prompt_field=str(args.prompt_field),
        answer_field=str(args.answer_field),
    )
    prompts = list(prompts_all)
    answers = list(answers_all)
    tokenizer_dir_used: Optional[Path] = None

    if int(args.max_prompt_len) > 0:
        tok_dir = _resolve_tokenizer_dir(
            run_dir=run_dir,
            step_dirs=step_dirs,
            tokenizer_dir=args.tokenizer_dir,
            base_model_dir=args.base_model_dir,
        )
        if tok_dir is None:
            raise FileNotFoundError("max_prompt_len set but tokenizer_dir/base_model_dir not found.")
        prompts, answers, lengths = _filter_prompts_by_length(
            prompts,
            answers,
            max_prompt_len=int(args.max_prompt_len),
            tokenizer_dir=tok_dir,
        )
        if not prompts:
            raise ValueError(f"All prompts exceed max_prompt_len={args.max_prompt_len}.")
        tokenizer_dir_used = tok_dir
        print(
            f"[INFO] Prompt length filter: max_len={args.max_prompt_len} kept={len(prompts)}"
            f" dropped={len(prompts_all) - len(prompts)}"
        )

    num_prompt_ratio = float(args.num_prompt_ratio)
    if num_prompt_ratio > 1.0:
        raise ValueError(f"--num_prompt_ratio must be in (0, 1], got {num_prompt_ratio}")
    num_prompts_eff = int(args.num_prompts)
    if num_prompt_ratio > 0.0:
        num_prompts_eff = int(round(len(prompts) * num_prompt_ratio))
        num_prompts_eff = max(1, min(num_prompts_eff, len(prompts)))
        print(f"[INFO] Prompt ratio: {num_prompt_ratio:.4f} -> num_prompts={num_prompts_eff}")

    if num_prompts_eff > 0 and num_prompts_eff < len(prompts):
        if str(args.subset_mode) == "first":
            prompts = list(prompts[: num_prompts_eff])
            answers = list(answers[: num_prompts_eff])
        else:
            rng = random.Random(int(args.seed))
            idx = rng.sample(range(len(prompts)), num_prompts_eff)
            idx_set = set(idx)
            prompts = [p for i, p in enumerate(prompts) if i in idx_set]
            answers = [a for i, a in enumerate(answers) if i in idx_set]

    print(f"[INFO] Prompts: {len(examples_all)} -> using {len(prompts)} (subset_mode={args.subset_mode})")
    print(f"[INFO] Steps: {steps}")

    ks = [int(x) for x in str(args.target_ks).split(",") if x.strip().isdigit()]
    ks = sorted({k for k in ks if k > 0 and k <= int(args.k_rollouts)})
    if not ks:
        ks = [1]

    run_name = run_dir.name
    config = {
        "run_dir": str(run_dir),
        "prompt_file": str(prompt_file),
        "prompt_field": str(args.prompt_field),
        "answer_field": str(args.answer_field),
        "data_name": str(args.data_name),
        "prompt_type": str(args.prompt_type),
        "num_prompts": int(args.num_prompts),
        "num_prompt_ratio": float(args.num_prompt_ratio),
        "subset_mode": str(args.subset_mode),
        "seed": int(args.seed),
        "k_rollouts": int(args.k_rollouts),
        "target_ks": ks,
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "gen_batch_size": int(args.gen_batch_size),
        "confidence_metric": str(args.confidence_metric),
        "entropy_topk": int(args.entropy_topk),
        "max_prompt_len": int(args.max_prompt_len),
        "tokenizer_dir": str(tokenizer_dir_used or args.tokenizer_dir) if (tokenizer_dir_used or args.tokenizer_dir) else None,
        "start_beta": float(args.start_beta),
        "end_beta": float(args.end_beta),
        "schedule_steps": int(args.schedule_steps),
        "beta_schedule_mode": str(args.beta_schedule_mode),
        "beta_ref_steps": list(beta_ref_steps),
        "beta_source": str(args.beta_source),
        "beta_log_path": str(args.beta_log_path) if args.beta_log_path else None,
        "beta_log_root": str(args.beta_log_root) if args.beta_log_root else None,
        "use_log_tau": bool(args.use_log_tau),
        "steps": list(steps),
        "use_vllm": bool(args.use_vllm),
        "vllm_tensor_parallel_size": int(args.vllm_tensor_parallel_size),
        "vllm_pipeline_parallel_size": int(args.vllm_pipeline_parallel_size),
        "vllm_gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
        "vllm_max_num_seqs": int(args.vllm_max_num_seqs),
        "part_id": int(args.part_id),
        "num_parts": int(args.num_parts),
    }

    tag = str(args.tag).strip() or _default_tag(config)
    suffix = f"__part{int(args.part_id)}" if int(args.num_parts) > 1 else ""
    out_json = out_dir / f"vi_curl_passk__{run_name}__{tag}{suffix}.json"
    out_png = out_dir / f"vi_curl_passk__{run_name}__{tag}{suffix}.png"

    rows_by_step: Dict[int, Dict[str, Any]] = {}
    if bool(args.resume) and out_json.exists():
        try:
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            for r in payload.get("rows", []):
                if "step" in r:
                    rows_by_step[int(r["step"])] = r
            print(f"[RESUME] Loaded {len(rows_by_step)} steps from {out_json}")
        except Exception as e:
            print(f"[WARN] Failed to load {out_json}: {e}")

    for step_dir in step_dirs:
        step = _step_num_from_dir(step_dir)
        if step in rows_by_step:
            continue
        beta_target = float(beta_by_step.get(step, 0.0))
        beta_target_log = None
        beta_actual_log = None
        tau_log = None
        if args.beta_source == "log":
            slot = beta_log.get(int(step), {})
            if "beta_target" in slot:
                beta_target_log = float(slot["beta_target"])
                beta_target = beta_target_log
            if "beta_actual" in slot:
                beta_actual_log = float(slot["beta_actual"])
            if "tau" in slot:
                tau_log = float(slot["tau"])

        if not bool(args.use_vllm):
            raise RuntimeError("Only vLLM backend is implemented for this script.")

        hf_dir, exported = _maybe_export_hf(
            step_dir,
            hf_export_root=args.hf_export_root,
            base_model_dir=args.base_model_dir,
            export_hf_if_needed=bool(args.export_hf_if_needed),
        )
        conf, score_mat = _generate_with_vllm(
            model_dir=hf_dir,
            prompts=prompts,
            answers=answers,
            data_name=str(args.data_name),
            prompt_type=str(args.prompt_type),
            k_rollouts=int(args.k_rollouts),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            confidence_metric=str(args.confidence_metric),
            entropy_topk=int(args.entropy_topk),
            gen_batch_size=int(args.gen_batch_size),
            do_sample=float(args.temperature) > 0.0,
            seed=int(args.seed),
            vllm_tensor_parallel_size=int(args.vllm_tensor_parallel_size),
            vllm_pipeline_parallel_size=int(args.vllm_pipeline_parallel_size),
            vllm_gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
            vllm_max_num_seqs=int(args.vllm_max_num_seqs),
        )

        use_log_tau = (
            args.beta_source == "log"
            and bool(args.use_log_tau)
            and tau_log is not None
            and (math.isinf(tau_log) or np.isfinite(tau_log))
        )
        if use_log_tau:
            keep = _compute_keep_mask_from_tau(conf, float(tau_log))
            tau = float(tau_log)
        else:
            keep, tau = _compute_keep_mask(conf, beta_target)
        beta_actual = float(np.mean(keep.astype(np.float64)))
        kept_scores = [s for s, k in zip(score_mat, keep) if bool(k)]
        drop_scores = [s for s, k in zip(score_mat, keep) if not bool(k)]

        pass_kept, counts_kept = _compute_pass_at_k(kept_scores, ks)
        pass_drop, counts_drop = _compute_pass_at_k(drop_scores, ks)

        row = {
            "step": int(step),
            "beta_target": float(beta_target),
            "beta_target_log": beta_target_log,
            "beta_actual": float(beta_actual),
            "beta_actual_log": beta_actual_log,
            "tau_log": tau_log,
            "tau": float(tau),
            "kept_count": int(np.sum(keep)),
            "dropped_count": int(np.sum(~keep)),
            "pass_at_k_kept": pass_kept,
            "pass_at_k_dropped": pass_drop,
            "valid_counts_kept": counts_kept,
            "valid_counts_dropped": counts_drop,
        }
        rows_by_step[int(step)] = row

        payload = {
            "run_name": run_name,
            "tag": tag,
            "num_parts": int(args.num_parts),
            "config": config,
            "steps": sorted(rows_by_step.keys()),
            "rows": [rows_by_step[s] for s in sorted(rows_by_step.keys())],
        }
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] Step {step}: kept={row['kept_count']} dropped={row['dropped_count']}")

        if exported and not bool(args.keep_hf_export):
            try:
                import shutil

                shutil.rmtree(hf_dir, ignore_errors=True)
            except Exception:
                pass

    steps_sorted = sorted(rows_by_step.keys())
    rows_sorted = [rows_by_step[s] for s in steps_sorted]
    payload = {
        "run_name": run_name,
        "tag": tag,
        "num_parts": int(args.num_parts),
        "config": config,
        "steps": steps_sorted,
        "rows": rows_sorted,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if not bool(args.no_plot):
        title = f"{run_name} kept vs dropped pass@k"
        _plot_passk(steps=steps_sorted, rows=rows_sorted, ks=ks, out_path=out_png, title=title)
        print(f"[OK] Plot saved: {out_png}")


if __name__ == "__main__":
    main()
