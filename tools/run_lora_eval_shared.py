#!/usr/bin/env python3
"""
Multi-LoRA Evaluation Shared Script
Evaluates multiple LoRA adapters using a single base model.
Output format matches run_qwen_eval_all_shared.py exactly.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LLM = None
SamplingParams = None
LoRARequest = None
VLLM_AVAILABLE = None

def _ensure_vllm():
    """Lazy-import vLLM to avoid CUDA init in eval worker processes."""
    global LLM, SamplingParams, LoRARequest, VLLM_AVAILABLE
    if VLLM_AVAILABLE is None:
        try:
            from vllm import LLM as _LLM, SamplingParams as _SamplingParams
            from vllm.lora.request import LoRARequest as _LoRARequest
            LLM = _LLM
            SamplingParams = _SamplingParams
            LoRARequest = _LoRARequest
            VLLM_AVAILABLE = True
        except ImportError:
            VLLM_AVAILABLE = False
            print('[WARN] vLLM with LoRA support not available')
    return VLLM_AVAILABLE

_DEFAULT_GROUP1 = 'aime25x8,amc23x8,aime24x8'
_DEFAULT_GROUP2 = 'minerva_math,olympiadbench,math500'

def _split_ds_list(datasets: str) -> List[str]:
    return [d.strip() for d in datasets.split(',') if d.strip()]

GROUP1_DATASETS = _split_ds_list(os.getenv('EVAL_GROUP1_DATASETS', _DEFAULT_GROUP1))
GROUP2_DATASETS = _split_ds_list(os.getenv('EVAL_GROUP2_DATASETS', _DEFAULT_GROUP2))

def get_group_idx(data_name: str) -> int:
    if data_name in GROUP1_DATASETS:
        return 1
    elif data_name in GROUP2_DATASETS:
        return 2
    return 1

@dataclass
class LoRAAdapter:
    name: str
    path: Path
    lora_id: int
    run_name: str
    step_name: str

    @classmethod
    def from_path(cls, path: str, lora_id: int):
        path = Path(path)
        # Handle adapter checkpoints under global_step_*/actor
        if path.name == 'actor' and path.parent.name.startswith('global_step_'):
            step_name = path.parent.name
            run_name = path.parent.parent.name
        else:
            step_name = path.name
            run_name = path.parent.name
        name = f"{run_name}_{step_name}"
        return cls(name=name, path=path, lora_id=lora_id, run_name=run_name, step_name=step_name)

def _now():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def _get_peft_type(adapter_path: Path) -> str:
    cfg_path = adapter_path / "adapter_config.json"
    if not cfg_path.exists():
        return ""
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(cfg.get("peft_type", "")).upper()

def _filter_supported_adapters(adapters: List["LoRAAdapter"]) -> List["LoRAAdapter"]:
    supported: List[LoRAAdapter] = []
    skipped: List[LoRAAdapter] = []
    for adapter in adapters:
        peft_type = _get_peft_type(adapter.path)
        if peft_type == "OFT":
            skipped.append(adapter)
            continue
        supported.append(adapter)
    if skipped:
        preview = ", ".join(a.name for a in skipped[:5])
        more = "" if len(skipped) <= 5 else f" (+{len(skipped) - 5} more)"
        print(f'[{_now()}] [WARN] Skipping OFT adapters (unsupported by vLLM LoRA): {preview}{more}')
    for i, adapter in enumerate(supported):
        adapter.lora_id = i + 1
    return supported

def _maybe_convert_adalora(adapter: "LoRAAdapter") -> Path:
    """
    vLLM expects LoRA weights to use *.lora_A.weight / *.lora_B.weight keys.
    AdaLoRA checkpoints store *.lora_A / *.lora_B / *.lora_E tensors instead.
    Convert AdaLoRA -> standard LoRA by folding lora_E into lora_B and renaming.
    """
    cfg_path = adapter.path / "adapter_config.json"
    if not cfg_path.exists():
        return adapter.path
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return adapter.path

    peft_type = str(cfg.get("peft_type", "")).upper()
    tensor_path = adapter.path / "adapter_model.safetensors"
    if not tensor_path.exists():
        return adapter.path

    need_convert = peft_type == "ADALORA"
    try:
        from safetensors import safe_open
        with safe_open(str(tensor_path), framework="pt") as f:
            if not need_convert:
                need_convert = any(
                    k.endswith(".lora_A") or k.endswith(".lora_B") or k.endswith(".lora_E")
                    for k in f.keys()
                )
    except Exception:
        return adapter.path

    if not need_convert:
        return adapter.path

    dest_dir = adapter.path.parent / f"{adapter.path.name}_vllm"
    dest_tensor = dest_dir / "adapter_model.safetensors"
    dest_cfg = dest_dir / "adapter_config.json"
    if dest_tensor.exists() and dest_cfg.exists():
        return dest_dir

    print(f'[{_now()}] [INFO] Converting AdaLoRA -> LoRA for vLLM: {adapter.path} -> {dest_dir}')
    dest_dir.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import load_file, save_file
    weights = load_file(str(tensor_path))
    new_weights = {}
    for key, val in weights.items():
        if key.endswith(".lora_E"):
            continue
        if key.endswith(".lora_A"):
            new_weights[key + ".weight"] = val
            continue
        if key.endswith(".lora_B"):
            e_key = key[:-len(".lora_B")] + ".lora_E"
            if e_key in weights:
                e = weights[e_key].view(-1).to(val.dtype)
                new_weights[key + ".weight"] = val * e.reshape(1, -1)
            else:
                new_weights[key + ".weight"] = val
            continue
        new_weights[key] = val

    save_file(new_weights, str(dest_tensor))

    cfg_out = dict(cfg)
    if str(cfg_out.get("peft_type", "")).upper() == "ADALORA":
        cfg_out["peft_type"] = "LORA"
    cfg_out["inference_mode"] = True
    dest_cfg.write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")
    return dest_dir

def _is_multi_choice(answer: str) -> bool:
    if answer is None:
        return False
    for c in str(answer):
        if c not in ['A', 'B', 'C', 'D', 'E']:
            return False
    return True

def _get_num_test_sample() -> int:
    val = os.getenv("EVAL_NUM_TEST_SAMPLE")
    if val is None or str(val).strip() == "":
        return -1
    try:
        return int(val)
    except ValueError:
        return -1

def load_multi_lora_llm(base_model: str, max_loras: int = 16, num_gpus: int = 1):
    if not _ensure_vllm():
        raise RuntimeError("vLLM with LoRA support not available")
    print(f'[{_now()}] Loading base model: {base_model}')
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
    gpu_mem_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.95"))
    enable_chunked_prefill = os.getenv("VLLM_ENABLE_CHUNKED_PREFILL", "false").lower() in ("1", "true", "yes", "y")
    swap_space = int(os.getenv("VLLM_SWAP_SPACE", "0"))
    disable_custom_all_reduce = os.getenv("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "false").lower() in ("1", "true", "yes", "y")
    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=64,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        enable_chunked_prefill=enable_chunked_prefill,
        enforce_eager=False,
        trust_remote_code=True,
        disable_log_stats=True,
        swap_space=swap_space,
        disable_custom_all_reduce=disable_custom_all_reduce,
    )
    print(f'[{_now()}] Base model loaded')
    return llm

def load_prompts(
    data_name: str,
    data_dir: str = None,
    prompt_type: str = 'think-boxed',
    split: str = 'test',
):
    from data_loader import load_data
    from parser import parse_question, parse_ground_truth
    from utils import construct_prompt

    if data_dir is None:
        data_dir = os.getenv("EVAL_DATA_DIR", "./data")
    
    examples = load_data(data_name, split, data_dir=data_dir)
    prompts = []
    
    for i, example in enumerate(examples):
        question = parse_question(example, data_name)
        if not question:
            continue
        example['question'] = question
        example['idx'] = i

        gt_cot, gt = parse_ground_truth(example, data_name)
        
        class Args:
            pass
        args = Args()
        args.prompt_type = prompt_type
        args.num_shots = 0
        args.adapt_few_shot = False
        
        prompt = construct_prompt(example, data_name, args)
        prompt_item = {
            'idx': i,
            'question': question,
            'prompt': prompt,
            'gt': gt,
            'gt_cot': gt_cot,
        }

        # Preserve common metadata fields for downstream analysis.
        for key in [
            'level', 'type', 'unit', 'solution_type', 'choices', 'solution',
            'ques_type', 'ans_type', 'answer_type', 'dataset', 'subfield', 'filed',
            'theorem', 'answer', 'difficulty'
        ]:
            if key in example:
                prompt_item[key] = example[key]

        prompts.append(prompt_item)
    return prompts

def _find_jsonl_for_metrics(metrics_path: Path) -> Optional[Path]:
    candidates = sorted(metrics_path.parent.glob("*.jsonl"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    matches = [p for p in candidates if metrics_path.name.startswith(p.stem + "_")]
    if matches:
        matches.sort(key=lambda p: len(p.stem), reverse=True)
        return matches[0]
    return None


def _load_score_lists(jsonl_path: Path) -> List[List[bool]]:
    scores: List[List[bool]] = []
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                score = obj.get("score") if isinstance(obj, dict) else None
                if isinstance(score, list):
                    scores.append([bool(x) for x in score])
    except Exception:
        return scores
    return scores


def _ensure_metrics_std(metrics_path: Path) -> bool:
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False

    has_pass = isinstance(data.get("pass_at_k_percent"), dict)
    need_acc = "acc_std" not in data
    need_total = "total_acc_std" not in data
    need_pass = has_pass and ("pass_at_k_std" not in data)
    if not (need_acc or need_total or need_pass):
        return True

    jsonl_path = _find_jsonl_for_metrics(metrics_path)
    if not jsonl_path:
        return False

    score_mat = _load_score_lists(jsonl_path)
    if not score_mat:
        return False

    try:
        from evaluate import _compute_sample_std_fields
        pass_keys = list((data.get("pass_at_k_percent") or {}).keys())
        acc_std, total_std, pass_std = _compute_sample_std_fields(
            score_mat=score_mat,
            pass_at_k_keys=pass_keys,
            decimals=1,
        )
    except Exception:
        return False

    if acc_std is None and total_std is None and pass_std is None:
        return False

    if need_acc and acc_std is not None:
        data["acc_std"] = acc_std
    if need_total and total_std is not None:
        data["total_acc_std"] = total_std
    if need_pass and pass_std is not None:
        data["pass_at_k_std"] = pass_std

    metrics_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    return True


def _load_existing_samples(jsonl_path: Path) -> List[dict]:
    if not jsonl_path.exists():
        return []
    samples: List[dict] = []
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    samples.append(obj)
    except Exception:
        return samples
    return samples


def _append_jsonl(samples: List[dict], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _get_gen_chunk_size() -> int:
    val = os.getenv("EVAL_GEN_CHUNK_SIZE")
    if val is None or str(val).strip() == "":
        return 64
    try:
        size = int(val)
        return max(1, size)
    except ValueError:
        return 64

def is_evaluation_complete(out_dir: Path) -> bool:
    metrics_files = list(out_dir.glob('*_metrics.json'))
    for mf in metrics_files:
        if '_part' not in mf.name:
            try:
                with open(mf) as f:
                    data = json.load(f)
                    if 'acc' in data or 'num_samples' in data:
                        has_pass = isinstance(data.get("pass_at_k_percent"), dict)
                        need_acc = "acc_std" not in data
                        need_total = "total_acc_std" not in data
                        need_pass = has_pass and ("pass_at_k_std" not in data)
                        if need_acc or need_total or need_pass:
                            return _ensure_metrics_std(Path(mf))
                        return True
            except Exception:
                pass
    return False

def evaluate_adapters(
    llm,
    adapters: List[LoRAAdapter],
    datasets: List[str],
    out_root: Path,
    prompt_type: str = 'think-boxed',
    data_dir: str = None,
    split: str = 'test',
    temperature_g1: float = 0.6,
    temperature_g2: float = 0.8,
    n_sampling_g1: int = 1,
    n_sampling_g2: int = 8,
    max_tokens: int = 8192,
    shard_id: int = 0,
    num_shards: int = 1,
):
    """Evaluate all adapters on all datasets with g1/g2 config split."""
    from parser import run_execute, choice_answer_clean
    from python_executor import PythonExecutor

    if not _ensure_vllm():
        raise RuntimeError("vLLM with LoRA support not available")

    if 'pal' in prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)
    
    # Create LoRARequest objects (convert AdaLoRA to vLLM-compatible LoRA if needed)
    lora_paths = {adapter.name: _maybe_convert_adalora(adapter) for adapter in adapters}
    lora_requests = {
        adapter.name: LoRARequest(
            lora_name=adapter.name,
            lora_int_id=adapter.lora_id,
            lora_local_path=str(lora_paths[adapter.name]),
        )
        for adapter in adapters
    }
    
    for data_name in datasets:
        # Determine g1/g2 config
        group_idx = get_group_idx(data_name)
        if group_idx == 1:
            temperature = temperature_g1
            n_sampling = n_sampling_g1
        else:
            temperature = temperature_g2
            n_sampling = n_sampling_g2
        
        # Create sampling params for this group
        stop_words = ['<|im_end|>', '</s>', '<|endoftext|>']
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_words,
            n=n_sampling,
        )
        
        print(f'[{_now()}] ===== {data_name} (T={temperature}, n={n_sampling}) =====')
        try:
            prompts = load_prompts(
                data_name,
                data_dir=data_dir,
                prompt_type=prompt_type,
                split=split,
            )
        except Exception as e:
            print(f'[ERROR] Failed to load {data_name}: {e}')
            continue

        if num_shards > 1:
            total_len = len(prompts)
            chunk_size = total_len // num_shards
            start_idx = shard_id * chunk_size
            if shard_id == num_shards - 1:
                end_idx = total_len
            else:
                end_idx = (shard_id + 1) * chunk_size
            prompts = prompts[start_idx:end_idx]
            print(f'[{_now()}] Shard {shard_id}/{num_shards} handles prompts [{start_idx}:{end_idx}]')

        num_test_sample = _get_num_test_sample()
        if num_test_sample > 0:
            prompts = prompts[:num_test_sample]
        
        for adapter in adapters:
            safe_run_name = adapter.run_name.replace('.', '_').replace('-', '_')
            run_tag = f"{adapter.run_name}__{adapter.step_name}"
            out_dir = out_root / safe_run_name / run_tag / f"g{group_idx}" / data_name
            
            if is_evaluation_complete(out_dir):
                print(f'[{_now()}] [SKIP] {adapter.name} on {data_name} - already complete')
                continue
            
            print(f'[{_now()}] Evaluating {adapter.name} on {data_name}')
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file_prefix = f'{split}_{prompt_type}_{num_test_sample}_seed0_t{temperature}'
            shard_suffix = f'_part{shard_id}' if num_shards > 1 else ''
            out_file = out_dir / f'{out_file_prefix}_s0_e-1{shard_suffix}.jsonl'

            existing_samples = _load_existing_samples(out_file)
            seen_idx = set()
            dedup_samples = []
            for sample in existing_samples:
                idx_val = sample.get("idx")
                if idx_val is None or idx_val in seen_idx:
                    continue
                seen_idx.add(idx_val)
                dedup_samples.append(sample)
            existing_samples = dedup_samples

            missing_prompts = [p for p in prompts if p.get("idx") not in seen_idx]
            if missing_prompts:
                chunk_size = _get_gen_chunk_size()
                print(f'[{_now()}] Resume: {len(existing_samples)} existing, {len(missing_prompts)} missing (chunk={chunk_size})')
                for start in range(0, len(missing_prompts), chunk_size):
                    chunk_prompts = missing_prompts[start:start + chunk_size]
                    chunk_prompt_strs = [p['prompt'] for p in chunk_prompts]
                    outputs = llm.generate(
                        chunk_prompt_strs,
                        sampling_params,
                        lora_request=lora_requests[adapter.name],
                    )

                    new_samples = []
                    for i, res in enumerate(outputs):
                        codes = [out.text for out in res.outputs]
                        results = [run_execute(executor, code, prompt_type, data_name) for code in codes]
                        preds = [item[0] for item in results]
                        reports = [item[1] for item in results]
                        for j in range(len(preds)):
                            pred_val = preds[j]
                            gt_val = chunk_prompts[i].get('gt')
                            if gt_val in ['A', 'B', 'C', 'D', 'E'] and pred_val not in ['A', 'B', 'C', 'D', 'E']:
                                preds[j] = choice_answer_clean(codes[j])
                            elif _is_multi_choice(gt_val) and not _is_multi_choice(pred_val):
                                preds[j] = ''.join([c for c in str(pred_val) if c in ['A', 'B', 'C', 'D', 'E']])
                        sample = dict(chunk_prompts[i])
                        sample.pop('prompt', None)
                        sample.update({
                            'code': codes,
                            'pred': preds,
                            'report': reports,
                        })
                        new_samples.append(sample)

                    _append_jsonl(new_samples, out_file)
                    existing_samples.extend(new_samples)
                    print(f'[{_now()}] Saved {len(new_samples)} samples (partial) to {out_file}')
            else:
                print(f'[{_now()}] Resume: all samples already in {out_file}')

            all_samples = existing_samples
            if not all_samples:
                print(f'[{_now()}] [WARN] No samples to evaluate for {adapter.name} on {data_name}')
                continue
            
            # Evaluate with shared grading logic (pass@k uses PASS_AT_KS env var).
            from evaluate import evaluate
            from utils import save_jsonl

            all_samples, metrics = evaluate(
                samples=all_samples,
                data_name=data_name,
                prompt_type=prompt_type,
                execute=False,
            )

            save_jsonl(all_samples, str(out_file))
            print(f'[{_now()}] Saved {len(all_samples)} samples to {out_file}')

            metrics_file = out_dir / out_file.name.replace('.jsonl', f'_{prompt_type}_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)

            print(f'[{_now()}] Saved metrics to {metrics_file}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--lora_adapters', type=str, required=True, help='Pipe-separated adapter paths')
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--datasets', type=str, default=','.join(GROUP1_DATASETS + GROUP2_DATASETS))
    parser.add_argument('--data_dir', type=str, default=os.getenv("EVAL_DATA_DIR", "./data"))
    parser.add_argument('--prompt_type', type=str, default='think-boxed')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--temperature_g1', type=float, default=0.6)
    parser.add_argument('--temperature_g2', type=float, default=0.8)
    parser.add_argument('--n_sampling_g1', type=int, default=1)
    parser.add_argument('--n_sampling_g2', type=int, default=8)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    args = parser.parse_args()
    
    adapter_paths = [p.strip() for p in args.lora_adapters.split('|') if p.strip()]
    adapters = []
    for i, path in enumerate(adapter_paths):
        path = Path(path)
        if path.exists() and (path / 'adapter_config.json').exists():
            adapters.append(LoRAAdapter.from_path(str(path), lora_id=i+1))

    adapters = _filter_supported_adapters(adapters)
    if not adapters:
        print('[ERROR] No supported LoRA adapters found (OFT is unsupported by vLLM LoRA)')
        return

    print(f'[{_now()}] Found {len(adapters)} LoRA adapters')
    
    llm = load_multi_lora_llm(
        args.base_model,
        max_loras=len(adapters) + 1,
        num_gpus=args.num_gpus,
    )
    
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    evaluate_adapters(
        llm,
        adapters,
        datasets,
        Path(args.out_root),
        prompt_type=args.prompt_type,
        data_dir=args.data_dir,
        split=args.split,
        temperature_g1=args.temperature_g1,
        temperature_g2=args.temperature_g2,
        n_sampling_g1=args.n_sampling_g1,
        n_sampling_g2=args.n_sampling_g2,
        max_tokens=args.max_tokens,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    
    print(f'[{_now()}] All evaluations complete')

if __name__ == '__main__':
    main()
