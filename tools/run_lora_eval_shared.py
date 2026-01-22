#!/usr/bin/env python3
"""
Multi-LoRA Evaluation Shared Script
Evaluates multiple LoRA adapters using a single base model.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print('[WARN] vLLM with LoRA support not available')

# Dataset groups (same as run_qwen_eval_all_shared.py)
GROUP1_DATASETS = ['aime24', 'amc23', 'aime25']
GROUP2_DATASETS = ['math500', 'minerva_math', 'olympiadbench']

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
        step_name = path.name
        run_name = path.parent.name
        name = f"{run_name}_{step_name}"
        return cls(name=name, path=path, lora_id=lora_id, run_name=run_name, step_name=step_name)

def _now():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def load_multi_lora_llm(base_model: str, max_loras: int = 16, num_gpus: int = 1):
    print(f'[{_now()}] Loading base model: {base_model}')
    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=64,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enable_prefix_caching=True,
        enforce_eager=False,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    print(f'[{ _now()}] Base model loaded')
    return llm

def load_prompts(data_name: str, data_dir: str = './data', prompt_type: str = 'think-boxed'):
    from data_loader import load_data
    from utils import construct_prompt, parse_question
    
    examples = load_data(data_name, data_dir=data_dir)
    prompts = []
    
    for i, example in enumerate(examples):
        question = parse_question(example, data_name)
        if not question:
            continue
        example['question'] = question
        
        class Args:
            pass
        args = Args()
        args.prompt_type = prompt_type
        
        prompt = construct_prompt(example, data_name, args)
        prompts.append({
            'idx': i,
            'question': question,
            'prompt': prompt,
            'gt_ans': example.get('answer', ''),
        })
    return prompts

def is_evaluation_complete(out_dir: Path, data_name: str, n_sampling: int = 1) -> bool:
    """Check if evaluation is already complete for this dataset+adapter."""
    gen_file = out_dir / f'{data_name}_generations.jsonl'
    metrics_file = out_dir / f'{data_name}_metrics.json'
    
    # Check if both files exist
    if not gen_file.exists():
        return False
    
    # If metrics file exists, consider complete
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                data = json.load(f)
                if 'pass@1' in data or 'accuracy' in data:
                    return True
        except Exception:
            pass
    
    # Check if jsonl has enough samples
    try:
        with open(gen_file) as f:
            line_count = sum(1 for _ in f)
        # Very rough check: needs at least some samples
        if line_count > 0:
            return True
    except Exception:
        pass
    
    return False


def evaluate_adapters(
    llm,
    adapters: List[LoRAAdapter],
    datasets: List[str],
    out_root: Path,
    n_sampling: int = 1,
    temperature: float = 0.6,
    max_tokens: int = 2048,
):
    """Evaluate all adapters on all datasets."""
    stop_words = ['<|im_end|>', '</s>', '<|endoftext|>']
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop_words,
        n=n_sampling,
    )
    
    # Create LoRARequest objects
    lora_requests = {
        adapter.name: LoRARequest(
            lora_name=adapter.name,
            lora_int_id=adapter.lora_id,
            lora_local_path=str(adapter.path),
        )
        for adapter in adapters
    }
    
    for data_name in datasets:
        print(f'\n[{ _now()}] ===== {data_name} =====')
        try:
            prompts = load_prompts(data_name)
        except Exception as e:
            print(f'[ERROR] Failed to load {data_name}: {e}')
            continue
        
        prompt_strs = [p['prompt'] for p in prompts]
        
        for adapter in adapters:
            run_tag = f"{adapter.run_name}__{adapter.step_name}"
            group_idx = get_group_idx(data_name)
            out_dir = out_root / run_tag / f"g{group_idx}" / data_name
            
            # Check if already complete
            if is_evaluation_complete(out_dir, data_name, n_sampling):
                print(f'[{_now()}] [SKIP] {adapter.name} on {data_name} - already complete')
                continue
            
            print(f'[{_now()}] Evaluating {adapter.name} on {data_name}')
            
            # Generate with LoRA
            outputs = llm.generate(
                prompt_strs,
                sampling_params,
                lora_request=lora_requests[adapter.name],
            )
            
            # Save results (out_dir already set above)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            results = []
            for i, res in enumerate(outputs):
                for j, out in enumerate(res.outputs):
                    results.append({
                        'idx': prompts[i]['idx'],
                        'question': prompts[i]['question'],
                        'gt_ans': prompts[i]['gt_ans'],
                        'sample_idx': j,
                        'output': out.text,
                    })
            
            out_file = out_dir / f'{data_name}_generations.jsonl'
            with open(out_file, 'w') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
            print(f'[{_now()}] Saved {len(results)} results to {out_file}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--lora_adapters', type=str, required=True,
                        help='Pipe-separated adapter paths')
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--datasets', type=str,
                        default='aime24,amc23,aime25,math500,minerva_math,olympiadbench')
    parser.add_argument('--n_sampling', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_tokens', type=int, default=2048)
    args = parser.parse_args()
    
    # Parse adapters
    adapter_paths = [p.strip() for p in args.lora_adapters.split('|') if p.strip()]
    adapters = []
    for i, path in enumerate(adapter_paths):
        path = Path(path)
        if path.exists() and (path / 'adapter_config.json').exists():
            adapters.append(LoRAAdapter.from_path(str(path), lora_id=i+1))
    
    if not adapters:
        print('[ERROR] No valid LoRA adapters found')
        return
    
    print(f'[{_now()}] Found {len(adapters)} LoRA adapters')
    
    # Load model
    llm = load_multi_lora_llm(
        args.base_model,
        max_loras=len(adapters) + 1,
        num_gpus=args.num_gpus,
    )
    
    # Evaluate
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    evaluate_adapters(
        llm,
        adapters,
        datasets,
        Path(args.out_root),
        n_sampling=args.n_sampling,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    print(f'[{_now()}] All evaluations complete')

if __name__ == '__main__':
    main()
