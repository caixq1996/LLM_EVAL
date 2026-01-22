"""
Tokenization Pre-caching Module

Pre-tokenizes prompts for each base model and caches them to disk.
This avoids redundant tokenization across multiple LoRA evaluations.

Usage:
    python tools/tokenize_cache.py --model_path Qwen/Qwen2.5-Math-1.5B --datasets "aime24,amc23"
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import hashlib

try:
    import torch
except ImportError:
    torch = None

from transformers import AutoTokenizer


class TokenCache:
    """Manages tokenized prompt caching per model."""
    
    def __init__(self, cache_dir: str = None, model_name: str = None):
        self.cache_dir = Path(cache_dir or os.environ.get('TOKEN_CACHE_DIR', '/tmp/token_cache'))
        self.model_name = model_name or 'unknown'
        self.model_cache_dir = self.cache_dir / self._safe_name(self.model_name)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _safe_name(name: str) -> str:
        """Convert model name to safe directory name."""
        return name.replace('/', '_').replace('\\', '_')
    
    def _get_cache_path(self, dataset_name: str) -> Path:
        return self.model_cache_dir / f"{dataset_name}.pt"
    
    def _get_meta_path(self, dataset_name: str) -> Path:
        return self.model_cache_dir / f"{dataset_name}_meta.json"
    
    def has_cache(self, dataset_name: str) -> bool:
        """Check if cache exists for dataset."""
        return self._get_cache_path(dataset_name).exists()
    
    def save_cache(self, dataset_name: str, token_ids: List[List[int]], metadata: Dict = None):
        """Save tokenized prompts to cache."""
        if torch is None:
            raise ImportError("PyTorch required for tokenization cache")
        
        cache_path = self._get_cache_path(dataset_name)
        meta_path = self._get_meta_path(dataset_name)
        
        # Save token IDs as list of tensors (variable length)
        cache_data = {
            'token_ids': [torch.tensor(ids, dtype=torch.long) for ids in token_ids],
            'lengths': [len(ids) for ids in token_ids],
        }
        torch.save(cache_data, cache_path)
        
        # Save metadata
        meta = {
            'model_name': self.model_name,
            'dataset_name': dataset_name,
            'num_samples': len(token_ids),
            'total_tokens': sum(len(ids) for ids in token_ids),
            **(metadata or {})
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"[TokenCache] Saved {len(token_ids)} prompts to {cache_path}")
    
    def load_cache(self, dataset_name: str) -> Optional[Dict]:
        """Load tokenized prompts from cache."""
        if torch is None:
            raise ImportError("PyTorch required for tokenization cache")
        
        cache_path = self._get_cache_path(dataset_name)
        if not cache_path.exists():
            return None
        
        cache_data = torch.load(cache_path, weights_only=False)
        print(f"[TokenCache] Loaded {len(cache_data['token_ids'])} prompts from {cache_path}")
        return cache_data


def preprocess_dataset(
    tokenizer,
    data_name: str,
    data_dir: str,
    prompt_type: str = 'cot',
    apply_chat_template: bool = False,
) -> List[List[int]]:
    """Load and tokenize a dataset."""
    from data_loader import load_data
    from utils import construct_prompt, parse_question
    
    examples = load_data(data_name, data_dir=data_dir)
    token_ids_list = []
    
    for example in examples:
        example['question'] = parse_question(example, data_name)
        if not example['question']:
            continue
        
        prompt = construct_prompt({'question': example['question']}, data_name, 
                                  type('Args', (), {'prompt_type': prompt_type})())
        
        if apply_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt.strip()}],
                tokenize=False, add_generation_prompt=True
            )
        
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        token_ids_list.append(token_ids)
    
    return token_ids_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--datasets', type=str, default='aime24,amc23,aime25,math500,minerva_math,olympiadbench')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--prompt_type', type=str, default='think-boxed')
    parser.add_argument('--apply_chat_template', action='store_true')
    args = parser.parse_args()
    
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    cache = TokenCache(cache_dir=args.cache_dir, model_name=args.model_path)
    
    for data_name in args.datasets.split(','):
        data_name = data_name.strip()
        if not data_name:
            continue
        
        if cache.has_cache(data_name):
            print(f"[Skip] {data_name} already cached")
            continue
        
        print(f"[Processing] {data_name}...")
        try:
            token_ids = preprocess_dataset(
                tokenizer, data_name, args.data_dir,
                args.prompt_type, args.apply_chat_template
            )
            cache.save_cache(data_name, token_ids, {
                'prompt_type': args.prompt_type,
                'apply_chat_template': args.apply_chat_template,
            })
        except Exception as e:
            print(f"[Error] Failed to process {data_name}: {e}")


if __name__ == '__main__':
    main()
