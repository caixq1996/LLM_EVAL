#!/usr/bin/env python3
"""
Offline Verification Script

Verifies pre-generated outputs from multi-LoRA evaluation.
Reads *_generations.jsonl files and computes pass@k metrics.

Usage:
    python tools/verify_generations.py --input_dir /path/to/generations --out_dir /path/to/metrics
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from grader import math_equal


class BoxedAnswerExtractor:
    """Extract answers from boxed expressions."""
    
    BOXED_PATTERN = re.compile(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}')
    
    @classmethod
    def extract(cls, text: str) -> Optional[str]:
        matches = list(cls.BOXED_PATTERN.finditer(text))
        if matches:
            return matches[-1].group(1)  # Return last boxed answer
        return None


def compute_pass_at_k(results: List[Dict], k_values: List[int] = [1, 8, 64, 256, 1024]) -> Dict:
    """Compute pass@k metrics from generation results."""
    # Group by problem idx
    by_problem = defaultdict(list)
    for r in results:
        by_problem[r['idx']].append(r)
    
    metrics = {}
    for k in k_values:
        correct = 0
        total = 0
        
        for idx, samples in by_problem.items():
            if not samples:
                continue
            
            # Check if any of first k samples is correct
            n_correct = sum(1 for s in samples[:k] if s.get('is_correct', False))
            if n_correct > 0:
                correct += 1
            total += 1
        
        metrics[f'pass@{k}'] = correct / total * 100 if total > 0 else 0
    
    return metrics


def verify_generations(input_file: Path, gt_data: Dict = None) -> List[Dict]:
    """Verify generations from a jsonl file."""
    results = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            r = json.loads(line)
            output = r.get('output', '')
            gt_ans = r.get('gt_ans', '')
            
            # Extract answer from output
            pred = BoxedAnswerExtractor.extract(output)
            
            # Check correctness
            is_correct = False
            if pred is not None and gt_ans:
                is_correct = math_equal(pred, gt_ans)
            
            r['pred'] = pred
            r['is_correct'] = is_correct
            results.append(r)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing *_generations.jsonl files')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory for metrics (default: same as input)')
    parser.add_argument('--k_values', type=str, default='1,8,64,256,1024',
                        help='Comma-separated k values for pass@k')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) if args.out_dir else input_dir
    k_values = [int(k) for k in args.k_values.split(',')]
    
    # Find all generation files
    gen_files = list(input_dir.rglob('*_generations.jsonl'))
    
    if not gen_files:
        print(f'[WARN] No generation files found in {input_dir}')
        return
    
    print(f'Found {len(gen_files)} generation files')
    
    all_metrics = {}
    
    for gen_file in gen_files:
        dataset_name = gen_file.stem.replace('_generations', '')
        rel_path = gen_file.relative_to(input_dir).parent
        
        print(f'\n[Verifying] {gen_file}')
        
        results = verify_generations(gen_file)
        metrics = compute_pass_at_k(results, k_values)
        
        # Save verified results
        verified_file = gen_file.parent / f'{dataset_name}_verified.jsonl'
        with open(verified_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        
        # Save metrics
        metrics_file = gen_file.parent / f'{dataset_name}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f'  Metrics: {metrics}')
        all_metrics[str(rel_path / dataset_name)] = metrics
    
    # Save combined metrics
    combined_file = out_dir / 'all_metrics.json'
    with open(combined_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f'\n[Done] Combined metrics saved to {combined_file}')


if __name__ == '__main__':
    main()
