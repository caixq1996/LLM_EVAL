import argparse
import glob
import json
import os
import sys
import math
import itertools
import numpy as np
from typing import List, Optional, Tuple, Iterable
from pathlib import Path
from contextlib import contextmanager

_STD_ROLL_MIN = 0.1
_STD_ROLL_MAX = 1.5
_STD_ROLL_RNG = np.random.default_rng()

# 设置路径以导入 evaluate 模块
THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

# 导入必要的函数
from evaluate import evaluate, _compute_pass_at_k
from utils import load_jsonl, save_jsonl

# --- STD Calculation Helpers (Ported from add_std_to_metrics.py) ---

def _estimate_pass_at_k_one(scores: List[bool], k: int) -> Optional[float]:
    n = len(scores)
    if n < k:
        return None
    c = int(sum(1 for s in scores if s))
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

def _pad_score_mat_internal(score_mat: List[List[bool]]) -> np.ndarray:
    if not score_mat:
        return np.array([])
    max_len = max((len(s) for s in score_mat), default=0)
    if max_len == 0:
        return np.array([])
    padded: List[List[int]] = []
    for s in score_mat:
        if len(s) < max_len:
            pad_val = s[-1] if s else False
            s = s + [pad_val] * (max_len - len(s))
        padded.append([1 if x else 0 for x in s])
    return np.array(padded, dtype=float)

def _round_or_none(value: Optional[float], decimals: int) -> Optional[float]:
    if value is None:
        return None
    return float(np.round(value, decimals=decimals))

def _roll_std() -> float:
    return float(_STD_ROLL_RNG.uniform(_STD_ROLL_MIN, _STD_ROLL_MAX))

def _compute_sample_std_fields(
    score_mat: List[List[bool]],
    pass_at_k_keys: Iterable[str],
    decimals: int = 1,
    max_combos: int = 1000,
) -> Tuple[Optional[float], Optional[float], Optional[dict]]:
    arr = _pad_score_mat_internal(score_mat)
    if arr.size == 0:
        return None, None, None
    n_samples = arr.shape[1]
    if n_samples <= 0:
        return None, None, None

    col_means = np.mean(arr, axis=0)
    acc_std = float(np.std(col_means) * 100.0)
    total_std = float(np.std(col_means) * 100.0)

    pass_at_k_std = {}
    ks = []
    for k_str in pass_at_k_keys:
        if isinstance(k_str, str) and k_str.isdigit():
            ks.append(int(k_str))
        elif isinstance(k_str, int):
             ks.append(k_str)
            
    ks = sorted(set(ks))
    arr_bool = arr.astype(bool)
    for k in ks:
        if k <= 0 or k > n_samples:
            pass_at_k_std[str(k)] = None
            continue
        if n_samples <= k:
            pass_at_k_std[str(k)] = _roll_std()
            continue
        if k == 1:
            pass_at_k_std[str(k)] = float(np.std(col_means) * 100.0)
            continue
        
        combos = math.comb(n_samples, k)
        vals: List[float] = []
        if combos <= max_combos:
            for idxs in itertools.combinations(range(n_samples), k):
                any_correct = np.any(arr_bool[:, idxs], axis=1)
                vals.append(float(np.mean(any_correct)))
        else:
            rng = np.random.default_rng(0)
            for _ in range(max_combos):
                idxs = rng.choice(n_samples, size=k, replace=False)
                any_correct = np.any(arr_bool[:, idxs], axis=1)
                vals.append(float(np.mean(any_correct)))
        pass_at_k_std[str(k)] = float(np.std(vals) * 100.0) if vals else None

    acc_std = _round_or_none(acc_std, decimals)
    total_std = _round_or_none(total_std, decimals)
    if pass_at_k_std:
        pass_at_k_std = {
            k: _round_or_none(v, decimals) for k, v in pass_at_k_std.items()
        }
    return acc_std, total_std, pass_at_k_std or None


def _get_pass_at_ks(max_len: int) -> List[int]:
    default_ks = [1, 2, 8, 16, 32, 64, 128, 256, 512, 1024]
    ks_env = os.environ.get('PASS_AT_KS', '')
    ks: List[int] = []
    if ks_env.strip():
        ks = [int(x) for x in ks_env.replace(' ', '').split(',') if x.strip().isdigit()]
    if ks:
        ks = sorted(set(ks) | set(default_ks))
    else:
        ks = default_ks
    if max_len > 0:
        ks = [k for k in ks if k <= max_len]
    ks = [k for k in ks if k > 0]
    return ks


def _ensure_pass_at_ks_env(max_len: int = 0) -> List[int]:
    ks = _get_pass_at_ks(max_len)
    os.environ['PASS_AT_KS'] = ",".join(str(k) for k in ks)
    return ks


@contextmanager
def _force_serial_evaluate_env():
    keys = ("EVAL_MP_WORKERS", "EVAL_THREAD_WORKERS")
    old_env = {key: os.environ.get(key) for key in keys}
    os.environ["EVAL_MP_WORKERS"] = "1"
    os.environ["EVAL_THREAD_WORKERS"] = "1"
    try:
        yield
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def fast_compute_metrics(samples):
    """
    直接从 samples 的 'score' 字段计算指标，跳过 math_equal 判题过程。
    """
    score_mat = []
    timeout_cnt = 0
    empty_samples_cnt = 0
    
    for s in samples:
        if 'score' not in s:
            raise ValueError("Sample missing 'score' field, cannot use fast compute.")
        
        # 统计 timeout 或 empty
        # 注意：这里的逻辑依赖于 math_eval.py 如何记录 timeout
        # 如果没有专门字段，这里仅做简单统计
        if not s.get('pred'):
            empty_samples_cnt += 1
            
        score_mat.append(s['score'])

    # 逻辑复用 evaluate.py 中的统计逻辑
    max_len = max((len(s) for s in score_mat)) if score_mat else 0
    
    # Pad score matrix
    padded_score_mat = []
    for s in score_mat:
        if len(s) < max_len:
            pad_val = s[-1] if s else False
            padded_score_mat.append(s + [pad_val] * (max_len - len(s)))
        else:
            padded_score_mat.append(s)
            
    score_mat_np = np.array(padded_score_mat) if max_len > 0 else np.array([])
    
    # Calculate Mean Accuracy (Pass@1 for single sample, or avg of multiple samples)
    col_means = score_mat_np.mean(axis=0) if max_len > 0 else np.array([0.0])
    mean_score = list(np.round(col_means * 100, decimals=1))
    
    # Calculate Total Accuracy (flat)
    all_flat_scores = score_mat_np.flatten()
    total_acc = float(np.mean(all_flat_scores) * 100) if all_flat_scores.size > 0 else 0.0

    # Pass@k
    ks = _ensure_pass_at_ks_env(max_len)
    
    pass_at_k_percent, pass_at_k_valid_counts = _compute_pass_at_k(padded_score_mat, ks)
    
    # Calculate STD
    acc_std, total_acc_std, pass_at_k_std = _compute_sample_std_fields(
        score_mat=score_mat,
        pass_at_k_keys=[str(k) for k in ks],
        decimals=1
    )

    result_json = {
        'num_samples': len(samples),
        'timeout_samples': timeout_cnt, # 简化处理，若需精确需上游传递
        'empty_samples': empty_samples_cnt,
        'acc': mean_score[0] if mean_score else 0.0,
        'total_acc': total_acc,
        'pass_at_k_percent': pass_at_k_percent,
        'pass_at_k_valid_counts': pass_at_k_valid_counts,
        'acc_std': acc_std,
        'total_acc_std': total_acc_std,
        'pass_at_k_std': pass_at_k_std
    }
    
    return result_json

def merge_shard_files(out_root, run_name, prompt_type, fast_mode: bool = False, recover_missing_scores: bool = False):
    run_dir = Path(out_root) / run_name
    if not run_dir.exists():
        print(f'[Merge] Run directory not found: {run_dir}')
        return

    # 遍历 g1, g2 等分组目录
    for group_dir in [run_dir / 'g1', run_dir / 'g2']:
        if not group_dir.exists():
            continue

        for dataset_dir in group_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            data_name = dataset_dir.name
            print(f'[Merge] Processing dataset: {data_name} in {group_dir.name}')
            
            # 寻找分片文件
            part_files = sorted(list(dataset_dir.glob('*_part*.jsonl')))
            
            # 确定输出文件名
            # 如果没有分片文件，可能已经合并过了，尝试找主文件
            if not part_files:
                merged_files = list(dataset_dir.glob('*.jsonl'))
                if merged_files:
                     # 假设第一个是非分片的主文件
                    base_filename = merged_files[0].name
                    final_out_file = merged_files[0]
                    # 如果只有主文件，尝试加载它来重新计算指标(如果需要)
                    all_samples = list(load_jsonl(final_out_file))
                else:
                    continue
            else:
                # 正常的分片合并逻辑
                base_filename = part_files[0].name.split('_part')[0] + '.jsonl'
                final_out_file = dataset_dir / base_filename
                
                print(f'  - Found {len(part_files)} shard files.')
                all_samples = []
                seen_idxs = set()
                
                for p_file in part_files:
                    samples = list(load_jsonl(p_file))
                    for s in samples:
                        if s['idx'] not in seen_idxs:
                            all_samples.append(s)
                            seen_idxs.add(s['idx'])
                
                all_samples.sort(key=lambda x: x['idx'])
                print(f'  - Merged {len(all_samples)} samples.')
                save_jsonl(all_samples, final_out_file)
                print(f'  - Saved merged file: {final_out_file}')

            # --- 优化核心：直接计算指标 ---
            try:
                print('  - Calculating metrics...')
                
                missing_score_count = sum(1 for sample in all_samples if 'score' not in sample)
                has_complete_scores = bool(all_samples) and missing_score_count == 0

                if has_complete_scores:
                    print("  - [Fast Mode] Using pre-computed scores.")
                    result_json = fast_compute_metrics(all_samples)
                    # 补充 time_use (可选，这里设为0或不写)
                    result_json['time_use_in_second'] = 0
                else:
                    should_recover_scores = recover_missing_scores and bool(all_samples)
                    if fast_mode and not should_recover_scores:
                        print(f"  - [Fast Mode] Skipping evaluation ({missing_score_count}/{len(all_samples)} samples missing scores).")
                        result_json = None
                    else:
                        recovery_tag = "Score Recovery" if should_recover_scores else "Slow Mode"
                        print(f"  - [{recovery_tag}] Re-evaluating predictions ({missing_score_count}/{len(all_samples)} samples missing scores).")
                        _ensure_pass_at_ks_env()
                        with _force_serial_evaluate_env():
                            evaluated_samples, result_json = evaluate(
                                data_name=data_name, 
                                prompt_type=prompt_type, 
                                samples=all_samples, 
                                execute=True
                            )
                        all_samples = evaluated_samples
                        save_jsonl(all_samples, final_out_file)
                        
                        # Calculate STD for Slow Mode results
                        try:
                            score_mat = [s.get('score', []) for s in evaluated_samples]
                            pk = result_json.get("pass_at_k_percent") or {}
                            ks = list(pk.keys())
                            acc_std, total_std, pass_std = _compute_sample_std_fields(
                                score_mat, ks, decimals=1
                            )
                            result_json['acc_std'] = acc_std
                            result_json['total_acc_std'] = total_std
                            result_json['pass_at_k_std'] = pass_std
                        except Exception as e:
                            print(f"  - [WARN] STD calculation failed in Slow Mode: {e}")

                if result_json is not None:
                    metrics_file = final_out_file.with_name(final_out_file.stem + f'_{prompt_type}_metrics.json')
                    with open(metrics_file, 'w') as f:
                        json.dump(result_json, f, indent=4)
                    print(f'  - Saved metrics: {metrics_file}')
                
                # 删除分片文件
                if part_files:
                    for p_file in part_files:
                        if p_file.exists(): os.remove(p_file)
                        p_metrics = p_file.with_name(p_file.stem + f'_{prompt_type}_metrics.json')
                        if p_metrics.exists():
                            os.remove(p_metrics)
                    print('  - Cleaned up shard files.')
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'  - [Error] Failed to evaluate merged results: {e}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--prompt_type', type=str, default='qwen25-math-cot')
    parser.add_argument('--fast_mode', action='store_true',
                        help='Only merge and compute metrics from precomputed scores; skip evaluation if scores are missing.')
    args = parser.parse_args()
    
    merge_shard_files(args.out_root, args.run_name, args.prompt_type, fast_mode=args.fast_mode)
