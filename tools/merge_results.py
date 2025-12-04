import argparse
import glob
import json
import os
import sys
import numpy as np
from pathlib import Path

# 设置路径以导入 evaluate 模块
THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

# 导入必要的函数
from evaluate import evaluate, _compute_pass_at_k
from utils import load_jsonl, save_jsonl

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
    ks_env = os.environ.get('PASS_AT_KS', '')
    if ks_env.strip():
        ks = [int(x) for x in ks_env.replace(' ', '').split(',') if x.strip().isdigit()]
    else:
        ks = [1, 8]
    ks = sorted({k for k in ks if k > 0 and (max_len == 0 or k <= max_len)})
    
    pass_at_k_percent, pass_at_k_valid_counts = _compute_pass_at_k(padded_score_mat, ks)

    result_json = {
        'num_samples': len(samples),
        'timeout_samples': timeout_cnt, # 简化处理，若需精确需上游传递
        'empty_samples': empty_samples_cnt,
        'acc': mean_score[0] if mean_score else 0.0,
        'total_acc': total_acc,
        'pass_at_k_percent': pass_at_k_percent,
        'pass_at_k_valid_counts': pass_at_k_valid_counts
    }
    
    return result_json

def merge_shard_files(out_root, run_name, prompt_type):
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
                
                # 检查是否包含 score 字段
                if all_samples and 'score' in all_samples[0]:
                    print("  - [Fast Mode] Using pre-computed scores.")
                    result_json = fast_compute_metrics(all_samples)
                    # 补充 time_use (可选，这里设为0或不写)
                    result_json['time_use_in_second'] = 0 
                else:
                    print("  - [Slow Mode] Re-evaluating predictions (scores not found).")
                    _, result_json = evaluate(
                        data_name=data_name, 
                        prompt_type=prompt_type, 
                        samples=all_samples, 
                        execute=True
                    )

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
    args = parser.parse_args()
    
    merge_shard_files(args.out_root, args.run_name, args.prompt_type)