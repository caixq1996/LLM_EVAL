# [Full Code] File: tools/merge_results.py
import argparse
import glob
import json
import os
import sys
from pathlib import Path

# 添加父目录到 path 以便导入 evaluate
THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from evaluate import evaluate
from utils import load_jsonl, save_jsonl

def merge_shard_files(out_root, run_name, prompt_type):
    run_dir = Path(out_root) / run_name
    if not run_dir.exists():
        print(f"[Merge] Run directory not found: {run_dir}")
        return

    # 遍历 g1 和 g2 下的数据集
    for group_dir in [run_dir / 'g1', run_dir / 'g2']:
        if not group_dir.exists():
            continue
        
        for dataset_dir in group_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            data_name = dataset_dir.name
            print(f"[Merge] Processing dataset: {data_name} in {group_dir.name}")
            
            # 查找所有分片文件 *_part*.jsonl
            part_files = sorted(list(dataset_dir.glob("*_part*.jsonl")))
            
            if not part_files:
                # 检查是否已经有合并好的文件
                merged_files = list(dataset_dir.glob("*.jsonl"))
                if merged_files:
                    print(f"  - No part files found, but merged file exists. Skipping.")
                continue
            
            print(f"  - Found {len(part_files)} shard files.")
            
            all_samples = []
            seen_idxs = set()
            
            # 确定最终文件名 (去掉 _partX)
            base_filename = part_files[0].name.split('_part')[0] + ".jsonl"
            final_out_file = dataset_dir / base_filename
            
            for p_file in part_files:
                samples = list(load_jsonl(p_file))
                for s in samples:
                    if s['idx'] not in seen_idxs:
                        all_samples.append(s)
                        seen_idxs.add(s['idx'])
            
            # 按 idx 排序
            all_samples.sort(key=lambda x: x['idx'])
            print(f"  - Merged {len(all_samples)} samples.")
            
            # 保存合并后的 JSONL
            save_jsonl(all_samples, final_out_file)
            print(f"  - Saved merged file: {final_out_file}")
            
            # 重新计算 Metrics
            try:
                print("  - Recalculating metrics...")
                _, result_json = evaluate(
                    data_name=data_name,
                    prompt_type=prompt_type,
                    samples=all_samples,
                    execute=True
                )
                
                metrics_file = final_out_file.with_name(final_out_file.stem + f'_{prompt_type}_metrics.json')
                with open(metrics_file, 'w') as f:
                    json.dump(result_json, f, indent=4)
                print(f"  - Saved metrics: {metrics_file}")
                
                # 删除分片文件
                for p_file in part_files:
                    os.remove(p_file)
                    # 同时删除对应的分片 metrics 文件
                    p_metrics = p_file.with_name(p_file.stem + f'_{prompt_type}_metrics.json')
                    if p_metrics.exists():
                        os.remove(p_metrics)
                print("  - Cleaned up shard files.")
                
            except Exception as e:
                print(f"  - [Error] Failed to evaluate merged results: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--prompt_type', type=str, default='qwen25-math-cot')
    args = parser.parse_args()
    
    merge_shard_files(args.out_root, args.run_name, args.prompt_type)