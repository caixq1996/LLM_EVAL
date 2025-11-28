# File: evaluation/evaluate.py
import os
import argparse
import numpy as np
from math import comb
from tqdm import tqdm
from grader import *
from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor


def _estimate_pass_at_k_one(scores, k):
    """
    scores: List[bool]，同一道题的多条采样是否命中
    k: int
    返回：None（n<k 时不可无偏估计，按 HumanEval 做法跳过），或 [0,1] 间的浮点数
    公式参考 Codex/HumanEval：1 - C(n-c, k)/C(n, k)
    """
    n = len(scores)
    if n < k:
        return None
    c = int(sum(1 for s in scores if s))
    if c == 0:
        return 0.0
    # comb(a,b) 在 b>a 时会报错，因此按数学定义处理 n-c<k 的情况：分子=0 => 结果为 1.0
    if n - c < k:
        return 1.0
    return 1.0 - (comb(n - c, k) / comb(n, k))


def _compute_pass_at_k(score_mat, ks):
    """
    score_mat: List[List[bool]]，每题一个列表，列表里是该题的多采样是否命中
    ks: 需要计算的 k 列表（均为正整数）
    返回：(pass_at_k_percent, valid_counts)
      - pass_at_k_percent: {str(k): 百分数(保留1位小数) 或 None(全题 n<k)}
      - valid_counts: {str(k): 参与该 k 估计的题数}
    """
    results = {}
    counts = {}
    for k in ks:
        vals = []
        for scores in score_mat:
            v = _estimate_pass_at_k_one(scores, k)
            if v is not None:
                vals.append(v)
        if len(vals) == 0:
            results[str(k)] = None
            counts[str(k)] = 0
        else:
            results[str(k)] = float(np.round(np.mean(vals) * 100.0, 1))
            counts[str(k)] = len(vals)
    return results, counts


def evaluate(data_name, prompt_type, samples=None, file_path=None, max_num_samples=None, execute=False):
    assert samples or file_path, 'samples or file_path must be provided'
    if not samples:
        samples = list(load_jsonl(file_path))

    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx'])
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f'max_num_samples: {max_num_samples} / {len(samples)}')
        samples = samples[:max_num_samples]

    # 解析 GT
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)

    # 逐条预测打分（按你已有的 math_equal）
    params = [(si, pred, sample['gt']) for si, sample in enumerate(samples) for pred in sample['pred']]
    n_tasks = len(params)
    scores = []
    timeout_cnt = 0
    with tqdm(total=n_tasks, desc='Evaluate') as bar:
        for _, pred, gt in params:
            try:
                s = math_equal(pred, gt)
            except Exception:
                s = False
            scores.append(s)
            bar.update(1)

    # 回填每题的 score 列表
    idx = 0
    score_mat = []
    for sample in samples:
        k = len(sample['pred'])
        sample_scores = scores[idx:idx + k]
        idx += k
        if len(sample_scores) < k:
            sample_scores = sample_scores + [False] * (k - len(sample_scores))
        sample['score'] = sample_scores
        score_mat.append(sample_scores)

    # 列平均（保留兼容：第一列相当于“第一条样本命中率”）
    max_len = max((len(s) for s in score_mat)) if score_mat else 0
    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            pad_val = s[-1] if s else False
            score_mat[i] = s + [pad_val] * (max_len - len(s))
    col_means = np.array(score_mat).mean(axis=0) if max_len > 0 else np.array([0.0])
    mean_score = list(np.round(col_means * 100, decimals=1))

    # total_acc：把所有候选一视同仁求均值（保留兼容）
    all_flat_scores = np.array([int(x) for row in score_mat for x in row]) if score_mat else np.array([])
    total_acc = float(np.mean(all_flat_scores) * 100) if all_flat_scores.size > 0 else 0.0

    # 空样本统计（保留兼容）
    empty_samples_cnt = sum((not s.get('pred') or not s['pred'][-1] for s in samples))

    # === 新增：pass@k ===
    # 读取待计算的 ks：优先环境变量 PASS_AT_KS（如 "1,8,10"），否则默认算 1 和 8
    ks_env = os.environ.get('PASS_AT_KS', '')
    if ks_env.strip():
        ks = [int(x) for x in ks_env.replace(' ', '').split(',') if x.strip().isdigit()]
    else:
        ks = [1, 8]
    # 只保留正数，并且不超过当前采样最大长度
    ks = sorted({k for k in ks if k > 0 and (max_len == 0 or k <= max_len)})
    pass_at_k_percent, pass_at_k_valid_counts = _compute_pass_at_k(score_mat, ks)

    result_json = {
        'num_samples': len(samples),
        'num_scores': len(scores),
        'timeout_samples': timeout_cnt,
        'empty_samples': empty_samples_cnt,
        'acc': mean_score[0] if mean_score else 0.0,   # 第一条样本命中率（保持兼容）
        'total_acc': total_acc,
        'pass_at_k_percent': pass_at_k_percent,        # 新增：{ '1':  xx.x, '8': xx.x, ... }（百分比）
        'pass_at_k_valid_counts': pass_at_k_valid_counts,  # 新增：每个 k 参与估计的题数
    }

    # 如有类型字段，给出各类型的“最后一条样本命中率”（保持兼容）以及可选的 type-wise pass@k
    if 'type' in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1] if sample['score'] else False)
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

        # 可选：type-wise pass@k（仅在你需要时查看）
        type_pass = {}
        for t in sorted({s['type'] for s in samples}):
            sub_scores = [s['score'] for s in samples if s['type'] == t]
            t_pass, _ = _compute_pass_at_k(sub_scores, ks)
            type_pass[t] = t_pass
        result_json['type_pass_at_k_percent'] = type_pass

    print(result_json)
    return (samples, result_json)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='math')
    parser.add_argument('--prompt_type', type=str, default='tool-integrated')
    parser.add_argument('--file_path', type=str, default=None, required=True)
    parser.add_argument('--max_num_samples', type=int, default=None)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
