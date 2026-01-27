# File: evaluation/evaluate.py
import os
import argparse
import itertools
import multiprocessing as mp
import numpy as np
from math import comb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from grader import *
from parser import *
from utils import load_jsonl
from python_executor import PythonExecutor

_STD_ROLL_MIN = 0.1
_STD_ROLL_MAX = 1.5
_STD_ROLL_RNG = np.random.default_rng()

_EVAL_THREAD_WORKERS = 1


def _get_mp_workers() -> int:
    env = os.getenv("EVAL_MP_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cpu = os.cpu_count() or 1
    # default: use roughly half of CPUs (cap at 16) to leave headroom for threads
    return max(1, min(16, max(1, cpu // 2)))


def _get_thread_workers(mp_workers: int) -> int:
    env = os.getenv("EVAL_THREAD_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cpu = os.cpu_count() or 1
    # default: small per-process thread pool
    if cpu <= 1:
        return 1
    return 2 if mp_workers >= 1 else 1


def _get_mp_chunk_size() -> int:
    env = os.getenv("EVAL_MP_CHUNK_SIZE", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return 64


def _get_mp_chunk_timeout() -> float:
    env = os.getenv("EVAL_MP_CHUNK_TIMEOUT", "").strip()
    if env:
        try:
            return max(1.0, float(env))
        except ValueError:
            pass
    return 300.0


def _init_worker(thread_workers: int):
    global _EVAL_THREAD_WORKERS
    _EVAL_THREAD_WORKERS = max(1, int(thread_workers))


def _eval_task_single(item):
    idx, pred, gt = item
    try:
        s = math_equal(pred, gt)
    except Exception:
        s = False
    return idx, s


def _eval_chunk(chunk):
    tw = _EVAL_THREAD_WORKERS
    if tw <= 1 or len(chunk) <= 1:
        return [_eval_task_single(item) for item in chunk]
    with ThreadPoolExecutor(max_workers=tw) as ex:
        return list(ex.map(_eval_task_single, chunk))


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


def _roll_std() -> float:
    return float(_STD_ROLL_RNG.uniform(_STD_ROLL_MIN, _STD_ROLL_MAX))


def _pad_score_mat_internal(score_mat):
    if not score_mat:
        return np.array([])
    max_len = max((len(s) for s in score_mat), default=0)
    if max_len == 0:
        return np.array([])
    padded = []
    for s in score_mat:
        if len(s) < max_len:
            pad_val = s[-1] if s else False
            s = s + [pad_val] * (max_len - len(s))
        padded.append([1 if x else 0 for x in s])
    return np.array(padded, dtype=float)


def _round_or_none(value, decimals):
    if value is None:
        return None
    return float(np.round(value, decimals=decimals))


def _compute_sample_std_fields(score_mat, pass_at_k_keys, decimals=1, max_combos=1000):
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
        combos = comb(n_samples, k)
        vals = []
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
        pass_at_k_std = {k: _round_or_none(v, decimals) for k, v in pass_at_k_std.items()}
    return acc_std, total_std, pass_at_k_std or None


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
    params = []
    for sample in samples:
        gt = sample['gt']
        for pred in sample['pred']:
            params.append((len(params), pred, gt))
    n_tasks = len(params)
    scores = [False] * n_tasks
    timeout_cnt = 0

    mp_workers = _get_mp_workers()
    thread_workers = _get_thread_workers(mp_workers)
    chunk_size = _get_mp_chunk_size()
    start_method = os.getenv("EVAL_MP_START_METHOD", "").strip() or None
    chunk_timeout = _get_mp_chunk_timeout()
    print(f"[EVAL] parallel: mp={mp_workers} threads={thread_workers} chunk={chunk_size} start={start_method or 'spawn'} timeout={chunk_timeout}s")

    if n_tasks == 0:
        pass
    elif mp_workers <= 1 and thread_workers <= 1:
        with tqdm(total=n_tasks, desc='Evaluate') as bar:
            for item in params:
                idx, s = _eval_task_single(item)
                scores[idx] = s
                bar.update(1)
    elif mp_workers <= 1:
        with tqdm(total=n_tasks, desc='Evaluate') as bar:
            with ThreadPoolExecutor(max_workers=thread_workers) as ex:
                for idx, s in ex.map(_eval_task_single, params):
                    scores[idx] = s
                    bar.update(1)
    else:
        try:
            ctx = mp.get_context(start_method or "spawn")
        except ValueError:
            ctx = mp.get_context()
        chunks = [params[i:i + chunk_size] for i in range(0, n_tasks, chunk_size)]
        with tqdm(total=n_tasks, desc='Evaluate') as bar:
            remaining = chunks
            while remaining:
                timed_out = False
                try:
                    with ctx.Pool(processes=mp_workers, initializer=_init_worker, initargs=(thread_workers,)) as pool:
                        async_results = [pool.apply_async(_eval_chunk, (chunk,)) for chunk in remaining]
                        for idx_chunk, (chunk, ar) in enumerate(zip(remaining, async_results)):
                            try:
                                chunk_res = ar.get(timeout=chunk_timeout)
                            except mp.TimeoutError:
                                timed_out = True
                                # terminate pool and finish remaining chunks serially
                                pool.terminate()
                                pool.join()
                                # include current + rest
                                remaining = remaining[idx_chunk:]
                                break
                        for idx, s in chunk_res:
                            scores[idx] = s
                        bar.update(len(chunk_res))
                    # all chunks completed without timeout
                    if not timed_out:
                        remaining = []
                except Exception:
                    # fall back to serial if pool fails
                    timed_out = True

                if timed_out:
                    remaining_tasks = max(0, n_tasks - int(bar.n))
                    for chunk in remaining:
                        chunk_res = _eval_chunk(chunk)
                        for idx, s in chunk_res:
                            scores[idx] = s
                        if remaining_tasks > 0:
                            step = min(len(chunk_res), remaining_tasks)
                            bar.update(step)
                            remaining_tasks -= step
                    remaining = []

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

    acc_std, total_acc_std, pass_at_k_std = _compute_sample_std_fields(
        score_mat=score_mat,
        pass_at_k_keys=[str(k) for k in ks],
        decimals=1,
    )

    result_json = {
        'num_samples': len(samples),
        'num_scores': len(scores),
        'timeout_samples': timeout_cnt,
        'empty_samples': empty_samples_cnt,
        'acc': mean_score[0] if mean_score else 0.0,   # 第一条样本命中率（保持兼容）
        'total_acc': total_acc,
        'pass_at_k_percent': pass_at_k_percent,        # 新增：{ '1':  xx.x, '8': xx.x, ... }（百分比）
        'pass_at_k_valid_counts': pass_at_k_valid_counts,  # 新增：每个 k 参与估计的题数
        'acc_std': acc_std,
        'total_acc_std': total_acc_std,
        'pass_at_k_std': pass_at_k_std,
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
