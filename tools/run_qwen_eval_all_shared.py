import argparse
import importlib
import os
import re
import sys
import json
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import shutil, stat
import gc
import subprocess

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
EVAL_ROOT = THIS_DIR.parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from math_eval import main as eval_one_dataset
from model_utils import load_hf_lm_and_tokenizer
from tools.export_fsdp_dtensor_to_hf import export_one_step_to_hf, list_step_dirs

try:
    from vllm import LLM
    from transformers import AutoTokenizer
except Exception:
    LLM = None
    AutoTokenizer = None

GROUP_DATASETS = ('aime25x8,amc23x8,aime24x8', 'minerva_math,olympiadbench,math500')
EXPORT_ROOT = (Path(os.getenv("WORK_HOME", "/data/giil/caixq")) / "export").resolve()

def _safe_rmtree(p: Path):
    def _onerror(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
            func(path)
        except Exception:
            pass
    if p and p.exists():
        shutil.rmtree(p, onerror=_onerror)

def _now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def has_hf_weights(hf_dir):
    if not hf_dir or not hf_dir.exists():
        return False
    if list(hf_dir.glob('*.safetensors')):
        return True
    if list(hf_dir.glob('pytorch_model*.bin')) or (hf_dir / 'pytorch_model.bin.index.json').exists():
        return True
    return False

def _norm(s):
    return re.sub('[^a-z0-9]+', '', s.lower())

def find_base_model_dir(base_root, run_name):
    if not base_root or not base_root.exists():
        return None
    run_key = _norm(run_name)
    best = None
    for d in base_root.iterdir():
        if not d.is_dir():
            continue
        key = _norm(d.name)
        if not key:
            continue
        if key in run_key or run_key in key:
            if best is None or len(key) > len(_norm(best.name)):
                best = d
    return best

def _split_ds_list(datasets):
    return [d.strip() for d in datasets.split(',') if d.strip()]

def _metrics_exists(ds_dir):
    if not ds_dir.exists():
        return False
    return bool(list(ds_dir.glob('*metrics.json')))

def check_missing_by_group(out_root, run_name):
    missing = {1: [], 2: []}
    run_out = out_root / run_name
    for group_idx, datasets in enumerate(GROUP_DATASETS, start=1):
        gdir = run_out / f'g{group_idx}'
        ds_list = _split_ds_list(datasets)
        for ds in ds_list:
            ds_dir = gdir / ds
            if not _metrics_exists(ds_dir):
                missing[group_idx].append(ds)
    return missing

# [FIX] 增加 shard_id 和 num_shards 参数
def build_args_template(prompt_type, max_tokens, use_vllm, vllm_batch_size, pipeline_parallel_size, shard_id=0, num_shards=1):
    import types
    args = types.SimpleNamespace()
    args.data_names = ''
    args.data_dir = './data'
    args.model_name_or_path = ''
    args.output_dir = ''
    args.prompt_type = prompt_type
    args.split = 'test'
    args.num_test_sample = -1
    args.seed = 0
    args.start = 0
    args.end = -1
    args.temperature = 0.0
    args.n_sampling = 1
    args.top_p = 1.0
    args.max_tokens_per_call = int(max_tokens)
    args.shuffle = False
    args.use_vllm = bool(use_vllm)
    args.vllm_batch_size = int(vllm_batch_size) if vllm_batch_size else 0
    args.save_outputs = True
    args.overwrite = True
    args.use_safetensors = True
    args.num_shots = 0
    args.apply_chat_template = False
    args.pipeline_parallel_size = int(pipeline_parallel_size)
    args.adapt_few_shot = False
    # [FIX] 显式设置分片参数到 args 对象中
    args.shard_id = int(shard_id)
    args.num_shards = int(num_shards)
    return args

def load_llm_and_tokenizer(model_dir, use_vllm, pipeline_parallel_size):
    if use_vllm:
        assert LLM is not None, 'vLLM 未安装，请去掉 --use_vllm 或安装 vLLM'
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        ngpus = len([x for x in visible.split(',') if x.strip()]) or 1
        tp = max(1, ngpus // max(1, pipeline_parallel_size))
        llm = LLM(
            model=str(model_dir),
            tensor_parallel_size=tp,
            gpu_memory_utilization=0.85,
            enable_chunked_prefill=True,
            enable_sleep_mode=True,
            enforce_eager=True,
            pipeline_parallel_size=pipeline_parallel_size,
            trust_remote_code=True
        )
        tokenizer = None
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=str(model_dir),
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=True
        )
    return (llm, tokenizer)

import torch
import time

# [FIX] 增加分片参数
def run_groups_with_shared_llm(
    run_name,
    model_dir,
    out_root,
    prompt_type,
    max_tokens,
    use_vllm,
    vllm_batch_size,
    pipeline_parallel_size,
    missing=None,
    temperature_g1=0.6,
    temperature_g2=0.0,
    n_sampling_g1=1,
    n_sampling_g2=1,
    shard_id=0,
    num_shards=1
):
    out_run = out_root / run_name
    out_run.mkdir(parents=True, exist_ok=True)

    if missing is None:
        missing = check_missing_by_group(out_root=out_root, run_name=run_name)

    print(f'[{_now()}] ▶ 加载模型（一次）：{model_dir}', flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    llm, tokenizer = load_llm_and_tokenizer(model_dir, use_vllm, pipeline_parallel_size)

    # 打印分片信息
    if num_shards > 1:
        print(f'[{_now()}] ℹ️  当前工作节点分片: {shard_id}/{num_shards}', flush=True)

    print(f'[{_now()}] ✓ 模型就绪，开始评测 {run_name}（共享同一 LLM，仅补缺数据集）', flush=True)
    group_cfgs = {
        1: (GROUP_DATASETS[0], float(temperature_g1), int(n_sampling_g1)),
        2: (GROUP_DATASETS[1], float(temperature_g2), int(n_sampling_g2)),
    }

    for group_idx in (1, 2):
        ds_need = list(missing.get(group_idx, []))
        if not ds_need:
            continue

        datasets, temperature, n_sampling = group_cfgs[group_idx]
        gdir = out_run / f'g{group_idx}'
        gdir.mkdir(parents=True, exist_ok=True)

        print(f'[{_now()}] ▶ {run_name}/g{group_idx}  待评测={ds_need}  T={temperature}  n={n_sampling}', flush=True)

        with tqdm(total=len(ds_need), desc=f'{run_name}/g{group_idx}', unit='ds') as pbar:
            for ds in ds_need:
                try:
                    # [FIX] 传递分片参数给模板构建函数
                    args = build_args_template(
                        prompt_type, max_tokens, use_vllm, vllm_batch_size, pipeline_parallel_size,
                        shard_id=shard_id, num_shards=num_shards
                    )
                    args.temperature = float(temperature)
                    args.n_sampling = int(n_sampling)
                    args.top_p = 1.0 if args.temperature == 0 else 1.0
                    args.output_dir = str(gdir)

                    result = eval_one_dataset(llm, tokenizer, ds, args)
                    print(f'[{_now()}] ✓ {run_name}/g{group_idx}/{ds}  acc={result.get("acc", "nan")} pass_at_k={result.get("pass_at_k_percent", {})}', flush=True)
                except Exception as e:
                    print(f'[{_now()}] ⚠ 数据集 {run_name}/g{group_idx}/{ds} 失败：{e}', flush=True)
                    # 打印堆栈以便调试
                    import traceback
                    traceback.print_exc()
                finally:
                    if use_vllm:
                        try:
                            if hasattr(llm, 'clear_cache'):
                                llm.clear_cache()
                        except Exception:
                            pass
                    pbar.update(1)

    print(f'[{_now()}] ✅ 完成：{run_name}（g1+g2 缺失数据集已补全）', flush=True)

def _execute_payload(payload, exit_on_done=False):
    if isinstance(payload.get('missing'), dict):
        payload['missing'] = {int(k): v for k, v in payload['missing'].items()}
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    try:
        # [FIX] 从 payload 中提取分片参数并传递
        run_groups_with_shared_llm(
            run_name=payload['run_name'],
            model_dir=Path(payload['model_dir']),
            out_root=Path(payload['out_root']),
            prompt_type=payload['prompt_type'],
            max_tokens=int(payload['max_tokens']),
            use_vllm=bool(payload['use_vllm']),
            vllm_batch_size=int(payload['vllm_batch_size']),
            pipeline_parallel_size=int(payload['pipeline_parallel_size']),
            missing=payload.get('missing'),
            temperature_g1=float(payload['temperature_g1']),
            temperature_g2=float(payload['temperature_g2']),
            n_sampling_g1=int(payload['n_sampling_g1']),
            n_sampling_g2=int(payload['n_sampling_g2']),
            shard_id=int(payload.get('shard_id', 0)),
            num_shards=int(payload.get('num_shards', 1))
        )
    finally:
        if exit_on_done:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

def _worker_entry(payload_json):
    payload = json.loads(payload_json)
    _execute_payload(payload, exit_on_done=True)

# ... (中间部分 _load_worker_env_plugin, _partition_visible_gpus, _worker_loop, _execute_with_timeout 保持不变) ...
def _load_worker_env_plugin():
    mod_path = os.environ.get('EVAL_WORKER_ENV_PLUGIN', '').strip()
    if not mod_path:
        return None
    try:
        module = importlib.import_module(mod_path)
    except Exception as exc:
        print(f'[WARN] 无法导入 worker 插件模块 {mod_path}: {exc}', flush=True)
        return None
    prepare = getattr(module, 'prepare_worker_env', None)
    if not callable(prepare):
        print(f'[WARN] 插件 {mod_path} 缺少可调用的 prepare_worker_env(worker_idx, total_workers)', flush=True)
        return None
    print(f'[{_now()}] [INFO] 使用 worker 插件 {mod_path} 提供环境变量', flush=True)
    return prepare

def _partition_visible_gpus(devices: List[str], groups: int) -> List[str]:
    groups = max(1, groups)
    if not devices:
        return [''] * groups
    groups = min(groups, len(devices))
    base, extra = divmod(len(devices), groups)
    partitions = []
    start = 0
    for idx in range(groups):
        chunk_len = base + (1 if idx < extra else 0)
        chunk = devices[start:start + chunk_len]
        partitions.append(','.join(chunk))
        start += chunk_len
    return partitions

def _worker_loop(task_queue, result_queue, cuda_devices, extra_env=None):
    if cuda_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
    else:
        # If not specified, inheriting current env or clearing it might be safer depending on setup
        # But usually we want to keep what was passed or set in main
        pass
    if extra_env:
        os.environ.update({k: str(v) for k, v in extra_env.items()})
    # Path to this script to run as subprocess
    script_path = str(THIS_FILE)
    while True:
        payload = task_queue.get()
        if payload is None:
            break

        run_name = payload.get('run_name', 'unknown')
        timeout = payload.pop('_timeout', None)
        
        # Serialize payload to pass to subprocess
        payload_json = json.dumps(payload)

        # Construct command: python tools/run_qwen_eval_all_shared.py --_one_model_worker --_worker_payload "..."
        cmd = [
            sys.executable, 
            script_path, 
            '--_one_model_worker', 
            '--_worker_payload', payload_json
        ]

        try:
            # We use subprocess.run to isolate the vLLM lifecycle.
            # timeout parameter in subprocess.run handles the per-model timeout.
            # capture_output=False lets stdout/stderr flow to the main log.
            subprocess.run(
                cmd, 
                env=os.environ, 
                check=True, 
                timeout=timeout
            )
            result_queue.put({'run_name': run_name, 'status': 'ok'})
            
        except subprocess.TimeoutExpired:
            print(f"[{_now()}] [TIMEOUT] Worker subprocess timed out for {run_name}", flush=True)
            result_queue.put({'run_name': run_name, 'status': 'error', 'error': 'Timeout'})
            
        except subprocess.CalledProcessError as e:
            print(f"[{_now()}] [ERROR] Worker subprocess failed for {run_name} with exit code {e.returncode}", flush=True)
            result_queue.put({'run_name': run_name, 'status': 'error', 'error': f'Exit code {e.returncode}'})
            
        except Exception as exc:
            print(f"[{_now()}] [ERROR] Unexpected exception in worker loop for {run_name}: {exc}", flush=True)
            result_queue.put({'run_name': run_name, 'status': 'error', 'error': repr(exc)})

def _execute_with_timeout(payload, timeout):
    handler = None
    if isinstance(timeout, (int, float)) and timeout > 0:
        timeout = int(timeout)
        def _on_timeout(signum, frame):
            raise TimeoutError(f'Evaluation timeout after {timeout}s')
        handler = signal.signal(signal.SIGALRM, _on_timeout)
        signal.alarm(timeout)
    try:
        _execute_payload(payload)
    finally:
        if handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, handler)

def main():
    import torch.multiprocessing as mp
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    ap = argparse.ArgumentParser()
    ap.add_argument('--model_root', type=Path, required=False, help='包含多个训练 run 的根目录（checkpoints）')
    ap.add_argument('--out_root', type=Path, required=False, help='评测输出根目录')
    ap.add_argument('--prompt_type', default='qwen25-math-cot')
    ap.add_argument('--max_tokens_per_call', default='3072')
    ap.add_argument('--nproc', type=int, default=1, help='单机 GPU 数')
    ap.add_argument('--worker_concurrency', type=int, default=1, help='并发 worker 数')
    ap.add_argument('--base_root', type=Path, default=Path('/hss/giil/caixq/model'), help='base 模型根目录')
    ap.add_argument('--use_vllm', action='store_true', help='使用 vLLM（推荐）')
    ap.add_argument('--vllm_batch_size', type=int, default=0)
    ap.add_argument('--pipeline_parallel_size', type=int, default=1)
    ap.add_argument('--temperature_g1', type=float, default=0.6)
    ap.add_argument('--temperature_g2', type=float, default=0.0)
    ap.add_argument('--n_sampling_g1', type=int, default=1)
    ap.add_argument('--n_sampling_g2', type=int, default=1)
    ap.add_argument('--per_model_timeout', type=int, default=0, help='每个模型评测的最大时长（秒）')
    ap.add_argument('--skip_base_eval', action='store_true', help='跳过 base 模型评测')
    ap.add_argument('--cleanup_exported', action='store_true', help='评测完成后删除导出的 HF 目录')
    ap.add_argument('--_one_model_worker', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--_worker_payload', type=str, default='', help=argparse.SUPPRESS)
    # [FIX] 确保 argparse 能接收这两个参数
    ap.add_argument('--shard_id', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    args = ap.parse_args()

    if args._one_model_worker:
        _worker_entry(args._worker_payload)
        return

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    num_gpus = max(1, args.nproc)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(num_gpus))))
    os.environ.setdefault('PYTHONUNBUFFERED', '1')

    runs = sorted([p for p in args.model_root.iterdir() if p.is_dir()])
    if not runs:
        print(f'[WARN] {args.model_root} 下未发现 run 目录', flush=True)
        return

    # ... (中间部分检查 runs 逻辑保持不变) ...
    if args.cleanup_exported:
        print(f'[{_now()}] [INFO] --cleanup_exported 已忽略，导出目录将保留在 {EXPORT_ROOT}', flush=True)
    print(f'[{_now()}] 发现 {len(runs)} 个 run。', flush=True)
    
    base_done = {}
    export_root = EXPORT_ROOT
    export_root.mkdir(parents=True, exist_ok=True)
    env_tmo = int(os.environ.get('EVAL_ONE_MODEL_TIMEOUT', '0'))
    per_model_timeout = args.per_model_timeout if args.per_model_timeout > 0 else env_tmo if env_tmo > 0 else None
    visible_gpus = [g.strip() for g in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if g.strip()]
    max_workers = len(visible_gpus) if visible_gpus else max(1, args.nproc)
    worker_env_factory = _load_worker_env_plugin()
    if args.use_vllm:
        requested = max(1, args.worker_concurrency)
        if visible_gpus:
            requested = min(requested, len(visible_gpus))
        device_slices = _partition_visible_gpus(visible_gpus, requested)
    else:
        worker_concurrency = max(1, min(args.worker_concurrency, max_workers))
        cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        device_slices = [cuda_env] * worker_concurrency
    worker_concurrency = len(device_slices)
    worker_envs = []
    for idx in range(worker_concurrency):
        if worker_env_factory:
            try:
                env = worker_env_factory(idx, worker_concurrency) or {}
            except Exception as exc:
                print(f'[WARN] worker 插件生成环境失败：idx={idx} err={exc}', flush=True)
                env = {}
        else:
            env = {}
        worker_envs.append(env)
    if args.use_vllm and worker_concurrency > 1:
        info = ', '.join((f"{idx}:{dev or '<all>'}@[port={worker_envs[idx].get('VLLM_WORKER_ASSIGNED_PORT', 'n/a')}]" for idx, dev in enumerate(device_slices)))
        print(f'[{_now()}] [INFO] vLLM 并发 worker = {worker_concurrency}，设备分片: {info}', flush=True)
    ctx = mp.get_context('spawn')
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = []
    for idx, cuda_devices in enumerate(device_slices):
        proc = ctx.Process(target=_worker_loop, args=(task_queue, result_queue, cuda_devices, worker_envs[idx]))
        proc.start()
        workers.append(proc)
    scheduled = 0
    def enqueue(payload):
        nonlocal scheduled
        job = dict(payload)
        if per_model_timeout and per_model_timeout > 0:
            job['_timeout'] = int(per_model_timeout)
        task_queue.put(job)
        scheduled += 1
    errors = []
    
    try:
        for run in runs:
            run_name = run.name
            base_dir = find_base_model_dir(args.base_root, run_name)
            if base_dir is None or not has_hf_weights(base_dir):
                print(f'[WARN] 跳过: run={run_name}', flush=True)
                continue
            base_key = base_dir.name
            if not base_done.get(base_key, False):
                if args.skip_base_eval:
                    print(f'[{_now()}] ⏭ 跳过 base-only：{base_key}', flush=True)
                else:
                    missing = check_missing_by_group(out_root=out_root, run_name=f'base__{base_key}')
                    need_any = any((missing[g] for g in missing))
                    if need_any:
                        payload = {
                            'run_name': f'base__{base_key}', 'model_dir': str(base_dir), 'out_root': str(out_root),
                            'prompt_type': args.prompt_type, 'max_tokens': int(args.max_tokens_per_call),
                            'use_vllm': bool(args.use_vllm), 'vllm_batch_size': int(args.vllm_batch_size),
                            'pipeline_parallel_size': int(args.pipeline_parallel_size), 'missing': missing,
                            'temperature_g1': float(args.temperature_g1), 'temperature_g2': float(args.temperature_g2),
                            'n_sampling_g1': int(args.n_sampling_g1), 'n_sampling_g2': int(args.n_sampling_g2),
                            # [FIX] 在 payload 中加入分片参数
                            'shard_id': args.shard_id, 'num_shards': args.num_shards
                        }
                        enqueue(payload)
                    else:
                        print(f'[{_now()}] ⏭ 跳过 base-only：{base_key}', flush=True)
                base_done[base_key] = True
            step_dirs = list_step_dirs(run, only_latest=False)
            if not step_dirs:
                print(f'[WARN] 该 run 无可导出的分片模型：{run_name}')
                continue
            for step_dir in step_dirs:
                tag = f'{run_name}__{step_dir.name}'
                missing = check_missing_by_group(out_root=out_root, run_name=tag)
                need_any = any((missing[g] for g in missing))
                if not need_any:
                    print(f'[{_now()}] ⏭ 跳过：{tag}', flush=True)
                    continue
                try:
                    hf_dir = export_one_step_to_hf(step_dir, base_dir, export_root)
                except Exception as e:
                    print(f'[{_now()}] [WARN] 导出失败：{tag} -> {e}', flush=True)
                    continue
                payload = {
                    'run_name': tag, 'model_dir': str(hf_dir), 'out_root': str(out_root),
                    'prompt_type': args.prompt_type, 'max_tokens': int(args.max_tokens_per_call),
                    'use_vllm': bool(args.use_vllm), 'vllm_batch_size': int(args.vllm_batch_size),
                    'pipeline_parallel_size': int(args.pipeline_parallel_size), 'missing': missing,
                    'temperature_g1': float(args.temperature_g1), 'temperature_g2': float(args.temperature_g2),
                    'n_sampling_g1': int(args.n_sampling_g1), 'n_sampling_g2': int(args.n_sampling_g2),
                    # [FIX] 在 payload 中加入分片参数
                    'shard_id': args.shard_id, 'num_shards': args.num_shards
                }
                enqueue(payload)
    finally:
        for _ in workers:
            task_queue.put(None)
        results_received = 0
        while results_received < scheduled:
            result = result_queue.get()
            results_received += 1
            run_name = result.get('run_name', 'unknown')
            if result.get('status') != 'ok':
                errors.append(result)
                print(f"[{_now()}] ⚠ 模型 {run_name} 评测失败：{result.get('error')}", flush=True)
        for proc in workers:
            proc.join()
    if errors:
        raise RuntimeError('部分模型评测失败，详见日志输出。')
    print(f'[{_now()}] ✅ 全部评测完成。输出目录：{out_root}', flush=True)

if __name__ == '__main__':
    main()