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
import torch
import time
try:
    from vllm.distributed.parallel_state import destroy_model_parallel
except ImportError:
    try:
        from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
    except ImportError:
        destroy_model_parallel = None

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

_DEFAULT_GROUP_DATASETS = (
    'aime25x8,amc23x8,aime24x8',
    'minerva_math,olympiadbench,math500',
)
GROUP_DATASETS = (
    os.getenv("EVAL_GROUP1_DATASETS", _DEFAULT_GROUP_DATASETS[0]),
    os.getenv("EVAL_GROUP2_DATASETS", _DEFAULT_GROUP_DATASETS[1]),
)
_export_root_env = os.getenv("EXPORT_ROOT")
_keep_exported_hf = os.getenv("KEEP_EXPORTED_HF", "false").lower() in ("true", "1", "yes")

# 如果 KEEP_EXPORTED_HF=false，使用 tmpfs 以避免磁盘配额问题
if not _keep_exported_hf and _export_root_env is None:
    # 使用 /dev/shm (tmpfs) 作为临时导出目录
    _tmpfs_export = Path("/dev/shm/eval_export") / str(os.getpid())
    _tmpfs_export.mkdir(parents=True, exist_ok=True)
    EXPORT_ROOT = _tmpfs_export
    print(f'[INFO] KEEP_EXPORTED_HF=false, using tmpfs for export: {EXPORT_ROOT}', flush=True)
else:
    EXPORT_ROOT = (
        Path(_export_root_env).expanduser().resolve()
        if _export_root_env
        else (Path(os.getenv("WORK_HOME", "/data/giil/caixq")) / "export").resolve()
    )

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

def _run_with_timeout(func, timeout_sec=30, desc="operation"):
    """
    在单独线程中执行函数，带超时保护。
    用于防止 cuda.synchronize() 或 destroy_model_parallel() 导致的死锁。
    """
    import threading
    result = [None]
    error = [None]
    
    def wrapper():
        try:
            result[0] = func()
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)
    
    if thread.is_alive():
        print(f"[{_now()}] [WARN] {desc} 超时 ({timeout_sec}s)，跳过", flush=True)
        return False
    
    if error[0] is not None:
        print(f"[{_now()}] [WARN] {desc} 异常: {error[0]}", flush=True)
        return False
    
    return True


def has_hf_weights(hf_dir):
    if not hf_dir or not hf_dir.exists():
        return False
    if list(hf_dir.glob('*.safetensors')):
        return True
    if list(hf_dir.glob('pytorch_model*.bin')) or (hf_dir / 'pytorch_model.bin.index.json').exists():
        return True
    return False


def is_adapter_checkpoint(step_dir: Path) -> bool:
    """检测是否为 adapter 格式 checkpoint（actor/ 下有 adapter_config.json）"""
    actor = step_dir / "actor"
    return (actor / "adapter_config.json").exists()


def get_adapter_base_model_suffix(step_dir: Path):
    """
    从路径名检测是否包含特殊算法名（如 pissa, qpissa），返回对应的 base model 后缀。
    通过环境变量 SPECIAL_ADAPTER_ALGORITHMS 配置映射，格式: "algo1:suffix1,algo2:suffix2,..."
    例如: "pissa:_pissa_base,qpissa:_qpissa_base"
    """
    # 从环境变量读取算法映射
    special_algos = os.environ.get("SPECIAL_ADAPTER_ALGORITHMS", "pissa:_pissa_base,qpissa:_qpissa_base")
    if not special_algos:
        return None
    
    # 解析映射配置
    algo_map = {}
    for pair in special_algos.split(','):
        pair = pair.strip()
        if ':' not in pair:
            continue
        algo, suffix = pair.split(':', 1)
        algo_map[algo.strip().lower()] = suffix.strip()
    
    if not algo_map:
        return None
    
    # 从路径中提取所有组件（父目录名 + checkpoint 目录名）
    path_parts = [step_dir.name, step_dir.parent.name]
    path_str = '_'.join(path_parts).lower()
    
    # 按算法名长度降序排序，避免短名称被子串匹配（如 pissa 匹配到 qpissa）
    sorted_algos = sorted(algo_map.items(), key=lambda x: len(x[0]), reverse=True)
    
    # 检查路径中是否包含特殊算法名（使用单词边界）
    for algo, suffix in sorted_algos:
        # 使用下划线或路径分隔符作为边界
        if f'_{algo}_' in f'_{path_str}_' or f'_{algo}/' in f'_{path_str}/':
            return suffix
    
    return None


def _norm(s):
    return re.sub('[^a-z0-9]+', '', s.lower())


def find_base_model_dir(base_root, run_name, adapter_suffix=None):
    """
    查找 base model 目录。
    adapter_suffix: 可选后缀（如 "_pissa_base"）用于 PiSSA/QPiSSA。
    """
    if not base_root or not base_root.exists():
        return None
    run_key = _norm(run_name)
    best = None
    
    # 如果有 adapter_suffix，优先查找带后缀的目录
    if adapter_suffix:
        for d in base_root.iterdir():
            if not d.is_dir():
                continue
            if d.name.endswith(adapter_suffix):
                # 提取不带后缀的名称进行匹配
                base_name = d.name[:-len(adapter_suffix)]
                key = _norm(base_name)
                if key and (key in run_key or run_key in key):
                    if best is None or len(key) > len(_norm(best.name.replace(adapter_suffix, ''))):
                        best = d
        if best:
            return best
    
    # 标准匹配逻辑
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

# 预期样本数量（从 base 模型结果中获取，或使用硬编码默认值）
_EXPECTED_SAMPLES_CACHE = {}

def _load_expected_samples():
    """Load expected sample counts from base model results."""
    global _EXPECTED_SAMPLES_CACHE
    if _EXPECTED_SAMPLES_CACHE:
        return _EXPECTED_SAMPLES_CACHE
    
    # 默认值（硬编码作为后备）
    defaults = {
        'aime24x8': 240,
        'aime25x8': 240,
        'amc23x8': 320,
        'math500': 500,
        'minerva_math': 272,
        'olympiadbench': 675,
    }
    
    # 尝试从 base 模型结果加载
    base_results_dirs = [
        Path(os.getenv('EVAL_BASE_RESULTS', '')),
        EVAL_ROOT / 'rl_reasoning_results' / 'base__Qwen2.5-math-7B',
        EVAL_ROOT / 'rl_reasoning_results' / 'base__Qwen2.5-Math-7B',
    ]
    
    for base_dir in base_results_dirs:
        if not base_dir or not base_dir.exists():
            continue
        for g in ['g1', 'g2']:
            gdir = base_dir / g
            if not gdir.exists():
                continue
            for ds_dir in gdir.iterdir():
                if not ds_dir.is_dir():
                    continue
                # 查找非 part 的 metrics 文件
                metrics_files = [f for f in ds_dir.glob('*metrics.json') if '_part' not in f.name]
                if metrics_files:
                    try:
                        with open(metrics_files[0]) as f:
                            data = json.load(f)
                            if 'num_samples' in data:
                                _EXPECTED_SAMPLES_CACHE[ds_dir.name] = data['num_samples']
                    except Exception:
                        pass
    
    # 使用默认值填充缺失的数据集
    for k, v in defaults.items():
        if k not in _EXPECTED_SAMPLES_CACHE:
            _EXPECTED_SAMPLES_CACHE[k] = v
    
    return _EXPECTED_SAMPLES_CACHE

def _is_dataset_complete(ds_dir, ds_name):
    """
    检查数据集评测是否完成。
    
    完成条件：
    1. 存在非 part 的最终 metrics.json，且 num_samples 等于预期值
    2. 或者：所有 part metrics.json 的 num_samples 之和等于预期值
    """
    if not ds_dir.exists():
        return False
    
    expected_samples = _load_expected_samples()
    expected = expected_samples.get(ds_name, 0)
    if expected <= 0:
        # 如果没有预期值，回退到简单检查
        return bool(list(ds_dir.glob('*metrics.json')))
    
    # 首先检查是否存在非 part 的最终 metrics 文件
    final_metrics = [f for f in ds_dir.glob('*metrics.json') if '_part' not in f.name]
    if final_metrics:
        try:
            with open(final_metrics[0]) as f:
                data = json.load(f)
                if data.get('num_samples', 0) >= expected:
                    return True
        except Exception:
            pass
    
    # 检查所有 part metrics 的样本数之和
    part_metrics = sorted(ds_dir.glob('*_part*_*metrics.json'))
    if not part_metrics:
        return False
    
    total_samples = 0
    for pm in part_metrics:
        try:
            with open(pm) as f:
                data = json.load(f)
                total_samples += data.get('num_samples', 0)
        except Exception:
            pass
    
    return total_samples >= expected

def check_missing_by_group(out_root, run_name):
    missing = {1: [], 2: []}
    run_out = out_root / run_name
    for group_idx, datasets in enumerate(GROUP_DATASETS, start=1):
        gdir = run_out / f'g{group_idx}'
        ds_list = _split_ds_list(datasets)
        for ds in ds_list:
            ds_dir = gdir / ds
            if not _is_dataset_complete(ds_dir, ds):
                missing[group_idx].append(ds)
    return missing

# [FIX] 增加 shard_id 和 num_shards 参数
def build_args_template(prompt_type, max_tokens, use_vllm, vllm_batch_size, pipeline_parallel_size, shard_id=0, num_shards=1):
    import types
    args = types.SimpleNamespace()
    args.data_names = ''
    args.data_dir = os.getenv("EVAL_DATA_DIR", "./data")
    args.model_name_or_path = ''
    args.output_dir = ''
    args.prompt_type = prompt_type
    args.split = 'test'
    num_test_sample = os.getenv("EVAL_NUM_TEST_SAMPLE")
    if num_test_sample is not None and str(num_test_sample).strip() != "":
        args.num_test_sample = int(num_test_sample)
    else:
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
        
        # [PERF-OPT] 优化 vLLM 配置以最大化吞吐量
        # 1. enable_prefix_caching: 关键优化！对于 pass@k，同一 prompt 的 n 个采样可以共享 KV cache
        # 2. enforce_eager=False: 启用 CUDA graphs 加速 kernel 执行
        # 3. gpu_memory_utilization=0.95: 使用更多显存用于 KV cache
        # 4. max_model_len: 限制单个序列长度以容纳更多并发序列
        # 5. disable_log_stats: 减少日志开销
        
        max_model_len = int(os.environ.get('VLLM_MAX_MODEL_LEN', '4096'))
        gpu_mem_util = float(os.environ.get('VLLM_GPU_MEMORY_UTILIZATION', '0.95'))
        
        print(f'[PERF] vLLM config: tp={tp}, max_model_len={max_model_len}, gpu_mem={gpu_mem_util}', flush=True)
        
        llm = LLM(
            model=str(model_dir),
            tensor_parallel_size=tp,
            pipeline_parallel_size=pipeline_parallel_size,
            gpu_memory_utilization=gpu_mem_util,  # 增加到 0.95
            max_model_len=max_model_len,  # 限制序列长度以增加并发
            enable_prefix_caching=True,  # 关键：prefix caching 让同一 prompt 的采样共享 KV cache
            enable_chunked_prefill=False,  # 禁用 chunked prefill，对 decoding-heavy 任务更快
            enforce_eager=False,  # 启用 CUDA graphs
            trust_remote_code=True,
            disable_log_stats=True,  # 减少日志开销
            # 额外优化
            swap_space=0,  # 禁用 swap 到 CPU，强制使用 GPU
            disable_custom_all_reduce=False,  # 保持高效 all-reduce
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
    temperature_g2=0.8,
    n_sampling_g1=1,
    n_sampling_g2=8,
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
    
    # 模型已加载到 GPU，如果使用 tmpfs 导出则立即删除以释放内存
    if not _keep_exported_hf and str(model_dir).startswith('/dev/shm'):
        try:
            model_dir_path = Path(model_dir)
            if model_dir_path.exists():
                _safe_rmtree(model_dir_path)
                print(f'[{_now()}] 🧹 已删除 tmpfs 导出目录: {model_dir}', flush=True)
        except Exception as e:
            print(f'[{_now()}] [WARN] 删除 tmpfs 导出失败: {e}', flush=True)

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
    if use_vllm:
        print(f'[{_now()}] 🧹 正在显式释放 vLLM 资源...', flush=True)
        # 定义显存打印辅助函数
        def _log_mem(tag):
            if torch.cuda.is_available():
                try:
                    # 获取当前设备索引
                    device = torch.cuda.current_device()
                    # 字节转GB
                    alloc = torch.cuda.memory_allocated(device) / (1024**3)
                    reserved = torch.cuda.memory_reserved(device) / (1024**3)
                    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                    print(f"[{_now()}] 📊 [{tag}] GPU Mem: Alloc={alloc:.2f}GB / Rsrv={reserved:.2f}GB / Total={total:.2f}GB", flush=True)
                except Exception:
                    pass
        
        # 辅助函数：杀死所有 vLLM EngineCore 子进程
        def _kill_vllm_child_processes():
            """Kill orphaned vLLM EngineCore child processes to free GPU memory."""
            import psutil
            current_pid = os.getpid()
            try:
                current_process = psutil.Process(current_pid)
                children = current_process.children(recursive=True)
                for child in children:
                    try:
                        cmdline = ' '.join(child.cmdline())
                        # Kill vLLM engine core processes and any CUDA-related children
                        if 'EngineCore' in cmdline or 'vllm' in cmdline.lower() or child.name() == 'python':
                            child.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                # Wait for processes to terminate
                _, alive = psutil.wait_procs(children, timeout=5)
                # Force kill any remaining
                for p in alive:
                    try:
                        p.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except Exception as e:
                print(f"[{_now()}] [WARN] 清理子进程时发生错误: {e}", flush=True)
        
        # 1. 打印清理前状态
        _log_mem("Before Cleanup")
        
        # 清理超时设置（秒）
        CLEANUP_TIMEOUT = int(os.environ.get('VLLM_CLEANUP_TIMEOUT', '30'))
        
        # 使用 stderr 重定向来抑制已知的 CUDAPluggableAllocator 错误
        import io
        import contextlib
        
        # 保存原始 stderr
        _orig_stderr = sys.stderr
        _captured_stderr = io.StringIO()
        
        try:
            # 临时捕获 stderr 以抑制 CUDAPluggableAllocator 噪音
            sys.stderr = _captured_stderr
            
            # 2. 删除对象并强制 GC
            del llm
            gc.collect()
            
            # 3. 杀死所有 vLLM 子进程以释放 GPU 内存
            _kill_vllm_child_processes()
            
            # 4. 销毁分布式组 (带超时保护，防止 NCCL 死锁)
            if destroy_model_parallel is not None:
                _run_with_timeout(
                    destroy_model_parallel,
                    timeout_sec=CLEANUP_TIMEOUT,
                    desc="destroy_model_parallel"
                )
            
            # 5. 清理 PyTorch 缓存 (带超时保护，防止 cuda.synchronize 死锁)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                _run_with_timeout(
                    torch.cuda.synchronize,
                    timeout_sec=CLEANUP_TIMEOUT,
                    desc="cuda.synchronize"
                )
            
            # 6. 再次尝试 GC 和缓存清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 7. 较长休眠确保驱动层完全回收
            time.sleep(2.0)

        except Exception as e:
            print(f"[{_now()}] [ERROR] 资源释放过程中发生错误: {e}", flush=True)
        finally:
            # 恢复 stderr
            sys.stderr = _orig_stderr
            
            # 过滤并打印非 CUDAPluggableAllocator 的错误
            captured = _captured_stderr.getvalue()
            if captured:
                for line in captured.splitlines():
                    # 过滤已知的非致命 CUDA 分配器错误
                    if 'CUDAPluggableAllocator' not in line and 'Trying to free a pointer' not in line:
                        print(line, file=sys.stderr, flush=True)

        # 8. 打印清理后状态
        _log_mem("After Cleanup")

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
        
        # 超时改为活跃度检测：只有在一段时间内没有任何输出时才超时
        # timeout 变量现在表示"无活动超时"（idle timeout），而不是总超时
        IDLE_TIMEOUT_SEC = int(os.environ.get('EVAL_IDLE_TIMEOUT', timeout or 1800))  # 默认30分钟无活动超时
        
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
            # 使用 Popen 进行基于活跃度的超时监控
            # 只要子进程还在产生输出（stdout/stderr），就不会超时
            import select
            
            proc = subprocess.Popen(
                cmd,
                env=os.environ,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line buffered
            )
            
            last_activity_time = time.time()
            
            while proc.poll() is None:  # 进程仍在运行
                # 使用 select 等待输出，最多等待 10 秒
                ready, _, _ = select.select([proc.stdout], [], [], 10.0)
                
                if ready:
                    line = proc.stdout.readline()
                    if line:
                        # 有输出，更新活跃时间并打印
                        last_activity_time = time.time()
                        print(line, end='', flush=True)
                else:
                    # 检查是否超过无活动超时
                    idle_time = time.time() - last_activity_time
                    if IDLE_TIMEOUT_SEC > 0 and idle_time > IDLE_TIMEOUT_SEC:
                        print(f"[{_now()}] [IDLE_TIMEOUT] No output for {idle_time:.0f}s, terminating {run_name}", flush=True)
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        result_queue.put({'run_name': run_name, 'status': 'error', 'error': f'Idle timeout ({idle_time:.0f}s)'})
                        break
            else:
                # 进程正常结束，读取剩余输出
                remaining = proc.stdout.read()
                if remaining:
                    print(remaining, end='', flush=True)
                
                if proc.returncode == 0:
                    result_queue.put({'run_name': run_name, 'status': 'ok'})
                else:
                    print(f"[{_now()}] [ERROR] Worker subprocess failed for {run_name} with exit code {proc.returncode}", flush=True)
                    result_queue.put({'run_name': run_name, 'status': 'error', 'error': f'Exit code {proc.returncode}'})
            
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
    ap.add_argument('--temperature_g2', type=float, default=0.8)
    ap.add_argument('--n_sampling_g1', type=int, default=1)
    ap.add_argument('--n_sampling_g2', type=int, default=8)
    ap.add_argument('--per_model_timeout', type=int, default=0, help='每个模型评测的最大时长（秒）')
    ap.add_argument('--skip_base_eval', action='store_true', help='跳过 base 模型评测')
    ap.add_argument('--skip_step_eval', action='store_true', help='跳过 global_step 评测（仅跑 base）')
    ap.add_argument('--cleanup_exported', action='store_true', help='评测完成后删除导出的 HF 目录')
    ap.add_argument('--steps', type=str, default='', help='Comma-separated step ids or names, e.g. 100,200 or global_step_100')
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

    # 检测 MODEL_ROOT 是否直接是 run 目录（包含 global_step_*）
    all_subdirs = [p for p in args.model_root.iterdir() if p.is_dir()]
    is_single_run = any(p.name.startswith('global_step_') for p in all_subdirs)
    
    # [NEW] 支持 DEEP_STEP_FILTER 环境变量（从 multi-submit-deep 传入）
    deep_step_filter = os.environ.get('DEEP_STEP_FILTER', '').strip()
    if deep_step_filter:
        # DEEP_STEP_FILTER 优先级高于 --steps 参数
        args.steps = deep_step_filter
        print(f'[{_now()}] [INFO] 使用 DEEP_STEP_FILTER 过滤 step: {deep_step_filter}', flush=True)
    
    def _filter_steps(step_dirs, steps_spec):
        if not steps_spec:
            return step_dirs
        wanted = set()
        for token in steps_spec.split(','):
            t = token.strip()
            if not t:
                continue
            if t.isdigit():
                wanted.add(f'global_step_{t}')
            else:
                wanted.add(t)
        if not wanted:
            return step_dirs
        return [p for p in step_dirs if p.name in wanted]

    if is_single_run:
        # MODEL_ROOT 本身就是一个 run 目录，使用其目录名查找 base 模型
        run_name_for_base = args.model_root.name
        runs = sorted([p for p in all_subdirs if p.name.startswith('global_step_')])
        if args.steps:
            runs = _filter_steps(runs, args.steps)
        print(f'[{_now()}] [INFO] 检测到单 run 目录模式，run_name_for_base={run_name_for_base}', flush=True)
    else:
        # MODEL_ROOT 下有多个 run 目录
        run_name_for_base = None
        runs = sorted(all_subdirs)
    
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
            # 优先使用 run_name_for_base（单 run 目录模式）
            lookup_name = run_name_for_base if run_name_for_base else run_name
            base_dir = find_base_model_dir(args.base_root, lookup_name)
            if base_dir is None or not has_hf_weights(base_dir):
                print(f'[WARN] 跳过: run={run_name} (lookup_name={lookup_name})', flush=True)
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

            if args.skip_step_eval:
                continue

            # 单 run 目录模式: runs 已经是 global_step_* 目录列表，直接使用
            # 多 run 目录模式: run 是 run_name 目录，需要在其下查找 global_step_*
            if is_single_run:
                step_dirs = [run]  # run 本身就是 step_dir
            else:
                step_dirs = list_step_dirs(run, only_latest=False)
            if args.steps:
                step_dirs = _filter_steps(step_dirs, args.steps)

            if not step_dirs:
                print(f'[WARN] 该 run 无可导出的分片模型：{run_name}')
                continue
            for step_dir in step_dirs:
                # 为保持输出目录结构一致，使用 lookup_name 作为父目录
                # 输出结构: out_root/{safe_run_name}/{run_name}__{step_name}/g{1,2}/...
                safe_run_name = run_name.replace('.', '_').replace('-', '_')
                if is_single_run:
                    tag = f'{safe_run_name}/{lookup_name}__{step_dir.name}'
                else:
                    tag = f'{safe_run_name}/{run_name}__{step_dir.name}'
                missing = check_missing_by_group(out_root=out_root, run_name=tag)
                need_any = any((missing[g] for g in missing))
                if not need_any:
                    print(f'[{_now()}] ⏭ 跳过：{tag}', flush=True)
                    continue

                # 检测 adapter 格式并获取可能的 PiSSA/QPiSSA 后缀
                step_base_dir = base_dir
                if is_adapter_checkpoint(step_dir):
                    adapter_suffix = get_adapter_base_model_suffix(step_dir)
                    if adapter_suffix:
                        # 重新查找带后缀的 base model（如 _pissa_base, _qpissa_base）
                        step_base_dir = find_base_model_dir(args.base_root, lookup_name, adapter_suffix=adapter_suffix)
                        if step_base_dir is None or not has_hf_weights(step_base_dir):
                            print(f'[{_now()}] [WARN] 跳过：{tag}（找不到 {adapter_suffix} base model）', flush=True)
                            continue
                        print(f'[{_now()}] ℹ️  使用特殊 base: {step_base_dir.name}', flush=True)
                    else:
                        print(f'[{_now()}] ℹ️  检测到 adapter checkpoint: {tag}', flush=True)

                try:
                    hf_dir = export_one_step_to_hf(step_dir, step_base_dir, export_root)
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
