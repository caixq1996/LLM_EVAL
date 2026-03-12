# [Full Code] File: math_eval.py
import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_names', default='gsm8k,math', type=str)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--model_name_or_path', default='gpt-4', type=str)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--prompt_type', default='tool-integrated', type=str)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--num_test_sample', default=-1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument('--temperature', default=0, type=float)
    parser.add_argument('--n_sampling', default=1, type=int)
    parser.add_argument('--top_p', default=1, type=float)
    parser.add_argument('--max_tokens_per_call', default=8192, type=int)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--use_vllm', action='store_true')
    parser.add_argument('--vllm_batch_size', default=0, type=int)
    parser.add_argument('--save_outputs', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--use_safetensors', action='store_true')
    parser.add_argument('--num_shots', type=int, default=0)
    parser.add_argument('--apply_chat_template', action='store_true', help='Apply chat template to prompt.')
    parser.add_argument('--pipeline_parallel_size', type=int, default=1)
    parser.add_argument('--adapt_few_shot', action='store_true', help='Few shot for multiple-choice questions, zero shot for others.')
    # [NEW] 分片参数
    parser.add_argument('--generation_only', action='store_true', help='Only generate outputs, skip evaluation')
    parser.add_argument('--verify_only', action='store_true', help='Only verify pre-generated outputs')
    # [NEW] 分片参数
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args

def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)
    
    # [NEW] 数据切片逻辑
    if args.num_shards > 1:
        total_len = len(examples)
        chunk_size = total_len // args.num_shards
        # 处理余数，分配给最后一个 shard
        start_idx = args.shard_id * chunk_size
        if args.shard_id == args.num_shards - 1:
            end_idx = total_len
        else:
            end_idx = (args.shard_id + 1) * chunk_size
        
        # 记录切片前的原始索引，方便后续合并
        print(f"[Info] Sharding enabled: Process {args.shard_id}/{args.num_shards} handling range [{start_idx}:{end_idx}]")
        examples = examples[start_idx:end_idx]

    if args.num_test_sample > 0:
        # 注意：分片模式下 num_test_sample 通常指每个分片跑多少，或者忽略
        examples = examples[:args.num_test_sample]

    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)
    
    examples = examples[args.start:len(examples) if args.end == -1 else args.end]
    
    dt_string = datetime.now().strftime('%m-%d_%H-%M')
    model_name = '/'.join(args.model_name_or_path.split('/')[-2:])
    out_file_prefix = f'{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}'
    output_dir = args.output_dir
    
    # [NEW] 如果是多卡并行，给文件名加上分片后缀，避免写入冲突
    filename_suffix = ""
    if args.num_shards > 1:
        filename_suffix = f"_part{args.shard_id}"
        
    out_file = f'{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}{filename_suffix}.jsonl'
    os.makedirs(f'{output_dir}/{data_name}', exist_ok=True)
    
    # [ENHANCED] 收集所有已完成样本（支持跨 GPU 数量变化的增量恢复）
    processed_samples = []
    if not args.overwrite:
        ds_output_dir = f'{output_dir}/{data_name}'
        
        # 1. 读取所有已存在的 part 文件（跨不同 GPU 配置）
        import glob
        for existing_part in glob.glob(os.path.join(ds_output_dir, '*_part*.jsonl')):
            try:
                processed_samples.extend(list(load_jsonl(existing_part)))
            except Exception as e:
                print(f"[Warn] Failed to load {existing_part}: {e}")
        
        # 2. 也读取已合并的主文件（如果存在）
        base_pattern = out_file.replace(filename_suffix, '')
        if os.path.exists(base_pattern):
            try:
                processed_samples.extend(list(load_jsonl(base_pattern)))
            except Exception as e:
                print(f"[Warn] Failed to load merged file {base_pattern}: {e}")
        
        if processed_samples:
            print(f"[Info] Loaded {len(processed_samples)} completed samples from existing files")
            
    processed_samples = {sample['idx']: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example['idx'] not in processed_idxs]
    return (examples, processed_samples, out_file)

def _get_gen_chunk_size() -> int:
    val = os.getenv("EVAL_GEN_CHUNK_SIZE")
    if val is None or str(val).strip() == "":
        return 0
    try:
        size = int(val)
        return max(1, size)
    except ValueError:
        return 0

def _get_vllm_max_model_len() -> int:
    val = os.getenv("VLLM_MAX_MODEL_LEN", "4096")
    try:
        return max(1, int(val))
    except ValueError:
        return 4096

def _get_vllm_tokenizer(llm, tokenizer, args):
    if tokenizer is not None:
        return tokenizer
    if llm is not None and hasattr(llm, "get_tokenizer"):
        try:
            return llm.get_tokenizer()
        except Exception:
            pass
    try:
        if args and getattr(args, "model_name_or_path", ""):
            return AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    except Exception:
        return None
    return None

def _adjust_prompts_for_max_len(prompts, tokenizer, max_model_len):
    if tokenizer is None or not prompts:
        return prompts, 0, False
    lengths = []
    for p in prompts:
        try:
            lengths.append(len(tokenizer.encode(p, add_special_tokens=False)))
        except Exception:
            lengths.append(0)
    max_prompt_len = max(lengths) if lengths else 0
    truncated = False
    if max_prompt_len > max_model_len:
        new_prompts = []
        for p, l in zip(prompts, lengths):
            if l > max_model_len:
                try:
                    ids = tokenizer.encode(p, add_special_tokens=False)
                    ids = ids[-max_model_len:]
                    p = tokenizer.decode(ids, skip_special_tokens=True)
                    truncated = True
                except Exception:
                    pass
            new_prompts.append(p)
        prompts = new_prompts
        max_prompt_len = max_model_len
    return prompts, max_prompt_len, truncated

def setup(args):
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if args.use_vllm:
        llm = LLM(model=args.model_name_or_path, tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size, gpu_memory_utilization=0.9, pipeline_parallel_size=args.pipeline_parallel_size, trust_remote_code=True)
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(model_name_or_path=args.model_name_or_path, load_in_half=True, use_fast_tokenizer=True, use_safetensors=args.use_safetensors)
    data_list = args.data_names.split(',')
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))
    data_list.append('avg')
    results.append({'acc': sum([result['acc'] for result in results]) / len(results)})
    pad = max([len(data_name) for data_name in data_list])
    print('\t'.join((data_name.ljust(pad, ' ') for data_name in data_list)))
    print('\t'.join([f"{result['acc']:.1f}".ljust(pad, ' ') for result in results]))

def is_multi_choice(answer):
    for c in answer:
        if c not in ['A', 'B', 'C', 'D', 'E']:
            return False
    return True

def _run_one_chunk(llm, tokenizer, samples, data_name, args):
    # [PERF-OPT] 使用 vLLM 的 n 参数进行批量采样，而非复制 prompt
    # 注意：vLLM 要求 temperature=0 时 n 必须为 1（贪婪采样）
    use_vllm_n_sampling = args.use_vllm and args.n_sampling > 1 and args.temperature > 0

    if use_vllm_n_sampling:
        unique_prompts = [sample['prompt'] for sample in samples]
        if args.apply_chat_template:
            unique_prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt.strip()}], tokenize=False, add_generation_prompt=True) for prompt in unique_prompts]
        input_prompts = unique_prompts
    else:
        input_prompts = [sample['prompt'] for sample in samples for _ in range(args.n_sampling)]
        if args.apply_chat_template:
            input_prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt.strip()}], tokenize=False, add_generation_prompt=True) for prompt in input_prompts]

    remain_prompts = [(i, prompt) for i, prompt in enumerate(input_prompts)]
    end_prompts = []
    max_func_call = 1 if args.prompt_type in ['cot', 'pal'] else 4
    stop_words = ['</s>', '<|im_end|>', '<|endoftext|>']
    if args.prompt_type in ['cot']:
        stop_words.append('\n\nQuestion:')
    if args.prompt_type in ['pal', 'tool-integrated', 'jiuzhang_tora']:
        stop_words.extend(['\n\n---', '```output'])
    elif args.prompt_type in ['wizard_zs', 'platypus_fs']:
        stop_words.extend(['Instruction', 'Response'])
    elif 'jiuzhang' in args.prompt_type:
        stop_words.append('\n\n## Question')
    elif 'numina' in args.prompt_type:
        stop_words.append('\n### Problem')
    elif 'pure' in args.prompt_type:
        stop_words.append('\n\n\n')

    if 'pal' in args.prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    result_prompts = []
    for epoch in range(max_func_call):
        print('-' * 20, 'Epoch', epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break
        prompts = [item[1] for item in current_prompts]
        result_prompts.extend(prompts)
        remain_prompts = []
        remain_codes = []

        epoch_use_vllm_n_sampling = use_vllm_n_sampling and epoch == 0
        n_per_prompt = args.n_sampling if epoch_use_vllm_n_sampling else 1
        if args.use_vllm:
            total_prompts = len(prompts)
            tok = _get_vllm_tokenizer(llm, tokenizer, args)
            max_model_len = _get_vllm_max_model_len()
            adj_prompts, max_prompt_len, truncated = _adjust_prompts_for_max_len(prompts, tok, max_model_len)
            if truncated:
                print(f"[WARN] Truncated prompt(s) to max_model_len={max_model_len}", flush=True)
            if adj_prompts != prompts:
                prompts = adj_prompts
                current_prompts = [(current_prompts[i][0], prompts[i]) for i in range(len(prompts))]
                if epoch == 0:
                    input_prompts = prompts
            if max_prompt_len > 0:
                available = max_model_len - max_prompt_len
                if available < 1:
                    available = 1
                if available < args.max_tokens_per_call:
                    print(f"[WARN] max_tokens_per_call {args.max_tokens_per_call} -> {available} (max_model_len={max_model_len}, max_prompt_len={max_prompt_len})", flush=True)
                    effective_max_tokens = available
                else:
                    effective_max_tokens = args.max_tokens_per_call
            else:
                effective_max_tokens = args.max_tokens_per_call
            print(f'  [Sampling] Starting generation: {total_prompts} prompts x {n_per_prompt} samples/prompt = {total_prompts * n_per_prompt} total', flush=True)
            gen_start = time.time()
            stop_token_ids = [151645, 151643] if 'qwen2' in args.model_name_or_path.lower() else None
            sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=effective_max_tokens,
                stop=stop_words,
                stop_token_ids=stop_token_ids,
                n=n_per_prompt,
                top_p=1.0 if args.temperature > 0 else 1.0,
            )
            print(f'  [Sampling] Sending all {total_prompts} prompts to vLLM (n={n_per_prompt} per prompt)...', flush=True)
            results = llm.generate(prompts, sampling_params, use_tqdm=True)
            results = sorted(results, key=lambda x: int(x.request_id))
            outputs = []
            for result in results:
                for out in result.outputs:
                    outputs.append(out.text)
            gen_time = time.time() - gen_start
            throughput = len(outputs) / gen_time if gen_time > 0 else 0
            print(f'  [Sampling] Generated {len(outputs)} outputs in {gen_time:.1f}s ({throughput:.1f} samples/s)', flush=True)
        else:
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )

        expected_outputs = len(current_prompts) * n_per_prompt
        assert len(outputs) == expected_outputs, f"Expected {expected_outputs} outputs, got {len(outputs)}"

        if epoch_use_vllm_n_sampling:
            output_idx = 0
            for (i, query) in current_prompts:
                for sample_idx in range(args.n_sampling):
                    output = outputs[output_idx].rstrip()
                    output_idx += 1
                    full_query = query + output
                    expanded_idx = i * args.n_sampling + sample_idx
                    if args.prompt_type == 'pal':
                        remain_prompts.append((expanded_idx, full_query))
                        if '```python' in output:
                            output = extract_program(full_query)
                        remain_codes.append(output)
                    elif args.prompt_type == 'cot':
                        end_prompts.append((expanded_idx, full_query))
                    elif 'boxed' not in output and output.endswith('```'):
                        program = extract_program(full_query)
                        remain_prompts.append((expanded_idx, full_query))
                        remain_codes.append(program)
                    else:
                        end_prompts.append((expanded_idx, full_query))
        else:
            for (i, query), output in zip(current_prompts, outputs):
                output = output.rstrip()
                query += output
                if args.prompt_type == 'pal':
                    remain_prompts.append((i, query))
                    if '```python' in output:
                        output = extract_program(query)
                    remain_codes.append(output)
                elif args.prompt_type == 'cot':
                    end_prompts.append((i, query))
                elif 'boxed' not in output and output.endswith('```'):
                    program = extract_program(query)
                    remain_prompts.append((i, query))
                    remain_codes.append(program)
                else:
                    end_prompts.append((i, query))

        if remain_codes:
            remain_results = executor.batch_apply(remain_codes)
            for k in range(len(remain_prompts)):
                i, query = remain_prompts[k]
                res, report = remain_results[k]
                exec_result = res if res else report
                if 'pal' in args.prompt_type:
                    exec_result = '\\boxed{' + exec_result + '}'
                exec_result = f'\n```output\n{exec_result}\n```\n'
                query += exec_result
                if epoch == max_func_call - 1:
                    query += '\nReach max function call limit.'
                remain_prompts[k] = (i, query)

    print('Unsolved samples:', len(remain_prompts))
    end_prompts.extend(remain_prompts)
    end_prompts = sorted(end_prompts, key=lambda x: x[0])
    expected_count = len(samples) * args.n_sampling
    assert len(end_prompts) == expected_count, f"Expected {expected_count} end_prompts, got {len(end_prompts)}"

    codes = []
    for i in range(len(end_prompts)):
        _, end_prompt = end_prompts[i]
        if use_vllm_n_sampling:
            base_prompt_idx = i // args.n_sampling
            base_prompt = input_prompts[base_prompt_idx]
        else:
            base_prompt = input_prompts[i]
        code = end_prompt.split(base_prompt)[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    results = [run_execute(executor, code, args.prompt_type, data_name) for code in codes]
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling:(i + 1) * args.n_sampling]
        result = results[i * args.n_sampling:(i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample['gt'] in ['A', 'B', 'C', 'D', 'E'] and preds[j] not in ['A', 'B', 'C', 'D', 'E']:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample['gt']) and (not is_multi_choice(preds[j])):
                preds[j] = ''.join([c for c in preds[j] if c in ['A', 'B', 'C', 'D', 'E']])
        sample.pop('prompt')
        sample.update({'code': code, 'pred': preds, 'report': reports})
        all_samples.append(sample)

    return all_samples, result_prompts

def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print('=' * 50)
    print('data:', data_name, ' ,remain samples:', len(examples))
    if len(examples) > 0:
        print(examples[0])

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example['idx']
        example['question'] = parse_question(example, data_name)
        if example['question'] == '':
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example['gt_ans'] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)
        if idx == args.start:
            print(full_prompt)
        sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': full_prompt}
        for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', 'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer', 'difficulty']:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    start_time = time.time()
    chunk_size = _get_gen_chunk_size()
    if chunk_size <= 0:
        chunk_size = len(samples) if samples else 1
    print(f"[INFO] Generation chunk size: {chunk_size}")
    all_samples = []
    result_prompts = []
    for start in range(0, len(samples), chunk_size):
        chunk_samples = samples[start:start + chunk_size]
        if not chunk_samples:
            continue
        chunk_outputs, chunk_prompts = _run_one_chunk(llm, tokenizer, chunk_samples, data_name, args)
        all_samples.extend(chunk_outputs)
        result_prompts.extend(chunk_prompts)
        if args.save_outputs:
            save_jsonl(processed_samples + all_samples, out_file)
            print(f"[INFO] Saved {len(processed_samples) + len(all_samples)} samples (partial) to {out_file}")

    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(samples=all_samples, data_name=data_name, prompt_type=args.prompt_type, execute=True)
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)
    result_json['time_use_in_second'] = time.time() - start_time
    result_json['time_use_in_minite'] = f'{int(result_json["time_use_in_second"] // 60)}:{int(result_json["time_use_in_second"] % 60):02d}'
    result_json['prompts'] = result_prompts
    with open(out_file.replace('.jsonl', f'_{args.prompt_type}_metrics.json'), 'w') as f:
        json.dump(result_json, f, indent=4)
    return result_json

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    setup(args)
