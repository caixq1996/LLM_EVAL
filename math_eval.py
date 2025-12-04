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
    
    processed_samples = []
    if not args.overwrite:
        # 在分片模式下，只读取自己分片的已完成文件
        if os.path.exists(out_file):
            processed_samples.extend(list(load_jsonl(out_file)))
            
    processed_samples = {sample['idx']: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example['idx'] not in processed_idxs]
    return (examples, processed_samples, out_file)

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

def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print('=' * 50)
    print('data:', data_name, ' ,remain samples:', len(examples))
    if len(examples) > 0:
        print(examples[0])
    if 'pal' in args.prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)
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
    input_prompts = [sample['prompt'] for sample in samples for _ in range(args.n_sampling)]
    if args.apply_chat_template:
        input_prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt.strip()}], tokenize=False, add_generation_prompt=True) for prompt in input_prompts]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
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
    start_time = time.time()
    result_prompts = []
    for epoch in range(max_func_call):
        print('-' * 20, 'Epoch', epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break
        prompts = [item[1] for item in current_prompts]
        result_prompts.extend(prompts)
        if args.use_vllm:
            if args.vllm_batch_size:
                outputs = []
                for i in range(0, len(prompts), args.vllm_batch_size):
                    batch_prompts = prompts[i:i + args.vllm_batch_size]
                    batch_outputs = llm.generate(batch_prompts, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens_per_call, stop=stop_words, stop_token_ids=[151645, 151643] if 'qwen2' in args.model_name_or_path.lower() else None))
                    batch_outputs = sorted(batch_outputs, key=lambda x: int(x.request_id))
                    batch_outputs = [output.outputs[0].text for output in batch_outputs]
                    outputs.extend(batch_outputs)
            else:
                outputs = llm.generate(prompts, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens_per_call, stop=stop_words, stop_token_ids=[151645, 151643] if 'qwen2' in args.model_name_or_path.lower() else None))
                outputs = sorted(outputs, key=lambda x: int(x.request_id))
                outputs = [output.outputs[0].text for output in outputs]
        else:
            outputs = generate_completions(model=llm, tokenizer=tokenizer, prompts=prompts, max_new_tokens=args.max_tokens_per_call, batch_size=16, stop_id_sequences=stop_words)
        assert len(outputs) == len(current_prompts)
        remain_prompts = []
        remain_codes = []
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
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)
    results = [run_execute(executor, code, args.prompt_type, data_name) for code in codes]
    time_use = time.time() - start_time
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
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(samples=all_samples, data_name=data_name, prompt_type=args.prompt_type, execute=True)
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)
    result_json['time_use_in_second'] = time_use
    result_json['time_use_in_minite'] = f'{int(time_use // 60)}:{int(time_use % 60):02d}'
    result_json['prompts'] = result_prompts
    with open(out_file.replace('.jsonl', f'_{args.prompt_type}_metrics.json'), 'w') as f:
        json.dump(result_json, f, indent=4)
    return result_json

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    setup(args)