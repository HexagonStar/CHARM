import json
import os
import random
import time
import numpy as np
import pandas as pd
from datasets import Dataset
# from openai import OpenAI
from tqdm import tqdm, trange
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch

from agent import VllmAgent
from utils import *


def inference_for_subset(args):
    torch.cuda.memory._set_allocator_settings('expandable_segments:False')
    vllm_kwargs = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        'max_num_seqs': args.max_num_seqs,
    }

    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    if args.is_dp:
        vllm_kwargs["data_parallel_size"] = torch.cuda.device_count()
    else:
        vllm_kwargs['tensor_parallel_size'] = torch.cuda.device_count()
    if "llama-3.1" in args.model_name.lower():
        generation_kwargs["stop_token_ids"] = [128001, 128008, 128009] # <|end_of_text|>, <|eom_id|>, <|eot_id|>
    elif "llama" in args.model_name.lower():
        generation_kwargs["stop_token_ids"] = [128001, 128009]

    print(f"VLLM kwargs: {vllm_kwargs}")
    print(f"Generation kwargs: {generation_kwargs}")


    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
 
    # Initialize the VLLM Agent with the specified translation model
    model = VllmAgent(model_name=args.model_name, model_kwargs=vllm_kwargs, generation_kwargs=generation_kwargs)


    ds = build_dataset(args.dataset_name, from_disk=args.from_disk, split=args.split).shuffle(seed=args.seed)

    prompt_list = []
    raw_prompt_list = []
    for data in ds:
        prompt = data['chosen'][0]
        prompt = tokenizer.apply_chat_template([prompt], add_generation_prompt=True, tokenize=False)
        prompt_list.append(prompt)
        raw_prompt_list.append(data['chosen'][0]['content'])
    
    # Inference
    outputs = []
    batch_output = model.generate(prompt_list)
    for i in range(len(batch_output)):
        outputs.append({
            'prompt': raw_prompt_list[i],
            'response': batch_output[i][0]
        })
    
    df = pd.DataFrame(outputs)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(args.output_path)


    

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Inference ')
    parser.add_argument('--dataset_name', type=str, default='hendrydong/preference_700K', help='Dataset name')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--from_disk', action='store_true', help='Load dataset from disk')
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Translation model name') #'THUDM/LongWriter-llama3.1-8b'
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization for VLLM Agent')
    parser.add_argument('--max_model_len', type=int, default=4096, help='Maximum model length')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for translation')
    parser.add_argument('--is_dp', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--max_num_seqs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    inference_for_subset(args)







