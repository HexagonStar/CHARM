import pandas as pd
import argparse
import torch
from datasets import Dataset

from agent import VllmAgent
from utils import *

def change_of_format(prompt):
    message = [
        {"role": "user", "content": prompt}
    ]
    return message

def inference_for_dataset(args):
    torch.cuda.memory._set_allocator_settings('expandable_segments:False')
    vllm_kwargs = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        'max_num_seqs': args.max_num_seqs,
        "trust_remote_code": True,
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

    policy_models = load_policy_model_config(args.config_path)
    model_path = policy_models[args.model_name].model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
 
    # Initialize the VLLM Agent with the specified translation model
    model = VllmAgent(model_name=model_path, model_kwargs=vllm_kwargs, generation_kwargs=generation_kwargs)
    ds = build_dataset(args.dataset_name, from_disk=args.from_disk, split=args.split)

    template_prompt_list = []
    prompt_list = []
    for data in ds:
        prompt = data['prompt']
        prompt = tokenizer.apply_chat_template(change_of_format(prompt), add_generation_prompt=True, tokenize=False)
        template_prompt_list.append(prompt)
        prompt_list.append(data['prompt'])

    # Inference
    outputs = []
    batch_output = model.generate(template_prompt_list)
    for i in range(len(batch_output)):
        outputs.append({
            'prompt': prompt_list[i],
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
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Inference model name')
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
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()

    inference_for_dataset(args)







