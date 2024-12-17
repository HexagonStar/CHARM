import argparse
from datasets import load_dataset, load_from_disk, Dataset
from typing import List
import pandas as pd


def construct_list_preference_dataset(args):
    MODEL_LIST = [
        'Qwen2.5-72B-Instruct',
        'gemma-2-27b-it',
        'gemma-2-9b-it-SimPO',
        'Llama-3.1-70B-Instruct',
        'Llama-3.1-8B-Instruct',
        ]
    RANK = [1, 2, 3, 4, 5]
    
    subset_list = [load_from_disk(f"./results/preference700K_subset/preference20K/temp0.7/skywork/{model}-test_sft_score") for model in MODEL_LIST]

    outputs = []
    def change_of_format(prompt, resp):
        message = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]
        return message

    for i in range(len(subset_list[0])):
        outputs.append({
            'response': [change_of_format(subset_list[j][i]['prompt'], subset_list[j][i]['response']) for j in range(len(MODEL_LIST))],
            'model': MODEL_LIST,
            'rank': RANK,
            'score': [ds[i]['score'] for ds in subset_list]
        })

    df = pd.DataFrame(outputs)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(args.output_path)

def construct_binary_preference_dataset(args):
    MODEL_LIST = [ 
    "gemma-2-27b-it",
    "gemma-2-9b-it-SimPO",
    "Qwen2.5-72B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct"
    ]
    subset_list = [load_from_disk(f"./results/preference700K_subset/preference20K/temp0.7/skywork/{model}-test_sft_score") for model in MODEL_LIST]
    

    outputs = []

    def change_of_format(prompt, resp):
        message = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]
        return message

    for i in range(len(subset_list[0])):
        scores = [ds[i]['score'] for ds in subset_list]
        max_idx = scores.index(max(scores))
        min_idx = scores.index(min(scores))
        chosen = change_of_format(subset_list[max_idx][i]['prompt'], subset_list[max_idx][i]['response'])
        rejected = change_of_format(subset_list[min_idx][i]['prompt'], subset_list[min_idx][i]['response'])
        outputs.append({
            'chosen': chosen,
            'rejected': rejected
        })

    
    df = pd.DataFrame(outputs)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(args.output_path)

def construct_binary_preference_dataset_with_calibrate(args):
    MODEL_LIST = [ 
    "gemma-2-27b-it",
    "gemma-2-9b-it-SimPO",
    "Qwen2.5-72B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct"
    ]

    shift_list = args.shift_list

    subset_list = [load_from_disk(f"{args.dataset_path}/{model}-test_sft_score") for model in MODEL_LIST]

    outputs = []

    def change_of_format(prompt, resp):
        message = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]
        return message

    for i in range(len(subset_list[0])):
        scores = []
        for j, ds in enumerate(subset_list):
            score = ds[i]['score'] + shift_list[j]
            scores.append(score)
        max_idx = scores.index(max(scores))
        min_idx = scores.index(min(scores))
        chosen = change_of_format(subset_list[max_idx][i]['prompt'], subset_list[max_idx][i]['response'])
        rejected = change_of_format(subset_list[min_idx][i]['prompt'], subset_list[min_idx][i]['response'])
        outputs.append({
            'chosen': chosen,
            'rejected': rejected
        })

    
    df = pd.DataFrame(outputs)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(args.output_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample subset from dataset.')
    parser.add_argument('--dataset_path', type=str, default="hendrydong/preference_700K", help="name or path to dataset")
    parser.add_argument('--output_path', type=str, help="path to save the sampled subset")
    parser.add_argument('--shift_list', type=list, help="list of shifts")
    args = parser.parse_args()

    construct_binary_preference_dataset_with_calibrate (args)

    
    