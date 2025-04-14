import argparse
import os
import pathlib

import pandas as pd
from tqdm import tqdm
from datasets import Dataset, load_from_disk

from pipeline import get_reward_model_pipeline
from utils import *

def score_for_dataset(args):
    # check if dataset exists
    if pathlib.Path(args.dataset_name).exists():
        ds = load_from_disk(args.dataset_name)
    else:
        raise ValueError(f"Dataset {args.dataset_name} not found")
    
    rm = get_reward_model_pipeline(args.model_name, args.config_path, args.model_path)

    outputs = []

    for i, example in enumerate(tqdm(ds)):
        score = rm([example['prompt'], example['response']])
        outputs.append({
            'prompt': example['prompt'],
            'response': example['response'],
            'score': score['score'],
        })

    df = pd.DataFrame(outputs)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(args.output_path)


if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Inference ')
    parser.add_argument('--dataset_name', type=str, help='Dataset to score')
    parser.add_argument('--model_name', type=str, default='Skywork-Reward-Llama-3.1-8B-v0.2', help='Reward model name')
    parser.add_argument('--model_path', type=str, help='Local path of reward model')
    parser.add_argument('--output_path', type=str, help='Output path of scored dataset')
    parser.add_argument('--config_path', type=str, help='Reward model config path')
    args = parser.parse_args()

    score_for_dataset(args)
