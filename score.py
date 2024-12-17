from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, AutoModelForSequenceClassification
from datasets import Dataset
import numpy as np
import pandas as pd
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
tqdm.pandas()

from utils import *

MODEL_LIST = ['Qwen2.5-72B-Instruct',
              'gemma-2-27b-it',
              'Llama-3.1-8B-Instruct',
              'Llama-3.1-70B-Instruct',
              'gemma-2-9b-it-SimPO']

class ArmoRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=True, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages):
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        
        if messages[1]['content'] == '':
            return {"score": 0}

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}

class SkyWorkPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=True, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages):
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        
        if messages[1]['content'] == '':
            return {"score": 0}

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(input_ids)
            score = output.logits[0][0].item()
        return {"score": score}



def score_for_subset(args):
    rm_name = args.model_name
    if "armorm" in rm_name.lower():
        rm = ArmoRMPipeline(rm_name)
    elif "skywork" in rm_name.lower():
        rm = SkyWorkPipeline(rm_name)
    else:
        print("Reward model not found")
        return

    ds = build_dataset(args.dataset_name, args.from_disk)

    def change_of_format(prompt, resp):
        message = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]

        return message

    outputs = []
    for i, example in enumerate(tqdm(ds)):
        score = rm(change_of_format(example['prompt'], example['response']))
        outputs.append({
            'prompt': example['prompt'],
            'response': example['response'],
            'score': score['score'],
        })

    df = pd.DataFrame(outputs)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(args.output_path)
    

def average_score(args):
    score_dir = args.score_dir
    # find all subdirectories under score_dir with suffix '-score'
    score_files = [f for f in os.listdir(score_dir) if os.path.isdir(os.path.join(score_dir, f))]
    for score_file in score_files:
        ds = build_dataset(f"{score_dir}/{score_file}", from_disk=True)
        scores = []
        for i, example in enumerate(tqdm(ds)):
            score = example['score']
            scores.append(score)
        print(f"{score_file}:{sum(scores)/len(scores)}")



if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Inference ')
    parser.add_argument('--dataset_name', type=str, default='hendrydong/preference_700K', help='Dataset name')
    parser.add_argument('--from_disk', action='store_true', help='Load dataset from disk')
    parser.add_argument('--model_name', type=str, default='RLHFlow/ArmoRM-Llama3-8B-v0.1', help='Reward model name')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--score_dir', type=str)
    parser.add_argument('--avg_score', action='store_true')
    args = parser.parse_args()

    if args.avg_score:
        average_score(args)
    else:
        score_for_subset(args)
