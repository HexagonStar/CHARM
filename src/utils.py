from dataclasses import dataclass
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from datasets import Dataset, load_dataset, load_from_disk
import pandas as pd
import numpy as np

def build_dataset(dataset_name, from_disk=False, split=None):
    # Load dataset
    if from_disk:
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name, split=split)
    return dataset

def calculate_mismatch_degree(overvalued_scores, ref_scores, overvalued_elo, ref_elo):
    '''
    Calculate the mismatch degree between the empirical winrate and the Elo winrate
    '''
    overvalued_winrate = np.mean(np.array(overvalued_scores) > np.array(ref_scores))
    elo_winrate = 1 / (1 + 10 ** ((ref_elo - overvalued_elo) / 400))
    return (overvalued_winrate - elo_winrate) / max(elo_winrate, 1 - elo_winrate)

@dataclass
class RewardModelConfig:
    model_name: str
    pipeline_class: str
    model_path: str

def load_reward_model_config(config_path):
    with open(config_path, 'r') as f:
        configs = json.load(f)
    reward_models = {}
    for config in configs:
        reward_models[config['model_name']] = RewardModelConfig(**config)
    return reward_models

@dataclass
class PolicyModelConfig:
    model_name: str
    model_path: str
    arena_score: int

def load_policy_model_config(config_path):
    with open(config_path, 'r') as f:
        configs = json.load(f)
    policy_models = {}
    for config in configs:
        policy_models[config['model_name']] = PolicyModelConfig(**config)
    return policy_models