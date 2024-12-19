import argparse
from datasets import load_from_disk
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from utils import *
import os

MODEL_LIST = ['Qwen2.5-72B-Instruct',
              'gemma-2-27b-it',
              'gemma-2-9b-it-SimPO',
              'Llama-3.1-70B-Instruct',
              'Llama-3.1-8B-Instruct']

MODEL_ELO = {'Qwen2.5-72B-Instruct': 1220,
                'gemma-2-27b-it': 1205,
                'gemma-2-9b-it-SimPO': 1197,
                'Llama-3.1-70B-Instruct': 1194,
                'Llama-3.1-8B-Instruct': 1142}


def calculate_shift(ds_list):
    scores_list = [[item['score'] for item in ds] for ds in ds_list]
    mean_list = [np.mean(scores) for scores in scores_list]
    var_list = [np.var(scores) for scores in scores_list]

    # Calculate shift
    delta_shift_list = []
    for i in range(len(ds_list) - 1):
        a_vs_b_winrate = 1 / (1 + 10 ** ((MODEL_ELO[MODEL_LIST[i+1]] - MODEL_ELO[MODEL_LIST[i]]) / 400))
        delta_shift = - (stats.norm.ppf(1 - a_vs_b_winrate) * np.sqrt(var_list[i] + var_list[i+1]))
        delta_shift_list.append(delta_shift)
    calibrated_mean_list =  [sum(delta_shift_list)] + [sum(delta_shift_list[i:]) for i in range(1, len(delta_shift_list)+1)]
    shift_list = [calibrated_mean_list[i] - mean_list[i] for i in range(len(mean_list))]
    return shift_list

def calibrate_dataset(ds_list, shift_list):
    output_datasets = []
    for i in range(len(ds_list)):
        outputs = []
        for item in ds_list[i]:
            outputs.append({
                'prompt' : item['prompt'],
                'response' : item['response'],
                'score' : item['score'] + shift_list[i]
            })
        df = pd.DataFrame(outputs)
        dataset = Dataset.from_pandas(df)
        output_datasets.append(dataset)
    return output_datasets
        

def main(args):
    # Load the dataset
    ds_list = [load_from_disk(f"{args.dataset_dir}/{model}-test_sft_score") for model in MODEL_LIST]

    # Calculate shift
    shift_list = calculate_shift(ds_list)
    print("Shift list: ", shift_list)

    # Calibrate the dataset
    calibrated_datasets = calibrate_dataset(ds_list, shift_list)

    _ , original_winrate_matrix = calculate_elo_ratings(prepare_for_elo_dataset(ds_list), len(MODEL_LIST))
    _ , calibrated_winrate_matrix = calculate_elo_ratings(prepare_for_elo_dataset(calibrated_datasets), len(MODEL_LIST))
    print("Original winrate matrix: ", original_winrate_matrix)
    print("Calibrated winrate matrix: ", calibrated_winrate_matrix)

    # Save the calibrated dataset
    for i in range(len(MODEL_LIST)):
        calibrated_datasets[i].save_to_disk(f"{args.output_dir}/{MODEL_LIST[i]}-test_sft_score")

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Calibrate dataset')
    parser.add_argument('--dataset_dir', type=str, help='Dataset directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    args = parser.parse_args()

    main(args)