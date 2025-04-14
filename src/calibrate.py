import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_from_disk
from torch.nn import functional as F
from tqdm import tqdm

from utils import *



class OffsetPredictor(nn.Module):
    """Simple MLP to predict score offsets"""
    def __init__(self, num_models):
        super().__init__()
        self.fc1 = nn.Linear(num_models, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_models)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def calculate_winrate_matrix(scores_list, offsets):
    """Calculate winrate matrix given scores and offsets"""
    N = len(scores_list)
    device = scores_list[0].device
    winrate_matrix = torch.zeros((N, N), device=device)
    temperature = 0.1  # Controls the sharpness of the sigmoid
    
    batch_size = 1000
    for i in range(N):
        for j in range(N):
            if i != j:
                scores_i = scores_list[i] + offsets[i]
                scores_j = scores_list[j] + offsets[j]
                
                wins = 0
                num_batches = (len(scores_i) + batch_size - 1) // batch_size
                
                for b in range(num_batches):
                    start_idx = b * batch_size
                    end_idx = min((b + 1) * batch_size, len(scores_i))
                    
                    batch_i = scores_i[start_idx:end_idx].unsqueeze(1)
                    batch_j = scores_j[start_idx:end_idx].unsqueeze(0)
                    
                    diff = (batch_i - batch_j) / temperature
                    wins += torch.sigmoid(diff).mean() * (end_idx - start_idx)
                
                winrate_matrix[i, j] = wins / len(scores_i)

    return winrate_matrix


def neural_network_calibration(ds_list, model_list, model_elo, num_epochs=100, lr=0.01):
    """
    Calibrate datasets using neural network to predict offsets that match target Elo-based winrates.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = len(ds_list)
    
    # Calculate target winrate matrix from Elo ratings
    target_winrate_matrix = torch.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                target_winrate_matrix[i, j] = 1 / (1 + 10 ** ((model_elo[model_list[j]] - model_elo[model_list[i]]) / 400))
    
    # Prepare scores and move to device
    scores_list = [[item['score'] for item in ds] for ds in ds_list]
    scores_list = [torch.tensor(scores, dtype=torch.float32, device=device) for scores in scores_list]
    target_winrate_matrix = target_winrate_matrix.to(device)
    
    model = OffsetPredictor(N).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    offsets = torch.zeros(N, device=device)

    # Training 
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        
        # Predict offsets
        input_offsets = torch.zeros(N, device=device)
        offsets = model(input_offsets.unsqueeze(0)).squeeze(0)
        current_winrate_matrix = calculate_winrate_matrix(scores_list, offsets)
        
        loss = F.mse_loss(current_winrate_matrix, target_winrate_matrix)
        loss.backward(retain_graph=True)

        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}')

    # Get final offsets
    with torch.no_grad():
        input_offsets = torch.zeros(N, device=device)
        offsets = model(input_offsets.unsqueeze(0)).squeeze(0).cpu().numpy()
    
    avg_scores = [np.mean(ds['score']) for ds in ds_list]
    scores_with_offset = [avg_scores[i] + offsets[i] for i in range(N)]

    print("Original scores ==> ", avg_scores)
    print("Offsets ==> ", offsets)
    print("Final scores ==> ", scores_with_offset)

    # Apply calibration
    calibrated_datasets = []
    for i in range(len(ds_list)):
        outputs = []
        for item in ds_list[i]:
            outputs.append({
                'prompt': item['prompt'],
                'response': item['response'],
                'score': item['score'] + offsets[i]
            })
        df = pd.DataFrame(outputs)
        dataset = Dataset.from_pandas(df)
        calibrated_datasets.append(dataset)
    
    return calibrated_datasets, offsets

def construct_binary_preference_dataset(ds_list, output_path):
    outputs = []

    def change_of_format(prompt, resp):
        message = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]
        return message

    for i in range(len(ds_list[0])):
        scores = [ds[i]['score'] for ds in ds_list]
        max_idx = scores.index(max(scores))
        min_idx = scores.index(min(scores))
        chosen = change_of_format(ds_list[max_idx][i]['prompt'], ds_list[max_idx][i]['response'])
        rejected = change_of_format(ds_list[min_idx][i]['prompt'], ds_list[min_idx][i]['response'])
        outputs.append({
            'chosen': chosen,
            'rejected': rejected
        })

    
    df = pd.DataFrame(outputs)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(output_path)



def main(args):
    if not os.path.exists(args.score_dataset_dir):
        print(f"Score dataset directory {args.score_dataset_dir} does not exist")
        return
    
    policy_models = load_policy_model_config(args.pm_config_path)
    model_list = [args.overvalued_model, args.ref_model]
    model_elo = {model: policy_models[model].arena_score for model in model_list}
    preference_output_path = args.preference_output_dir

    ds_list = [load_from_disk(f"{args.score_dataset_dir}/{model}_score") for model in model_list]

    md = calculate_mismatch_degree(ds_list[0]['score'], ds_list[1]['score'], model_elo[args.overvalued_model], model_elo[args.ref_model])
    print(f"Mismatch degree: {md}")
    calibrated_datasets, offset_list = neural_network_calibration(ds_list, model_list, model_elo)

    os.makedirs(args.preference_output_dir, exist_ok=True)
    construct_binary_preference_dataset(calibrated_datasets, preference_output_path)



if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Calibrate dataset')
    parser.add_argument('--score_dataset_dir', type=str, help='Dataset directory for score dataset')
    parser.add_argument('--preference_output_dir', type=str, help='Output directory for preference dataset')
    parser.add_argument('--pm_config_path', type=str, help='Path to the policy model config file')
    parser.add_argument('--ref_model', type=str, help='Reference model name')
    parser.add_argument('--overvalued_model', type=str, help='Overvalued model name')
    args = parser.parse_args()

    main(args)