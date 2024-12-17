from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from datasets import Dataset, load_dataset, load_from_disk
import pandas as pd
import numpy as np

def build_model_and_tokenizer(model_name):
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def build_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer

def build_dataset(dataset_name, from_disk=False, split=None):
    # Load dataset
    if from_disk:
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name, split=split)
    return dataset

def prepare_for_elo_dataset(ds_list):
    outputs = []
    for i in range(len(ds_list[0])):
        outputs.append({
            'prompt' : ds_list[0][i]['prompt'],
            'responses' : [ds_list[j][i]['response'] for j in range(len(ds_list))],
            'scores' : [ds_list[j][i]['score'] for j in range(len(ds_list))]
        })
    ds = Dataset.from_pandas(pd.DataFrame(outputs))
    return ds
    

def calculate_elo_ratings(dataset, num_models):
    """
    Calculate Elo ratings from dataset
    
    Args:
    - dataset: Calibrated dataset with scores
    - num_models: Number of models
    
    Returns:
    - Elo ratings for each model
    """
    def expected_score(rating_diff):
        return 1.0 / (1.0 + 10 ** (rating_diff / 400))
    
    # Initialize Elo ratings with 1000
    ratings = np.full(num_models, 1000.0)
    k_factor = 4.0
    winrate_matrix = [[0 for _ in range(5)] for _ in range(5)]
    
    # Process each comparison
    for item in dataset:
        for i in range(len(item['scores'])):
            for j in range(i+1, len(item['scores'])):
                # Compute score difference
                score_diff = item['scores'][i] - item['scores'][j]
                
                # Determine match outcome
                if score_diff > 0:
                    actual = 1
                    winrate_matrix[i][j] += 1
                elif score_diff < 0:
                    actual = 0
                    winrate_matrix[j][i] +=1
                else:
                    actual = 0.5
                    
                
                # Expected score based on current ratings
                expected = expected_score(ratings[j] - ratings[i])
                
                # Update ratings
                ratings[i] += k_factor * (actual - expected)
                ratings[j] -= k_factor * (actual - expected)
    
    return ratings, winrate_matrix
