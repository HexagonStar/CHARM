from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk

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

def prepare_prompts_from_dataset(dataset):
    prompt_list = []
    for data in dataset:
        prompt = data['chosen'][0]['content']
        prompt_list.append(prompt)
    return prompt_list

def calculte_max_length(tokenizer, dataset):
    prompt_list = prepare_prompts_from_dataset(dataset)
    len_list = []
    for prompt in prompt_list:
        input_ids = tokenizer.encode(prompt)
        len_list.append(len(input_ids))
    return max(len_list)


