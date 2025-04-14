import pathlib

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import load_reward_model_config

tqdm.pandas()



class LlamaPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=True, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = self.model.device

    def __call__(self, messages):
        messages = self.change_of_format(messages[0], messages[1])
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(input_ids)
            score = output.logits[0][0].item()
        return {"score": score}
    
    def change_of_format(self, prompt, resp):
        message = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]
        return message

class GRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=True, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, messages):
        messages = self.change_of_format(messages[0], messages[1])

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )

        kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
        tokens = self.tokenizer.encode_plus(input_ids, **kwargs)
        with torch.no_grad():
            reward_tensor = self.model(tokens["input_ids"][0].view(1,-1).to(self.device), attention_mask=tokens["attention_mask"][0].view(1,-1).to(self.device)).logits
            reward = reward_tensor.cpu().detach().item()

        return {"score": reward}
    
    def change_of_format(self, prompt, resp):
        message = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]

        return message


def get_reward_model_pipeline(rm_name, config_path, model_path=None):
    reward_models = load_reward_model_config(config_path)
    if rm_name not in reward_models:
        raise ValueError(f"Reward model {rm_name} not found in config.")
    config = reward_models[rm_name]

    if model_path is None:
        print(f"Model path not provided, using default model path: {config.model_path}")
        model_path = config.model_path

    if config.pipeline_class == "LlamaPipeline":
        return LlamaPipeline(model_path)
    elif config.pipeline_class == "GRMPipeline":
        return GRMPipeline(model_path)
    else:
        raise ValueError(f"Invalid reward model class: {config.pipeline_class}")