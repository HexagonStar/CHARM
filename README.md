# CHARM: Calibrating Reward Models With Chatbot Arena Scores

This is the repository for the paper [CHARM: Calibrating Reward Models With Chatbot Arena Scores](https://arxiv.org/abs/2504.10045).


## Setup
Our implementation is based on the [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling/) repository. You may follow the instructions in the [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/bradley-terry-rm) to install the requirements.

```python
conda create -n charm 
conda activate charm
git clone https://github.com/HexagonStar/CHARM.git
cd CHARM
bash scripts/setup.sh
```

## Dataset
We randomly sampled 20K instructions from the [Preference700K](https://huggingface.co/datasets/hendrydong/preference_700K). You can download it from [HERE](https://huggingface.co/datasets/shawnxzhu/CHARM-preference20K) and put it under the `./datasets/preference20K` folder.

We also provide the generated responses using selected policy models. You can download the datasets from [this collection](https://huggingface.co/collections/shawnxzhu/charm-datasets-67fa21cb8ee69d220e369fc0) and put them under the `./datasets/preference20K` folder too.

Or you can generate your own datasets following the steps:

```bash
# Step1. Download the preference20K dataset and put it under the `./datasets/preference20K` folder.

# Step2. Add policy model information to `./configs/pm_config.json`

# Step3. Run the inference script.
bash scripts/inference.sh

```

## Calibrate

To generate the calibrated preference dataset, follow these steps:

1. Ensure model responses are placed in the `./datasets/preference20K` directory.

2. Assign the model name to the `MODEL_LIST` variable in the `scripts/score.sh` script and run the script to evaluate model responses:
```bash
bash scripts/score.sh
```

3. Assign the model name to the `RM_NAME`, `OVERVALUED_MODEL`, and `REF_MODEL` variables in the `scripts/calibrate.sh` script and run the script to generate the calibrated preference dataset:
```bash
bash scripts/calibrate.sh
```
The calibrated preference dataset will be saved in the `./datasets/calibrate` directory.

## Train 

To train the reward model on calibrated preference dataset, follow these steps:

1. Ensure the calibrated preference dataset is placed in the `./datasets/calibrate` directory.

2. Set your wandb key in `./src/train_rm.py` and run the training script:
```bash
bash scripts/train_rm.sh
```

3. The trained reward model will be saved in the `./models` directory.

4. You can also download the calibrated Skywork-RM from [HERE](https://huggingface.co/collections/shawnxzhu/charm-models-67fa222b7cfcb309de09ab93).

## Evaluation

You can evaluate the calibrated reward model on benchmarks like [RM-Bench](https://github.com/THU-KEG/RM-Bench) and [RewardBench](https://github.com/allenai/reward-bench). 


## Citation
If you find this work useful, please consider citing:

```bibtex
@article{zhu2025charmcalibratingrewardmodels,
  title={CHARM: Calibrating Reward Models With Chatbot Arena Scores},
  author={Xiao Zhu and Chenmien Tan and Pinzhen Chen and Rico Sennrich and Yanlin Zhang and Hanxu Hu},
  journal={https://arxiv.org/abs/2504.10045},
  year={2025}
}
```
