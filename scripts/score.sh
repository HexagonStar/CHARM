RM_NAME=Skywork-Reward-Llama-3.1-8B-v0.2

export CUDA_VISIBLE_DEVICES=0

MODEL_LIST=(
gemma-2-9b-it-SimPO 
GPT-4o-mini-2024-07-18
)

for model_name in ${MODEL_LIST[@]}
do
    python ./src/score.py --dataset_name ./datasets/preference20K/${model_name} \
                              --model_name ${RM_NAME} \
                              --output_path ./datasets/scores/${RM_NAME}/${model_name}_score \
                              --config_path ./configs/rm_config.json
done


