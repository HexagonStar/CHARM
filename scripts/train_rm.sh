RM_NAME=Skywork-Reward-Llama-3.1-8B-v0.2
OVERVALUED_MODEL=gemma-2-9b-it-SimPO
REF_MODEL=GPT-4o-mini-2024-07-18

TRAIN_SET_PATH=./datasets/calibrate/${RM_NAME}/${OVERVALUED_MODEL}_${REF_MODEL}
OUTPUT_PATH=./models/${RM_NAME}/${OVERVALUED_MODEL}_${REF_MODEL}

export CUDA_VISIBLE_DEVICES=0

accelerate launch ./src/train_rm.py \
    --model_name ${RM_NAME} \
    --num_train_epochs 1 \
    --max_length 4096 \
    --per_device_train_batch_size 2\
    --train_set_path ${TRAIN_SET_PATH} \
    --output_path ${OUTPUT_PATH} \
    --config_path ./configs/rm_config.json \

