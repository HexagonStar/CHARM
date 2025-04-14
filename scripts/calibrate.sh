RM_NAME=Skywork-Reward-Llama-3.1-8B-v0.2
OVERVALUED_MODEL=gemma-2-9b-it-SimPO
REF_MODEL=GPT-4o-mini-2024-07-18

SCORE_DATASET_DIR=./datasets/scores/${RM_NAME}
PREFERENCE_OUTPUT_DIR=./datasets/calibrate/${RM_NAME}/${OVERVALUED_MODEL}_${REF_MODEL}

python src/calibrate.py --score_dataset_dir ${SCORE_DATASET_DIR} \
                        --preference_output_dir ${PREFERENCE_OUTPUT_DIR} \
                        --pm_config_path ./configs/pm_config.json \
                        --overvalued_model ${OVERVALUED_MODEL} \
                        --ref_model ${REF_MODEL}