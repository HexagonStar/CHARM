MODEL_NAME=gemma-2-9b-it-SimPO

python ./src/inference.py --dataset_name ./datasets/preference700K_subset1_2 \
                        --from_disk \
                        --model_name ${MODEL_NAME} \
                        --output_path ./datasets/preference20K/${MODEL_NAME} \
                        --temperature 0.7 \
                        --is_dp





            
   