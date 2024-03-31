#!/bin/bash

python train.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-v1 \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path "lattice-opt1.3b" \
    --block_size 128  \
    --output_dir "./opt-models/1gram-n2-model" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs  3   \
    --checkpointing_steps 200 \
    --parallel_data \
# --output_dir "./corrected_models/5gram_n2_model_8sample" \
# --output_dir "./tmp/tmp" \
# --output_dir "./tmp/tmp2" \
# --output_dir "./corrected_models/corrected_models/5gram_n3_model_8sample" \
