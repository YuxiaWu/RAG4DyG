#!/bin/bash
Timestamp=15
alpha=0.4 
eta=0.8  
gamma=0.6 
tdecay=10 
batch_size=128
n_layer=2 
n_head=2 
n_embed=256  
lr=1e-5
seed=42 

export TRAIN_FILE="./resources/dialog/$Timestamp/train.link_prediction"
export TRAIN_PAIR_FILE="./resources/dialog/$Timestamp/train_retrieval/train_index.retrieval"
export TEST_FILE="./resources/dialog/$Timestamp/test.link_prediction"
export TEST_GT_FILE="./resources/dialog/$Timestamp/train_retrieval/test_score.retrieval"
export VAL_FILE="./resources/dialog/$Timestamp/val.link_prediction"
export VAL_GT_FILE="./resources/dialog/$Timestamp/train_retrieval/val_score.retrieval"
export output="output/dialog/retrieval/$Timestamp/{$n_layer}_{$n_head}_{$n_embed}_{$batch_size}_{$lr}_{$seed}/gpt2"

CUDA_VISIBLE_DEVICES=3 python main_retriever.py \
    --dataset 'dialog' \
    --lambda_decay=$tdecay \
    --alpha $alpha \
    --gamma $gamma \
    --eta $eta \
    --lrdecay 1 \
    --should_continue \
    --warmup_steps 0 \
    --temperature=0.1 \
    --output_dir=$output \
    --model_type 'gpt2' \
    --model_name_or_path 'gpt2' \
    --train_data_file=$TRAIN_FILE \
    --train_pair_data_file=$TRAIN_PAIR_FILE \
    --do_train \
    --eval_data_file=$VAL_FILE \
    --eval_data_gt_file=$VAL_GT_FILE \
    --test_data_file=$TEST_FILE \
    --test_data_gt_file=$TEST_GT_FILE \
    --save_steps 250 \
    --logging_steps 500 \
    --per_gpu_train_batch_size=$batch_size \
    --num_train_epochs 50 \
    --block_size 1024 \
    --eval_all_checkpoints \
    --timestamp $Timestamp \
    --patience 5 \
    --n_layer=$n_layer \
    --n_head=$n_head \
    --n_embed=$n_embed \
    --learning_rate=$lr \
    --seed=$seed 
