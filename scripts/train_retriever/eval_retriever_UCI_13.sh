#!/bin/bash
Timestamp=12
alpha=1 
eta=0.8 
gamma=0.4 
tdecay=0.0001 
tmp=0.1 
batch_size=64
n_layer=4
n_head=2
n_embed=512
lr=1e-5
seed=42 

export TRAIN_FILE="./resources/UCI_13/$Timestamp/train.link_prediction"
export TRAIN_PAIR_FILE="./resources/UCI_13/$Timestamp/train_retrieval/train_index.retrieval"
export TEST_FILE="./resources/UCI_13/$Timestamp/test.link_prediction"
export TEST_GT_FILE="./resources/UCI_13/$Timestamp/train_retrieval/test_score.retrieval"
export VAL_FILE="./resources/UCI_13/$Timestamp/val.link_prediction"
export VAL_GT_FILE="./resources/UCI_13/$Timestamp/train_retrieval/val_score.retrieval"
export output="output/UCI_13/retrieval/$Timestamp/{$n_layer}_{$n_head}_{$n_embed}_{$batch_size}_{$lr}_{$seed}/gpt2"

CUDA_VISIBLE_DEVICES=3 python main_retriever.py \
    --dataset 'UCI_13' \
    --eta $eta \
    --gamma $gamma \
    --temperature=$tmp \
    --alpha $alpha \
    --lambda_decay=$tdecay \
    --lrdecay 1 \
    --warmup_steps 0 \
    --output_dir=$output \
    --model_type 'gpt2' \
    --model_name_or_path 'gpt2' \
    --train_data_file=$TRAIN_FILE \
    --train_pair_data_file=$TRAIN_PAIR_FILE \
    --do_eval \
    --eval_data_file=$VAL_FILE \
    --eval_data_gt_file=$VAL_GT_FILE \
    --test_data_file=$TEST_FILE \
    --test_data_gt_file=$TEST_GT_FILE \
    --save_steps 250 \
    --logging_steps 500 \
    --per_gpu_train_batch_size=$batch_size \
    --num_train_epochs 50 \
    --block_size 512 \
    --eval_all_checkpoints \
    --timestamp $Timestamp \
    --patience 10 \
    --n_layer=$n_layer \
    --n_head=$n_head \
    --n_embed=$n_embed \
    --learning_rate=$lr \
    --seed=$seed \
    --run_seed 
