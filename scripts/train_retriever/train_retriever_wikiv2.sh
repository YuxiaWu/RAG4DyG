#!/bin/bash
N_NODES=10
TASK='classification'
THRESHOLD=0.8
Timestamp=15
alpha=0.2 
eta=0.6 
gamma=0.8 
beta=0 
tdecay=1 
tmp=0.1 
batch_size=128
n_layer=2
n_head=6
n_embed=768
lr=1e-6 
seed=1 
export TRAIN_FILE="./resources/wikiv2/$Timestamp/train.link_prediction"
export TRAIN_PAIR_FILE="./resources/wikiv2/$Timestamp/train_retrieval/train_index.retrieval"
export TEST_FILE="./resources/wikiv2/$Timestamp/test.link_prediction"
export TEST_GT_FILE="./resources/wikiv2/$Timestamp/train_retrieval/test_score.retrieval"
export VAL_FILE="./resources/wikiv2/$Timestamp/val.link_prediction"
export VAL_GT_FILE="./resources/wikiv2/$Timestamp/train_retrieval/val_score.retrieval"
export output="output/wikiv2/retrieval/$Timestamp/{$n_layer}_{$n_head}_{$n_embed}_{$batch_size}_{$lr}_{$seed}/gpt2"

CUDA_VISIBLE_DEVICES=3 python main_retriever.py \
    --run_seed \
    --seed=$seed \
    --dataset 'wikiv2' \
    --loss_type 'augclhardtime' \
    --should_continue \
    --eta $eta \
    --beta $beta \
    --gamma $gamma \
    --temperature=$tmp \
    --alpha $alpha \
    --lambda_decay=$tdecay \
    --lrdecay 1 \
    --projector 0 \
    --warmup_steps 0 \
    --output_dir=$output \
    --model_type 'gpt2' \
    --model_name_or_path 'gpt2' \
    --mask_file=$MASK_FILE \
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
    --block_size 512 \
    --eval_all_checkpoints \
    --timestamp $Timestamp \
    --patience 5 \
    --n_layer=$n_layer \
    --n_head=$n_head \
    --n_embed=$n_embed \
    --learning_rate=$lr \
    --task $TASK \
    --threshold $THRESHOLD 


