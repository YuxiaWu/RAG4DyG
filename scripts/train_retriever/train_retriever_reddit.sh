#!/bin/bash
N_NODES=10
TASK='classification'
THRESHOLD=0.8
Timestamp=11
alpha=0.2 
eta=0.2 
gamma=0.8
beta=0 # 
tdecay=10 
tmp=0.1
batch_size=128
n_layer=2
n_head=8
n_embed=512
lr=1e-6 
seed=42 

export TRAIN_FILE="./resources/reddit/$Timestamp/train.link_prediction"
export TRAIN_PAIR_FILE="./resources/reddit/$Timestamp/train_retrieval/train_index.retrieval"
export TEST_FILE="./resources/reddit/$Timestamp/test.link_prediction"
export TEST_GT_FILE="./resources/reddit/$Timestamp/train_retrieval/test_score.retrieval"
export VAL_FILE="./resources/reddit/$Timestamp/val.link_prediction"
export VAL_GT_FILE="./resources/reddit/$Timestamp/train_retrieval/val_score.retrieval"
export output="output/reddit/retrieval/$Timestamp/{$alpha}_{$eta}_{$gamma}_{$tdecay}_{$lr}_{$seed}/gpt2"
export NODE_FILE="./resources/reddit/node_features.npy" 


CUDA_VISIBLE_DEVICES=2 python main_retriever.py \
    --seed=$seed \
    --run_seed \
    --should_continue \
    --dataset 'reddit' \
    --loss_type 'augclhardtime' \
    --node_feat_file=$NODE_FILE \
    --gradient_accumulation_steps 1 \
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
    --patience 3 \
    --n_layer=$n_layer \
    --n_head=$n_head \
    --n_embed=$n_embed \
    --learning_rate=$lr \
    --task $TASK \
    --threshold $THRESHOLD 
