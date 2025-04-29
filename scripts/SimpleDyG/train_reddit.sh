#!/bin/bash
Timestamp = 11
batch_size = 32 
n_layer = 2 
n_head = 8 
n_embed = 512 
lr = 0.0001 
for seed in 42 #0 1 2 3 4 5 6 7 8
    do
    export TRAIN_FILE="./resources/reddit/$Timestamp/train.link_prediction"
    export TEST_FILE="./resources/reddit/$Timestamp/test.link_prediction"
    export TEST_GT_FILE="./resources/reddit/$Timestamp/test_gt.link_prediction"
    export VAL_FILE="./resources/reddit/$Timestamp/val.link_prediction"    
    export VAL_GT_FILE="./resources/reddit/$Timestamp/val_gt.link_prediction"    
    export output="output/reddit/$Timestamp/{$n_layer}_{$n_head}_{$n_embed}_{$batch_size}_{$lr}_{$seed}/gpt2"
    export NODE_FILE="./resources/reddit/node_features.npy" 

    CUDA_VISIBLE_DEVICES=3 python main_SimpleDyG.py \
        --task 'SimpleDyG' \
        --gradient_accumulation_steps 1 \
        --node_feat_file=$NODE_FILE \
        --dataset 'reddit' \
        --output_dir=$output \
        --model_type 'gpt2' \
        --model_name_or_path 'gpt2' \
        --train_data_file=$TRAIN_FILE \
        --do_train \
        --eval_data_file=$VAL_FILE \
        --eval_data_gt_file=$VAL_GT_FILE \
        --test_data_file=$TEST_FILE \
        --test_data_gt_file=$TEST_GT_FILE \
        --save_steps 250 \
        --logging_steps 500 \
        --per_gpu_train_batch_size=$batch_size \
        --num_train_epochs 100 \
        --block_size 512 \
        --eval_all_checkpoints \
        --timestamp $Timestamp \
        --patience 5 \
        --n_layer=$n_layer \
        --n_head=$n_head \
        --n_embed=$n_embed \
        --learning_rate=$lr \
        --seed=$seed \
        --run_seed                                               
    done

