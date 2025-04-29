#!/bin/bash
Timestamp={11..11} 
batch_size=32  
n_layer=12
n_head=2
n_embed=256
lr=0.0001
for seed in 4 #42 0 1 2 3 4 5 6 7 8 
    do               
    export TRAIN_FILE="./resources/hepth/$Timestamp/train.link_prediction"
    export TEST_FILE="./resources/hepth/$Timestamp/test.link_prediction"
    export TEST_GT_FILE="./resources/hepth/$Timestamp/test_gt.link_prediction"
    export VAL_FILE="./resources/hepth/$Timestamp/val.link_prediction"    
    export VAL_GT_FILE="./resources/hepth/$Timestamp/val_gt.link_prediction"      
    export output="simpledyg_ckpt/hepth/$Timestamp/{$seed}/gpt2"
    export NODE_FILE="./resources/hepth/node_features.npy" 

    CUDA_VISIBLE_DEVICES=3 python main_SimpleDyG.py \
        --dataset 'hepth' \
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
        --patience 10 \
        --n_layer=$n_layer \
        --n_head=$n_head \
        --n_embed=$n_embed \
        --node_feat_file=$NODE_FILE \
        --learning_rate=$lr \
        --seed=$seed \
        --run_seed
    done
