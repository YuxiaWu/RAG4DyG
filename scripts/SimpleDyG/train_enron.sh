#!/bin/bash
#1,enron,SimpleDyG,16,2,6,768,32,0.0001,42,0.0448,0.1011,0.1119,0.1209,0.0729,0.0729,
Timestamp=16
batch_size=32 
n_layer=2
n_head=6
n_embed=768
lr=0.0001 
for seed in 0 1 2 3 4 5 6 7 8 #42
    do
    export TRAIN_FILE="./resources/enron/$Timestamp/train.link_prediction"
    export TEST_FILE="./resources/enron/$Timestamp/test.link_prediction"
    export TEST_GT_FILE="./resources/enron/$Timestamp/test_gt.link_prediction"
    export VAL_FILE="./resources/enron/$Timestamp/val.link_prediction"    
    export VAL_GT_FILE="./resources/enron/$Timestamp/val_gt.link_prediction"    
    export output="output/enron/simpledyg_ckpt/$Timestamp/{$seed}/gpt2"
    

    CUDA_VISIBLE_DEVICES=2 python main_SimpleDyG.py \
        --task 'SimpleDyG' \
        --dataset 'enron' \
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
        --learning_rate=$lr \
        --seed=$seed \
        --run_seed                                               
    done
