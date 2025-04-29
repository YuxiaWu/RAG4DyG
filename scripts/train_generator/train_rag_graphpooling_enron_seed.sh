#!/bin/bash
TOPK=7 
GNN_layer=1
MLP_Layer=1 
ParaM=1
Timestamp=16
batch_size=32
n_layer=2
n_head=6
n_embed=768
lr=0.0001 
for RETRIEVAL_TYPE in 'augclhardtime'
do
    for seed in 1 2 3 #4 5 6 7 8 
    do
        export TRAIN_FILE="./resources/enron/$Timestamp/train.link_prediction"
        export TEST_FILE="./resources/enron/$Timestamp/test.link_prediction"
        export TEST_GT_FILE="./resources/enron/$Timestamp/test_gt.link_prediction"
        export VAL_FILE="./resources/enron/$Timestamp/val.link_prediction"
        export VAL_GT_FILE="./resources/enron/$Timestamp/val_gt.link_prediction"
        
        export output="output/enron/generator/graphfusion/$Timestamp/{$seed}/gpt2"
        
        export TRAIN_INDEX_FILE="./resources/train_generator/enron/$Timestamp/train_gt_topk/train_index.gen"
        export TRAIN_SCORE_FILE="./resources/train_generator/enron/$Timestamp/train_gt_topk/train_score.gen"

        export TEST_INDEX_FILE="./resources/retrieval_result/enron/$RETRIEVAL_TYPE/test_index.gen"
        export TEST_SCORE_FILE="./resources/retrieval_result/enron/$RETRIEVAL_TYPE/test_score.gen"
        
        export VAL_INDEX_FILE="./resources/retrieval_result/enron/$RETRIEVAL_TYPE/val_index.gen"
        export VAL_SCORE_FILE="./resources/retrieval_result/enron/$RETRIEVAL_TYPE/val_score.gen"

        CUDA_VISIBLE_DEVICES=3 python main_generator.py \
            --task 'RAG' \
            --retrieval_type=$RETRIEVAL_TYPE \
            --dataset 'enron' \
            --fusion 'graphpooling' \
            --m=$ParaM \
            --topK=$TOPK \
            --mlp_layers=$MLP_Layer \
            --gnn_layer=$GNN_layer \
            --output_dir=$output \
            --model_type 'gpt2' \
            --model_name_or_path 'gpt2' \
            --train_data_file=$TRAIN_FILE \
            --do_train \
            --eval_data_file=$VAL_FILE \
            --eval_data_gt_file=$VAL_GT_FILE \
            --test_data_file=$TEST_FILE \
            --test_data_gt_file=$TEST_GT_FILE \
            --train_index_file=$TRAIN_INDEX_FILE \
            --test_index_file=$TEST_INDEX_FILE \
            --val_index_file=$VAL_INDEX_FILE \
            --train_score_file=$TRAIN_SCORE_FILE \
            --test_score_file=$TEST_SCORE_FILE \
            --val_score_file=$VAL_SCORE_FILE \
            --save_steps 250 \
            --logging_steps 250 \
            --evaluate_during_training \
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
            --seed=$seed \
            --simpledyg_checkpoint "output/enron/simpledyg_ckpt/16/{42}/gpt2/checkpoint-0" \
            --freeze \
            --run_seed
    done
done