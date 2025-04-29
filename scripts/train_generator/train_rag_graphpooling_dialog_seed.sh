#!/bin/bash
decay=1 
TOPK=7
MLP_Layer=1
GNN_layer=1
ParaM=1
Timestamp=15
batch_size=64
n_layer=2
n_head=2
n_embed=256
lr=0.001 

for seed in 42 0 1 2 3 4 5 6 7 8
do
    export TRAIN_FILE="./resources/dialog/$Timestamp/train.link_prediction"
    export TEST_FILE="./resources/dialog/$Timestamp/test.link_prediction"
    export TEST_GT_FILE="./resources/dialog/$Timestamp/test_gt.link_prediction"
    export VAL_FILE="./resources/dialog/$Timestamp/val.link_prediction"
    export VAL_GT_FILE="./resources/dialog/$Timestamp/val_gt.link_prediction"
    
    export output="output/dialog/generator/graphfusion/$Timestamp/{$seed}/gpt2"
    
    export TRAIN_INDEX_FILE="./resources/train_generator/dialog/$Timestamp/train_gt_topk/train_index.gen"
    export TRAIN_SCORE_FILE="./resources/train_generator/dialog/$Timestamp/train_gt_topk/train_score.gen"
    
    export TEST_INDEX_FILE="./resources/retrieval_result_ablation/dialog/${RETRIEVAL_TYPE}/test_index.gen"
    export TEST_SCORE_FILE="./resources/retrieval_result_ablation/dialog/${RETRIEVAL_TYPE}/test_score.gen"
    
    export VAL_INDEX_FILE="./resources/retrieval_result_ablation/dialog/${RETRIEVAL_TYPE}/val_index.gen"
    export VAL_SCORE_FILE="./resources/retrieval_result_ablation/dialog/${RETRIEVAL_TYPE}/val_score.gen"

    export simpledyg_ckpt="simpledyg_ckpt/dialog/15/{7}/gpt2/checkpoint-0" 

    CUDA_VISIBLE_DEVICES=2 python main_generator.py \
        --dataset 'dialog' \
        --retrieval_type=$RETRIEVAL_TYPE \
        --fusion 'graphpooling' \
        --lambda_decay=$decay \
        --warmup_steps 0 \
        --lrdecay 0 \
        --weight_decay 1e-6 \
        --m=$ParaM \
        --topK $TOPK \
        --mlp_layers=$MLP_Layer \
        --gnn_layer=$GNN_layer \
        --learning_rate=$lr \
        --run_seed \
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
        --per_gpu_train_batch_size=$batch_size \
        --evaluate_during_training \
        --num_train_epochs 50 \
        --block_size 512 \
        --eval_all_checkpoints \
        --timestamp $Timestamp \
        --patience 10 \
        --n_layer=$n_layer \
        --n_head=$n_head \
        --n_embed=$n_embed \
        --seed=$seed \
        --simpledyg_checkpoint=$simpledyg_ckpt \
        --freeze 
done