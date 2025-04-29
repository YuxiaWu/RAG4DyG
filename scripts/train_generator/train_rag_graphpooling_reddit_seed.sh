#!/bin/bash
#TOPK=7 
GNN_layer=1
MLP_Layer=1 
ParaM=1
Timestamp=11
batch_size=32
n_layer=2
n_head=8
n_embed=512
lr=0.0001 
#RETRIEVAL_TYPE='overlap_out_0.0'
RETRIEVAL_TYPE='augclhardtime'
for TOPK in 7 
do
    for seed in 42 0 1 #2 3 4 5 6 7 8 
    do
        export TRAIN_FILE="./resources/reddit/$Timestamp/train.link_prediction"
        export TEST_FILE="./resources/reddit/$Timestamp/test.link_prediction"
        export TEST_GT_FILE="./resources/reddit/$Timestamp/test_gt.link_prediction"
        export VAL_FILE="./resources/reddit/$Timestamp/val.link_prediction"
        export VAL_GT_FILE="./resources/reddit/$Timestamp/val_gt.link_prediction"
        
        export output="output/reddit/generator/graphfusion/$Timestamp/{$seed}/gpt2"
        
        export TRAIN_INDEX_FILE="./resources/train_generator/reddit/$Timestamp/train_gt_topk/train_index.gen"
        export TRAIN_SCORE_FILE="./resources/train_generator/reddit/$Timestamp/train_gt_topk/train_score.gen"

        #export TEST_INDEX_FILE="./resources/reddit/$Timestamp/train_retrieval/test_index.retrieval"
        #export TEST_SCORE_FILE="./resources/reddit/$Timestamp/train_retrieval/test_score.retrieval"
        
        #export VAL_INDEX_FILE="./resources/reddit/$Timestamp/train_retrieval/val_index.retrieval"
        #export VAL_SCORE_FILE="./resources/reddit/$Timestamp/train_retrieval/val_score.retrieval"
        export NODE_FILE="./resources/reddit/node_features.npy" 
        #--node_feat_file=$NODE_FILE \
        export TEST_INDEX_FILE="./resources/retrieval_result/reddit/$RETRIEVAL_TYPE/42/test_index.gen"
        export TEST_SCORE_FILE="./resources/retrieval_result/reddit/$RETRIEVAL_TYPE/42/test_score.gen"
        
        export VAL_INDEX_FILE="./resources/retrieval_result/reddit/$RETRIEVAL_TYPE/42/val_index.gen"
        export VAL_SCORE_FILE="./resources/retrieval_result/reddit/$RETRIEVAL_TYPE/42/val_score.gen"
        
        # --warmup_steps 0 \
        #--max_grad_norm 10 \
        #--weight_decay 1e-6 \
        #--lrdecay 0 \
        CUDA_VISIBLE_DEVICES=3 python main_generator.py \
            --task 'RAG' \
            --dataset 'reddit' \
            --fusion 'graphpooling' \
            --retrieval_type=$RETRIEVAL_TYPE \
            --node_feat_file=$NODE_FILE \
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
            --simpledyg_checkpoint "./output/reddit/simpledyg_ckpt/11/{42}/gpt2/checkpoint-0" \
            --freeze \
            --run_seed
    done
done