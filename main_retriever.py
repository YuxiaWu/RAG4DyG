"""
Fine-tuning pretrained language model (GPT2) on Task-oriented Dialogue
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import pandas as pd
import json
import numpy as np
import torch
from tqdm import tqdm, trange
import copy
from transformers import (
    WEIGHTS_NAME,
    GPT2Tokenizer,
    GPT2Tokenizer,
)
# comment this if you want to load gpt2 class from transformers
from models.modeling_rag import GPT2LMHeadModel
from models import GPT2Config, GPT2SmallConfig

from dataloader.retriever import *
from utils.model import *
from utils.args_parser_retriever import ArgsParser
from utils.tokenizer import get_model_tokenizer
from train.train_retriever import train, test
import torch
import wandb

logger = logging.getLogger(__name__)

torch.set_num_threads(50)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2-small": (GPT2SmallConfig, GPT2LMHeadModel, GPT2Tokenizer),
}

def main():
    args = ArgsParser().parse()
    set_seed(args)
    

    # set weight decay
    if args.dataset == "UCI_13":
        args.weight_decay = 1e-3
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "--eval_data_file should be specified when do_eval is true"
        )

    #if args.dataset == "enron":
    #    from train.train_retriever_step import train,test

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # initialize distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    lr_type = 'y' if args.learning_rate > 0 else 'n'
    ckpt = 1 if args.should_continue==1 else 0

    args.para_names = ['d', 'alpha','eta','gamma',
                        'nl','nh','emb','bz',
                        'lr', 'lrdecay','tdecay', 
                        'se', 'temp','ckpt','wd','loss']
    args.para_values = [args.dataset, args.alpha, args.eta, args.gamma, 
                        args.n_layer, args.n_head, args.n_embed, args.per_gpu_train_batch_size, 
                        args.learning_rate, lr_type, args.lambda_decay,
                        args.seed, args.temperature, ckpt, args.weight_decay, args.loss_type]
    
    run_name = ''
    for para_name, para_value in zip(args.para_names, args.para_values):
        run_name += para_name + ':' + str(para_value) + '_'
    args.run_name = run_name
    
    proj_name = 'RAG4DyG_retriever'+args.dataset
    wandb.login(key='your key')
    wandb.init(project=proj_name, name=args.run_name, save_code=True, config=args)
    wandb.run.log_code(".")

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # if not the first process, do not load pretrained model & vocab

    # Train model from scatch
    model, tokenizer, model_class, args = get_model_tokenizer(args, MODEL_CLASSES)

    if args.should_continue:
        print('load model from checkpoint')
        if args.dataset == "UCI_13":
            simpledyg_checkpoint = "simpledyg_ckpt/UCI_13/12/{42}/gpt2/checkpoint-0" 
        elif args.dataset == "hepth":
            simpledyg_checkpoint =  "simpledyg_ckpt/hepth/11/{4}/gpt2/checkpoint-0" 
        elif args.dataset == "dialog":
            simpledyg_checkpoint =  "simpledyg_ckpt/dialog/15/{7}/gpt2/checkpoint-0" 
        elif args.dataset == "wikiv2":
            simpledyg_checkpoint =  "simpledyg_ckpt/wikiv2/15/{42}/gpt2/checkpoint-0" 
            #simpledyg_checkpoint =  "output/wikiv2/simpledyg/15/{42}/gpt2/checkpoint-0"
        elif args.dataset == "enron":
            simpledyg_checkpoint =  "output/enron/simpledyg_ckpt/16/{42}/gpt2/checkpoint-0"    
        elif args.dataset == "reddit":
            simpledyg_checkpoint =  "output/reddit/simpledyg_ckpt/11/{42}/gpt2/checkpoint-0"    

        model.transformer = model.transformer.from_pretrained(simpledyg_checkpoint)
        model.resize_token_embeddings(len(tokenizer)) 
    model = model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # finish barrier, when first process has loaded pretrained model & vocab
    # Training
    try:
        if args.do_train:
            if args.local_rank not in [-1, 0]:
                torch.distributed.barrier()  # only first process will preprocess data/caching

            train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

            if args.local_rank == 0:
                torch.distributed.barrier() # end of barrier

            global_step, train_loss = train(args, train_dataset, model, tokenizer)
            print(" global_step = {}, average loss = {}".format(global_step, train_loss))

        # Evaluation
        if args.do_eval and args.local_rank in [-1, 0]:
            checkpoints = [args.output_dir]

            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
            print("Evaluate the following checkpoints: {}".format(checkpoints))

            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                #model = model_class.from_pretrained(checkpoint)
                state_dict = torch.load(os.path.join(checkpoint, "pytorch_model.bin"))
                model.load_state_dict(state_dict)
                model.to(args.device)
                if args.n_gpu > 1:
                    model = torch.nn.DataParallel(model)     
                #_, _ = test(args, model, tokenizer, prefix=prefix)
                test_metrics= test(0, args, model, tokenizer, evaluate=False, prefix="best")
                print('test_metrics: ', test_metrics)

        wandb.finish()
    except KeyboardInterrupt:
        print("Interrupted! Stopping wandb logging.")
        wandb.finish()

if __name__ == "__main__":
    main()
