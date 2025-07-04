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
    PreTrainedTokenizerFast,
    GPT2Tokenizer,
)
from models import GPT2LMHeadModel
from models import GPT2Config, GPT2SmallConfig

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from dataloader.SimpleDyG import *
from utils.model import *
from utils.args_parser_SimpleDyG import ArgsParser

from tensorboardX import SummaryWriter

from tokenizers import Tokenizer
import torch
import torch.nn as nn
from utils.Evaluation_SimpleDyG import Evaluation, get_eval_metrics
import time
import wandb
logger = logging.getLogger(__name__)

torch.set_num_threads(50)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2-small": (GPT2SmallConfig, GPT2LMHeadModel, GPT2Tokenizer),
}

def get_model_tokenizer(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    
    # new added for fine-tuning the hyperparameters
    config.n_head = args.n_head
    config.n_layer = args.n_layer
    config.n_embd = args.n_embed
    

    spl_tokens = ['<|history|>','<|endofhistory|>','<|pre|>','<|endofpre|>'] + ['<|time'+str(i)+'|>' for i in range(int(args.timestamp)+1)]
    args.spl_tokens = spl_tokens
    dataset = args.dataset
    vocab_file = os.path.join('./vocabs', dataset, args.timestamp, 'vocab.json')
    my_vocab = WordLevel.read_file(vocab_file)
    tokenizer = Tokenizer(WordLevel(vocab=my_vocab) )
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer_path = os.path.join('./tokenizers', dataset, args.timestamp)
    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path)

    tokenizer.save(os.path.join(tokenizer_path, "tokenizer.json"))

    tokenizer_file = os.path.join(tokenizer_path,"tokenizer.json" )

    gpt_tokenizer = PreTrainedTokenizerFast(tokenizer_file = tokenizer_file, use_fast=False)
    print('vocab size: ',gpt_tokenizer.vocab_size)
    gpt_tokenizer.truncation = True
    gpt_tokenizer.max_len = 1024
    gpt_tokenizer.truncation_side='left'
    gpt_tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})
    gpt_tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})    

    gpt_tokenizer.add_special_tokens({'additional_special_tokens':spl_tokens})
    gpt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    gpt_tokenizer.save_pretrained(os.path.join(os.path.join('./tokenizers/',args.dataset, args.timestamp)))

    max_token_id =   gpt_tokenizer.vocab_size
    config.max_token_id = max_token_id

    model = model_class(config=config)

    model.to(args.device)

    model.resize_token_embeddings(len(gpt_tokenizer)) # 3245

    if args.dataset=='hepth' or args.dataset=='reddit':
        # node_feature
        node_raw_features = np.load(args.node_feat_file)
        node_raw_features_vocab = node_raw_features[:gpt_tokenizer.vocab_size]
        # cheack the feat dim, if feat_dim<768, then pad to 768
        if node_raw_features_vocab.shape[1] < args.n_embed:
            pad_feat = np.zeros((node_raw_features_vocab.shape[0],args.n_embed-node_raw_features_vocab.shape[1]))
            node_raw_features_vocab = np.concatenate((node_raw_features_vocab,pad_feat),axis=1)
            
        sp_feat = model.transformer.wte.weight[gpt_tokenizer.vocab_size:]
        node_raw_features_vocab = torch.FloatTensor(node_raw_features_vocab).cuda()
        weights = torch.cat([node_raw_features_vocab,sp_feat])
        model.transformer.wte = nn.Embedding.from_pretrained(embeddings = weights, freeze = False)


    return model, gpt_tokenizer, model_class, args

def get_training_info(dataloader, args):
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(dataloader) // args.gradient_accumulation_steps)

            print("  Continuing training from checkpoint, will skip to saved global_step")
            print("  Continuing training from epoch %d", epochs_trained)
            print("  Continuing training from global step %d", global_step)
            print("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            print("  Starting fine-tuning.")
    return global_step, epochs_trained, steps_trained_in_current_epoch



def train_epoch(model, tokenizer, optimizer, scheduler, train_dataloader, tr_loss, logging_loss, global_step, steps_trained_in_current_epoch, tb_writer, args):
    """train one epoch"""
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):

        # Skip past any already trained steps if resuming training
        if steps_trained_in_current_epoch > 0:
            steps_trained_in_current_epoch -= 1
            continue

        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        model.train()
        outputs = model(inputs, labels=labels)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        tr_loss += loss.item()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            # Log metrics
            
            '''
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                if (args.local_rank == -1 and args.evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well
                    results, val_loss = evaluate(args, model, tokenizer)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss
            '''
            logging_loss = tr_loss
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break

    return model, optimizer, scheduler, global_step, tr_loss, logging_loss


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter('./runs/{}/{}/{}'.format(args.dataset, args.timestamp, args.run_name))

    # Prepare dataloader
    train_dataloader, args = get_dataloader(train_dataset, tokenizer, args)

    # total iteration and batch size
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1)

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer, scheduler = get_optimizer_scheduler(args, model, t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    print("***** Running training *****")
    print("  Num examples = {}".format(len(train_dataset)))
    print("  Num Epochs = {}".format(args.num_train_epochs))
    print("  Instantaneous batch size per GPU = {}".format(args.per_gpu_train_batch_size))
    print("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(total_batch_size))
    print("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    print("  Total optimization steps = {}".format(t_total))

    global_step, epochs_trained, steps_trained_in_current_epoch = get_training_info(train_dataloader, args)

    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()

    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    best_score = None
    early_stop = False
    counter = 0

    start_time = time.time()

    for epoch in train_iterator:
        model, optimizer, scheduler, global_step, tr_loss, logging_loss = train_epoch(model, tokenizer, optimizer, scheduler, train_dataloader, tr_loss, logging_loss, global_step,
                                  steps_trained_in_current_epoch, tb_writer, args)

        #val_loss = evaluate(args, model, tokenizer)
        top_k_scores = get_eval_metrics(args, model, tokenizer, global_step, mode="val")
        #print("  val_loss = {}".format(val_loss))
        
        score = top_k_scores['NDCG'][0]

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch} | Step: {global_step} | train loss: {tr_loss/global_step}  | val_NDCG@5: {top_k_scores['NDCG'][0]} | lr: {lr} ")
        

        print("  val_NDCG@1 = {}".format(score))

        #tb_writer.add_scalar("val_loss", val_loss, global_step)   
        tb_writer.add_scalar("val_NDCG_1", score, global_step)     

        
        wandb.log({
            'lr': lr,
            "train_loss": tr_loss/global_step,
            #"val_loss": val_loss, 
            "val_NDCG@1": top_k_scores['NDCG'][0],
            "val_NDCG@5": top_k_scores['NDCG'][1],
            'val_jaccard': top_k_scores['jaccard'][0],
            })  


        if best_score is None:
            best_score = score
            save_checkpoint(model, optimizer, scheduler, tokenizer, args,0)
            best_model = copy.deepcopy(model)
            best_step = global_step
        elif score < best_score:
            counter+=1
            print('Score: {} < Best_score {}'.format(score, best_score))

            print('EarlyStopping counter: {} out of {}'.format(counter, args.patience))
            if counter >= args.patience:
                early_stop = True
        else:
            best_score = score
            save_checkpoint(model, optimizer, scheduler, tokenizer, args, 0)
            best_model = copy.deepcopy(model)
            best_step = global_step
            counter = 0

        if early_stop:
            print('Early Stopping.....')
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    
    end_time = time.time()
    cost_time = (end_time - start_time) / 3600
    print("***** Train cost time: {} hours *****".format(cost_time))

    print("***** Running val *****")
    top_k_scores_val = get_eval_metrics(args, best_model, tokenizer, best_step, mode="val")
    print('top_k_scores_val: ', top_k_scores_val)
    print("***** Running test *****")
    top_k_scores_test = get_eval_metrics(args, best_model, tokenizer, best_step, mode="test")
    print('top_k_scores_test: ', top_k_scores_test)
    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)
    # Prepare dataloader
    eval_dataloader, args = get_dataloader(eval_dataset, tokenizer, args, split='eval')
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    return eval_loss


def main():
    args = ArgsParser().parse()
    set_seed(args)
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "--eval_data_file should be specified when do_eval is true"
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("--should_continue is true, but no checkpoint found in --output_dir")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

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

    args.para_names = ['dataset', 'method', 'time', 'nlayer','nhead','nemb','bz','lr','seed']
    args.para_values = [args.dataset, 'SimpleDyG', args.timestamp, args.n_layer, args.n_head, args.n_embed, args.per_gpu_train_batch_size, args.learning_rate, args.seed]
    
    run_name = ''
    for para_name, para_value in zip(args.para_names, args.para_values):
        run_name += para_name + '_' + str(para_value) + '_'
    args.run_name = run_name
    log_path = os.path.join('./all_logs', args.dataset, args.timestamp, args.run_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # Setup logging
    if args.do_train:
        log_save_name = os.path.join(log_path, 'train.log')
    else:
        log_save_name = os.path.join(log_path, 'eval.log' )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        filename = log_save_name,
        filemode = 'w',
        force =True,
        level=logging.INFO #if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # if not the first process, do not load pretrained model & vocab

    model, tokenizer, model_class, args = get_model_tokenizer(args)

    print("model.config {}".format(model.config)) 
    print("model {}".format(model))

    proj_name = 'SimpleDyG_'+args.dataset
    wandb.login(key='5059699e3ffaa4bc3836fde49c086cd29048546e')
    wandb.init(project=proj_name, name=args.run_name, save_code=True, config=args)
    wandb.run.log_code(".") 

   
    if args.local_rank == 0:
        torch.distributed.barrier()  # finish barrier, when first process has loaded pretrained model & vocab

    print("Training/evaluation parameters {}".format(args))

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
        results = {}
        if args.do_eval and args.local_rank in [-1, 0]:
            checkpoints = [args.output_dir]

            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("models.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            print("Evaluate the following checkpoints: {}".format(checkpoints))

            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)
                if args.n_gpu > 1:
                    model = torch.nn.DataParallel(model)     

                _ = get_eval_metrics(args, model, tokenizer, 0, mode="test")
        wandb.finish()
    except KeyboardInterrupt:
        print("Interrupted! Stopping wandb logging.")
        wandb.finish()            

    return results


if __name__ == "__main__":
    main()
