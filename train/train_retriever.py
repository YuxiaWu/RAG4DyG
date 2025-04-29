import time
import torch
import math
import numpy as np
import os
import logging
import copy
import torch.nn as nn
import torch.nn.functional as F
from utils.model import get_optimizer_scheduler
from dataloader.retriever import *
from utils.model import *
from tqdm import tqdm, trange
import wandb
import pandas as pd

def ndcg_k(sorted_indices, ground_truth, k):
    dcg, pdcg = 0, 0
    
    for i, index in enumerate(sorted_indices[:k]):
        if index in ground_truth:
            relevance = 1  # Assuming binary relevance, change this if using graded relevance
            dcg += (2 ** relevance - 1) / math.log(i + 2, 2)
    
    for i in range(k):
        relevance = 1  # Assuming binary relevance, change this if using graded relevance
        pdcg += (2 ** relevance - 1) / math.log(i + 2, 2)
    
    return dcg / pdcg if pdcg > 0 else 0

def hit_rate_at_k(predictions, targets, k=1):
    indices = predictions[:k]
    gt = set(targets)
    pred = set(indices)
    for i in pred:
        if i in gt:
            return 1
    return 0

def CLtime_loss(args, anchors, positives, hard_negatives, anchors_time, positives_time, negatives_time):
    temperature = args.temperature
    decay_rate = args.lambda_decay  # Decay rate parameter
    batch_size = anchors.size(0)
    all_embeddings = torch.cat([anchors, positives], dim=0)
    all_embeddings = torch.cat([all_embeddings, hard_negatives], dim=0)
    # Compute similarity matrix
    similarity_matrix = F.cosine_similarity(all_embeddings.unsqueeze(1), all_embeddings.unsqueeze(0), dim=2)
    # Compute time differences and decay factors for positive and negative pairs
    # pair-wise time differences of anchors and positives
    time_diff_pos = torch.abs(anchors_time.unsqueeze(1) - positives_time) #[128,128,1]
    # decay factor for positive pairs
    decay_factor_pos = torch.exp(-decay_rate * time_diff_pos.squeeze())
    decay_factor_pos =decay_factor_pos.to(anchors.device)
    # Apply decay to positive similarities
    pos_similarities = similarity_matrix[:batch_size, batch_size:2*batch_size] * decay_factor_pos
    # Create labels for cross-entropy
    labels = torch.arange(batch_size).to(anchors.device)
    # Compute time differences for negatives and apply decay
    time_diff_neg = torch.abs(anchors_time.unsqueeze(1) - anchors_time)
    decay_factor_neg = torch.exp(-decay_rate * time_diff_neg.squeeze())
    decay_factor_neg.fill_diagonal_(0)  # Remove self-comparison
    decay_factor_neg =decay_factor_neg.to(anchors.device)
    # Apply decay to negative similarities
    neg_similarities = similarity_matrix[:batch_size, :batch_size] * decay_factor_neg
    time_diff_hard_neg = torch.abs(anchors_time.unsqueeze(1) - negatives_time)
    decay_factor_hard_neg = torch.exp(-decay_rate * time_diff_hard_neg.squeeze())
    decay_factor_hard_neg =decay_factor_hard_neg.to(anchors.device)
    hard_neg_similarities = similarity_matrix[:batch_size, 2*batch_size:] * decay_factor_hard_neg
    logits = torch.cat([pos_similarities, neg_similarities, hard_neg_similarities], dim=1) / temperature
    # Loss computation
    loss = F.cross_entropy(logits, labels)
    return loss

def mask_correlated_samples(batch_size):
    # this is for inco nce loss
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask

def info_nce(args, z_i, z_j, temp, batch_size, mask):
    N = 2 * batch_size
    z = torch.cat((z_i, z_j), dim=0)
    sim = torch.mm(z, z.T) / temp
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    if batch_size != args.per_gpu_train_batch_size:
        mask = mask_correlated_samples(batch_size)
    negative_samples = sim[mask].reshape(N, -1)
    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    cl_loss_func = nn.CrossEntropyLoss()
    info_nce_loss = cl_loss_func(logits, labels)
    return info_nce_loss

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

def adjust_learning_rate(args, optimizer, epoch, base_lr, i, iteration_per_epoch):
    T = epoch * iteration_per_epoch + i
    warmup_iters = args.warmup_steps * iteration_per_epoch
    total_iters = (args.num_train_epochs - args.warmup_steps) * iteration_per_epoch
    if epoch < args.warmup_steps:
        lr = base_lr * 1.0 * T / warmup_iters
    else:
        T = T - warmup_iters
        lr = 0.5 * base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_epoch(
    all_query_time,
    epoch,
    model, 
    optimizer, 
    scheduler, 
    train_dataloader, 
    tr_loss, 
    logging_loss, 
    global_step, 
    steps_trained_in_current_epoch, 
    args,
    mask_nce
    ):
    """train one epoch"""
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    
    epoch_iterator = tqdm(train_dataloader,  
                        desc="Iteration", 
                        disable=args.local_rank not in [-1, 0])
    i = 0
    step = 0
    tr_loss = 0
    tr_cl_loss=0
    tr_aug_loss=0
    model.train()
    for batch in epoch_iterator:
        if args.lrdecay==1:               
            adjust_learning_rate(args, optimizer, epoch, args.learning_rate, i, train_dataloader.__len__())
        if steps_trained_in_current_epoch > 0:
            steps_trained_in_current_epoch -= 1
            continue
        anchor_seq, pos_seq, neg_seq, anchor_idx, pos_idx, neg_idx = batch
        """
        seq: the sequence data
        idx: the index of the seq in full train_data
        """
        anchor_seq = anchor_seq.to(args.device)
        pos_seq = pos_seq.to(args.device)
        neg_seq = neg_seq.to(args.device)
        
        _, h_ego = model(input_ids=anchor_seq)  # (batch_size, num_nodes, hidden_size)
        _, h_pos = model(input_ids=pos_seq)
        _, h_neg = model(input_ids=neg_seq)

        i += 1

        h_egos = torch.mean(h_ego, dim=1)
        h_i = torch.mean(h_pos, dim=1)
        h_j = torch.mean(h_neg, dim=1)

        cl_loss = CLtime_loss(args, h_egos, h_i, h_j, all_query_time[anchor_idx], all_query_time[pos_idx], all_query_time[neg_idx])

        aug_seq1, aug_seq2 = model._aug(anchor_seq)  
        _, seq_output1 = model(input_ids=aug_seq1)
        _, seq_output2 = model(input_ids=aug_seq2)
        h_1 = torch.mean(seq_output1, dim=1)
        h_2 = torch.mean(seq_output2, dim=1)
        
        aug_loss = args.alpha*info_nce(args,h_1, h_2, args.temperature, aug_seq1.size(0), mask_nce)
        loss = cl_loss + aug_loss
        tr_aug_loss+=aug_loss.item()
        tr_cl_loss+=cl_loss.item()

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
            #if args.lrdecay==0:
            #    scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break
        step += 1

    return model, optimizer, scheduler, global_step, tr_loss, tr_cl_loss, tr_aug_loss, logging_loss


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    print("Dataset: ", train_dataset)
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
                model, 
                device_ids=[args.local_rank], 
                output_device=args.local_rank, 
                find_unused_parameters=True
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
    mask_nce = mask_correlated_samples(args.per_gpu_train_batch_size)
    start_time = time.time()
    # load the query_time.pt
    query_time_file = os.path.join("resources/", args.dataset+'_train_query_time.pt')
    all_query_time = torch.load(query_time_file)

    for epoch in train_iterator:
        # show the initial results based on the loaded ckpt
        print('==> Training Epoch: ', epoch)
        model, optimizer, scheduler, global_step, tr_loss, cl_loss, aug_loss, logging_loss = train_epoch(all_query_time, epoch, model, optimizer, scheduler, train_dataloader, tr_loss, logging_loss, global_step,
                                steps_trained_in_current_epoch, args, mask_nce)
        val_metrics, val_loss = test(epoch, args, model, tokenizer, evaluate=True)
        score = val_metrics['hit@3']
        
        wandb.log({
            'lr': optimizer.param_groups[0]['lr'],
            "train_loss": tr_loss/global_step,
            'val_loss': val_loss,
            'val_hit@3': val_metrics['hit@3'],
        })        

        print('best_score: ', best_score)
        if epoch >args.warmup_steps:
            if best_score is None or score > best_score:
                best_score = score
                save_checkpoint(model, optimizer, scheduler, tokenizer, args, 0)
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                counter = 0            
            else:
                counter+=1
                print('  Score: {} < Best {}'.format(score, best_score))
                print('best_epoch: ', best_epoch)
                print('  EarlyStopping counter: {} out of {}'.format(counter, args.patience))
                if counter >= args.patience:
                    early_stop = True
        
        if early_stop:
            print('  Early Stopping.....')
            train_iterator.close()
            break
        
        save_checkpoint(model, optimizer, scheduler, tokenizer, args, 1)
        last_model = copy.deepcopy(model)

    
    end_time = time.time()
    cost_time = (end_time - start_time) / 3600
    print("***** Train cost time: {} hours *****".format(cost_time))

    # testing
    print("***** Running testing *****")
    print('best_epoch: ', best_epoch)
    test_metrics= test(best_epoch, args, best_model, tokenizer, evaluate=False, prefix="best")
    print("test_metrics best epoch : ", test_metrics)
    end_time = time.time()
    # time cost hours
    cost_time = (end_time - start_time) / 3600
    print("***** Total cost time: {} hours *****".format(cost_time))

    print("save eval files for best epoch")
    val_metrics, val_loss = test(best_epoch, args, best_model, tokenizer, evaluate=True, prefix="best")

    print("***** Running testing on last epoch *****")
    test_metrics= test(epoch, args, last_model, tokenizer, evaluate=False)
    print("test_metrics last epoch : ", test_metrics) 

    return global_step, tr_loss / global_step


def save_index_score(score_matrix, save_index_file, save_score_file, steps):
    indices = np.argsort(-score_matrix, axis=1)
    if steps==0:
        with open(save_index_file, 'w') as f, open(save_score_file, 'w') as g:
            for i in range(score_matrix.shape[0]):
                f.write(' '.join([str(x) for x in indices[i]]) + '\n')
                g.write(' '.join([f"{x:.4f}" for x in score_matrix[i]]) + '\n')
    else:
        with open(save_index_file, 'a') as f, open(save_score_file, 'a') as g:
            for i in range(score_matrix.shape[0]):
                f.write(' '.join([str(x) for x in indices[i]]) + '\n')
                g.write(' '.join([f"{x:.4f}" for x in score_matrix[i]]) + '\n')

def normalize_scores(scores):
    min_val = scores.min(axis=1, keepdims=True)
    max_val = scores.max(axis=1, keepdims=True)
    normalized_scores = (scores - min_val) / (max_val - min_val)
    return normalized_scores

def test(epoch, args, model, tokenizer, evaluate=True, prefix=""):
    eval_output_dir = args.output_dir
    test = False if evaluate else True

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=evaluate, test=test)
    
    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    # Prepare dataloader
    eval_dataloader, args = get_dataloader(eval_dataset, tokenizer, args, split='eval')
    train_dataset = LineByLineTextDatasetHistory(tokenizer, args, file_path=args.train_data_file, block_size=args.block_size)
    train_dataloader, args = get_dataloader(train_dataset, tokenizer, args, split='eval')

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    print("Num examples = {}".format(len(eval_dataset)))
    print("Batch size = {}".format(args.eval_batch_size))
    hit_1, hit_3 = 0.0, 0.0
    nb_eval_steps = 0
    eval_loss=0
    cnt = 0
    model.eval()

    if evaluate:
        score_file = args.eval_data_gt_file
    else:    
        score_file = args.test_data_gt_file

    with open(score_file, encoding="utf-8") as f:
        scores = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    scores = [list(map(float, item.split())) for item in scores]
    scores = torch.Tensor(scores)
    batch_size = args.eval_batch_size
    scores = DataLoader(scores, batch_size=batch_size, shuffle=False, num_workers=0,drop_last=False)

    train_embeddings = []
    for batch in tqdm(train_dataloader, desc="Getting train embeddings"):
        inputs = batch
        inputs = inputs.to(args.device)
        with torch.no_grad():
            _, h = model(input_ids=inputs)
            h_egos = torch.mean(h, dim=1)
            train_embeddings.append(h_egos)
    train_embeddings = torch.cat(train_embeddings, dim=0)
    print('size of train_embeddings: ', train_embeddings.size())

    for batch, score in tqdm(zip(eval_dataloader, scores), desc="Evaluating", total=len(eval_dataloader)):
        inputs = batch.to(args.device)
        score = score.to(args.device)
        
        with torch.no_grad():
            _, h = model(input_ids=inputs)
            
            h_egos = torch.mean(h, dim=1) #[batch size, hidden]
            h_egos_norm = h_egos / h_egos.norm(dim=1, keepdim=True)

            train_embeddings = train_embeddings.to("cuda")
            train_embeddings_norm = train_embeddings / train_embeddings.norm(dim=1, keepdim=True)
            dot_products = torch.matmul(h_egos_norm, train_embeddings_norm.t())
            dot_products = (dot_products + 1) / 2
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(dot_products, score)            
            eval_loss += loss
            score_array = score.detach().cpu().numpy()
            dot_products_array = dot_products.detach().cpu().numpy()
            save_file_path = f'resources/retrieval_result/{args.dataset}/'

            if not os.path.exists(save_file_path):
                os.makedirs(save_file_path)

            if evaluate:
                save_index_file = os.path.join(save_file_path, f'val_index.gen')
                save_score_file = os.path.join(save_file_path, f'val_score.gen')
            else:
                save_index_file = os.path.join(save_file_path, f'test_index.gen')
                save_score_file = os.path.join(save_file_path, f'test_score.gen')
            if prefix=="best":
                save_index_score(dot_products_array, save_index_file, save_score_file, nb_eval_steps)

            hit_batch_1, hit_batch_3 = 0, 0
            cnt_0 = 0
            for i in range(dot_products.shape[0]):
                gt = np.argsort(-score_array[i])
                gt = gt[:3]
                
                if len(gt) == 0:
                    cnt_0 += 1
                    continue
                pred = np.argsort(-dot_products_array[i])
                cnt+=1
                hit_batch_1 += hit_rate_at_k(pred, gt, 1)
                hit_batch_3 += hit_rate_at_k(pred, gt, 3)

            n = score_array.shape[0] - cnt_0
            hit_1 += hit_batch_1/n
            hit_3 += hit_batch_3/n
            
        nb_eval_steps += 1
    eval_loss = eval_loss / len(eval_dataset)
    hit_1 = round(hit_1 / nb_eval_steps, 4)
    hit_3 = round(hit_3 / nb_eval_steps, 4)
    eval_metrics = {
        'hit@1':hit_1,
        'hit@3':hit_3,
    }

    if evaluate:
        result_save_file = os.path.join(save_file_path, "val_results.csv")
    else:
        result_save_file = os.path.join(save_file_path, "test_results.csv")

    if prefix=="best":

        with open(result_save_file, "w") as f:
            f.write(f"{'epoch'}, ")
            for param in args.para_names:
                f.write(f"{param}, ")
            f.write("Hit@1, Hit@3\n")
            f.write(f"{epoch}, ")
            for param in args.para_values:
                f.write(f"{param}, ")
            f.write(f"{hit_1},{hit_3}\n")
            f.write('\n')
            f.flush()

    if evaluate:
        return eval_metrics, eval_loss
    elif prefix=="best":
        if args.run_seed:
            save_folder = os.path.join('topk_scores_seed_retrieval')
        else:
            save_folder = os.path.join('topk_scores_finetune')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        result_save_test = os.path.join(save_folder, args.dataset + '_retrieval.csv')

        test_results = pd.read_csv(result_save_file)

        if os.path.exists(result_save_test):
            test_results.to_csv(result_save_test, mode='a', header=False, index=False)
        else:
            test_results.to_csv(result_save_test, index=False)

        return eval_metrics
    else:
        return eval_metrics
    
