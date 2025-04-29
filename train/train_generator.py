import time
import torch
import os
import copy
from utils.model import get_optimizer_scheduler
from dataloader.generator import get_dataloader, load_and_cache_examples
from utils.Evaluation_generator import  get_eval_metrics_generator
from utils.model import *
from tqdm import trange
import math
import wandb

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

def train_epoch(epoch,
                model, 
                tokenizer,
                optimizer, 
                scheduler, 
                train_dataloader,
                train_dataset, 
                tr_loss, 
                logging_loss, 
                global_step, 
                steps_trained_in_current_epoch, 
                args):
    """train one epoch"""
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    i=0
    print('==>training epoch: ', epoch)
    model.train()
    for step, batch in enumerate(train_dataloader):

        if args.lrdecay==1:
            adjust_learning_rate(args, optimizer, epoch, args.learning_rate, i, train_dataloader.__len__())
            
        if steps_trained_in_current_epoch > 0:
            steps_trained_in_current_epoch -= 1
            continue
        input_tokens, idxs_sim, scores_sim, egoids = batch
        idxs_sim =  idxs_sim.to(args.device) 
        inputs, labels = (input_tokens, input_tokens)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        i+=1
        
        if args.fusion=="mlp":
            from utils.model import fusion_mlp as fusion_v

        if args.fusion=="graphpooling":
            from utils.model import fusion_graphpooling as fusion_v


        lm_logits = fusion_v(args, model, tokenizer, train_dataset, 
                            input_tokens, idxs_sim,
                            args.m, top_k=args.topK)    

        t_tensor = torch.zeros((labels.size(0), args.m))
        augmented_labels = torch.ones_like(t_tensor) * -100
        augmented_labels = augmented_labels.long().to(args.device) # Bxm
        labels = torch.cat([labels[:, :2], augmented_labels, labels[:, 2:]], dim=1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

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
            if args.lrdecay==0:
                scheduler.step()  # Update learning rate schedule

            model.zero_grad()
            global_step += 1
            logging_loss = tr_loss

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return model, optimizer, scheduler, global_step, tr_loss, logging_loss


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    # Prepare dataloader
    train_dataloader, args = get_dataloader(train_dataset, tokenizer, args)
    print('iter one batch',train_dataloader.__len__())

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
        # train
        model, optimizer, scheduler, global_step, \
        tr_loss, logging_loss = train_epoch(epoch,
                                            model, 
                                            tokenizer, 
                                            optimizer, 
                                            scheduler, 
                                            train_dataloader, 
                                            train_dataset,
                                            tr_loss, 
                                            logging_loss, 
                                            global_step,
                                            steps_trained_in_current_epoch, 
                                            args)

        _, val_loss = evaluate(args, model, tokenizer)
        top_k_scores = get_eval_metrics_generator(args, epoch, model, tokenizer, global_step, mode="val", is_rag=True)
        score = top_k_scores['NDCG'][0]
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch} | Step: {global_step} | train loss: {tr_loss/global_step} | val loss: {val_loss} | val_NDCG@5: {top_k_scores['NDCG'][0]} | lr: {lr} ")
        
        wandb.log({
            'lr': lr,
            "train_loss": tr_loss/global_step,
            "val_loss": val_loss, 
            "val_NDCG@5": top_k_scores['NDCG'][0],
            'val_jaccard': top_k_scores['jaccard'][0],
            })  

        if epoch >args.warmup_steps:
            if best_score is None or score > best_score:
                best_score = score
                save_checkpoint(model, optimizer, scheduler, tokenizer, args, 0)
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                best_step = global_step
                counter = 0
            else:
                counter+=1
                print('Score: {} < Best_score {}'.format(score, best_score))
                print('best_epoch: ', best_epoch)
                print('best_step: ', global_step)
                print('EarlyStopping counter: {} out of {}'.format(counter, args.patience))
                if counter >= args.patience:
                    early_stop = True
            
            if early_stop:
                print('Early Stopping.....')
                train_iterator.close()
                break
    
    end_time = time.time()
    cost_time = (end_time - start_time) / 3600
    print("***** Train cost time: {} hours *****".format(cost_time))

    print("***** Running testing *****")
    top_k_scores_test  = get_eval_metrics_generator(args, best_epoch, best_model, tokenizer, best_step, mode="test", is_rag=True, is_best=True)
    end_time = time.time()
    cost_time = (end_time - start_time) / 3600
    print("test_metrics best epoch : ", top_k_scores_test) 
    print("***** Total cost time: {} hours *****".format(cost_time))

    print("***** Running val *****")
    top_k_scores_val  = get_eval_metrics_generator(args, best_epoch, best_model, tokenizer, best_step, mode="val", is_rag=True, is_best=True)
    print("val_metrics best epoch : ", top_k_scores_val)

    print("***** Running testing on last epoch *****")
    top_k_scores_last = get_eval_metrics_generator(args, epoch, model, tokenizer, global_step, mode="test", is_rag=True)
    print("test_metrics last epoch : ", top_k_scores_last) 
    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):

    if args.fusion=="mlp":
        from utils.model import fusion_mlp as fusion_v
    if args.fusion=="graphpooling":
        from utils.model import fusion_graphpooling as fusion_v
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)
    # Prepare dataloader
    eval_dataloader, args = get_dataloader(eval_dataset, tokenizer, args, split='eval')
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    print("***** Running evaluation {} *****".format(prefix))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    with torch.no_grad():
        for batch in eval_dataloader:
            inputs_, index, _, egoids = batch
            inputs, labels = (inputs_, inputs_)
            index = index.to(args.device)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            lm_logits = fusion_v(args, model, tokenizer, train_dataset, 
                                inputs, index, args.m, args.topK)
            t_tensor = torch.zeros((labels.size(0), args.m))
            augmented_labels = torch.ones_like(t_tensor) * -100
            augmented_labels = augmented_labels.long().to(args.device)
            labels = torch.cat([labels[:, :2], augmented_labels, labels[:, 2:]], dim=1)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            eval_loss += loss.item()
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        result = eval_loss
    return result, eval_loss
