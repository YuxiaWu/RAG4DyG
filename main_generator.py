import sys
sys.path.append('..')
sys.path.append('../')
import glob
import os
import torch
from transformers import (
    WEIGHTS_NAME,
    GPT2Tokenizer,
)
from models.modeling_rag import GPT2LMHeadModel
from models import GPT2Config, GPT2SmallConfig
from dataloader.generator import  load_and_cache_examples
from utils.model import *
from utils.args_parser_generator import ArgsParser
from utils.tokenizer_generator import get_model_tokenizer
from train.train_generator import train
import torch
import wandb
from utils.Evaluation_generator import get_eval_metrics_generator

torch.set_num_threads(50)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2-small": (GPT2SmallConfig, GPT2LMHeadModel, GPT2Tokenizer),
}

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

    args.para_names = ['d', 
                    'nl','nh','nd','bz', 'lr','se', 
                    'fus', 'm', 'k', 'mlp','gnn', 
                    'lrdecay', 'wd','wm','re']
    args.para_values = [args.dataset,  
                        args.n_layer, args.n_head, args.n_embed, args.per_gpu_train_batch_size, args.learning_rate, args.seed, 
                        args.fusion, args.m, args.topK, args.mlp_layers, args.gnn_layers, 
                        args.lrdecay, args.weight_decay, args.warmup_steps, args.retrieval_type]

    run_name = ''
    for para_name, para_value in zip(args.para_names, args.para_values):
        run_name += para_name + ':' + str(para_value) + '_'
    args.run_name = run_name

    
    proj_name = 'RAG4DyG_generator_'+ args.task + '_'+args.dataset
    wandb.login(key='your key')
    wandb.init(project=proj_name, name=args.run_name, save_code=True, config=args)
    wandb.run.log_code(".")    

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # if not the first process, do not load pretrained model & vocab

    model, tokenizer, model_class, args = get_model_tokenizer(args, MODEL_CLASSES)

    if args.fusion=="mlp":
        _= model.get_mlp(512, args.m, args.mlp_layers)

    if  "graphpooling" in args.fusion: 
        _ = model.get_gnn(args.n_embed, int(args.n_embed/2), args.n_embed, args.gnn_layers, 0.2)

    if args.local_rank == 0:
        torch.distributed.barrier()  

    try:
        if args.do_train:
            if args.local_rank not in [-1, 0]:
                torch.distributed.barrier()  

            train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

            if args.local_rank == 0:
                torch.distributed.barrier()

            print("Retrieval checkpoint: ", args.retrieval_checkpoint)

            if args.freeze:
                print("Freezing the parameters of the transformer")
                model = load_and_freeze_params(model, args.simpledyg_checkpoint)

            model.to(args.device)
            global_step, train_loss = train(args, train_dataset, model, tokenizer)

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

                state_dict = torch.load(os.path.join(checkpoint, "pytorch_model.bin"))
                model.load_state_dict(state_dict)

                model.to(args.device)
                if args.n_gpu > 1:
                    model = torch.nn.DataParallel(model)   
                _ = get_eval_metrics_generator(args, 0, model, tokenizer, global_step, mode="test", is_rag=True)
        wandb.finish()
    except KeyboardInterrupt:
        print("Interrupted! Stopping wandb logging.")
        wandb.finish()

if __name__ == "__main__":
    main()
