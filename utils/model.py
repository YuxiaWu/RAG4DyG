import regex as re
import torch
import glob
import random
import numpy as np
import os
import shutil
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
import networkx as nx
from torch_geometric.utils import from_networkx

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False):
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        shutil.rmtree(checkpoint)


def save_checkpoint(model, optimizer, scheduler, tokenizer, args, global_step):
    checkpoint_prefix = "checkpoint"
    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))

    _rotate_checkpoints(args, checkpoint_prefix)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

def load_and_freeze_params(model, checkpoint_path):
    model.transformer = model.transformer.from_pretrained(checkpoint_path)
    model = model.cuda()
    # Freeze the parameters of the transformer
    for name, param in model.named_parameters():
        if 'transformer' in name:
            param.requires_grad = False
    return model

def get_optimizer_scheduler(args, model, t_total):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if (
            args.model_name_or_path
            and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    return optimizer, scheduler


def fusion_mlp(args, model, tokenizer, dataset, tokens_tensor, idxs_sim, m, top_k=5):
    """Agument the input sequence with the retrieved samples not used the similar score
        datase: train_dataset   train_text
        tokens_tensor: one batch data: input sequence  
        idxs_sim: index of the retrieved samples, padded with -1
        m: parameter transform the length of retrieved tokens
    """
    # convert tokens, idxs_sim to 3D tensor if they are 2D tensor
    if len(tokens_tensor.size()) == 1:
        tokens_tensor = tokens_tensor.unsqueeze(0)
    if len(idxs_sim.size()) == 1:
        idxs_sim = idxs_sim.unsqueeze(0)
    idxs_sim = idxs_sim[:, :top_k] 

    concat_sim_tokens = []
    for i in range(idxs_sim.size(0)): # [batch_size,10] 
        history_topK_i = []
        for node in idxs_sim[i]:
            #if node == -1:
            #    continue
            list_seq = dataset.retrieval_sources[node]
            history_topK_i += list_seq
        concat_sim_tokens.append(history_topK_i)
        
    # padding to the same length in concat_sim_tokens
    # batch tokens: [batch_size, max_len_sim]
    pad_value = tokenizer.pad_token_id
    max_len_sim=512
    for i in range(len(concat_sim_tokens)):
        n_i = len(concat_sim_tokens[i])
        if n_i < max_len_sim:
            concat_sim_tokens[i] = torch.tensor(concat_sim_tokens[i], dtype=torch.long)
            concat_sim_tokens[i] = torch.nn.functional.pad(concat_sim_tokens[i], (0, max_len_sim - n_i), value=pad_value)
        else:
            concat_sim_tokens[i] = torch.tensor(concat_sim_tokens[i][:max_len_sim], dtype=torch.long)    
    concat_sim_tokens = [torch.tensor(elem, dtype=torch.long) if isinstance(elem, list) else elem for elem in concat_sim_tokens]
    concat_sim_tokens = torch.stack(concat_sim_tokens)
    concat_sim_tokens = concat_sim_tokens.to(args.device)
    tokens_tensor = tokens_tensor.to(args.device)

    # get the hidden state of similar tokens and the input tokens
    # concat_sim_tokens: batch,max_len_sim
    H_sim = model.transformer.wte(concat_sim_tokens) # batch,max_len_sim, hidden
    H = model.transformer.wte(tokens_tensor) # batch,len_tokens, hidden

    # transform the embedding of similar tokens to m tokens
    # H_sim: [batch_size, max_len_sim, emb_size] -> [batch_size, m, emb_size]
    mlp_fusion = model.mlp_fusion #model.get_mlp(H_sim.size(1), args.m)
    mlp_fusion = mlp_fusion.to(args.device)

    H_sim = H_sim.view(-1, H_sim.size(1))
    H_sim_ = mlp_fusion(H_sim)
    H_sim_ = H_sim_.view(-1, m, args.n_embed) # [batch_size, max_len_sim, emb_size]
    
    H_aug = torch.cat([H[:, :2], H_sim_, H[:, 2:]], dim=1)
    # get the prediction from the model
    outputs, hidden_states = model(inputs_embeds=H_aug)
    
    lm_logits = outputs[0]
    return lm_logits


def fusion_graphpooling(args, model, tokenizer, dataset, tokens_tensor, idxs_sim, m, top_k=5):
    """
    Fusion encoder with graph neural network.
    We first build a graph from retrieved nodes and then use the graph to update the node embeddings.
    """
    
    fused_graphs = []  # Fused graphs for each sample in the batch
    if tokens_tensor.dim() == 1:
        tokens_tensor = tokens_tensor.unsqueeze(0)
    if idxs_sim.dim() == 1:
        idxs_sim = idxs_sim.unsqueeze(0)
    idxs_sim = idxs_sim[:, :top_k] 

    # Construct and fuse graphs for each sample in the batch
    for i in range(idxs_sim.size(0)): # batch_size
        # Initialize a graph for the current sample
        fused_graph = nx.Graph()
        # Iterate over top_k retrieved sequences
        #for node in idxs_sim[i][:top_k]:
        for node in idxs_sim[i]:
            list_seq = dataset.retrieval_sources[node]
            ego_id = int(list_seq[2])
            list_seq = [int(elem) for elem in list_seq]
            fused_graph.add_edges_from([(ego_id, elem) for elem in list_seq])
        
        # After fusing, collect node IDs and features
        node_ids = list(fused_graph.nodes)
        node_ids_tensor = torch.tensor(node_ids).to(args.device)
        feats = model.transformer.wte(node_ids_tensor)  # Get node features from the transformer model

        # Convert the fused graph to PyTorch Geometric Data format
        data = from_networkx(fused_graph)
        data.x = feats
        fused_graphs.append(data)
    
    gnn_model = model.gnn_fusion
    gnn_model = gnn_model.to(args.device)
    # GNN model forward pass for each fused graph in the batch
    all_embeddings = []
    for data in fused_graphs:
        data = data.to(args.device)
        embeddings = gnn_model(data.x, data.edge_index)
        # Calculate mean pooling of embeddings
        mean_embedding = torch.mean(embeddings, dim=0)
        all_embeddings.append(mean_embedding)

    H_sim = torch.stack(all_embeddings, dim=0)
    H_sim = H_sim.unsqueeze(1)
    H = model.transformer.wte(tokens_tensor.to(args.device)) # batch,len_tokens, hidden

    H_sim_ = H_sim
    # H before augmentation:
    H_aug = torch.cat([H[:, :2], H_sim_, H[:, 2:]], dim=1)

    # get the prediction from the model
    outputs, hidden_states = model(inputs_embeds=H_aug)
    lm_logits = outputs[0]
    return lm_logits 