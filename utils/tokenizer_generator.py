import torch
import glob
import random
import numpy as np
from typing import Dict, List, Tuple
import os
import shutil
import json
import logging
from transformers import (
    WEIGHTS_NAME,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from transformers import GPT2Tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer

## a pretokenizer to segment the text into words
from tokenizers.pre_tokenizers import Whitespace
# uncomment this if you want to load gpt2 class from transformers
# from transformers import GP2Config, GPT2LMHeadModel
logger = logging.getLogger(__name__)

def get_model_tokenizer(args, MODEL_CLASSES):
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

    #model = model_class(config=config)

    #model.to(args.device)
    # '<|similar|>', '<|input|>', '<|output|>'
    spl_tokens = ['<|history|>','<|endofhistory|>','<|pre|>','<|endofpre|>'] + ['<|time'+str(i)+'|>' for i in range(int(args.timestamp)+1)]
    args.spl_tokens = spl_tokens
    data_path = './data_pre'
    dataset = args.dataset
    #data = pd.read_csv(os.path.join(data_path, dataset, dataset + '.csv'))
    # the vocab file path
    vocab_file = os.path.join('./vocabs', dataset, args.timestamp, 'vocab.json')

    # read the vocab file
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    my_vocab = WordLevel.read_file(vocab_file)
    tokenizer = Tokenizer(WordLevel(vocab=my_vocab))

    # Customize tokenizer settings
    tokenizer.pre_tokenizer = Whitespace()


    tokenizer_path = os.path.join('./tokenizers', dataset, args.timestamp)
    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path)

    tokenizer.save(os.path.join(tokenizer_path, "tokenizer.json"))

    tokenizer_file = os.path.join(tokenizer_path,"tokenizer.json" )

    gpt_tokenizer = PreTrainedTokenizerFast(tokenizer_file = tokenizer_file, use_fast=False)
    print('vocab size: ',gpt_tokenizer.vocab_size)
    gpt_tokenizer.truncation = True
    # truncation max length
    gpt_tokenizer.max_len = 1024
    #tokenizer.truncation = True
    gpt_tokenizer.truncation_side='left'
  
    #tokenizer.add_special_tokens(special_tokens) 
    gpt_tokenizer.add_special_tokens({'bos_token': '<|endoftext|>'})
    gpt_tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'}) 

    gpt_tokenizer.add_special_tokens({'additional_special_tokens':spl_tokens})
    gpt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    
    #tokenizer.add_tokens(add_special_tokens, special_tokens=True)
    gpt_tokenizer.save_pretrained(os.path.join(os.path.join('./tokenizers/',args.dataset, args.timestamp)))
    print('vocab size: ', gpt_tokenizer.vocab_size) # 3239

    # refine to work with new loss
    max_token_id =   gpt_tokenizer.vocab_size
    config.max_token_id = max_token_id

    model = model_class(config=config)

    model.to(args.device)

    model.resize_token_embeddings(len(gpt_tokenizer)) # 3245

    

    
    if args.dataset=='hepth':
        node_raw_features = np.load(args.node_feat_file)
        node_raw_features_vocab = node_raw_features[:gpt_tokenizer.vocab_size]
        if node_raw_features_vocab.shape[1] < args.n_embed:
            pad_feat = np.zeros((node_raw_features_vocab.shape[0],args.n_embed-node_raw_features_vocab.shape[1]))
            node_raw_features_vocab = np.concatenate((node_raw_features_vocab,pad_feat),axis=1)
            
        sp_feat = model.transformer.wte.weight[gpt_tokenizer.vocab_size:]
        node_raw_features_vocab = torch.FloatTensor(node_raw_features_vocab).cuda()
        weights = torch.cat([node_raw_features_vocab,sp_feat])
        model.transformer.wte = nn.Embedding.from_pretrained(embeddings = weights, freeze = False)

    return model, gpt_tokenizer, model_class, args
