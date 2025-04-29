import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer
import os
import logging
logger = logging.getLogger(__name__)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size,truncation='longest_first')["input_ids"]
        #print("Example: ", self.examples[0])

        # 
        #self.examples = tokenizer.batch_encode_plus(lines[:2], add_special_tokens=True, max_length=block_size,truncation='longest_first')["input_ids"]
        #['<|belief|> 6076 20 4903 4996 <|endofbelief|> <|action|> 6077 6063 <|endofaction|>', '<|belief|> 6076 20 4903 4996 6077 6063 6076 4924 5080 5048 <|endofbelief|> <|action|> 6077 750 4996 <|endofaction|>']
        #[[55432, 52081, 1238, 52115, 53563, 55433, 55434, 50420, 51556, 55435], [55432, 52081, 1238, 52115, 53563, 50420, 51556, 52081, 53049, 54684, 54158, 55433, 55434, 50420, 15426, 53563, 55435]]
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

class LineByLineTextDatasetHistory(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
        
            lines_all = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            lines = [line.split('<|pre|>')[0].strip() for line in lines_all]
            #train_lines = [line.split('<|pre|>')[0].split('<|endoftext|>')[1].strip() for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            # extract the historical sequence
            print('line0: ', lines[0])

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size,truncation='longest_first')["input_ids"]
        #print("Example: ", self.examples[0])

        # 
        #self.examples = tokenizer.batch_encode_plus(lines[:2], add_special_tokens=True, max_length=block_size,truncation='longest_first')["input_ids"]
        #['<|belief|> 6076 20 4903 4996 <|endofbelief|> <|action|> 6077 6063 <|endofaction|>', '<|belief|> 6076 20 4903 4996 6077 6063 6076 4924 5080 5048 <|endofbelief|> <|action|> 6077 750 4996 <|endofaction|>']
        #[[55432, 52081, 1238, 52115, 53563, 55433, 55434, 50420, 51556, 55435], [55432, 52081, 1238, 52115, 53563, 50420, 51556, 52081, 53049, 54684, 54158, 55433, 55434, 50420, 15426, 53563, 55435]]
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class PairSequenceDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512):
        assert os.path.isfile(file_path)
        logger.info("Loading sequence from file at %s", file_path)
        
        with open(args.train_data_file, encoding="utf-8") as f:
            train_lines_all = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            train_lines = [line.split('<|pre|>')[0].strip() for line in train_lines_all]
            #train_lines = [line.split('<|pre|>')[0].split('<|endoftext|>')[1].strip() for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            # extract the historical sequence
            #print('line0: ', train_lines[0])
            #seq.split('<|endoftext|>')
            #seq = seq.split('<|history|>')[1].split('<|endofhistory|>')[0].split(' ')
        # !!! Here, the line data contain history and prediction(output), 
        # !!! We should only use the history part(from <|history|> to <|endofhistory|>), and feed it into Transformer.
        # !!! <|history|> 0 <|time0|> 1 108 8 216 <|time1|> 670 608 815 <|time2|> ... <|time7|> <|endofhistory|>
        # pair of sequences in train data, <index, index>
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            # convert to list of list of int
            lines = [list(map(int, line.split())) for line in lines]
        
        examples = tokenizer.batch_encode_plus(train_lines, add_special_tokens=True, max_length=block_size,truncation='longest_first')["input_ids"]
        
        # !!! the length of examples should be co

        # get the sequence from train data and index pair
        self.anchor, self.positive, self.negative = [], [], []
        self.anchor_idx, self.positive_idx, self.negative_idx = [], [], []
        for line in lines:
            self.anchor.append(examples[line[0]])  # !!! line[0] should be the indice
            self.positive.append(examples[line[1]])  # !!! line[1] should be the indice
            self.negative.append(examples[line[2]])
            self.anchor_idx.append([line[0]])
            self.positive_idx.append([line[1]])
            self.negative_idx.append([line[2]])
        # print("Anchor: ", self.anchor[0])
        # print("Positive: ", self.positive[0])

    def __len__(self):
        return len(self.anchor)
    
    def __getitem__(self, i):
        return torch.tensor(self.anchor[i], dtype=torch.long), torch.tensor(self.positive[i], dtype=torch.long), torch.tensor(self.negative[i], dtype=torch.long), torch.tensor(self.anchor_idx[i], dtype=torch.long), torch.tensor(self.positive_idx[i], dtype=torch.long), torch.tensor(self.negative_idx[i], dtype=torch.long)
    
def load_and_cache_examples(args, tokenizer, evaluate=False, test=False, pair=True):
    if evaluate:
        file_path = args.eval_data_file 
    elif test:
        file_path = args.test_data_file
    elif pair:
        file_path = args.train_pair_data_file
    else:
        file_path = args.train_data_file
    if not evaluate and pair and not test:
            return PairSequenceDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)       


def get_dataloader(dataset, tokenizer, args, split='train'):
    def collate(examples):
        if split == 'train':
            anchor, positive, negative = [], [], []
            anchor_idx, positive_idx, negative_idx = [], [], []
            for example in examples:
                anchor.append(example[0])
                positive.append(example[1])
                negative.append(example[2])
                anchor_idx.append(example[3])
                positive_idx.append(example[4])
                negative_idx.append(example[5])
            if tokenizer._pad_token is None:
                return pad_sequence(anchor, batch_first=True), \
                        pad_sequence(positive, batch_first=True), \
                        pad_sequence(negative, batch_first=True), \
                        pad_sequence(anchor_idx, batch_first=True), \
                        pad_sequence(positive_idx, batch_first=True), \
                        pad_sequence(negative_idx, batch_first=True)
            return pad_sequence(anchor, batch_first=True, padding_value=tokenizer.pad_token_id), \
                    pad_sequence(positive, batch_first=True, padding_value=tokenizer.pad_token_id), \
                    pad_sequence(negative, batch_first=True, padding_value=tokenizer.pad_token_id),\
                    pad_sequence(anchor_idx, batch_first=True, padding_value=tokenizer.pad_token_id), \
                    pad_sequence(positive_idx, batch_first=True, padding_value=tokenizer.pad_token_id), \
                    pad_sequence(negative_idx, batch_first=True, padding_value=tokenizer.pad_token_id)
        else:
            if tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    if split == 'train':
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        batch_size = args.train_batch_size
        sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    else:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        batch_size = args.eval_batch_size
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate,drop_last=False)

    return dataloader, args
