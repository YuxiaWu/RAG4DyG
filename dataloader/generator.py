import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer
import os
import logging
logger = logging.getLogger(__name__)


class TextIndexScoreDataset(Dataset):
    def __init__(self, 
                 tokenizer, 
                 args, 
                 text_file_path,
                 index_file_path,
                 score_file_path, 
                 block_size=512):
        assert os.path.isfile(text_file_path)
        assert os.path.isfile(index_file_path)
        assert os.path.isfile(score_file_path)

        with open(args.train_data_file, encoding="utf-8") as f:
            train_data = [line for line in f.read().splitlines() \
                          if (len(line) > 0 and not line.isspace())]

        with open(text_file_path, encoding="utf-8") as f:
            text_lines = [line for line in f.read().splitlines() \
                          if (len(line) > 0 and not line.isspace())]

        with open(index_file_path, encoding="utf-8") as f:
            index_lines = [line for line in f.read().splitlines() \
                          if (len(line) > 0 and not line.isspace())]
        with open(score_file_path, encoding="utf-8") as f:
            score_lines = [line for line in f.read().splitlines() \
                          if (len(line) > 0 and not line.isspace())]
        # store egoId in a dictionary
        self.egolist = []
        self.egoId = {} # key: egoId, value: index of the line
        for i, line in enumerate(text_lines):
            egoId = int(line.split('<|history|>')[1].split(' ')[1])
            self.egoId[egoId] = i
            self.egolist.append(egoId)
        # convert each line index to int, each line contain list of index
        index_lines = [list(map(int, line.split())) for line in index_lines]
        # convert each line score to float
        score_lines = [list(map(float, line.split())) for line in score_lines]

        self.text = tokenizer.batch_encode_plus(text_lines, 
                                                add_special_tokens=True, 
                                                max_length=block_size,truncation='longest_first')["input_ids"]
        self.retrieval_sources = tokenizer.batch_encode_plus(train_data, 
                                                add_special_tokens=True, 
                                                max_length=block_size,truncation='longest_first')["input_ids"]
        self.index = index_lines
        self.score = score_lines
        # print("Len of egoId: ", len(self.egoId))
    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        # print("sample: ", i)
        if i >= len(self.text):
            print("Index out of range for text: ", i)
            return None
        return torch.tensor(self.text[i], dtype=torch.long), \
                torch.tensor(self.index[i], dtype=torch.long), \
                torch.tensor(self.score[i], dtype=torch.float), \
                torch.tensor(self.egolist[i], dtype=torch.long)
    def __gethistory__(self, i):
        return torch.tensor(self.text[i], dtype=torch.long)
    
    def get_item_by_egoId(self, egoId):
        # print("EgoId: ", egoId)
        if egoId not in self.egoId:
            return None
        if self.egoId[egoId] >= len(self.text):
            return None
        return self.__gethistory__(self.egoId[egoId])
        

def load_and_cache_examples(args, tokenizer, evaluate=False, test=False):
    if evaluate:
        text_file_path = args.eval_data_file
        index_file_path = args.val_index_file 
        score_file_path = args.val_score_file
    elif test:
        text_file_path = args.test_data_file
        index_file_path = args.test_index_file
        score_file_path = args.test_score_file
    else: 
        text_file_path = args.train_data_file
        index_file_path = args.train_index_file
        score_file_path = args.train_score_file
        
    return TextIndexScoreDataset(tokenizer, 
                                 args, 
                                 text_file_path, 
                                 index_file_path,
                                 score_file_path, 
                                 block_size=args.block_size)       


def get_dataloader(dataset, tokenizer, args, split='train'):

    def collate(examples):
        text, index, score, ego = [], [], [], []
        for example in examples:
            text.append(example[0])
            index.append(example[1])
            score.append(example[2])
            ego.append(example[3])
        idx = torch.stack(index, dim=0)
        score = torch.stack(score, dim=0)
        ego = torch.stack(ego, dim=0)
        if tokenizer._pad_token is None:
            return pad_sequence(text, batch_first=True), idx, score, ego
        return pad_sequence(text, batch_first=True, padding_value=tokenizer.pad_token_id), idx, score, ego

    if split == 'train':
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        batch_size = args.train_batch_size
        sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    else:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        batch_size = args.eval_batch_size
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, 
                            sampler=sampler, 
                            batch_size=batch_size, 
                            collate_fn=collate,
                            drop_last=True)

    return dataloader, args
