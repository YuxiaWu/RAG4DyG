
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import os
import logging
logger = logging.getLogger(__name__)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512):
        print('file_path', file_path)
        assert os.path.isfile(file_path)

        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size,truncation='longest_first')["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if not evaluate:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)

def get_dataloader(dataset, tokenizer, args, split='train'):

    def collate(examples):
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

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate,drop_last=True)

    return dataloader, args
