import csv
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import random_split

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

def read_csv(path):
    data = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            description = row[1]
            diagmosis = row[2]
            data.append((description, diagmosis))
            for w in description.split(' '):
                w = int(w)
    return data

def generate_tokens(data):
    src_tokens = [row[0].strip().split(' ') for row in data]
    tgt_tokens = [row[1].strip().split(' ') for row in data]
    return src_tokens, tgt_tokens


def generate_vocab(data, val_size=None):
    train_src, train_tgt = generate_tokens(data)

    # 保存原始的未经vocab的tgt
    val_tgt_ = None

    if val_size != None:
        num_val = int(len(train_src) * val_size)
        num_train = int(len(train_src) * (1-val_size))

        train_src, val_src = random_split(train_src, [num_train, num_val], generator=torch.Generator().manual_seed(42))
        train_tgt, val_tgt = random_split(train_tgt, [num_train, num_val], generator=torch.Generator().manual_seed(42))
        val_tgt_ = val_tgt

    vocab = build_vocab_from_iterator(train_src + train_tgt, min_freq=2, specials=special_symbols)
    vocab.set_default_index(UNK_IDX)

    train_src = [vocab(x) for x in train_src]
    train_tgt = [vocab(x) for x in train_tgt]

    if val_size != None:
        val_src = [vocab(x) for x in val_src]
        val_tgt = [vocab(x) for x in val_tgt]
        return train_src, train_tgt, vocab, val_src, val_tgt, val_tgt_
    else:
        return train_src, train_tgt, vocab


def add_bos_eos(tokens):
    ret = []
    for x in tokens:
        ret.append(torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(x),
                      torch.tensor([EOS_IDX]))))
    return ret

def collate_fn(batch):
    src_batch, tgt_batch = [x[0] for x in batch], [x[1] for x in batch]
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

def get_dataloader(path, batch_size, val_size=None):
    data = read_csv(path)

    if val_size != None:
        train_src, train_tgt, vocab, val_src, val_tgt, val_tgt_ = generate_vocab(data, val_size=val_size)
        val_src = add_bos_eos(val_src)
        val_tgt = add_bos_eos(val_tgt)
        val_iter = [(x, y) for x, y in zip(val_src, val_tgt)] 
        val_dataloader = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn)
    else:
        train_src, train_tgt, vocab = generate_vocab(data)

    train_src = add_bos_eos(train_src)
    train_tgt = add_bos_eos(train_tgt)
    train_iter = [(x, y) for x, y in zip(train_src, train_tgt)] 
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)
    
    if val_size != None:
        return train_dataloader, val_dataloader, vocab, val_src, val_tgt_
    
    return train_dataloader, vocab


if __name__ == '__main__':

    train_dataloader, val_dataloader = get_dataloader("data/train.csv", val_size=0.2)
