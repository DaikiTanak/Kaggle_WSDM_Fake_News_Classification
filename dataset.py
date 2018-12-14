import pandas as pd
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class BERTDataset(Dataset):
    def __init__(self, titles1_en, titles2_en, labels, tokenizer, seq_length=100):

        self.titles1_en = titles1_en
        self.titles2_en = titles2_en
        self.labels = labels
        self.tokenizer = tokenizer
        self.seq_length=seq_length

    def __len__(self):
        return len(self.titles1_en)

    def __getitem__(self, idx):
        seq_length = self.seq_length
        tokenizer = self.tokenizer

        title1_en = self.titles1_en[idx]
        tokens_a = tokenizer.tokenize(title1_en)

        title2_en = self.titles2_en[idx]
        tokens_b = tokenizer.tokenize(title2_en)


        def _truncate_seq_pair(tokens_a, tokens_b, max_length):
            """Truncates a sequence pair in place to the maximum length."""

            # This is a simple heuristic which will always truncate the longer sequence
            # one token at a time. This makes more sense than truncating an equal percent
            # of tokens from each, since if one sequence is very short then each token
            # that's truncated likely contains more information than a longer sequence.
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()


        _truncate_seq_pair(tokens_a, tokens_b, seq_length-3)

        tokens = []
        input_type_ids = []

        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            input_type_ids.append(1)
        tokens.append("[SEP]")
        input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero padding.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        input_type_ids = torch.tensor(input_type_ids)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)

        sample = {'input_ids': input_ids, 'input_mask': input_mask,
                    'input_type_ids':input_type_ids, 'label': labels}

        return sample

# Dataset
class TitleDataset(Dataset):

    def __init__(self, titles1_en, titles2_en,
    titles1_zh, titles2_zh, labels, dic_en=None, dic_zh=None,
    transform=None, seq_length_en=50, seq_length_zh=140,
    if_test=False):

        self.titles1_en = titles1_en
        self.titles2_en = titles2_en
        self.titles1_zh = titles1_zh
        self.titles2_zh = titles2_zh

        self.labels = labels
        self.transform = transform
        self.dic_en=dic_en
        self.dic_zh=dic_zh

        self.seq_length_en=seq_length_en
        self.seq_length_zh=seq_length_zh

        self.if_test=if_test

    def __len__(self):
        return len(self.titles1_en)

    def __getitem__(self, idx):
        title1_en = self.titles1_en[idx]
        title2_en = self.titles2_en[idx]
        title1_zh = self.titles1_zh[idx]
        title2_zh = self.titles2_zh[idx]

        if self.if_test:
            # dummy label
            label = title1_en
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)

        sample = {'t1_en': title1_en, 't2_en': title2_en, 't1_zh': title1_zh, 't2_zh': title2_zh, 'label': label}

        if self.transform:
            sample = self.transform(sample, self.dic_en, self.dic_zh, self.seq_length_en, self.seq_length_zh)

        return sample


class Toidx(object):
    def __call__(self, sample, word_to_idx_en, word_to_idx_zh, max_seq_length_en, max_seq_length_zh):

        def prepare_sequence(seq, to_ix, max_seq_length, language="english"):
            seq = str(seq)
            #zero padding and word--->ix in seq.
            if language == "english":
                idxs = [to_ix[w] for w in seq.split()]
            elif language == "chinese":
                idxs = [to_ix[w] for w in seq]


            if len(idxs) > max_seq_length:
                idxs = idxs[:max_seq_length]
            else:
                idxs += [0] * (max_seq_length - len(idxs))
            return torch.tensor(idxs, dtype=torch.long)

        t1_en, t2_en, t1_zh, t2_zh, label = sample['t1_en'], sample['t2_en'], sample['t1_zh'], sample['t2_zh'], sample["label"]
        return {'t1_en': prepare_sequence(t1_en, word_to_idx_en, max_seq_length_en, language="english"),
                    't2_en': prepare_sequence(t2_en, word_to_idx_en, max_seq_length_en,language="english"),
                    't1_zh': prepare_sequence(t1_zh, word_to_idx_zh, max_seq_length_zh,language="chinese"),
                    't2_zh': prepare_sequence(t2_zh, word_to_idx_zh, max_seq_length_zh,language="chinese"),
                    'label': label}
