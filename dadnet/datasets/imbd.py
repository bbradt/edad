from torch.utils.data.dataset import Dataset
from torchtext import data, datasets, vocab
import torch
import numpy as np
from dadnet.datasets.numpy import NumpyDataset


class ImdbDataset(NumpyDataset):
    def __init__(self, train=False):
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
        LABEL = data.Field(sequential=False)
        self.train, self.test = datasets.IMDB.splits(TEXT, LABEL)
        # self.train, self.test = self.train.split(split_ratio=0.9)
        if train:
            self.dataset = self.train
        else:
            self.dataset = self.test
        TEXT.build_vocab(
            self.dataset, max_size=50_000 - 2
        )  # - 2 to make space for <unk> and <pad>
        LABEL.build_vocab(self.dataset)
        self.x = []
        rows = []
        max_len = 64
        for sample in self.dataset:
            row = []
            for i, token in enumerate(sample.text):
                row.append(TEXT.vocab.stoi[token])
                # if row[-1] == -2:
                #    row[-1] = 0
            while len(row) < max_len:
                row.append(0)
            if len(row) > max_len:
                row = row[:max_len]
            rows.append(np.array(row))
        self.x = np.stack(rows, 0)
        self.y = [int(d.label == "pos") for d in self.dataset]
        super(ImdbDataset, self).__init__(self.x, self.y)
        self.x = self.x.long()

    def shuffle(self):
        pass
