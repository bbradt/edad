import os
from torch.utils.data.dataset import Dataset
import numpy as np
import torch


class NumpyDataset(Dataset):
    def __init__(self, x, y, indices=None, classify=True):
        self.x = x
        self.y = y
        self.classify = classify
        if type(self.x) is np.ndarray:
            self.x = torch.from_numpy(self.x)
            self.y = torch.from_numpy(self.y)
        self.x = self.x.float()
        if classify:
            self.y = self.y.flatten().long()
        if indices is not None:
            self.x = x[indices, ...]
            self.y = y[indices, ...]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, k):
        return self.x[k, ...], self.y[k, ...]

    def get_subset_by_condition(self, condition_func, label=False):
        if not label:
            x = self.x[condition_func(self.x), ...]
            y = self.y[condition_func(self.x), ...]
        else:
            x = self.x[condition_func(self.y), ...]
            y = self.y[condition_func(self.y), ...]
        return NumpyDataset(x, y)

    def get_subset_by_indices(self, indices):
        return NumpyDataset(self.x, self.y, indices, classify=self.classify)

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self


class NumpyFileDataset(NumpyDataset):
    def __init__(self, xfile, yfile, indices):
        x = np.load(xfile)
        y = np.load(yfile)
        super(NumpyDataset, self).__init__(x, y)
