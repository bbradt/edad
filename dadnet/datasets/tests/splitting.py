import torch
import numpy as np
from dadnet.datasets.splitting import random_split, class_split
from dadnet.datasets.numpy import NumpyDataset


def _test_random_split(N=1024, M=(32, 32), L=10, S=2):
    x = torch.randn((N, *M))
    y = torch.randint(0, L, (N,))
    dataset = NumpyDataset(x, y)
    datasets = random_split(dataset, S)
    for datasubset in datasets:
        assert len(datasubset) < len(dataset)
        assert len(datasubset) <= len(dataset) / S


def _test_class_split(N=1024, M=(32, 32), L=10, S=2):
    x = torch.randn((N, *M))
    y = torch.randint(0, L, (N,))
    dataset = NumpyDataset(x, y)
    datasets = list(class_split(dataset, S))
    for i, datasubset in enumerate(datasets):
        print("%d/%d %s" % (i, len(datasets), np.unique(datasubset.y)))


if __name__ == "__main__":
    for s in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        _test_random_split(N=1024, M=(32, 32), L=10, S=s)
    for s in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        _test_class_split(N=1024, M=(32, 32), L=10, S=s)
