import torch
import numpy as np
from dadnet.utils import split_list


def even_split(dataset, n_sites):
    indices = np.arange(len(dataset))
    split_indices = split_list(indices, n_sites)
    for split_index in split_indices:
        yield dataset.get_subset_by_indices(split_index)


def random_split(dataset, n_sites, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split_indices = split_list(indices, n_sites)
    for split_index in split_indices:
        yield dataset.get_subset_by_indices(split_index)


def class_split(dataset, n_sites):
    unique_labels = np.unique(dataset.y)
    split_labels = split_list(unique_labels, n_sites)
    for split_label in split_labels:
        split_index = np.argwhere(np.isin(dataset.y, split_label)).flatten()
        yield dataset.get_subset_by_indices(split_index)


def split_dataset(mode, dataset, n_sites, seed=0):
    if mode.lower() == "class":
        return class_split(dataset, n_sites)
    else:
        return random_split(dataset, n_sites)
