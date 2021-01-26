import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self, dim):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.flatten(self.dim)
