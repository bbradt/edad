import torch
import torch.nn as nn
from dadnet.hooks.model_hook import ModelHook
from dadnet.modules.flatten import Flatten
from dadnet.modules.fake_linear_layer import FakeLinear
import numpy as np


FC1_OUT = 512
FC2_OUT = 256


class SimpleFFEncoder(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(SimpleFFEncoder, self).__init__()
        if len(input_shape) > 2:
            input_shape = (input_shape[0], np.prod(input_shape[1:]))
        self.flatten = Flatten(1)
        self.fc1 = nn.Linear(input_shape[-1], FC1_OUT, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = FakeLinear(FC1_OUT, FC2_OUT, bias=False)
        self.hook = ModelHook(
            self,
            verbose=False,
            layer_names=[
                "Linear",
                "Conv2d",
                "ReLU",
                "MaxPool2d",
                "Dropout",
                "FakeLinear",
            ],
            register_self=True,
        )

    def set_delta(self, delta, indices):
        self.hook.backward_return = delta
        self.hook.batch_indices = indices

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class SimpleFFDecoder(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(SimpleFFDecoder, self).__init__()
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(FC2_OUT, n_classes, bias=False)
        self.hook = ModelHook(
            self,
            verbose=False,
            layer_names=[
                "Linear",
                "Conv2d",
                "ReLU",
                "MaxPool2d",
                "Dropout",
                "FakeLinear",
            ],
            register_self=True,
        )

    def forward(self, x):
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class SimpleFFEncoderDecoder(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(SimpleFFEncoderDecoder, self).__init__()
        if len(input_shape) > 2:
            input_shape = (input_shape[0], np.prod(input_shape[1:]))
        self.flatten = Flatten(1)
        self.fc1 = nn.Linear(input_shape[-1], FC1_OUT, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = FakeLinear(FC1_OUT, FC2_OUT, bias=False)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(FC2_OUT, n_classes, bias=False)
        self.hook = ModelHook(
            self,
            verbose=False,
            layer_names=[
                "Linear",
                "Conv2d",
                "ReLU",
                "MaxPool2d",
                "Dropout",
                "FakeLinear",
            ],
            register_self=True,
        )

    def set_delta(self, delta, indices):
        self.hook.backward_return = delta
        self.hook.batch_indices = indices

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
