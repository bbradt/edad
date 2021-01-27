import numpy as np
import torch.nn as nn
from dadnet.modules.flatten import Flatten
from dadnet.hooks.model_hook import ModelHook


class SimpleFeedForward(nn.Module):
    def __init__(self):
        super(SimpleFeedForward, self).__init__()
        self.flatten = Flatten(1)
        self.fc1 = nn.Linear(32 * 32, 128, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10, bias=False)
        self.hook = ModelHook(
            self,
            verbose=False,
            layer_names=["Linear", "ReLU", "Sigmoid", "Tanh"],
            register_self=True,
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        # x = self.relu1(x)
        x = self.fc2(x)
        return x


class FeedForward(nn.Module):
    def __init__(
        self, input_shape, n_classes, hidden_dims=[], activation_function=nn.ReLU
    ):
        super(FeedForward, self).__init__()
        self.hook = ModelHook(
            self,
            verbose=False,
            layer_names=["Linear", "ReLU", "Tanh", "Sigmoid"],
            register_self=True,
        )
        self.layer_names = []
        self.flatten = None
        if len(input_shape) > 2:
            self.flatten = Flatten(1)
            input_shape = (input_shape[0], np.prod(input_shape[1:]))
            self.layer_names.append("flatten")
        batch, input_features = input_shape
        hidden_dims.append(n_classes)

        for i, output_features in enumerate(hidden_dims):
            setattr(self, "fc%d" % i, nn.Linear(input_features, output_features))
            self.layer_names.append("fc%d" % i)
            if activation_function and i != len(hidden_dims) - 1:
                setattr(self, "act%d" % i, activation_function())
                self.layer_names.append("act%d" % i)
            input_features = output_features

    def forward(self, x):
        for layer_name in self.layer_names:
            x = getattr(self, layer_name)(x)
        return x


class LogisticRegression(FeedForward):
    def __init__(self, input_shape, n_classes):
        super(LogisticRegression, self).__init__(
            input_shape, n_classes, hidden_dims=[], activation_function=None
        )
