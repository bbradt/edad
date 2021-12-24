import torch
import torch.nn as nn
from dadnet.hooks.model_hook import ModelHook
from dadnet.modules.flatten import Flatten
from dadnet.modules.fake_linear_layer import FakeLinear


CONV1_OUT = 16
CONV1_KWARGS = dict(kernel_size=2, stride=1, padding=1)
MAXPOOL1_KWARGS = dict(kernel_size=2, stride=2)
CONV2_OUT = 32
CONV2_KWARGS = dict(kernel_size=2, stride=1, padding=1)
MAXPOOL2_KWARGS = dict(kernel_size=2, stride=2)
FC1_OUT = 1024
FC2_OUT = 1024


class SimpleConv2dEncoder(nn.Module):
    def __init__(self, input_shape, n_classes):
        batch, chan, xdim, ydim = input_shape
        test_input = torch.zeros((1, chan, xdim, ydim))
        super(SimpleConv2dEncoder, self).__init__()
        self.conv1 = nn.Conv2d(chan, CONV1_OUT, **CONV1_KWARGS)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(**MAXPOOL1_KWARGS)
        self.conv2 = nn.Conv2d(CONV1_OUT, CONV2_OUT, **CONV2_KWARGS)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(**MAXPOOL1_KWARGS)
        # self.dropout = nn.Dropout()
        self.flatten = Flatten(1)
        test_encoding = self.flatten(
            self.maxpool2(self.conv2(self.maxpool1(self.relu1(self.conv1(test_input)))))
        )
        self.fc1 = FakeLinear(test_encoding.shape[-1], FC1_OUT, bias=False)
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
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


class SimpleConv2dDecoder(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(SimpleConv2dDecoder, self).__init__()
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(FC1_OUT, FC2_OUT, bias=False)
        self.relu3 = nn.ReLU()
        self.fc3 = nn.Linear(FC2_OUT, n_classes, bias=False)
        self.hook = ModelHook(
            self,
            verbose=False,
            layer_names=["Linear", "Conv2d", "ReLU", "MaxPool2d", "Dropout"],
            register_self=True,
        )

    def forward(self, x):
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.fc3(x)
        return x


class SimpleConv2dEncoderDecoder(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(SimpleConv2dEncoderDecoder, self).__init__()
        batch, chan, xdim, ydim = input_shape
        test_input = torch.zeros((batch, chan, xdim, ydim))
        self.conv1 = nn.Conv2d(chan, CONV1_OUT, **CONV1_KWARGS)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(**MAXPOOL1_KWARGS)
        self.conv2 = nn.Conv2d(CONV1_OUT, CONV2_OUT, **CONV2_KWARGS)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(**MAXPOOL1_KWARGS)
        # self.dropout = nn.Dropout()
        self.flatten = Flatten(1)
        test_encoding = self.flatten(
            self.maxpool2(self.conv2(self.maxpool1(self.relu1(self.conv1(test_input)))))
        )
        self.fc1 = FakeLinear(test_encoding.shape[-1], FC1_OUT, bias=False)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(FC1_OUT, FC2_OUT, bias=False)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(FC2_OUT, n_classes, bias=False)
        # self.fc2 = nn.Linear(32, 10, bias=False)
        self.hook = ModelHook(
            self,
            verbose=False,
            layer_names=["Linear", "Conv2d", "ReLU", "MaxPool2d", "Dropout"],
            register_self=True,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
