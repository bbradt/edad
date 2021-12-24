import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F

from dadnet.modules.flatten import Flatten
from dadnet.modules.fake_linear_layer import FakeLinear
from dadnet.hooks.accumulate_hook import AccumulateHook

GRU_HIDDEN = 64
FC1_OUT = 512
FC2_OUT = 256


class GRUEncoder(nn.Module):
    def __init__(
        self, input_shape, n_classes, hidden_dim=GRU_HIDDEN, layer_dim=1, bias=False
    ):
        super(GRUEncoder, self).__init__()
        batch, seq_len, n_features = input_shape
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.x2h = nn.Linear(n_features, 3 * hidden_dim, bias=bias)
        self.h2h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)

        self.flatten = Flatten(1)

        self.fc1 = FakeLinear(seq_len * hidden_dim, FC1_OUT, bias=bias)
        self.hook = AccumulateHook(
            self,
            verbose=False,
            layer_names=["Linear", "ReLU", "FakeLinear",],
            register_self=True,
        )

    def set_delta(self, delta, indices):
        self.hook.backward_return = delta
        self.hook.batch_indices = indices

    def forward(self, x):
        h0 = Variable(
            torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        )
        outs = []
        hn = h0[0, :, :]
        for seq in range(x.size(1)):
            xt = x[:, seq, :]
            xt = xt.view(-1, xt.size(1))
            gate_x = self.x2h(xt)
            gate_h = self.h2h(hn)

            gate_x = gate_x.squeeze()
            gate_h = gate_h.squeeze()

            i_r, i_i, i_n = gate_x.chunk(3, 1)
            h_r, h_i, h_n = gate_h.chunk(3, 1)

            resetgate = F.sigmoid(i_r + h_r)
            inputgate = F.sigmoid(i_i + h_i)
            newgate = F.tanh(i_n + (resetgate * h_n))

            hn = newgate + inputgate * (hn - newgate)
            outs.append(hn)

        out = torch.stack(outs, 1)
        out = self.flatten(out)
        out = self.fc1(out)
        return out


class GRUDecoder(nn.Module):
    def __init__(
        self, input_shape, n_classes, hidden_dim=GRU_HIDDEN, layer_dim=1, bias=False
    ):
        super(GRUDecoder, self).__init__()
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(FC1_OUT, FC2_OUT, bias=bias)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(FC2_OUT, n_classes, bias=bias)
        self.hook = AccumulateHook(
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
        out = self.relu1(x)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


class GRUEncoderDecoder(nn.Module):
    def __init__(
        self, input_shape, n_classes, hidden_dim=GRU_HIDDEN, layer_dim=1, bias=False
    ):
        super(GRUEncoderDecoder, self).__init__()
        batch, seq_len, n_features = input_shape
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.x2h = nn.Linear(n_features, 3 * hidden_dim, bias=bias)
        self.h2h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)

        self.flatten = Flatten(1)

        self.fc1 = FakeLinear(seq_len * hidden_dim, FC1_OUT, bias=bias)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(FC1_OUT, FC2_OUT, bias=bias)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(FC2_OUT, n_classes, bias=bias)
        self.hook = AccumulateHook(
            self,
            verbose=False,
            layer_names=["Linear", "ReLU", "FakeLinear", "Flatten"],
            register_self=True,
        )

    def forward(self, x):
        h0 = Variable(
            torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        )
        outs = []
        hn = h0[0, :, :]
        for seq in range(x.size(1)):
            xt = x[:, seq, :]
            xt = xt.view(-1, xt.size(1))
            gate_x = self.x2h(xt)
            gate_h = self.h2h(hn)

            gate_x = gate_x.squeeze()
            gate_h = gate_h.squeeze()

            i_r, i_i, i_n = gate_x.chunk(3, 1)
            h_r, h_i, h_n = gate_h.chunk(3, 1)

            resetgate = F.sigmoid(i_r + h_r)
            inputgate = F.sigmoid(i_i + h_i)
            newgate = F.tanh(i_n + (resetgate * h_n))

            hn = newgate + inputgate * (hn - newgate)
            outs.append(hn)

        out = torch.stack(outs, 1)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


if __name__ == "__main__":
    test_input = torch.randn(32, 123, 4)
    test_classes = torch.randint(0, 1, (32,))
    gru = GRUEncoderDecoder(test_input.shape, 2)
    yhat = gru(test_input)
    loss = nn.CrossEntropyLoss()(yhat, test_classes)
    loss.backward()
