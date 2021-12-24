import torch
import torch.nn as nn
from dadnet.hooks.model_hook import ModelHook
from dadnet.modules.flatten import Flatten
from dadnet.modules.fake_linear_layer import FakeLinear


class SimpleLSTMEncoder(nn.Module):
    """
    model for timeseries classification
    """

    def __init__(
        self, input_shape, n_classes, out_size=128, num_layers=1, hidden_size=1024
    ):
        super(SimpleLSTMEncoder, self).__init__()
        batch, seq_len, input_size = input_shape
        test_input = torch.zeros(1, seq_len, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.flatten = Flatten(1)
        test_embed, _ = self.lstm(test_input)
        test_embed = self.flatten(test_embed)
        self.fc1 = nn.Linear(test_embed.shape[-1], out_size, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = FakeLinear(out_size, out_size, bias=False)
        self.hook = ModelHook(
            self,
            verbose=False,
            layer_names=["Linear", "ReLU", "LSTM", "FakeLinear"],
            register_self=True,
        )

    def set_delta(self, delta, indices):
        self.hook.backward_return = delta
        self.hook.batch_indices = indices

    def forward(self, batch_input):
        out, _ = self.lstm(batch_input)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)  # Extract outout of lasttime step
        return out


class SimpleLSTMDecoder(nn.Module):
    """
    model for timeseries classification
    """

    def __init__(
        self, input_shape, n_classes, out_size=128, num_layers=1, hidden_size=32
    ):
        super(SimpleLSTMDecoder, self).__init__()
        batch, seq_len, input_size = input_shape
        self.fc3 = nn.Linear(out_size, n_classes, bias=False)
        self.hook = ModelHook(
            self,
            verbose=False,
            layer_names=["Linear", "ReLU", "LSTM"],
            register_self=True,
        )

    def forward(self, out):
        out = self.fc3(out)
        return out


class SimpleLSTMEncoderDecoder(nn.Module):
    pass
