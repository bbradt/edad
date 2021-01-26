import torch
import torch.nn as nn

from dadnet.hooks.model_hook import ModelHook
from dadnet.modules.flatten import Flatten
from dadnet.distnets.dsgdnet import DsgdNet


class SimpleLinear(nn.Module):
    def __init__(self):
        super(SimpleLinear, self).__init__()
        self.flatten = Flatten(1)
        self.fc1 = nn.Linear(32 * 32, 128, bias=False)
        self.fc2 = nn.Linear(128, 10, bias=False)
        self.hook = ModelHook(
            self, verbose=False, layer_names=["Linear"], register_self=True
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    x1 = torch.randn(64, 32, 32)
    y1 = torch.randint(0, 10, (64,))
    x2 = torch.randn(64, 32, 32)
    y2 = torch.randint(0, 10, (64,))

    model1 = SimpleLinear()
    model2 = SimpleLinear()
    dist_net = DsgdNet(model1, model2)
    yhats = dist_net.forward(x1, x2)
    dist_net.backward([y1, y2], yhats, nn.CrossEntropyLoss)
    dist_net.aggregate()
    dist_net.recompute_gradients()
    print()
