import torch
import torch.nn as nn

from dadnet.hooks.model_hook import ModelHook
from dadnet.modules.flatten import Flatten
from dadnet.distnets.dadnet import DadNet
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
    torch.manual_seed(0)
    x1 = torch.randn(8, 32, 32)
    y1 = torch.randint(0, 10, (8,))
    x2 = torch.randn(8, 32, 32)
    y2 = torch.randint(0, 10, (8,))

    # Edad
    torch.manual_seed(0)
    model1 = SimpleLinear()
    torch.manual_seed(0)
    model2 = SimpleLinear()
    dad_net = DadNet(model1, model2)
    yhats_dad = dad_net.forward(x1, x2)
    dad_net.backward([y1, y2], yhats_dad, nn.CrossEntropyLoss)
    dad_net.aggregate()
    dad_net.recompute_gradients()

    # dSGD
    torch.manual_seed(0)
    model1 = SimpleLinear()
    torch.manual_seed(0)
    model2 = SimpleLinear()
    dsgd_net = DsgdNet(model1, model2)
    yhats_dsgd = dsgd_net.forward(x1, x2)
    dsgd_net.backward([y1, y2], yhats_dsgd, nn.CrossEntropyLoss)
    dsgd_net.aggregate()
    dsgd_net.recompute_gradients()
    print()
