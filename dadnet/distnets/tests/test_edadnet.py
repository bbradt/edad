import torch
import torch.nn as nn

from dadnet.hooks.model_hook import ModelHook
from dadnet.modules.flatten import Flatten
from dadnet.distnets.edadnet import EdadNet
from dadnet.distnets.dadnet import DadNet
from dadnet.distnets.dsgdnet import DsgdNet
from dadnet.distnets.distnet import DistNet


class SimpleLinear(nn.Module):
    def __init__(self):
        super(SimpleLinear, self).__init__()
        self.flatten = Flatten(1)
        self.fc1 = nn.Linear(32 * 32, 128, bias=False)
        # self.relu1 = nn.Tanh()
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


if __name__ == "__main__":
    torch.manual_seed(0)
    x1 = torch.randn(8, 32, 32)
    y1 = torch.randint(0, 10, (8,))
    x2 = torch.randn(8, 32, 32)
    y2 = torch.randint(0, 10, (8,))
    xc = torch.cat([x1, x2], 0)
    yc = torch.cat([y1, y2])

    # Pooled
    torch.manual_seed(0)
    model1_pooled = SimpleLinear()
    torch.manual_seed(0)
    yhats = model1_pooled(xc)
    pooled_optim_1 = torch.optim.Adam(model1_pooled.parameters())
    loss_pooled = nn.CrossEntropyLoss()(yhats, yc)
    loss_pooled.backward()

    # Edad
    torch.manual_seed(0)
    model1_dist = SimpleLinear()
    torch.manual_seed(0)
    model2_dist = SimpleLinear()
    dist_net = DistNet(model1_dist, model2_dist)
    yhats = dist_net.forward(x1, x2)
    dist_optim_1 = torch.optim.Adam(model1_dist.parameters())
    dist_optim_2 = torch.optim.Adam(model2_dist.parameters())
    dist_net.backward([y1, y2], yhats, nn.CrossEntropyLoss)
    dist_net.aggregate()
    dist_net.recompute_gradients()

    torch.manual_seed(0)
    model1_dad = SimpleLinear()
    torch.manual_seed(0)
    model2_dad = SimpleLinear()
    dad_net = DadNet(model1_dad, model2_dad)
    dad_optim_1 = torch.optim.Adam(model1_dad.parameters())
    dad_optim_2 = torch.optim.Adam(model2_dad.parameters())
    yhats = dad_net.forward(x1, x2)
    dad_net.backward([y1, y2], yhats, nn.CrossEntropyLoss)
    dad_net.aggregate()
    dad_net.recompute_gradients()

    # Edad
    torch.manual_seed(0)
    model1_edad = SimpleLinear()
    torch.manual_seed(0)
    model2_edad = SimpleLinear()
    edad_optim_1 = torch.optim.Adam(model1_edad.parameters())
    edad_optim_2 = torch.optim.Adam(model2_edad.parameters())
    edad_net = EdadNet(model1_edad, model2_edad)
    yhats = edad_net.forward(x1, x2)
    edad_net.backward([y1, y2], yhats, nn.CrossEntropyLoss)
    edad_net.aggregate()
    edad_net.recompute_gradients()

    # dSGD
    torch.manual_seed(0)
    model1_dsgd = SimpleLinear()
    torch.manual_seed(0)
    model2_dsgd = SimpleLinear()
    dsgd_net = DsgdNet(model1_dsgd, model2_dsgd)
    yhats = dsgd_net.forward(x1, x2)
    dsgd_optim_1 = torch.optim.Adam(model1_dsgd.parameters())
    dsgd_optim_2 = torch.optim.Adam(model2_dsgd.parameters())
    dsgd_net.backward([y1, y2], yhats, nn.CrossEntropyLoss)
    dsgd_net.aggregate()
    dsgd_net.recompute_gradients()

    def pooled_net_grad_diff(n1, n2):
        return torch.norm(
            model1_pooled.fc1.weight.grad
            - (n1.fc1.weight.grad + n2.fc1.weight.grad) / 2
        ) + torch.norm(
            model1_pooled.fc2.weight.grad
            - (n1.fc2.weight.grad + n2.fc2.weight.grad) / 2
        )

    def net_grad_diff(n1, n2):
        return torch.norm(n1.fc1.weight.grad - n2.fc1.weight.grad) + torch.norm(
            n1.fc2.weight.grad - n2.fc2.weight.grad
        )

    def net_weight_diff(n1, n2):
        return torch.norm(n1.fc1.weight - n2.fc1.weight) + torch.norm(
            n1.fc2.weight - n2.fc2.weight
        )

    assert net_grad_diff(model1_dsgd, model1_edad) < 1e-5
    assert net_grad_diff(model1_dsgd, model1_dad) < 1e-5
    assert net_grad_diff(model1_dsgd, model1_dist) > 1e-5
    assert net_grad_diff(model1_dist, model1_edad) > 1e-5
    assert net_grad_diff(model1_dist, model1_dad) > 1e-5

    dist_optim_1.step()
    dist_optim_2.step()
    dsgd_optim_1.step()
    dsgd_optim_2.step()
    edad_optim_1.step()
    edad_optim_2.step()
    dad_optim_1.step()
    dad_optim_2.step()

    assert net_weight_diff(model1_dsgd, model1_edad) < 1e-5
    assert net_weight_diff(model1_dsgd, model1_dad) < 1e-5
    assert net_weight_diff(model1_dsgd, model1_dist) > 1e-5
    assert net_weight_diff(model1_dist, model1_edad) > 1e-5
    assert net_weight_diff(model1_dist, model1_dad) > 1e-5
