import numpy as np
import torch
import torch.nn as nn

from dadnet.hooks.model_hook import ModelHook
from dadnet.modules.flatten import Flatten
from dadnet.distnets.edadnet import EdadNet
from dadnet.distnets.dadnet import DadNet
from dadnet.distnets.dsgdnet import DsgdNet
from dadnet.distnets.distnet import DistNet
from dadnet.distnets.jointnet import JointNet

from dadnet.model.feed_forward import (
    SimpleFFEncoder,
    SimpleFFDecoder,
    SimpleFFEncoderDecoder,
)


if __name__ == "__main__":
    torch.manual_seed(0)
    x1 = torch.randn(8, 1, 28, 28)
    y1 = torch.randint(0, 10, (8,))
    x2 = torch.randn(8, 1, 28, 28)
    y2 = torch.randint(0, 10, (8,))
    xc = torch.cat([x1, x2], 0)
    yc = torch.cat([y1, y2])

    # Pooled
    torch.manual_seed(0)
    np.random.seed(0)
    model1_pooled = SimpleFFEncoderDecoder(x1.shape, 10)
    torch.manual_seed(0)
    np.random.seed(0)
    yhats = model1_pooled(xc)
    pooled_optim_1 = torch.optim.Adam(model1_pooled.parameters())
    loss_pooled = nn.CrossEntropyLoss()(yhats, yc)
    loss_pooled.backward()

    # dSGD
    torch.manual_seed(0)
    np.random.seed(0)
    model1_dsgd = SimpleFFEncoderDecoder(x1.shape, 10)
    torch.manual_seed(0)
    np.random.seed(0)
    model2_dsgd = SimpleFFEncoderDecoder(x1.shape, 10)
    dsgd_net = DsgdNet(model1_dsgd, model2_dsgd)
    yhats = dsgd_net.forward(x1, x2)
    dsgd_optim_1 = torch.optim.Adam(model1_dsgd.parameters())
    dsgd_optim_2 = torch.optim.Adam(model2_dsgd.parameters())
    dsgd_net.backward([y1, y2], yhats, nn.CrossEntropyLoss)
    dsgd_net.aggregate()
    dsgd_net.recompute_gradients()

    # Dad
    torch.manual_seed(0)
    np.random.seed(0)
    model1_dad = SimpleFFEncoderDecoder(x1.shape, 10)
    torch.manual_seed(0)
    np.random.seed(0)
    model2_dad = SimpleFFEncoderDecoder(x1.shape, 10)
    dad_optim_1 = torch.optim.Adam(model1_dad.parameters())
    dad_optim_2 = torch.optim.Adam(model2_dad.parameters())
    dad_net_2 = DadNet(model1_dad, model2_dad)

    yhats = dad_net_2.forward(x1, x2)
    dad_net_2.backward([y1, y2], yhats, nn.CrossEntropyLoss)
    dad_net_2.aggregate()
    dad_net_2.recompute_gradients()

    # Eda
    torch.manual_seed(0)
    np.random.seed(0)
    model1_edad = SimpleFFEncoderDecoder(x1.shape, 10)
    torch.manual_seed(0)
    np.random.seed(0)
    model2_edad = SimpleFFEncoderDecoder(x1.shape, 10)
    edad_optim_1 = torch.optim.Adam(model1_edad.parameters())
    edad_optim_2 = torch.optim.Adam(model2_edad.parameters())
    edad_net_2 = EdadNet(model1_edad, model2_edad)

    yhats = edad_net_2.forward(x1, x2)
    edad_net_2.backward([y1, y2], yhats, nn.CrossEntropyLoss)
    edad_net_2.aggregate()
    edad_net_2.recompute_gradients()

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

    print("*****fc3*****")
    print(
        "pooled v dsgd ",
        torch.norm(model1_pooled.fc3.weight.grad - model1_dsgd.fc3.weight.grad).item(),
    )
    print(
        "pooled v dad ",
        torch.norm(model1_pooled.fc3.weight.grad - model1_dad.fc3.weight.grad).item(),
    )
    print(
        "dsgd v dad ",
        torch.norm(model1_dsgd.fc3.weight.grad - model1_dad.fc3.weight.grad).item(),
    )
    print(
        "pooled v edad",
        torch.norm(model1_pooled.fc3.weight.grad - model1_edad.fc3.weight.grad).item(),
    )
    print(
        "dsgd v edad",
        torch.norm(model1_dsgd.fc3.weight.grad - model1_edad.fc3.weight.grad).item(),
    )
    print("*****fc2*****")
    print(
        "pooled v dsgd ",
        torch.norm(model1_pooled.fc2.weight.grad - model1_dsgd.fc2.weight.grad).item(),
    )
    print(
        "pooled v dad ",
        torch.norm(model1_pooled.fc2.weight.grad - model1_dad.fc2.weight.grad).item(),
    )
    print(
        "dsgd v dad ",
        torch.norm(model1_dsgd.fc2.weight.grad - model1_dad.fc2.weight.grad).item(),
    )
    print(
        "pooled v edad F",
        torch.norm(model1_pooled.fc2.weight.grad - model1_edad.fc2.weight.grad).item(),
    )
    print(
        "dsgd v edad F",
        torch.norm(model1_dsgd.fc2.weight.grad - model1_edad.fc2.weight.grad).item(),
    )
    print("*****fc1*****")
    print(
        "pooled v dsgd ",
        torch.norm(model1_pooled.fc1.weight.grad - model1_dsgd.fc1.weight.grad).item(),
    )
    print(
        "pooled v dad ",
        torch.norm(model1_pooled.fc1.weight.grad - model1_dad.fc1.weight.grad).item(),
    )
    print(
        "dsgd v dad ",
        torch.norm(model1_dsgd.fc1.weight.grad - model1_dad.fc1.weight.grad).item(),
    )
    print(
        "pooled v edad F",
        torch.norm(model1_pooled.fc1.weight.grad - model1_edad.fc1.weight.grad).item(),
    )
    print(
        "dsgd v edad F",
        torch.norm(model1_dsgd.fc1.weight.grad - model1_edad.fc1.weight.grad).item(),
    )
    print("**done**")
