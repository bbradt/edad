import numpy as np
import torch
import torch.nn as nn

from dadnet.hooks.model_hook import ModelHook
from dadnet.modules.flatten import Flatten
from dadnet.distnets.tedadnet import TeDadNet
from dadnet.distnets.dadnet import DadNet
from dadnet.distnets.dsgdnet import DsgdNet
from dadnet.distnets.distnet import DistNet
from dadnet.distnets.jointnet import JointNet

from dadnet.model.cnn import (
    SimpleConv2dEncoder,
    SimpleConv2dDecoder,
    SimpleConv2dEncoderDecoder,
)


if __name__ == "__main__":
    torch.manual_seed(0)
    x1 = torch.randn(8, 1, 32, 32)
    y1 = torch.randint(0, 10, (8,))
    x2 = torch.randn(8, 1, 32, 32)
    y2 = torch.randint(0, 10, (8,))
    xc = torch.cat([x1, x2], 0)
    yc = torch.cat([y1, y2])

    # Pooled
    torch.manual_seed(0)
    np.random.seed(0)
    model1_pooled = SimpleConv2dEncoderDecoder()
    torch.manual_seed(0)
    np.random.seed(0)
    yhats = model1_pooled(xc)
    pooled_optim_1 = torch.optim.Adam(model1_pooled.parameters())
    loss_pooled = nn.CrossEntropyLoss()(yhats, yc)
    loss_pooled.backward()

    # dSGD
    torch.manual_seed(0)
    np.random.seed(0)
    model1_dsgd = SimpleConv2dEncoderDecoder()
    torch.manual_seed(0)
    np.random.seed(0)
    model2_dsgd = SimpleConv2dEncoderDecoder()
    dsgd_net = DsgdNet(model1_dsgd, model2_dsgd)
    yhats = dsgd_net.forward(x1, x2)
    dsgd_optim_1 = torch.optim.Adam(model1_dsgd.parameters())
    dsgd_optim_2 = torch.optim.Adam(model2_dsgd.parameters())
    dsgd_net.backward([y1, y2], yhats, nn.CrossEntropyLoss)
    dsgd_net.aggregate()
    dsgd_net.recompute_gradients()

    # Edad
    torch.manual_seed(0)
    np.random.seed(0)
    model1_edad_encoder = SimpleConv2dEncoder()
    model1_edad_decoder = SimpleConv2dDecoder()
    torch.manual_seed(0)
    model2_edad_encoder = SimpleConv2dEncoder()
    model2_edad_decoder = SimpleConv2dDecoder()
    edad_optim_1_e = torch.optim.Adam(model1_edad_encoder.parameters())
    edad_optim_2_e = torch.optim.Adam(model2_edad_encoder.parameters())
    edad_optim_1_d = torch.optim.Adam(model1_edad_decoder.parameters())
    edad_optim_2_d = torch.optim.Adam(model2_edad_decoder.parameters())
    edad_net = TeDadNet(model1_edad_decoder, model2_edad_decoder, shared_layers=[0, 1])
    edad_joint_net = JointNet([model1_edad_encoder, model2_edad_encoder], edad_net)
    yhats = edad_joint_net.forward(x1, x2)
    edad_joint_net.backward([y1, y2], yhats, nn.CrossEntropyLoss)
    # edad_net.aggregate()
    # edad_net.recompute_gradients()

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
