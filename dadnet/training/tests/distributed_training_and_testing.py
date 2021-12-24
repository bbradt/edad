import torch
import torch.nn as nn

from dadnet.hooks.model_hook import ModelHook
from dadnet.modules.flatten import Flatten
from dadnet.distnets.distnet import DistNet
from dadnet.datasets.numpy import NumpyDataset
from dadnet.datasets.mnist import MnistDataset
from dadnet.training.distributed_trainer import DistributedTrainer
from dadnet.training.distributed_tester import DistributedTester
from dadnet.datasets.loaders import get_dataloaders
from dadnet.model.feed_forward import FeedForward


def test_two_sites():
    torch.manual_seed(0)
    x1 = torch.randn(64, 32, 32)
    y1 = torch.randint(0, 10, (64,))
    x2 = torch.randn(64, 32, 32)
    y2 = torch.randint(0, 10, (64,))
    data_1 = NumpyDataset(x1, y1)
    data_2 = NumpyDataset(x2, y2)
    loader_1, _, _ = get_dataloaders(train_data=data_1)
    loader_2, _, _ = get_dataloaders(train_data=data_2)

    # Edad
    torch.manual_seed(0)
    model1 = FeedForward(x1.shape, 10)
    torch.manual_seed(0)
    model2 = FeedForward(x2.shape, 10)
    dist_net = DistNet(model1, model2)
    trainer = DistributedTrainer(
        dist_net,
        [loader_1, loader_2],
        optimizer_class=torch.optim.SGD,
        loss_class=nn.CrossEntropyLoss,
    )
    trainer.run(0)
    assert (model1.fc0.weight.grad != 0).any()
    assert (model2.fc0.weight.grad != 0).any()


def test_one_site():
    torch.manual_seed(0)
    x1 = torch.randn(64, 32, 32)
    y1 = torch.randint(0, 10, (64,))
    data_1 = NumpyDataset(x1, y1)
    loader_1, _, _ = get_dataloaders(train_data=data_1)

    # Edad
    torch.manual_seed(0)
    model1 = FeedForward(x1.shape, 10)
    dist_net = DistNet(model1)
    trainer = DistributedTrainer(
        dist_net,
        [loader_1],
        optimizer_class=torch.optim.SGD,
        loss_class=nn.CrossEntropyLoss,
    )
    trainer.run(0)
    assert (model1.fc0.weight.grad != 0).any()


def test_one_site_mnist(epochs=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    train_data = MnistDataset()
    test_data = MnistDataset(train=False)
    train_loader, test_loader, _ = get_dataloaders(
        train_data=train_data, test_data=test_data
    )
    torch.manual_seed(0)
    model1 = FeedForward(train_data.x.shape, 10, hidden_dims=[128]).to(device)
    dist_net = DistNet(model1)
    trainer = DistributedTrainer(
        dist_net,
        [train_loader],
        optimizer_class=torch.optim.Adam,
        loss_class=nn.CrossEntropyLoss,
        device=device,
        lr=1e-3,
    )

    tester = DistributedTester(
        [model1], test_loader, loss_class=nn.CrossEntropyLoss, device=device
    )
    for e in range(epochs):
        acc, loss = trainer.run(0)
        print("train ", e, acc, loss)
        acc, loss = tester.run(0)
        print("test ", e, acc, loss)


if __name__ == "__main__":
    test_one_site()
    test_two_sites()
    test_one_site_mnist()
    # d_net.backward([y1, y2], yhats, nn.CrossEntropyLoss)
    # dad_net.aggregate()
    # dad_net.recompute_gradients()
