import torch
import torch.nn as nn

from dadnet.hooks.model_hook import ModelHook
from dadnet.modules.flatten import Flatten
from dadnet.distnets.distnet import DistNet
from dadnet.datasets.numpy import NumpyDataset
from dadnet.datasets.mnist import MnistDataset
from dadnet.training.trainer import Trainer
from dadnet.training.tester import Tester
from dadnet.datasets.loaders import get_dataloaders
from dadnet.model.feed_forward import FeedForward


def test_mnist():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_train = MnistDataset()
    data_test = MnistDataset(train=False)
    train_loader, test_loader, _ = get_dataloaders(
        train_data=data_train, test_data=data_test
    )

    # Edad
    torch.manual_seed(0)
    model1 = FeedForward(data_train.x.shape, 10, hidden_dims=[512]).to(device)
    trainer = Trainer(
        model1,
        train_loader,
        optimizer_class=torch.optim.Adam,
        loss_class=nn.CrossEntropyLoss,
        lr=1e-3,
        device=device,
    )
    tester = Tester(model1, test_loader, loss_class=nn.CrossEntropyLoss, device=device)
    for i in range(10):
        acc, loss = trainer.run(i)
        print("train", i, acc, loss)
        acc, loss = tester.run(i)
        print("test", i, acc, loss)


if __name__ == "__main__":
    test_mnist()
