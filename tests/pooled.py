import torch
import torch.nn as nn
import copy
import pandas as pd

from dadnet.hooks.model_hook import ModelHook
from dadnet.modules.flatten import Flatten
from dadnet.distnets.distnet import DistNet
from dadnet.datasets.mnist import MnistDataset
from dadnet.datasets.loaders import get_dataloaders
from dadnet.datasets.splitting import random_split, class_split
from dadnet.utils import get_average_model


class SimpleLinear(nn.Module):
    def __init__(self):
        super(SimpleLinear, self).__init__()
        self.flatten = Flatten(1)
        self.fc1 = nn.Linear(28 * 28, 128, bias=False)
        self.relu1 = nn.Tanh()
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
        x = self.relu1(x)
        x = self.fc2(x)
        return x


def run_pool(
    lr=1e-3,
    batch_size=32,
    seed=0,
    epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    batch_size *= 2
    torch.manual_seed(seed)
    train_data = MnistDataset(train=True)
    test_data = MnistDataset(train=False)
    train_1 = list(random_split(train_data, 1))[0]

    # train_1 = train_1.to(device)
    # train_2 = train_2.to(device)
    dataloader_1 = torch.utils.data.DataLoader(
        train_1, shuffle=True, batch_size=batch_size
    )
    # dataloader_2 = prefetch_map(prefetch_to_gpu, dataloader_2, prefetch=128)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=10000, shuffle=False
    )

    # Edad
    torch.manual_seed(seed)
    model1 = SimpleLinear().to(device)
    pool_net = DistNet(model1)
    optimizer_1 = torch.optim.Adam(model1.parameters(), lr=lr)
    rows = []
    for e in range(epochs):
        totals_1 = correct_1 = 0
        for i, batch_1 in enumerate(dataloader_1):
            optimizer_1.zero_grad()
            data_1, label_1 = batch_1
            data_1 = data_1.to(device)
            label_1 = label_1.to(device)
            yhats = pool_net.forward(data_1)
            loss = pool_net.backward([label_1], yhats, nn.CrossEntropyLoss)
            # pool_net.aggregate()
            # pool_net.recompute_gradients()
            loss_1 = loss[0]

            # Training
            yhat_1 = model1(data_1)
            pred_1 = yhat_1.argmax(dim=1, keepdim=True)
            correct_1 += pred_1.eq(label_1.view_as(pred_1)).sum().item()
            totals_1 += len(data_1)
            # pool_net.aggregate()
            # pool_net.recompute_gradients()
            optimizer_1.step()
            pool_net.clear()
        # Testing
        with torch.no_grad():
            for i, (test_data_x, test_data_y) in enumerate(test_dataloader):
                test_data_x = test_data_x.to(device)
                test_data_y = test_data_y.to(device)
                yhat_1 = model1(test_data_x)
                test_loss_1 = nn.CrossEntropyLoss()(yhat_1, test_data_y)
                pred_1 = yhat_1.argmax(dim=1, keepdim=True)
                correct_test_1 = pred_1.eq(test_data_y.view_as(pred_1)).sum().item()
                totals_test = len(test_data)
                avg_model = get_average_model(SimpleLinear().to(device), model1).to(
                    device
                )
                yhat_a = avg_model(test_data_x)
                test_loss_a = nn.CrossEntropyLoss()(yhat_a, test_data_y)
                pred_a = yhat_a.argmax(dim=1, keepdim=True)
                correct_test_a = pred_a.eq(test_data_y.view_as(pred_a)).sum().item()

        print(
            "Epoch {epoch}\tTrain Loss Site 1 {ls1}\tTrain Acc Site 1 {acc1}".format(
                epoch=e, ls1=loss_1.item(), acc1=correct_1 / totals_1,
            )
        )
        print(
            "\tTest Loss Site 1 {tls1}\tTest Acc Site 1 {tacc1}".format(
                tls1=test_loss_1.item(), tacc1=correct_test_1 / totals_test,
            )
        )
        print(
            "\tTest Loss Avg {tlsa} Test Acc Avg {tacca}".format(
                tlsa=test_loss_a.item(), tacca=correct_test_a / totals_test
            )
        )
        row = dict(
            epoch=e,
            mode="pooled",
            ls1=loss_1.item(),
            ls2="NaN",
            acc1=correct_1 / totals_1,
            acc2="NaN",
            tls1=test_loss_1.item(),
            tls2="NaN",
            tacc1=correct_test_1 / totals_test,
            tacc2="NaN",
            tlsa=test_loss_a.item(),
            tacca=correct_test_a / totals_test,
        )
        rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv("results/pooled.csv")
    return rows

    # Edad


if __name__ == "__main__":
    rows = run_pool()

