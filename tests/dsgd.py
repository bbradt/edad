import torch
import torch.nn as nn
import copy
import pandas as pd
import numpy as np

from dadnet.hooks.model_hook import ModelHook
from dadnet.modules.flatten import Flatten
from dadnet.distnets.dsgdnet import DsgdNet
from dadnet.distnets.dadnet import DadNet
from dadnet.distnets.dsgdnet import DsgdNet
from dadnet.datasets.mnist import MnistDataset
from dadnet.datasets.loaders import get_dataloaders
from dadnet.datasets.splitting import random_split, class_split, even_split
from dadnet.utils import get_average_model
from dadnet.datasets.prefetch import prefetch_map, prefetch_to_gpu


class SimpleLinear(nn.Module):
    def __init__(self):
        super(SimpleLinear, self).__init__()
        self.flatten = Flatten(1)
        self.fc1 = nn.Linear(28 * 28, 128, bias=False)
        self.relu1 = nn.ReLU()
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


def run_dsgd(
    lr=1e-3,
    batch_size=32,
    seed=0,
    epochs=500,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_data = MnistDataset(train=True)
    test_data = MnistDataset(train=False)
    # train_1, train_2 = random_split(train_data, 2, seed=seed)
    train_1, train_2 = class_split(train_data, 2)
    # train_1 = train_1.to(device)
    # train_2 = train_2.to(device)
    dataloader_1 = torch.utils.data.DataLoader(
        train_1, shuffle=False, batch_size=batch_size
    )
    # dataloader_1 = prefetch_map(prefetch_to_gpu, dataloader_1, prefetch=128)
    dataloader_2 = torch.utils.data.DataLoader(
        train_1, shuffle=False, batch_size=batch_size
    )
    # dataloader_2 = prefetch_map(prefetch_to_gpu, dataloader_2, prefetch=128)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=10000, shuffle=False
    )

    # Dsgd
    torch.manual_seed(seed)
    np.random.seed(seed)
    model1 = SimpleLinear().to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model2 = SimpleLinear().to(device)
    dsgd_net = DsgdNet(model1, model2)
    optimizer_1 = torch.optim.Adam(model1.parameters(), lr=lr)
    optimizer_2 = torch.optim.Adam(model2.parameters(), lr=lr)
    rows = []
    for e in range(epochs):
        totals_1 = totals_2 = correct_1 = correct_2 = 0
        for i, (batch_1, batch_2) in enumerate(zip(dataloader_1, dataloader_2)):
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            data_1, label_1 = batch_1
            data_2, label_2 = batch_2
            data_1 = data_1.to(device)
            data_2 = data_2.to(device)
            label_1 = label_1.to(device)
            label_2 = label_2.to(device)
            yhats = dsgd_net.forward(data_1, data_2)
            loss = dsgd_net.backward([label_1, label_2], yhats, nn.CrossEntropyLoss)
            dsgd_net.aggregate()
            dsgd_net.recompute_gradients()
            loss_1, loss_2 = loss
            optimizer_1.step()
            optimizer_2.step()
            dsgd_net.clear()
            with torch.no_grad():
                yhat_1 = model1(data_1)
                pred_1 = yhat_1.argmax(dim=1, keepdim=True)
                correct_1 += pred_1.eq(label_1.view_as(pred_1)).sum().item()
                totals_1 += len(data_1)
                yhat_2 = model2(data_2)
                pred_2 = yhat_2.argmax(dim=1, keepdim=True)
                correct_2 += pred_2.eq(label_2.view_as(pred_2)).sum().item()
                totals_2 += len(data_2)
            # Training

            # dsgd_net.aggregate()
            # dsgd_net.recompute_gradients()

        # Testing
        with torch.no_grad():
            for i, (test_data_x, test_data_y) in enumerate(test_dataloader):
                test_data_x = test_data_x.to(device)
                test_data_y = test_data_y.to(device)
                yhat_1 = model1(test_data_x)
                test_loss_1 = nn.CrossEntropyLoss()(yhat_1, test_data_y)
                pred_1 = yhat_1.argmax(dim=1, keepdim=True)
                correct_test_1 = pred_1.eq(test_data_y.view_as(pred_1)).sum().item()
                yhat_2 = model2(test_data_x)
                test_loss_2 = nn.CrossEntropyLoss()(yhat_2, test_data_y)
                pred_2 = yhat_2.argmax(dim=1, keepdim=True)
                correct_test_2 = pred_2.eq(test_data_y.view_as(pred_2)).sum().item()
                totals_test = len(test_data)
                avg_model = get_average_model(
                    SimpleLinear().to(device), model1, model2
                ).to(device)
                yhat_a = avg_model(test_data_x)
                test_loss_a = nn.CrossEntropyLoss()(yhat_a, test_data_y)
                pred_a = yhat_a.argmax(dim=1, keepdim=True)
                correct_test_a = pred_a.eq(test_data_y.view_as(pred_a)).sum().item()

        print(
            "Epoch {epoch}\tTrain Loss Site 1 {ls1}\tTrain Loss Site 2 {ls2}\tTrain Acc Site 1 {acc1}\tTrain Acc Site 2 {acc2}".format(
                epoch=e,
                ls1=loss_1.item(),
                ls2=loss_2.item(),
                acc1=correct_1 / totals_1,
                acc2=correct_2 / totals_2,
            )
        )
        print(
            "\tTest Loss Site 1 {tls1}\tTest Loss Site 2 {tls2}\tTest Acc Site 1 {tacc1}\tTest Acc Site 2 {tacc2}".format(
                tls1=test_loss_1.item(),
                tls2=test_loss_2.item(),
                tacc1=correct_test_1 / totals_test,
                tacc2=correct_test_2 / totals_test,
            )
        )
        print(
            "\tTest Loss Avg {tlsa} Test Acc Avg {tacca}".format(
                tlsa=test_loss_a.item(), tacca=correct_test_a / totals_test
            )
        )
        row = dict(
            epoch=e,
            mode="dsgd",
            ls1=loss_1.item(),
            ls2=loss_2.item(),
            acc1=correct_1 / totals_1,
            acc2=correct_2 / totals_2,
            tls1=test_loss_1.item(),
            tls2=test_loss_2.item(),
            tacc1=correct_test_1 / totals_test,
            tacc2=correct_test_2 / totals_test,
            tlsa=test_loss_a.item(),
            tacca=correct_test_a / totals_test,
        )
        rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv("results/dsgd.csv")
    return rows

    # Dsgd


if __name__ == "__main__":
    rows = run_dsgd()

