import torch
import torch.nn as nn
import copy
import pandas as pd
import numpy as np

from dadnet.hooks.model_hook import ModelHook
from dadnet.modules.flatten import Flatten
from dadnet.distnets.edadnet import EdadNet
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


def run_all(
    lr=1e-5,
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
        train_2, shuffle=False, batch_size=batch_size
    )
    # dataloader_2 = prefetch_map(prefetch_to_gpu, dataloader_2, prefetch=128)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=10000, shuffle=False
    )

    # Dsgd
    torch.manual_seed(seed)
    np.random.seed(seed)
    model1_dsgd = SimpleLinear().to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model2_dsgd = SimpleLinear().to(device)
    dsgd_net = DsgdNet(model1_dsgd, model2_dsgd)
    optimizer_1_dsgd = torch.optim.Adam(model1_dsgd.parameters(), lr=lr)
    optimizer_2_dsgd = torch.optim.Adam(model2_dsgd.parameters(), lr=lr)

    # Edad
    torch.manual_seed(seed)
    np.random.seed(seed)
    model1_edad = SimpleLinear().to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model2_edad = SimpleLinear().to(device)
    edad_net = EdadNet(model1_edad, model2_edad)
    optimizer_1_edad = torch.optim.Adam(model1_edad.parameters(), lr=lr)
    optimizer_2_edad = torch.optim.Adam(model2_edad.parameters(), lr=lr)

    # Pooled
    torch.manual_seed(seed)
    np.random.seed(seed)
    model1_pooled = SimpleLinear().to(device)
    optimizer_1_pooled = torch.optim.Adam(model1_pooled.parameters(), lr=lr)

    # Noshare
    torch.manual_seed(seed)
    np.random.seed(seed)
    model1_noshare = SimpleLinear().to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model2_noshare = SimpleLinear().to(device)
    optimizer_1_noshare = torch.optim.Adam(model1_noshare.parameters(), lr=lr)
    optimizer_2_noshare = torch.optim.Adam(model2_noshare.parameters(), lr=lr)

    rows = []
    for e in range(epochs):
        totals_1 = totals_2 = 0
        correct_edad_1 = correct_edad_2 = 0
        correct_dsgd_1 = correct_dsgd_2 = 0
        correct_noshare_1 = correct_noshare_2 = 0
        correct_pooled_1 = 0
        for i, (batch_1, batch_2) in enumerate(zip(dataloader_1, dataloader_2)):
            # Zero Grad
            optimizer_1_dsgd.zero_grad()
            optimizer_2_dsgd.zero_grad()
            optimizer_1_edad.zero_grad()
            optimizer_2_edad.zero_grad()
            optimizer_1_noshare.zero_grad()
            optimizer_2_noshare.zero_grad()
            optimizer_1_pooled.zero_grad()

            # Data Gathering
            data_1, label_1 = batch_1
            data_2, label_2 = batch_2
            data_1 = data_1.to(device)
            data_2 = data_2.to(device)
            label_1 = label_1.to(device)
            label_2 = label_2.to(device)
            data_c = torch.cat([data_1, data_2], 0)
            label_c = torch.cat([label_1, label_2])

            # DSGD
            yhats_dsgd = dsgd_net.forward(data_1, data_2)
            yhats_dsgd_1, yhats_dsgd_2 = yhats_dsgd
            loss_dsgd = dsgd_net.backward(
                [label_1, label_2], yhats_dsgd, nn.CrossEntropyLoss
            )
            loss_dsgd_1, loss_dsgd_2 = loss_dsgd
            dsgd_net.aggregate()
            dsgd_net.recompute_gradients()

            # EDAD
            yhats_edad = edad_net.forward(data_1, data_2)
            yhats_edad_1, yhats_edad_2 = yhats_edad
            loss_edad = edad_net.backward(
                [label_1, label_2], yhats_edad, nn.CrossEntropyLoss
            )
            loss_edad_1, loss_edad_2 = loss_edad
            edad_net.aggregate()
            edad_net.recompute_gradients()

            # Noshare
            yhats_noshare_1 = model1_noshare(data_1)
            yhats_noshare_2 = model2_noshare(data_2)
            loss_noshare_1 = nn.CrossEntropyLoss()(yhats_noshare_1, label_1)
            loss_noshare_2 = nn.CrossEntropyLoss()(yhats_noshare_2, label_2)
            loss_noshare_1.backward()
            loss_noshare_2.backward()

            # Pooled
            yhats_pooled_1 = model1_pooled(data_c)
            loss_pooled_1 = nn.CrossEntropyLoss()(yhats_pooled_1, label_c)
            loss_pooled_1.backward()

            # Step
            optimizer_1_dsgd.step()
            optimizer_2_dsgd.step()
            optimizer_1_edad.step()
            optimizer_2_edad.step()
            optimizer_1_noshare.step()
            optimizer_2_noshare.step()
            optimizer_1_pooled.step()

            # Clear
            edad_net.clear()
            dsgd_net.clear()

            # Accuracy
            # edad
            pred_edad_1 = yhats_edad_1.argmax(dim=1, keepdim=True)
            correct_edad_1 += pred_edad_1.eq(label_1.view_as(pred_edad_1)).sum().item()
            pred_edad_2 = yhats_edad_2.argmax(dim=1, keepdim=True)
            correct_edad_2 += pred_edad_2.eq(label_2.view_as(pred_edad_2)).sum().item()
            # dsgd
            pred_dsgd_1 = yhats_dsgd_1.argmax(dim=1, keepdim=True)
            correct_dsgd_1 += pred_dsgd_1.eq(label_1.view_as(pred_dsgd_1)).sum().item()
            pred_dsgd_2 = yhats_dsgd_2.argmax(dim=1, keepdim=True)
            correct_dsgd_2 += pred_dsgd_2.eq(label_2.view_as(pred_dsgd_2)).sum().item()
            # noshare
            pred_noshare_1 = yhats_noshare_1.argmax(dim=1, keepdim=True)
            correct_noshare_1 += (
                pred_noshare_1.eq(label_1.view_as(pred_noshare_1)).sum().item()
            )
            pred_noshare_2 = yhats_noshare_2.argmax(dim=1, keepdim=True)
            correct_noshare_2 += (
                pred_noshare_2.eq(label_2.view_as(pred_noshare_2)).sum().item()
            )
            # pooled
            pred_pooled_1 = yhats_pooled_1.argmax(dim=1, keepdim=True)
            correct_pooled_1 += (
                pred_pooled_1.eq(label_c.view_as(pred_pooled_1)).sum().item()
            )

            totals_1 += len(data_1)
            totals_2 += len(data_2)

        # Testing
        with torch.no_grad():
            for i, (test_data_x, test_data_y) in enumerate(test_dataloader):
                test_data_x = test_data_x.to(device)
                test_data_y = test_data_y.to(device)
                # edad
                yhat_edad_1 = model1_edad(test_data_x)
                test_loss_edad_1 = nn.CrossEntropyLoss()(yhat_edad_1, test_data_y)
                pred_edad_1 = yhat_edad_1.argmax(dim=1, keepdim=True)
                correct_test_edad_1 = (
                    pred_edad_1.eq(test_data_y.view_as(pred_edad_1)).sum().item()
                )
                yhat_edad_2 = model2_edad(test_data_x)
                test_loss_edad_2 = nn.CrossEntropyLoss()(yhat_edad_2, test_data_y)
                pred_edad_2 = yhat_edad_2.argmax(dim=1, keepdim=True)
                correct_test_edad_2 = (
                    pred_edad_2.eq(test_data_y.view_as(pred_edad_2)).sum().item()
                )
                avg_model_edad = get_average_model(
                    SimpleLinear().to(device), model1_edad, model2_edad
                ).to(device)
                yhat_a_edad = avg_model_edad(test_data_x)
                test_loss_a_edad = nn.CrossEntropyLoss()(yhat_a_edad, test_data_y)
                pred_a_edad = yhat_a_edad.argmax(dim=1, keepdim=True)
                correct_test_a_edad = (
                    pred_a_edad.eq(test_data_y.view_as(pred_a_edad)).sum().item()
                )
                # dsgd
                yhat_dsgd_1 = model1_dsgd(test_data_x)
                test_loss_dsgd_1 = nn.CrossEntropyLoss()(yhat_dsgd_1, test_data_y)
                pred_dsgd_1 = yhat_dsgd_1.argmax(dim=1, keepdim=True)
                correct_test_dsgd_1 = (
                    pred_dsgd_1.eq(test_data_y.view_as(pred_dsgd_1)).sum().item()
                )
                yhat_dsgd_2 = model2_dsgd(test_data_x)
                test_loss_dsgd_2 = nn.CrossEntropyLoss()(yhat_dsgd_2, test_data_y)
                pred_dsgd_2 = yhat_dsgd_2.argmax(dim=1, keepdim=True)
                correct_test_dsgd_2 = (
                    pred_dsgd_2.eq(test_data_y.view_as(pred_dsgd_2)).sum().item()
                )
                avg_model_dsgd = get_average_model(
                    SimpleLinear().to(device), model1_dsgd, model2_dsgd
                ).to(device)
                yhat_a_dsgd = avg_model_dsgd(test_data_x)
                test_loss_a_dsgd = nn.CrossEntropyLoss()(yhat_a_dsgd, test_data_y)
                pred_a_dsgd = yhat_a_dsgd.argmax(dim=1, keepdim=True)
                correct_test_a_dsgd = (
                    pred_a_dsgd.eq(test_data_y.view_as(pred_a_dsgd)).sum().item()
                )
                # noshare
                yhat_noshare_1 = model1_noshare(test_data_x)
                test_loss_noshare_1 = nn.CrossEntropyLoss()(yhat_noshare_1, test_data_y)
                pred_noshare_1 = yhat_noshare_1.argmax(dim=1, keepdim=True)
                correct_test_noshare_1 = (
                    pred_noshare_1.eq(test_data_y.view_as(pred_noshare_1)).sum().item()
                )
                yhat_noshare_2 = model2_noshare(test_data_x)
                test_loss_noshare_2 = nn.CrossEntropyLoss()(yhat_noshare_2, test_data_y)
                pred_noshare_2 = yhat_noshare_2.argmax(dim=1, keepdim=True)
                correct_test_noshare_2 = (
                    pred_noshare_2.eq(test_data_y.view_as(pred_noshare_2)).sum().item()
                )
                avg_model_noshare = get_average_model(
                    SimpleLinear().to(device), model1_noshare, model2_noshare
                ).to(device)
                yhat_a_noshare = avg_model_noshare(test_data_x)
                test_loss_a_noshare = nn.CrossEntropyLoss()(yhat_a_noshare, test_data_y)
                pred_a_noshare = yhat_a_noshare.argmax(dim=1, keepdim=True)
                correct_test_a_noshare = (
                    pred_a_noshare.eq(test_data_y.view_as(pred_a_noshare)).sum().item()
                )
                # pooled
                yhat_pooled_1 = model1_pooled(test_data_x)
                test_loss_pooled_1 = nn.CrossEntropyLoss()(yhat_pooled_1, test_data_y)
                pred_pooled_1 = yhat_pooled_1.argmax(dim=1, keepdim=True)
                correct_test_pooled_1 = (
                    pred_pooled_1.eq(test_data_y.view_as(pred_pooled_1)).sum().item()
                )
                # averages
                totals_test = len(test_data)
        print("*****Epoch {epoch}*****".format(epoch=e))
        # Edad
        print("*Edad*")
        print(
            "\tTrain Loss Site 1 {ls1}\tTrain Loss Site 2 {ls2}\tTrain Acc Site 1 {acc1}\tTrain Acc Site 2 {acc2}".format(
                ls1=loss_edad_1.item(),
                ls2=loss_edad_2.item(),
                acc1=correct_edad_1 / totals_1,
                acc2=correct_edad_2 / totals_2,
            )
        )
        print(
            "\tTest Loss Site 1 {tls1}\tTest Loss Site 2 {tls2}\tTest Acc Site 1 {tacc1}\tTest Acc Site 2 {tacc2}".format(
                tls1=test_loss_edad_1.item(),
                tls2=test_loss_edad_2.item(),
                tacc1=correct_test_edad_1 / totals_test,
                tacc2=correct_test_edad_2 / totals_test,
            )
        )
        print(
            "\tTest Loss Avg {tlsa} Test Acc Avg {tacca}".format(
                tlsa=test_loss_a_edad.item(), tacca=correct_test_a_edad / totals_test
            )
        )
        row_edad = dict(
            epoch=e,
            mode="edad",
            ls1=loss_edad_1.item(),
            ls2=loss_edad_2.item(),
            acc1=correct_edad_1 / totals_1,
            acc2=correct_edad_2 / totals_2,
            tls1=test_loss_edad_1.item(),
            tls2=test_loss_edad_2.item(),
            tacc1=correct_test_edad_1 / totals_test,
            tacc2=correct_test_edad_2 / totals_test,
            tlsa=test_loss_a_edad.item(),
            tacca=correct_test_a_edad / totals_test,
        )
        rows.append(row_edad)
        # dSGD
        print("*dSGD*")
        print(
            "\tTrain Loss Site 1 {ls1}\tTrain Loss Site 2 {ls2}\tTrain Acc Site 1 {acc1}\tTrain Acc Site 2 {acc2}".format(
                ls1=loss_dsgd_1.item(),
                ls2=loss_dsgd_2.item(),
                acc1=correct_dsgd_1 / totals_1,
                acc2=correct_dsgd_2 / totals_2,
            )
        )
        print(
            "\tTest Loss Site 1 {tls1}\tTest Loss Site 2 {tls2}\tTest Acc Site 1 {tacc1}\tTest Acc Site 2 {tacc2}".format(
                tls1=test_loss_dsgd_1.item(),
                tls2=test_loss_dsgd_2.item(),
                tacc1=correct_test_dsgd_1 / totals_test,
                tacc2=correct_test_dsgd_2 / totals_test,
            )
        )
        print(
            "\tTest Loss Avg {tlsa} Test Acc Avg {tacca}".format(
                tlsa=test_loss_a_dsgd.item(), tacca=correct_test_a_dsgd / totals_test
            )
        )
        row_dsgd = dict(
            epoch=e,
            mode="dsgd",
            ls1=loss_dsgd_1.item(),
            ls2=loss_dsgd_2.item(),
            acc1=correct_dsgd_1 / totals_1,
            acc2=correct_dsgd_2 / totals_2,
            tls1=test_loss_dsgd_1.item(),
            tls2=test_loss_dsgd_2.item(),
            tacc1=correct_test_dsgd_1 / totals_test,
            tacc2=correct_test_dsgd_2 / totals_test,
            tlsa=test_loss_a_dsgd.item(),
            tacca=correct_test_a_dsgd / totals_test,
        )
        rows.append(row_dsgd)
        # Noshare
        print("*Noshare*")
        print(
            "\tTrain Loss Site 1 {ls1}\tTrain Loss Site 2 {ls2}\tTrain Acc Site 1 {acc1}\tTrain Acc Site 2 {acc2}".format(
                ls1=loss_noshare_1.item(),
                ls2=loss_noshare_2.item(),
                acc1=correct_noshare_1 / totals_1,
                acc2=correct_noshare_2 / totals_2,
            )
        )
        print(
            "\tTest Loss Site 1 {tls1}\tTest Loss Site 2 {tls2}\tTest Acc Site 1 {tacc1}\tTest Acc Site 2 {tacc2}".format(
                tls1=test_loss_noshare_1.item(),
                tls2=test_loss_noshare_2.item(),
                tacc1=correct_test_noshare_1 / totals_test,
                tacc2=correct_test_noshare_2 / totals_test,
            )
        )
        print(
            "\tTest Loss Avg {tlsa} Test Acc Avg {tacca}".format(
                tlsa=test_loss_a_noshare.item(),
                tacca=correct_test_a_noshare / totals_test,
            )
        )
        row_noshare = dict(
            epoch=e,
            mode="noshare",
            ls1=loss_noshare_1.item(),
            ls2=loss_noshare_2.item(),
            acc1=correct_noshare_1 / totals_1,
            acc2=correct_noshare_2 / totals_2,
            tls1=test_loss_noshare_1.item(),
            tls2=test_loss_noshare_2.item(),
            tacc1=correct_test_noshare_1 / totals_test,
            tacc2=correct_test_noshare_2 / totals_test,
            tlsa=test_loss_a_noshare.item(),
            tacca=correct_test_a_noshare / totals_test,
        )
        rows.append(row_noshare)
        # Pooled
        print("*Pooled*")
        print(
            "\tTrain Loss Site 1 {ls1}\tTrain Loss Site 2 {ls2}\tTrain Acc Site 1 {acc1}\tTrain Acc Site 2 {acc2}".format(
                ls1=loss_pooled_1.item(),
                ls2=loss_pooled_1.item(),
                acc1=correct_pooled_1 / (totals_1 + totals_2),
                acc2=correct_pooled_1 / (totals_1 + totals_2),
            )
        )
        print(
            "\tTest Loss Site 1 {tls1}\tTest Loss Site 2 {tls2}\tTest Acc Site 1 {tacc1}\tTest Acc Site 2 {tacc2}".format(
                tls1=test_loss_pooled_1.item(),
                tls2=test_loss_pooled_1.item(),
                tacc1=correct_test_pooled_1 / totals_test,
                tacc2=correct_test_pooled_1 / totals_test,
            )
        )
        row_pooled = dict(
            epoch=e,
            mode="pooled",
            ls1=loss_pooled_1.item(),
            ls2=loss_pooled_1.item(),
            acc1=correct_pooled_1 / totals_1,
            acc2=correct_pooled_1 / totals_2,
            tls1=test_loss_pooled_1.item(),
            tls2=test_loss_pooled_1.item(),
            tacc1=correct_test_pooled_1 / totals_test,
            tacc2=correct_test_pooled_1 / totals_test,
            tlsa=loss_pooled_1.item(),
            tacca=correct_test_pooled_1 / totals_test,
        )
        rows.append(row_pooled)
        df = pd.DataFrame(rows)
        df.to_csv("results/all_methods.csv")
    return rows

    # Edad


if __name__ == "__main__":
    rows = run_all()

