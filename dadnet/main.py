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
    epochs=50,
    device="cuda" if torch.cuda.is_available() else "cpu",
    mode="noshare",
    n_sites=2,
    split="class",
):
    if mode == "pooled":
        n_sites = 1
        batch_size *= 2
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_data = MnistDataset(train=True)
    test_data = MnistDataset(train=False)
    if split == "class":
        site_datasets = class_split(train_data, n_sites)
    elif split == "even":
        site_datasets = even_split(train_data, n_sites)
    elif split == "uniform":
        site_datasets = random_split(train_data, n_sites, seed=seed)
    site_dataloaders = []
    for site_dataset in site_datasets:
        site_dataloaders.append(
            torch.utils.data.DataLoader(
                site_dataset, shuffle=False, batch_size=batch_size
            )
        )
    # dataloader_1 = prefetch_map(prefetch_to_gpu, dataloader_1, prefetch=128)
    # dataloader_2 = torch.utils.data.DataLoader(
    #    train_2, shuffle=False, batch_size=batch_size
    # )
    # dataloader_2 = prefetch_map(prefetch_to_gpu, dataloader_2, prefetch=128)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=10000, shuffle=False
    )

    # Dsgd
    site_models = []
    site_optimizers = []
    for site in range(n_sites):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SimpleLinear().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        site_models.append(model)
        site_optimizers.append(optimizer)
    dist_net = None
    if mode == "dsgd":
        dist_net = DsgdNet(*site_models)
    elif mode == "edad":
        dist_net = EdadNet(*site_models)

    rows = []
    for e in range(epochs):
        totals = [0 for i in range(n_sites)]
        correct = [0 for i in range(n_sites)]
        correct_test = [0 for i in range(n_sites)]
        # totals_1 = totals_2 = 0

        # correct_edad_1 = correct_edad_2 = 0
        # correct_dsgd_1 = correct_dsgd_2 = 0
        # correct_noshare_1 = correct_noshare_2 = 0
        # correct_pooled_1 = 0
        for i, batches in enumerate(zip(*site_dataloaders)):
            # Zero Grad
            for optimizer in site_optimizers:
                optimizer.zero_grad()
            # optimizer_1_dsgd.zero_grad()
            # optimizer_2_dsgd.zero_grad()
            # optimizer_1_edad.zero_grad()
            # optimizer_2_edad.zero_grad()
            # optimizer_1_noshare.zero_grad()
            # optimizer_2_noshare.zero_grad()
            # optimizer_1_pooled.zero_grad()

            # Data Gathering
            site_data = []
            site_labels = []
            for batch in batches:
                site_data.append(batch[0].to(device))
                site_labels.append(batch[1].to(device))

            # DSGD
            yhats_dist = []
            yhats_loss = []
            if dist_net:
                yhats_dist = dist_net.forward(*site_data)
                yhats_loss = dist_net.backward(
                    site_labels, yhats_dist, nn.CrossEntropyLoss
                )
                dist_net.aggregate()
                dist_net.recompute_gradients()
            else:
                for model, data, label in zip(site_models, site_data, site_labels):
                    yhats = model(data)
                    loss = nn.CrossEntropyLoss()(yhats, label)
                    loss.backward()
                    yhats_loss.append(loss)
                    yhats_dist.append(yhats)
            for optimizer in site_optimizers:
                optimizer.step()
            if dist_net:
                dist_net.clear()

            # Accuracy
            # edad
            for i, (label, yhats) in enumerate(zip(site_labels, yhats_dist)):
                pred = yhats.argmax(dim=1, keepdim=True)
                correct[i] += pred.eq(label.view_as(pred)).sum().item()
                totals[i] += len(label)

        # Testing
        yhats_test_loss = []
        with torch.no_grad():
            for i, (test_data_x, test_data_y) in enumerate(test_dataloader):
                test_data_x = test_data_x.to(device)
                test_data_y = test_data_y.to(device)
                # edad
                for i, model in enumerate(site_models):
                    yhat = model(test_data_x)
                    test_loss = nn.CrossEntropyLoss()(yhat, test_data_y)
                    yhats_test_loss.append(test_loss)
                    pred = yhat.argmax(dim=1, keepdim=True)
                    correct_test[i] = pred.eq(test_data_y.view_as(pred)).sum().item()
                avg_model = get_average_model(
                    SimpleLinear().to(device), *site_models
                ).to(device)
                yhat_a = avg_model(test_data_x)
                test_loss_a = nn.CrossEntropyLoss()(yhat_a, test_data_y)
                pred_a = yhat_a.argmax(dim=1, keepdim=True)
                correct_test_a = pred_a.eq(test_data_y.view_as(pred_a)).sum().item()

                # averages
        totals_test = len(test_data)
        print("*****Epoch {epoch}*****".format(epoch=e))
        # Edad
        print("*%s*" % mode)
        print(
            "\tTrain Losses {losses}\tTrain Acc {acc}".format(
                losses=[l.item() for l in yhats_loss],
                acc=[c / t for c, t in zip(correct, totals)],
            )
        )
        print(
            "\tTest Losses {losses}\tTest Acc {acc}".format(
                losses=[l.item() for l in yhats_test_loss],
                acc=[c / t for c, t in zip(correct, totals)],
            )
        )
        print(
            "\tTest Loss Avg {tlsa} Test Acc Avg {tacca}".format(
                tlsa=test_loss_a.item(), tacca=correct_test_a / totals_test
            )
        )
        row = dict(epoch=e, mode=mode, lr=lr, batch_size=batch_size)
        for site, (train_loss, test_loss, correct_, correct_test_, total) in enumerate(
            zip(yhats_loss, yhats_test_loss, correct, correct_test, totals)
        ):
            row["train_loss_site_%d" % site] = train_loss.item()
            row["test_loss_site_%d" % site] = test_loss.item()
            row["train_acc_site_%d" % site] = correct_ / total
            row["test_acc_site_%d" % site] = correct_test_ / totals_test
        row["test_acc_avg"] = correct_test_a / totals_test

        rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv("results/%s_%dsites_split%s_mnist.csv" % (mode, n_sites, split))
    return rows

    # Edad


if __name__ == "__main__":
    import argparse
    import inspect

    parser = argparse.ArgumentParser("Edad")
    argspec = inspect.getfullargspec(run_all)
    for arg, default in zip(argspec.args, argspec.defaults):
        parser.add_argument(
            "--%s" % arg.replace("_", "-"), default=default, type=type(default)
        )
    args = parser.parse_args()

    rows = run_all(**args.__dict__)

