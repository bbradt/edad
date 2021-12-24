from dadnet.datasets.imbd import ImdbDataset
import torch
import torch.nn as nn
import copy
import pandas as pd
import numpy as np
import os
import time
import torchtext

from sklearn.model_selection import KFold
from dadnet.modules.sequential_model import SequentialModel
from dadnet.distnets.edadnet import EdadNet
from dadnet.distnets.tedadnet import TeDadNet
from dadnet.distnets.dadnet import DadNet
from dadnet.distnets.rankdad import RankDadNet
from dadnet.distnets.rankdad2way import RankDad2WayNet
from dadnet.distnets.dsgdnet import DsgdNet
from dadnet.distnets.powersgdnet import PowerSgdNet
from dadnet.datasets.mnist import MnistDataset
from dadnet.datasets.cifar10 import Cifar10Dataset
from dadnet.datasets.tapnet import TapnetDataset
from dadnet.datasets.torchtext import ImdbDataset
from dadnet.datasets.splitting import random_split, class_split, even_split
from dadnet.utils import get_average_model

from dadnet.model.cnn import (
    SimpleConv2dEncoder,
    SimpleConv2dDecoder,
    SimpleConv2dEncoderDecoder,
)

from dadnet.model.feed_forward import (
    SimpleFFEncoder,
    SimpleFFDecoder,
    SimpleFFEncoderDecoder,
)
from dadnet.model.gru import GRUEncoder, GRUDecoder, GRUEncoderDecoder

from dadnet.model.resnet import ResNetEncoder, ResNetDecoder, ResNetEncoderDecoder
from dadnet.model.lstm import SimpleLSTMDecoder, SimpleLSTMEncoder
from dadnet.model.transformer import (
    CTransformerEncoder,
    CTransformerDecoder,
    CTransformerEncoderDecoder,
)
from dadnet.distnets.jointnet import JointNet
from dadnet.distnets.jointnet_sergey import JointNetSergey

# For these modes, all layers in the network _must_ be shared
ALL_SHARED_MODES = [
    "dad",
    "dsgd",
    "edad",
    "rankdad",
    "tedad-all-shared",
    "powersgd",
    "rankdad2way",
]
# These modes use a jointnet construction, i.e. we backflow to the encoder
JOINT_DIST_MODES = ["tedad", "tedad-sergey"]

# Modes for distnets
DSGD_MODES = ["dsgd", "dsgd-untouched-encoder"]
ADAD_MODES = ["tedad", "tedad-all-shared", "tedad-untouched-encoder", "tedad-sergey"]
DAD_MODES = ["dad", "dad-untouched-encoder"]
RANKDAD_MODES = ["rankdad", "rankdad-untouched-encoder", "rankdad-all-shared"]
RANKDAD2_MODES = ["rankdad2way"]
EDAD_MODES = ["edad", "edad-untouched-encoder"]
PSGD_MODES = ["powersgd", "powersgd-untouched-encoder"]
OTHER_MODES = ["pooled", "noshare"]  # these modes do not have distnets

ALL_MODES = [
    "pooled",
    "noshare",
    "dsgd",
    "dad",
    "tedad",
    "edad",
    "rankdad",
    "powersgd",
    "tedad-all-shared",
    "edad-all-shared",
    "tedad-untouched-encoder",
    "dsgd-untouched-encoder",  # sanity check
    "edad-untouched-encoder",  # sanity check - should be equivalent to dsgd untouched encoder
    "dad-untouched-encoder",  # sanity check - should be equivalent ot dsgd untouched encoder
    "powersgd-untouched-encoder",
    "rankdad-untouched-encoder",
]


def run_all(
    name="default",
    lr=0.0001,
    batch_size=4,
    seed=0,
    epochs=50,
    device="cuda" if torch.cuda.is_available() else "cpu",
    mode="pooled",
    n_sites=2,
    # split="class",
    split="random",
    # dataset="mnist",
    # model="simpleff",
    # dataset="cifar10",
    # model="resnet",
    # dataset="tapnet-SpokenArabicDigits",
    # model="simplelstm",
    # dataset="tapnet-BasicMotions",
    dataset="imdb",
    model="transformer",
    # dataset="mnist",
    # model="simpleconv",
    # dataset="mnist",
    # model="simpleff",
    k=-1,
    kf=5,
    dry_run=False,
    rank=8,
    numiterations=10,
    num_heads=4,
    embedding_size=32,
    depth=1,
    vocab_size=50_000,
    max_pool=True,
    max_length=32,
):
    lossFun = nn.CrossEntropyLoss
    if model == "transformer":
        lossFun = nn.NLLLoss

    filename = (
        "results/%s/%s_model=%s_mode=%s_sites=%d_split=%s_data=%s_lr=%s_batch_size=%s_seed=%s_k=%s_kf=%s_rank=%s_numiterations=%s.csv"
        % (
            name,
            name,
            model,
            mode,
            n_sites,
            split,
            dataset,
            lr,
            batch_size,
            seed,
            k,
            kf,
            rank,
            numiterations,
        )
    )
    os.makedirs("results/%s" % name, exist_ok=True)
    rows = []

    # Make effective batch size of pooled match
    if mode == "pooled":
        batch_size *= n_sites
        n_sites = 1

    # Data Loading
    torch.manual_seed(seed)
    np.random.seed(seed)
    val_data = None
    start = time.time()
    if dataset == "mnist":
        train_data = MnistDataset(train=True)
        test_data = MnistDataset(train=False)
        train_data.add_singleton()
        test_data.add_singleton()
    elif dataset == "cifar10":
        train_data = Cifar10Dataset(train=True)
        test_data = Cifar10Dataset(train=False)
        train_data.permute()
        test_data.permute()
    elif "tapnet" in dataset:
        tapnet_name = dataset.replace("tapnet-", "")
        train_data = TapnetDataset(tapnet_name, train=True)
        test_data = TapnetDataset(tapnet_name, train=False)
    elif "imdb" in dataset:
        train_data = ImdbDataset(train=True)
        test_data = ImdbDataset(train=False)
    print("Building dataset took %f " % (time.time() - start))
    torch.manual_seed(seed)
    np.random.seed(seed)
    # train_data.shuffle()

    # Model Args
    start = time.time()
    if model == "transformer":
        n_classes = 2
        model_args = (
            embedding_size,
            num_heads,
            depth,
            max_length,
            vocab_size,
            2,
            max_pool,
        )
        # Cross-Validation
        if k >= 0:
            train_ind = np.arange(0, len(train_data))
            kf_obj = KFold(n_splits=kf, shuffle=False, random_state=seed)
            train_ind, val_ind = list(kf_obj.split(train_ind))[k]
            val_data = train_data.get_subset_by_indices(val_ind)
            train_data = train_data.get_subset_by_indices(train_ind)
    else:
        n_classes = len(np.unique(train_data.y))
        input_shape = train_data.x.shape
        model_args = (input_shape, n_classes)

        # Cross-Validation
        if k >= 0:
            train_ind = np.arange(0, train_data.x.shape[0])
            kf_obj = KFold(n_splits=kf, shuffle=False, random_state=seed)
            train_ind, val_ind = list(kf_obj.split(train_ind))[k]
            val_data = train_data.get_subset_by_indices(val_ind)
            train_data = train_data.get_subset_by_indices(train_ind)
    # End Data Loading

    # Split Data
    if split == "class":
        site_datasets = class_split(train_data, n_sites)
    elif split == "even":
        site_datasets = even_split(train_data, n_sites)
    elif split == "uniform":
        site_datasets = random_split(train_data, n_sites, seed=seed)
    else:
        site_datasets = random_split(train_data, n_sites, seed=seed)
    # End Split Data
    print("Splitting dataset took %f " % (time.time() - start))

    # Data Loaders
    site_dataloaders = []
    if model == "transformer":
        for site_dataset in site_datasets:
            train_iter = torchtext.data.BucketIterator(
                site_dataset.train, batch_size, device=device
            )
            site_dataloaders.append(train_iter)
        test_dataloader = torchtext.data.BucketIterator(
            test_data.test, batch_size, device=device
        )
    else:
        for site_dataset in site_datasets:
            site_dataloaders.append(
                torch.utils.data.DataLoader(
                    site_dataset, shuffle=False, batch_size=batch_size, drop_last=True
                )
            )
        test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False
        )
        val_dataloader = None
        if val_data is not None:
            val_dataloader = torch.utils.data.DataLoader(
                val_data, batch_size=batch_size, shuffle=False
            )
    # End Data Loaders

    # Set Encoder/Decoder/E2E classes
    if model == "resnet":
        encoder_class = ResNetEncoder
        decoder_class = ResNetDecoder
        e2e_class = ResNetEncoderDecoder
    elif model == "simpleconv":
        encoder_class = SimpleConv2dEncoder
        decoder_class = SimpleConv2dDecoder
        e2e_class = SimpleConv2dEncoderDecoder
    elif model == "simpleff":
        encoder_class = SimpleFFEncoder
        decoder_class = SimpleFFDecoder
        e2e_class = SimpleFFEncoderDecoder
    elif model == "simplelstm":
        encoder_class = SimpleLSTMEncoder
        decoder_class = SimpleLSTMDecoder
    elif model == "gru":
        encoder_class = GRUEncoder
        decoder_class = GRUDecoder
        e2e_class = GRUEncoderDecoder
    elif model == "transformer":
        encoder_class = CTransformerEncoder
        decoder_class = CTransformerDecoder
        e2e_class = CTransformerEncoderDecoder
    if mode in ALL_SHARED_MODES:
        decoder_class = e2e_class
    # End Model Setup

    # Setup Models
    site_encoders = []
    site_decoders = []
    site_encoder_optimizers = []
    site_decoder_optimizers = []
    site_encoder_schedulers = []
    site_decoder_schedulers = []
    for site in range(n_sites):
        encoder_model = None
        encoder_optimizer = None

        if mode not in ALL_SHARED_MODES:
            torch.manual_seed(seed)
            np.random.seed(seed)
            encoder_model = encoder_class(*model_args).to(device)
            encoder_optimizer = torch.optim.Adam(encoder_model.parameters(), lr=lr)
        torch.manual_seed(seed)
        np.random.seed(seed)
        decoder_model = decoder_class(*model_args).to(device)
        decoder_optimizer = torch.optim.Adam(decoder_model.parameters(), lr=lr)

        site_encoders.append(encoder_model)
        site_encoder_optimizers.append(encoder_optimizer)
        site_decoders.append(decoder_model)
        site_decoder_optimizers.append(decoder_optimizer)
        encoder_sch = None
        if encoder_optimizer is not None:
            encoder_sch = torch.optim.lr_scheduler.LambdaLR(
                encoder_optimizer, lambda i: min(i / (10000 / batch_size), 1.0)
            )
        decoder_sch = torch.optim.lr_scheduler.LambdaLR(
            decoder_optimizer, lambda i: min(i / (10000 / batch_size), 1.0)
        )
        site_encoder_schedulers.append(encoder_sch)
        site_decoder_schedulers.append(decoder_sch)
    # End Setup models

    # Dist Net
    dist_net = None
    dist_net_encoder = None

    if mode in DSGD_MODES:
        dist_net = DsgdNet(*site_decoders)
    elif mode in DAD_MODES:
        dist_net = DadNet(*site_decoders)
    elif mode in ADAD_MODES:
        dist_net = TeDadNet(*site_decoders)
    elif mode in EDAD_MODES:
        dist_net = EdadNet(*site_decoders)
    elif mode in RANKDAD_MODES:
        dist_net = RankDadNet(*site_decoders, rank=rank, numiterations=numiterations)
    elif mode in RANKDAD2_MODES:
        dist_net = RankDad2WayNet(
            *site_decoders, rank=rank, numiterations=numiterations
        )
    elif mode in PSGD_MODES:
        dist_net = PowerSgdNet(*site_decoders, rank=rank)
    # End Dist Net

    # Joint Net
    joint_net = None
    if mode in JOINT_DIST_MODES:
        joint_net = JointNet(site_encoders, dist_net)
        if mode == "tedad-sergey":
            joint_net = JointNetSergey(site_encoders, dist_net)

    for e in range(epochs):
        print("on epoch ", e)
        start_epoch = time.time()
        # runtimes = dict()
        totals = [0 for i in range(n_sites)]
        correct = [0 for i in range(n_sites)]
        correct_test = [0 for i in range(n_sites)]
        correct_test_a = 0
        correct_test_e = 0
        correct_val = [0 for i in range(n_sites)]
        correct_val_a = 0
        correct_val_e = 0
        effrank = {i: dict() for i in range(n_sites)}
        effrank["agg"] = dict()

        # Do Save statistics during training
        for encoder, decoder in zip(site_encoders, site_decoders):
            if encoder:
                encoder.hook.start_save()
                encoder.train()
            decoder.hook.start_save()
            decoder.train()
        start_train = time.time()
        for batch_iter, batches in enumerate(zip(*site_dataloaders)):
            batch_time = time.time()
            # Zero Grad
            # batch_train_start_time = time.time()
            for optimizer in site_decoder_optimizers:
                optimizer.zero_grad()
            for optimizer in site_encoder_optimizers:
                if optimizer:
                    optimizer.zero_grad()

            # Data Gathering
            site_data = []
            site_labels = []
            # print("WHAT THE HELL IS THE ", model)
            if model == "transformer":
                for batch in batches:
                    text = batch.text[0]
                    if text.shape[1] > max_length:
                        text = text[:, :(max_length)]
                    site_data.append(text)
                    site_labels.append(batch.label - 1)
            else:
                for batch in batches:
                    site_data.append(batch[0].to(device))
                    site_labels.append(batch[1].to(device))

            # DSGD
            yhats_dist = []
            yhats_loss = []
            if joint_net:
                yhats_dist = joint_net.forward(*site_data)
                if mode == "tedad-sergey":
                    yhats_loss = joint_net.backward(
                        site_labels,
                        yhats_dist,
                        lossFun,
                        site_data=site_data,
                        decoder_optimizers=site_decoder_optimizers,
                    )
                else:
                    yhats_loss = joint_net.backward(site_labels, yhats_dist, lossFun)
            elif dist_net:
                encodings = site_data
                for i, encoder in enumerate(site_encoders):
                    if encoder is not None:
                        encodings[i] = encoder(site_data[i])
                yhats_dist = dist_net.forward(*encodings)
                for network in dist_net.networks:
                    for module in network.modules():
                        for parameter in module.parameters():
                            parameter.retain_grad()
                back_time = time.time()
                yhats_loss = dist_net.backward(site_labels, yhats_dist, lossFun)
                print("\t\tbackward pass took ", time.time() - back_time)
                agg_time = time.time()
                dist_net.aggregate()
                print("\t\taggregate took ", time.time() - agg_time)
                recomp_time = time.time()
                dist_net.recompute_gradients()
                print("\t\trecompute gradients took ", time.time() - recomp_time)
            else:
                for encoder, decoder, data, label in zip(
                    site_encoders, site_decoders, site_data, site_labels
                ):
                    xs = encoder(data)
                    yhats = decoder(xs)
                    loss = lossFun()(yhats, label)
                    loss.backward()
                    yhats_loss.append(loss)
                    yhats_dist.append(yhats)
            for decoder in site_decoders:
                nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            for encoder in site_encoders:
                if encoder is not None:
                    nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            if mode != "tedad-sergey":
                for optimizer, sch in zip(
                    site_decoder_optimizers, site_decoder_schedulers
                ):
                    optimizer.step()
                    sch.step()
            for optimizer, sch in zip(site_encoder_optimizers, site_encoder_schedulers):
                if optimizer:
                    optimizer.step()
                if sch:
                    sch.step()
            for encoder, decoder in zip(site_encoders, site_decoders):
                if encoder:
                    encoder.hook.clear()
                decoder.hook.clear()
            if joint_net:
                joint_net.clear()
            elif dist_net:
                if hasattr(dist_net, "saved_ranks"):
                    for site, saved_ranks in dist_net.saved_ranks.items():
                        for mname, effective_rank in saved_ranks.items():
                            if mname not in effrank[site].keys():
                                effrank[site][mname] = 0
                            effrank[site][mname] += effective_rank
                dist_net.clear()
            elif dist_net_encoder:
                dist_net_encoder.clear()

            # Accuracy
            for i, (label, yhats) in enumerate(zip(site_labels, yhats_dist)):
                pred = yhats.argmax(dim=1, keepdim=True)
                correct[i] += pred.eq(label.view_as(pred)).sum().item()
                totals[i] += len(label)
            if dry_run:
                break
            print(
                "\ttraining batch %d/%d took " % (batch_iter, len(train_iter)),
                time.time() - batch_time,
            )
        print("all training took ", time.time() - start_train)
        # do not save statistics during testing
        for encoder, decoder in zip(site_encoders, site_decoders):
            if encoder:
                encoder.hook.clear()
                encoder.hook.stop_save()
                encoder.eval()
            decoder.hook.stop_save()
            decoder.hook.clear()

            decoder.eval()
        # Testing and Validation
        yhats_test_loss = [torch.zeros((1,)) for e in site_encoders]
        yhats_val_loss = [torch.zeros((1,)) for e in site_encoders]
        test_loss_e = 0
        test_loss_a = torch.zeros((1,))
        val_loss_e = 0
        val_loss_a = torch.zeros((1,))
        with torch.no_grad():
            # Testing
            test_time = time.time()
            for batch_iter, (test_data_x, test_data_y) in enumerate(test_dataloader):
                test_batch_time = time.time()
                if model == "transformer":
                    test_data_y = test_data_x[1] - 1
                    test_data_x = test_data_x[0][0]
                    if test_data_x.shape[1] > max_length:
                        test_data_x = test_data_x[:, :max_length]

                test_data_x = test_data_x.to(device)
                test_data_y = test_data_y.to(device)
                for i, (encoder_model, decoder_model) in enumerate(
                    zip(site_encoders, site_decoders)
                ):
                    yhat = None
                    encoding = None
                    if encoder_model:
                        encoding = encoder_model(test_data_x)
                    else:
                        encoding = test_data_x
                    yhat = decoder_model(encoding)
                    test_loss = lossFun()(yhat, test_data_y)
                    yhats_test_loss[i] += test_loss.item()
                    pred = yhat.argmax(dim=1, keepdim=True)
                    correct_test[i] += pred.eq(test_data_y.view_as(pred)).sum().item()
                avg_encoder = avg_decoder = None
                # Average Model
                """
                if site_encoders[0]:
                    avg_encoder = get_average_model(
                        encoder_class(*model_args).to(device), *site_encoders
                    ).to(device)
                    avg_encoder.hook.stop_save()
                    encoding = avg_encoder(test_data_x)
                else:
                    encoding = test_data_x
                avg_decoder = get_average_model(
                    decoder_class(*model_args).to(device), *site_decoders
                ).to(device)
                avg_decoder.hook.stop_save()
                yhat_a = avg_decoder(encoding)
                test_loss_a = lossFun()(yhat_a, test_data_y)
                pred_a = yhat_a.argmax(dim=1, keepdim=True)
                correct_test_a += pred_a.eq(test_data_y.view_as(pred_a)).sum().item()
                # End Average Model

                # Ensemble Model
                yhat_e = []
                for i, (encoder, decoder) in enumerate(
                    zip(site_encoders, site_decoders)
                ):
                    if encoder:
                        encoding = encoder(test_data_x)
                    else:
                        encoding = test_data_x
                    yhat_e.append(decoder(encoding))

                test_loss_e = np.mean(
                    [lossFun()(ye, test_data_y).item() for ye in yhat_e]
                )
                yhat_e = torch.sigmoid(torch.cat(yhat_e, 1))

                pred_e = torch.fmod(yhat_e.argmax(dim=1, keepdim=True), n_classes)
                correct_test_e += pred_e.eq(test_data_y.view_as(pred_e)).sum().item()
                """
                # End Ensemble Model
                for encoder, decoder in zip(site_encoders, site_decoders):
                    if encoder:
                        encoder.hook.clear()
                    decoder.hook.clear()
                if dry_run:
                    break
                print(
                    "\ttest batch %d/%d took " % (batch_iter, len(test_dataloader)),
                    time.time() - test_batch_time,
                )
            print("testing took ", time.time() - test_time)

        totals_test = len(test_data)
        totals_val = len(val_data) if val_data is not None else 1

        # Console Logging
        print("*****Epoch {epoch}*****".format(epoch=e))
        # Edad
        print("TIME ", time.time() - start_epoch)
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
                acc=[c / totals_test for c in correct_test],
            )
        )
        print(
            "\tTest Loss Avg {tlsa} Test Acc Avg {tacca}".format(
                tlsa=test_loss_a.item(), tacca=correct_test_a / totals_test
            )
        )
        print(
            "\tTest Loss Ensemble {tlsa} Test Acc Ensemble {tacca}".format(
                tlsa=test_loss_e, tacca=correct_test_e / totals_test
            )
        )
        print(
            "\tVal Losses {losses}\tVal Acc {acc}".format(
                losses=[l.item() for l in yhats_val_loss],
                acc=[c / totals_val for c in correct_val],
            )
        )
        print(
            "\tVal Loss Avg {tlsa} Val Acc Avg {tacca}".format(
                tlsa=val_loss_a.item(), tacca=correct_val_a / totals_val
            )
        )
        print(
            "\tVal Loss Ensemble {tlsa} Val Acc Ensemble {tacca}".format(
                tlsa=val_loss_e, tacca=correct_val_e / totals_val
            )
        )
        # End Console Logging

        # Results Logging
        row = dict(
            epoch=e,
            mode=mode,
            lr=lr,
            batch_size=batch_size,
            k=k,
            kf=kf,
            model=model,
            dataset=dataset,
            rank=rank,
            numiterations=numiterations,
        )
        for site, effrank_ in effrank.items():
            for mname, effective_rank in effrank_.items():
                row["effective_rank_site_%s_%s" % (site, mname)] = effective_rank / len(
                    site_dataloaders[0]
                )

        for (
            site,
            (train_loss, test_loss, correct_, correct_test_, correct_val_, total),
        ) in enumerate(
            zip(yhats_loss, yhats_test_loss, correct, correct_test, correct_val, totals)
        ):
            row["train_loss_site_%d" % site] = train_loss.item()
            row["test_loss_site_%d" % site] = test_loss.item()
            row["train_acc_site_%d" % site] = correct_ / total
            row["test_acc_site_%d" % site] = correct_test_ / totals_test
            row["val_acc_site_%d" % site] = correct_val_ / totals_val
        row["test_acc_avg"] = correct_test_a / totals_test
        row["test_loss_avg"] = test_loss_a.item()
        row["test_acc_ensemble"] = correct_test_e / totals_test
        row["test_loss_ensemble"] = test_loss_e
        row["val_acc_avg"] = correct_val_a / totals_val
        row["val_loss_avg"] = val_loss_a.item()
        row["val_acc_ensemble"] = correct_val_e / totals_val
        row["val_loss_ensemble"] = val_loss_e

        rows.append(row)
        df = pd.DataFrame(rows)
        if not dry_run:
            df.to_csv(
                filename, index=False,
            )
        # End Results logging
    return rows


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

