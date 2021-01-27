from dadnet.datasets.mnist import MnistDataset
from dadnet.datasets.loaders import get_dataloaders
from dadnet.datasets.splitting import split_dataset
from dadnet.distnets import get_distributed_model
from dadnet.model import get_model
import numpy as np
from dadnet.training.distributed_trainer import DistributedTrainer
from dadnet.training.distributed_tester import DistributedTester
import torch.nn as nn
import torch
import time

OPTIMIZERS = dict(
    sgd=torch.optim.SGD, adam=torch.optim.Adam, adadelta=torch.optim.Adadelta
)


def main(
    epochs=1000,
    n_sites=2,
    mode="dsgd",
    batch_size=32,
    test_batch_size=10000,
    dataname="mnist",
    split_type="class",
    model_type="simpleff",
    model_kwargs={},
    optimizer="adam",
    lr=1e-5,
    device="cuda",
):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    optimizer = OPTIMIZERS.get(optimizer.lower(), torch.optim.SGD)
    if dataname.lower() == "mnist":
        train_dataset = MnistDataset(train=True)
        test_dataset = MnistDataset(train=False)
    else:
        raise (ValueError("The dataset %s not implemented" % dataname))
    if mode.lower() == "pooled":
        n_sites = 1
    train_datasets = split_dataset(split_type, train_dataset, n_sites)
    train_loaders = []
    for train_dataset in train_datasets:
        loader, _, _ = get_dataloaders(train_data=train_dataset, batch_size=batch_size)
        train_loaders.append(loader)
    _, test_loader, _ = get_dataloaders(
        test_data=test_dataset, batch_size=test_batch_size
    )
    input_shape = train_dataset.x.shape
    n_classes = len(np.unique(train_dataset.y))
    models = []
    for i in range(n_sites):
        torch.manual_seed(0)
        np.random.seed(0)
        models.append(
            get_model(model_type)(input_shape, n_classes, **model_kwargs).to(device)
        )

    distributed_model = get_distributed_model(mode)(
        *models, layer_names=["Linear", "ReLU", "Sigmoid", "Tanh"]
    )
    trainer = DistributedTrainer(
        distributed_model,
        train_loaders,
        optimizer,
        lr=lr,
        loss_class=nn.CrossEntropyLoss,
        device=device,
    )
    tester = DistributedTester(
        models, test_loader, lr=lr, loss_class=nn.CrossEntropyLoss, device=device
    )
    for e in range(epochs):
        train_acc, train_loss = trainer.run(e, verbose=False)
        test_acc, test_loss = tester.run(e)
        print(
            "Epoch {epoch}, Train Loss {train_loss}, Train Acc {train_acc}, Test Loss {test_loss}, Test Acc {test_acc}".format(
                epoch=e,
                train_loss=train_loss,
                train_acc=train_acc,
                test_acc=test_acc,
                test_loss=test_loss,
            )
        )
        distributed_model.clear()

    pass


if __name__ == "__main__":
    main()
