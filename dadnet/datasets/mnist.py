import torchvision
from dadnet.datasets.numpy import NumpyDataset


class MnistDataset(NumpyDataset):
    def __init__(self, train=True, download=True, normalize=True, data_root="data"):
        transforms = [torchvision.transforms.ToTensor()]
        if normalize:
            transforms.append(torchvision.transforms.Normalize((0.1307,), (0.3081,)))
        dataset = torchvision.datasets.MNIST(
            data_root,
            train=train,
            download=download,
            transform=torchvision.transforms.Compose(transforms),
        )
        if train:
            x = dataset.train_data
            y = dataset.train_labels
        else:
            x = dataset.test_data
            y = dataset.test_labels
        super(MnistDataset, self).__init__(x, y)

