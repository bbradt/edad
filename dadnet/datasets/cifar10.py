import torchvision
from dadnet.datasets.numpy import NumpyDataset


class Cifar10Dataset(NumpyDataset):
    def __init__(self, train=True, download=True, normalize=True, data_root="data"):
        transforms = [torchvision.transforms.ToTensor()]
        if train:
            transforms.append(torchvision.transforms.RandomCrop(32, padding=4))
            transforms.append(torchvision.transforms.RandomHorizontalFlip())
        if normalize:
            transforms.append(
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                )
            )
        dataset = torchvision.datasets.CIFAR10(
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
        super(Cifar10Dataset, self).__init__(x, y)

