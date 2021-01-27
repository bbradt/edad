from dadnet.distnets.dadnet import DadNet
from dadnet.distnets.edadnet import EdadNet
from dadnet.distnets.dsgdnet import DsgdNet
from dadnet.distnets.distnet import DistNet

ALIASES = dict(pooled=DistNet, dsgd=DsgdNet, edad=EdadNet, dad=DadNet)


def get_distributed_model(name):
    return ALIASES.get(name.lower(), DistNet)
