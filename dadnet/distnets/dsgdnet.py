import torch
from dadnet.distnets.distnet import DistNet


class DsgdNet(DistNet):
    def __init__(self, *networks, **kwargs):
        self.grads = None
        super(DsgdNet, self).__init__(*networks, **kwargs)

    def recompute_gradients(self):
        for network in self.networks:
            for module in network.modules():
                mname = str(module)
                for p_i, parameter in enumerate(module.parameters()):
                    try:
                        agg_grad = torch.sum(
                            torch.stack(self.aggregate_grads[mname][p_i], -1), -1
                        ) / len(self.networks)
                        parameter.grad = agg_grad.clone()
                        assert (parameter.grad == agg_grad).all()
                    except KeyError:
                        continue
