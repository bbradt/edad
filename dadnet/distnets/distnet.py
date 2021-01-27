import torch
import torch.nn as nn
from dadnet.hooks.model_hook import ModelHook
from dadnet.utils import n_bits


class DistNet:
    def __init__(self, *networks, layer_names=[]):
        self.networks = networks
        self.hooks = [
            ModelHook(
                network, verbose=False, layer_names=layer_names, register_self=True
            )
            for network in networks
        ]
        self.bandwidth_sent = dict()
        self.aggregate_grads = dict()
        self.aggregate_forward = dict()
        self.aggregate_backward = dict()
        self.network_module_map, self.module_orders = self.build_network_module_map()
        super(DistNet, self).__init__()

    def clear(self):
        self.bandwidth_sent = dict()
        self.aggregate_grads = dict()
        self.aggregate_forward = dict()
        self.aggregate_backward = dict()

    def build_network_module_map(self):
        result = dict()
        orders = []
        for i, network in enumerate(self.networks):
            result[i] = dict()
            for m_i, module in enumerate(network.modules()):
                mname = str(module)
                result[i][mname] = module
                if i == 0:
                    orders.append(mname)
        return result, orders

    def get_dact(self, module):
        mname = module.__class__.__name__

        def _d_relu(x):
            dr = torch.where(
                x > 0, torch.ones_like(x).to(x.device), torch.zeros_like(x).to(x.device)
            )
            return dr.to(x.device)

        def _d_sigmoid(x):
            s = nn.functional.sigmoid(x)
            return s * (1 - s)

        def _d_tanh(x):
            t = nn.functional.tanh(x)
            return 1 - t * t

        def _lin(x):
            return torch.ones_like(x).to(x.device)

        result = {
            "ReLU": _d_relu,
            "Sigmoid": _d_sigmoid,
            "Tanh": _d_tanh,
            "Linear": _lin,
        }
        return result.get(mname, _lin)

    def get_next_module(self, mname):
        index = self.module_orders.index(mname)
        if index + 1 < len(self.module_orders):
            return self.module_orders[index + 1]

    def get_backward_module(self, mname):
        index = self.module_orders.index(mname)
        if index - 1 > 0:
            return self.module_orders[index - 1]

    def forward(self, *args):
        ys = []
        for x, network in zip(args, self.networks):
            ys.append(network(x))
        return ys

    def backward(self, ys, yhats, loss_class):
        losses = []
        for y, yhat in zip(ys, yhats):
            loss = loss_class()(yhat, y)
            loss.backward()
            losses.append(loss)
        return losses

    def aggregate(self):
        for n_i, network in enumerate(self.networks):
            for m_i, module in enumerate(self.reverse_modules(network)):
                mname = str(module)
                if mname not in self.bandwidth_sent.keys():
                    self.bandwidth_sent[mname] = dict(
                        grad=dict(), forward=dict(), backward=dict()
                    )
                    self.aggregate_grads[mname] = dict()
                    self.aggregate_forward[mname] = dict()
                    self.aggregate_backward[mname] = dict()
                for p_i, parameters in enumerate(module.parameters()):
                    if p_i not in self.aggregate_grads[mname]:
                        self.aggregate_grads[mname][p_i] = list()
                        self.bandwidth_sent[mname]["grad"][p_i] = 0
                    self.aggregate_grads[mname][p_i].append(parameters.grad)
                    self.bandwidth_sent[mname]["grad"][p_i] += n_bits(parameters.grad)
                if module in network.hook.forward_stats.keys():
                    for statname, stat in network.hook.forward_stats[module].items():
                        if statname not in self.aggregate_forward.keys():
                            self.aggregate_forward[mname][statname] = dict()
                            self.bandwidth_sent[mname]["forward"][statname] = dict()
                        for i, s in enumerate(stat):
                            if i not in self.aggregate_forward[mname][statname].keys():
                                self.aggregate_forward[mname][statname][i] = list()
                                self.bandwidth_sent[mname]["forward"][statname][i] = 0
                            self.aggregate_forward[mname][statname][i].append(stat[i])
                            self.bandwidth_sent[mname]["forward"][statname][
                                i
                            ] += n_bits(stat[i])
                if mname in network.hook.backward_stats.keys():
                    for statname, stat in network.hook.backward_stats[mname].items():
                        if statname not in self.aggregate_backward.keys():
                            self.aggregate_backward[mname][statname] = dict()
                            self.bandwidth_sent[mname]["backward"][statname] = dict()
                        for i, s in enumerate(stat):
                            if i not in self.aggregate_backward[mname][statname].keys():
                                self.aggregate_backward[mname][statname][i] = list()
                                self.bandwidth_sent[mname]["backward"][statname][i] = 0
                            self.aggregate_backward[mname][statname][i].append(stat[i])
                            self.bandwidth_sent[mname]["backward"][statname][
                                i
                            ] += n_bits(stat[i])

    def broadcast(self):
        pass

    def recompute_gradients(self):
        pass

    def reverse_modules(self, network):
        return list(network.modules())[::-1]
