import torch
import torch.nn as nn
from dadnet.utils import n_bits
from dadnet.hooks.accumulate_hook import AccumulateHook


class DistNet:
    def __init__(self, *networks, layer_names=[], shared_layers=None):
        self.networks = networks
        self.bandwidth_sent = dict()
        self.aggregate_grads = dict()
        self.aggregate_forward = dict()
        self.aggregate_backward = dict()
        self.shared_layers = shared_layers
        if not self.shared_layers:
            self.shared_layers = []
            for i, _ in enumerate(self.reverse_modules(self.networks[0])):
                self.shared_layers.append(i)
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
            for m_i, module in enumerate(self.reverse_modules(network)):
                if not hasattr(module, "_order"):
                    continue
                mname = module._order
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
        if index + 1 < len(self.module_orders):
            return self.module_orders[index + 1]

    def forward(self, *args):
        ys = []
        for x, network in zip(args, self.networks):
            ys.append(network(x))
        return ys

    def backward(self, ys, yhats, loss_class):
        losses = []
        for y, yhat in zip(ys, yhats):
            loss = loss_class()(yhat, y)
            loss.backward(retain_graph=True)
            losses.append(loss)
        return losses

    def toggle_compute_grads(self):
        for network in self.networks:
            for parameter in network.parameters():
                parameter.requires_grad = not (parameter.requires_grad)

    def aggregate_from_dict(
        self, source_dict, agg_dict, bandwidth_dict, mname, fkey="forward"
    ):
        if mname in source_dict.keys():
            for statname, stat in source_dict[mname].items():
                if statname not in agg_dict.keys():
                    agg_dict[mname][statname] = dict()
                    bandwidth_dict[mname][fkey][statname] = dict()
                for i, s in enumerate(stat):
                    if i not in agg_dict[mname][statname].keys():
                        agg_dict[mname][statname][i] = list()
                        bandwidth_dict[mname][fkey][statname][i] = 0
                    agg_dict[mname][statname][i].append(stat[i])
                    bandwidth_dict[mname][fkey][statname][i] += n_bits(stat[i])
        return agg_dict, bandwidth_dict

    def aggregate_stack_from_dict(
        self, source_dict, agg_dict, bandwidth_dict, mname, fkey="forward"
    ):
        if mname in source_dict.keys():
            for statname, stat in source_dict[mname].items():
                if statname not in agg_dict.keys():
                    agg_dict[mname][statname] = dict()
                    bandwidth_dict[mname][fkey][statname] = dict()
                for i, s in enumerate(stat):
                    if i not in agg_dict[mname][statname].keys():
                        agg_dict[mname][statname][i] = list()
                        bandwidth_dict[mname][fkey][statname][i] = 0
                    if len(stat[i]) > 1:
                        stat_val = [s for s in stat[i] if s is not None]
                        if len(stat_val) > 0:
                            stat[i] = torch.stack(stat_val, 1)
                        else:
                            stat[i] = torch.empty((0,))
                    else:
                        stat[i] = stat[i][0]
                    agg_dict[mname][statname][i].append(stat[i])
                    bandwidth_dict[mname][fkey][statname][i] += n_bits(stat[i])
        return agg_dict, bandwidth_dict

    def aggregate(self):
        for n_i, network in enumerate(self.networks):
            for m_i, module in enumerate(self.reverse_modules(network)):
                if not hasattr(module, "_order"):
                    continue
                mname = module._order
                if mname not in self.bandwidth_sent.keys():
                    self.bandwidth_sent[mname] = dict(
                        grad=dict(),
                        forward=dict(),
                        backward=dict(),
                        forward_accumulated=dict(),
                        backward_accumulated=dict(),
                    )
                    self.aggregate_grads[mname] = dict()
                    self.aggregate_forward[mname] = dict()
                    self.aggregate_backward[mname] = dict()
                for p_i, parameters in enumerate(module.parameters()):
                    grad = network.hook.grads[mname]
                    if p_i not in self.aggregate_grads[mname]:
                        self.aggregate_grads[mname][p_i] = list()
                        self.bandwidth_sent[mname]["grad"][p_i] = 0
                    self.aggregate_grads[mname][p_i].append(grad)
                    self.bandwidth_sent[mname]["grad"][p_i] += n_bits(grad)
                if isinstance(network.hook, AccumulateHook):
                    (
                        self.aggregate_forward,
                        self.bandwidth_sent,
                    ) = self.aggregate_stack_from_dict(
                        network.hook.forward_accumulated,
                        self.aggregate_forward,
                        self.bandwidth_sent,
                        mname,
                    )
                    (
                        self.aggregate_backward,
                        self.bandwidth_sent,
                    ) = self.aggregate_stack_from_dict(
                        network.hook.backward_accumulated,
                        self.aggregate_backward,
                        self.bandwidth_sent,
                        mname,
                    )
                else:
                    (
                        self.aggregate_forward,
                        self.bandwidth_sent,
                    ) = self.aggregate_from_dict(
                        network.hook.forward_stats,
                        self.aggregate_forward,
                        self.bandwidth_sent,
                        mname,
                    )
                    (
                        self.aggregate_backward,
                        self.bandwidth_sent,
                    ) = self.aggregate_from_dict(
                        network.hook.backward_stats,
                        self.aggregate_backward,
                        self.bandwidth_sent,
                        mname,
                    )

    def broadcast(self):
        pass

    def recompute_gradients(self):
        pass

    def reverse_modules(self, network):
        return list(network.modules())[::-1]
