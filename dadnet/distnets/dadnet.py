from logging import PlaceHolder
import torch
from dadnet.distnets.distnet import DistNet
from dadnet.hooks.accumulate_hook import AccumulateHook


class DadNet(DistNet):
    def __init__(self, *networks, **kwargs):
        self.grads = None

        super(DadNet, self).__init__(*networks, **kwargs)

    def recompute_gradients(self):
        seed_network = self.networks[0]
        new_delta = None
        for m_i, seed_module in enumerate(self.reverse_modules(seed_network)):
            if not hasattr(seed_module, "_order"):
                continue
            seed_mname = seed_module._order
            if seed_mname not in seed_network.hook.backward_stats.keys():
                continue
            agg_delta = []
            agg_input_activations = []
            agg_output_activations = []
            for n_i, network in enumerate(self.networks):
                module = self.network_module_map[n_i][seed_mname]
                if isinstance(network.hook, AccumulateHook):
                    module_delta = network.hook.backward_accumulated[seed_mname][
                        "output"
                    ][0]
                    module_input_activations = network.hook.forward_accumulated[
                        seed_mname
                    ]["input"][0]
                    if type(module_delta) is list and len(module_delta) > 1:
                        module_delta = torch.cat(module_delta, 0)
                    elif type(module_delta) is list:
                        module_delta = module_delta[0]
                    if (
                        type(module_input_activations) is list
                        and len(module_input_activations) > 1
                    ):
                        module_input_activations = torch.cat(
                            module_input_activations, 0
                        )
                    elif type(module_input_activations) is list:
                        module_input_activations = module_input_activations[0]
                else:
                    module_delta = network.hook.backward_stats[seed_mname]["output"][0]
                    module_input_activations = network.hook.forward_stats[seed_mname][
                        "input"
                    ][0]

                agg_delta.append(module_delta)
                agg_input_activations.append(module_input_activations)
            try:
                agg_delta = torch.cat(agg_delta, 0)
            except RuntimeError:
                max_shapes = max([d.shape[1] for d in agg_delta])
                for i, d in enumerate(agg_delta):
                    if d.shape[1] < max_shapes:
                        zero_placeholder = torch.zeros(
                            (d.shape[0], max_shapes, d.shape[2])
                        ).to(d.device)
                        zero_placeholder[:, : d.shape[1], :] = d
                        agg_delta[i] = zero_placeholder.clone()
                agg_delta = torch.cat(agg_delta, 0)
            try:
                agg_input_activations = torch.cat(agg_input_activations, 0)
            except RuntimeError:
                max_shapes = max([d.shape[1] for d in agg_input_activations])
                for i, d in enumerate(agg_input_activations):
                    if d.shape[1] < max_shapes:
                        zero_placeholder = torch.zeros(
                            (d.shape[0], max_shapes, d.shape[2])
                        ).to(d.device)
                        zero_placeholder[:, : d.shape[1], :] = d
                        agg_input_activations[i] = zero_placeholder.clone()
                agg_input_activations = torch.cat(agg_input_activations, 0)
            agg_grad = None
            if agg_delta.ndim == 3:
                agg_delta = agg_delta.reshape(
                    agg_delta.shape[0] * agg_delta.shape[1], agg_delta.shape[2]
                )
                agg_input_activations = agg_input_activations.reshape(
                    agg_input_activations.shape[0] * agg_input_activations.shape[1],
                    agg_input_activations.shape[2],
                )
                # agg_delta = torch.cat([agg_delta, agg_delta, agg_delta], 1)
            if hasattr(seed_module, "weight"):
                if agg_delta.shape[0] != agg_input_activations.shape[0]:
                    agg_delta = agg_delta.t()
                agg_grad = agg_delta.t() @ agg_input_activations / len(self.networks)
            for n_i, network in enumerate(self.networks):
                module = self.network_module_map[n_i][seed_mname]
                for p_i, parameter in enumerate(module.parameters()):
                    try:
                        if len(parameter.grad.shape) == 1:
                            parameter.grad = torch.sum(agg_grad, 1)
                        else:
                            parameter.grad = agg_grad.clone()
                        # assert (parameter.grad == agg_grad).all()
                    except KeyError:
                        continue
        return new_delta
