import torch
from dadnet.distnets.distnet import DistNet


class EdadNet(DistNet):
    def __init__(self, *networks, **kwargs):
        self.grads = None

        super(EdadNet, self).__init__(*networks, **kwargs)

    def recompute_gradients(self):
        seed_network = self.networks[0]
        for m_i, seed_module in enumerate(self.reverse_modules(seed_network)):
            seed_mname = str(seed_module)
            if seed_mname not in seed_network.hook.backward_stats.keys():
                continue
            agg_delta = []
            agg_input_activations = []
            agg_output_activations = []
            dact = self.get_dact(seed_module)
            for n_i, network in enumerate(self.networks):
                module = self.network_module_map[n_i][seed_mname]
                module_delta = network.hook.backward_stats[seed_mname]["output"][0]
                module_input_activations = network.hook.forward_stats[seed_mname][
                    "input"
                ][0]
                module_output_activations = network.hook.forward_stats[seed_mname][
                    "output"
                ][0]

                # true_delta = module_delta / module_dacts
                # if m_i == 0:
                agg_delta.append(module_delta)
                agg_input_activations.append(module_input_activations)
                agg_output_activations.append(module_output_activations)
            if m_i == 0:
                agg_delta = torch.cat(agg_delta, 0)
            else:
                agg_delta = agg_delta[0]
            agg_input_activations = torch.cat(agg_input_activations, 0)
            agg_output_activations = torch.cat(agg_output_activations, 0)
            agg_grad = agg_delta.t() @ agg_input_activations
            for n_i, network in enumerate(self.networks):
                module = self.network_module_map[n_i][seed_mname]
                weight = None
                new_delta = agg_delta
                if hasattr(module, "weight"):
                    weight = module.weight
                    if m_i == 0:
                        new_delta = (weight.t() @ (agg_delta).t()).t()
                    else:
                        agg_dacts = dact(agg_output_activations)
                        new_delta = weight.t() @ (agg_delta * agg_dacts).t()
                else:
                    agg_dacts = dact(agg_input_activations)
                    new_delta = agg_delta * agg_dacts
                next_mname = self.get_backward_module(seed_mname)
                if next_mname and next_mname in network.hook.backward_stats.keys():
                    network.hook.backward_stats[next_mname]["output"][0] = new_delta
                for p_i, parameter in enumerate(module.parameters()):
                    try:
                        parameter.grad = agg_grad.clone()
                    except KeyError:
                        continue
