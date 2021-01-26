import torch
from dadnet.distnets.distnet import DistNet


class DadNet(DistNet):
    def __init__(self, *networks, **kwargs):
        self.grads = None
        super(DadNet, self).__init__(*networks, **kwargs)

    def recompute_gradients(self):
        seed_network = self.networks[0]
        for m_i, seed_module in enumerate(self.reverse_modules(seed_network)):
            seed_mname = str(seed_module)
            if seed_mname not in seed_network.hook.backward_stats.keys():
                continue
            agg_delta = []
            agg_input_activations = []
            for n_i, network in enumerate(self.networks):
                module = self.network_module_map[n_i][seed_mname]
                module_delta = network.hook.backward_stats[seed_mname]["output"][0]
                module_input_activations = network.hook.forward_stats[seed_mname][
                    "input"
                ][0]
                agg_delta.append(module_delta)
                agg_input_activations.append(module_input_activations)
                # agg_output_activations.append(module_output_activations)
            agg_delta = torch.cat(agg_delta, 0)
            agg_input_activations = torch.cat(agg_input_activations, 0)
            # utput_activations = torch.cat(agg_output_activations, 0)
            agg_grad = (agg_input_activations.t().mm(agg_delta)).t()
            for n_i, network in enumerate(self.networks):
                module = self.network_module_map[n_i][seed_mname]
                for p_i, parameter in enumerate(module.parameters()):
                    try:
                        parameter.grad = agg_grad.clone()
                    except KeyError:
                        continue
