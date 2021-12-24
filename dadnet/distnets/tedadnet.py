import torch
from dadnet.distnets.distnet import DistNet
from dadnet.hooks.accumulate_hook import AccumulateHook


class TeDadNet(DistNet):
    def __init__(self, *networks, **kwargs):
        self.grads = None

        super(TeDadNet, self).__init__(*networks, **kwargs)

    def recompute_gradients(self):
        seed_network = self.networks[0]
        new_delta = None
        for m_i, seed_module in enumerate(self.reverse_modules(seed_network)):
            seed_mname = str(seed_module) + str(m_i)
            if seed_mname not in seed_network.hook.backward_stats.keys():
                continue
            agg_delta = []
            agg_input_activations = []
            agg_output_activations = []
            dact = self.get_dact(seed_module)
            for n_i, network in enumerate(self.networks):
                module = self.network_module_map[n_i][seed_mname]
                if isinstance(network.hook, AccumulateHook):
                    module_delta = network.hook.backward_accumulated[seed_mname][
                        "output"
                    ][0]
                    module_input_activations = network.hook.forward_accumulated[
                        seed_mname
                    ]["input"][0]
                    module_output_activations = network.hook.forward_accumulated[
                        seed_mname
                    ]["output"][0]
                    if len(module_delta) > 1:
                        module_delta = torch.cat(module_delta, 0)
                    else:
                        module_delta = module_delta[0]
                    if len(module_input_activations) > 1:
                        module_input_activations = torch.cat(
                            module_input_activations, 0
                        )
                    else:
                        module_input_activations = module_input_activations[0]
                    if len(module_output_activations) > 1:
                        module_output_activations = torch.cat(
                            module_output_activations, 0
                        )
                    else:
                        module_output_activations = module_output_activations[0]
                else:
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
            if agg_delta.ndim == 3:
                agg_delta = agg_delta.reshape(
                    agg_delta.shape[0] * agg_delta.shape[1], agg_delta.shape[2]
                )
                agg_delta = torch.cat([agg_delta, agg_delta, agg_delta], 1)
            agg_input_activations = torch.cat(agg_input_activations, 0)
            agg_output_activations = torch.cat(agg_output_activations, 0)
            agg_grad = None
            if hasattr(seed_module, "weight"):
                agg_grad = agg_delta.t() @ agg_input_activations / len(self.networks)
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
                        new_delta = (weight.t() @ (agg_delta * agg_dacts).t()).t()
                else:
                    # new_delta = agg_delta
                    agg_dacts = dact(agg_input_activations)
                    new_delta = agg_delta.view_as(agg_dacts) * agg_dacts
                next_mname = self.get_backward_module(seed_mname)
                if next_mname and next_mname in network.hook.backward_stats.keys():
                    network.hook.backward_stats[next_mname]["output"][0] = new_delta
                    if isinstance(network.hook, AccumulateHook):
                        if (
                            seed_mname
                            != "Linear(in_features=128, out_features=384, bias=False)6"
                        ):
                            network.hook.backward_accumulated[next_mname]["output"][
                                0
                            ] = [new_delta]
                        if seed_mname == "Flatten()5":
                            next_mname = self.get_backward_module(next_mname)
                            network.hook.backward_accumulated[next_mname]["output"][
                                0
                            ] = [new_delta]
                for p_i, parameter in enumerate(module.parameters()):
                    try:
                        parameter.grad = agg_grad.clone()
                        # assert (parameter.grad == agg_grad).all()
                    except KeyError:
                        continue
        return new_delta
