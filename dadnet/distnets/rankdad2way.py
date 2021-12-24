import torch
from dadnet.distnets.distnet import DistNet
from dadnet.hooks.accumulate_hook import AccumulateHook
from dadnet.power_iterations.power_iteration_BC import power_iteration_BC


class RankDad2WayNet(DistNet):
    def __init__(self, *networks, rank=10, numiterations=10, **kwargs):
        self.rank = rank
        self.numiterations = numiterations
        self.saved_ranks = dict()
        super(RankDad2WayNet, self).__init__(*networks, **kwargs)

    def clear(self):
        super(RankDad2WayNet, self).clear()
        self.saved_ranks = dict()

    def recompute_gradients(self):
        seed_network = self.networks[0]
        new_delta = None
        for m_i, seed_module in enumerate(self.reverse_modules(seed_network)):
            seed_mname = str(seed_module) + str(m_i)
            if seed_mname not in seed_network.hook.backward_stats.keys():
                continue
            agg_delta = []
            agg_input_activations = []
            if not hasattr(seed_module, "weight"):
                continue
            for n_i, network in enumerate(self.networks):
                if n_i not in self.saved_ranks.keys():
                    self.saved_ranks[n_i] = dict()
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
                if module_delta.ndim == 3:
                    module_delta = module_delta.reshape(
                        module_delta.shape[0] * module_delta.shape[1],
                        module_delta.shape[2],
                    )
                if module_input_activations.ndim == 3:
                    module_input_activations = module_input_activations.reshape(
                        module_input_activations.shape[0]
                        * module_input_activations.shape[1],
                        module_input_activations.shape[2],
                    )
                module_B, module_C = power_iteration_BC(
                    module_input_activations.t(),
                    module_delta.t(),
                    rank=self.rank,
                    numiterations=self.numiterations,
                    device=module_delta.device,
                )
                self.saved_ranks[n_i][seed_mname] = module_B.shape[1]

                agg_delta.append(module_C.t())
                agg_input_activations.append(module_B.t())
            agg_C = torch.cat(agg_delta, 0)
            agg_B = torch.cat(agg_input_activations, 0)
            agg_input_activations, agg_delta = power_iteration_BC(
                agg_B.t(),
                agg_C.t(),
                rank=self.rank,
                numiterations=self.numiterations,
                device=agg_B.device,
            )
            agg_input_activations = agg_input_activations.t()
            agg_delta = agg_delta.t()
            if "agg" not in self.saved_ranks.keys():
                self.saved_ranks["agg"] = dict()
            self.saved_ranks["agg"][seed_mname] = agg_delta.shape[0]

            agg_grad = None
            if hasattr(seed_module, "weight"):
                if agg_delta.shape[0] != agg_input_activations.shape[0]:
                    agg_delta = agg_delta.t()
                agg_grad = agg_delta.t() @ agg_input_activations / len(self.networks)
            for n_i, network in enumerate(self.networks):
                module = self.network_module_map[n_i][seed_mname]
                for p_i, parameter in enumerate(module.parameters()):
                    try:
                        parameter.grad = agg_grad.clone()
                    except KeyError:
                        continue
        return new_delta
