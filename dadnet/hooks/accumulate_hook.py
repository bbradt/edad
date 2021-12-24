import torch.nn as nn

try:
    import pydevd
except Exception:
    pass

from dadnet.hooks.model_hook import ModelHook


class AccumulateHook(ModelHook):
    def __init__(
        self, model, verbose=False, layer_names=None, register_self=False, save=True
    ):
        super(AccumulateHook, self).__init__(
            model, verbose, layer_names, register_self, save
        )

    def clear(self):
        super(AccumulateHook, self).clear()
        self.forward_accumulated = dict()
        self.backward_accumulated = dict()

    def forward_hook_fn(self, module, input, output):
        super(AccumulateHook, self).forward_hook_fn(module, input, output)
        mname = module._order
        if self.save:
            if mname not in self.forward_accumulated.keys():
                self.forward_accumulated[mname] = dict()
            for key, stat in self.forward_stats[mname].items():
                if key not in self.forward_accumulated[mname].keys():
                    self.forward_accumulated[mname][key] = [[s] for s in stat]
                else:
                    for i, stat_i in enumerate(stat):
                        self.forward_accumulated[mname][key][i].append(stat[i])

    def backward_hook_fn(self, module, input, output):
        super(AccumulateHook, self).backward_hook_fn(module, input, output)
        mname = module._order
        if self.save:
            if mname not in self.backward_accumulated.keys():
                self.backward_accumulated[mname] = dict()
            for key, stat in self.backward_stats[mname].items():
                if key not in self.backward_accumulated[mname].keys():
                    self.backward_accumulated[mname][key] = [[s] for s in stat]
                else:
                    for i, stat_i in enumerate(stat):
                        self.backward_accumulated[mname][key][i].append(stat[i])
