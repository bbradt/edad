import torch.nn as nn

try:
    import pydevd
except Exception:
    pass


class ModelHook:
    def __init__(self, model, verbose=False, layer_names=None, register_self=False):
        self.model = model
        self.verbose = verbose
        self.layer_names = layer_names
        self.keys = []
        if register_self:
            self.register_hooks(self.model, layer_names)
        self.clear()

    def clear(self):
        self.forward_stats = dict()
        self.backward_stats = dict()

    def forward_hook_fn(self, module, input, output):
        try:
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        except Exception:
            pass
        mname = str(module)
        if type(output) is not tuple:
            output = [output]
        else:
            output = list(output)
        self.forward_stats[mname] = dict(input=list(input), output=output)

    def backward_hook_fn(self, module, input, output):
        try:
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        except Exception:
            pass
        mname = str(module)
        self.backward_stats[mname] = dict(input=list(input), output=list(output))

    def register_hooks(self, model, layer_names):
        modules = list(
            [
                module
                for i, module in enumerate(model.modules())
                if module.__class__.__name__ in layer_names
            ]
        )

        for i, module in enumerate(modules):
            self.keys.append(module)
            module.register_forward_hook(self.forward_hook_fn)
            module.register_backward_hook(self.backward_hook_fn)
            if self.verbose:
                print("Registered hook on module %s" % i)
