import torch.nn as nn

try:
    import pydevd
except Exception:
    pass


class ModelHook:
    def __init__(
        self, model, verbose=False, layer_names=None, register_self=False, save=True
    ):
        self.model = model
        self.verbose = verbose
        self.layer_names = layer_names
        self.keys = []
        self.orders = dict()
        if register_self:
            self.register_hooks(self.model, layer_names)
        self.save = save
        self.clear()

    def start_save(self):
        self.save = True

    def stop_save(self):
        self.save = False

    def clear(self):
        self.grads = dict()
        self.forward_stats = dict()
        self.backward_stats = dict()
        self.backward_return = None
        self.batch_indices = None

    def forward_hook_fn(self, module, input, output):
        try:
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        except Exception:
            pass
        mname = module._order
        if self.save:
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
        mname = module._order
        if self.save:
            self.backward_stats[mname] = dict(input=list(input), output=list(output))
        if self.backward_return is not None:
            # grad_output = output[0]
            delta = self.backward_return[self.batch_indices, :]
            # delta = self.backward_return.t()
            # batch_mul = int(delta.shape[1] / grad_output.shape[0])
            # batch_res = int(grad_output.shape[0])
            # delta_res = delta.view(delta.shape[0], batch_res, batch_mul).mean(-1)
            # delta = delta_res.t().clone()
            self.backward_return = None
            if self.save:
                self.backward_stats[mname]["delta"] = delta
            act_input = self.forward_stats[mname]["input"][0]
            grad_weight = act_input.t().mm(delta)
            new_delta = (delta * self.forward_stats[mname]["output"][0]) @ module.weight
            return (new_delta, grad_weight.t())

    def get_hook_fun(self, module):
        mname = module._order

        def hook_fun(grad):
            if self.save:
                self.grads[mname] = grad

        return hook_fun

    def register_hooks(self, model, layer_names):
        modules = list(
            [
                module
                for i, module in enumerate(model.modules())
                if module.__class__.__name__ in layer_names
            ]
        )
        rev_mod = modules[::-1]
        self.orders = dict()
        for m_i, module in enumerate(rev_mod):
            self.keys.append(str(module) + str(m_i))
            module._order = str(module) + str(m_i)

            module.register_forward_hook(self.forward_hook_fn)
            module.register_backward_hook(self.backward_hook_fn)
            for parameter in module.parameters():
                parameter.register_hook(self.get_hook_fun(module))
            # for parameter in module.parameters():
            #    parameter.retain_grad()
            if self.verbose:
                print("Registered hook on module %s" % i)
