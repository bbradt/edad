import torch
import torch.nn as nn

from dadnet.hooks.model_hook import ModelHook
from dadnet.modules.flatten import Flatten

x = torch.randn(64, 32, 32)
y = torch.randint(0, 10, (64,))


class SimpleLinear(nn.Module):
    def __init__(self):
        super(SimpleLinear, self).__init__()
        self.flatten = Flatten(1)
        self.fc1 = nn.Linear(32 * 32, 128, bias=False)
        self.fc2 = nn.Linear(128, 10, bias=False)
        self.hook = ModelHook(
            self, verbose=False, layer_names=["Linear"], register_self=True
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def recompute_gradients(self):
        for module in list(self.modules())[::-1]:
            if module not in self.hook.keys:
                continue
            module_delta = self.hook.backward_stats[module]["output"][0]
            module_input_activations = self.hook.forward_stats[module]["input"][0]
            for parameter in module.parameters():
                assert (
                    parameter.grad == module_delta.t() @ module_input_activations
                ).all()
        pass


if __name__ == "__main__":
    model = SimpleLinear()
    yhat = model(x)
    loss = nn.CrossEntropyLoss()(yhat, y)
    loss.backward()
    model.recompute_gradients()
