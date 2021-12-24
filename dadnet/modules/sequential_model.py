import torch.nn as nn
from dadnet.hooks.model_hook import ModelHook


class SequentialModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(SequentialModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.layers = []
        for module in encoder.modules():
            if module.__class__.__name__ == encoder.__class__.__name__:
                continue
            setattr(self, str(module), module)
            self.layers.append(str(module))
        for module in decoder.modules():
            if module.__class__.__name__ == decoder.__class__.__name__:
                continue
            setattr(self, str(module), module)
            self.layers.append(str(module))
        # self.encoder = encoder
        # self.decoder = decoder
        layer_names = encoder.hook.layer_names + decoder.hook.layer_names
        self.hook = ModelHook(
            self, verbose=False, layer_names=list(layer_names), register_self=True
        )

    def forward(self, x):
        for attr in self.layers:
            x = getattr(self, attr)(x)
        return x

    def get_state_dicts(self):
        return self.encoder.state_dict(), self.decoder.state_dict()
