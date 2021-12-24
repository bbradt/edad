import torch
import numpy as np


class JointNet:
    def __init__(self, encoders, dist_net):
        self.encoders = encoders
        self.dist_net = dist_net
        self.forward_encodings = None
        self.inputs = None

    def forward(self, *args):
        self.inputs = args
        ys = []
        for x, encoder in zip(args, self.encoders):
            x = encoder(x)
            ys.append(x)
        self.forward_encodings = ys
        ys = self.dist_net.forward(*ys)
        return ys

    def get_last_parameter(self, encoder):
        return list(encoder.parameters())[-1]

    def backward(self, ys, yhats, loss_class):
        self.dist_net.clear()
        losses = self.dist_net.backward(ys, yhats, loss_class)
        self.dist_net.aggregate()
        errors = self.dist_net.recompute_gradients()
        self.dist_net.toggle_compute_grads()
        if errors is not None:
            batch_size_per_size = int(errors.shape[0] / len(yhats))
            for site1, _ in enumerate(losses):
                batch_start = 0
                batch_end = batch_size_per_size
                all_batch_grads = []
                for site, (loss, encoder, encoding) in enumerate(
                    zip(losses, self.encoders, self.forward_encodings)
                ):
                    batch_indices = np.arange(batch_start, batch_end)
                    encoder.set_delta(errors, batch_indices)
                    batch_start += batch_size_per_size
                    batch_end += batch_size_per_size
                    loss.backward(retain_graph=True)
                    grads = {
                        p_i: param.grad.clone()
                        for p_i, param in enumerate(encoder.parameters())
                    }
                    all_batch_grads.append(grads)
                for p_i, param in enumerate(encoder.parameters()):
                    param.grad = torch.stack(
                        [g[p_i] for g in all_batch_grads], -1
                    ).mean(-1)
        self.dist_net.toggle_compute_grads()
        return losses

    def clear(self):
        self.dist_net.clear()
        self.forward_encodings = None
        self.inputs = None
        for encoder in self.encoders:
            encoder.set_delta(None, None)