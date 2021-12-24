import torch
import numpy as np


class JointNetSergey:
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

    def backward(self, ys, yhats, loss_class, decoder_optimizers=None, site_data=None):
        self.dist_net.clear()
        losses = self.dist_net.backward(ys, yhats, loss_class)
        self.dist_net.aggregate()
        self.dist_net.recompute_gradients()
        if decoder_optimizers is not None:
            for optimizer in decoder_optimizers:
                optimizer.step()
        self.dist_net.toggle_compute_grads()
        if site_data is not None:
            yhats = self.forward(*site_data)
        losses = self.dist_net.backward(ys, yhats, loss_class)
        self.dist_net.aggregate()
        errors = self.dist_net.recompute_gradients()
        batch_size_per_size = int(errors.shape[0] / len(yhats))
        batch_start = 0
        batch_end = batch_size_per_size
        for encoder, loss in zip(self.encoders, losses):
            batch_indices = np.arange(batch_start, batch_end)
            encoder.set_delta(errors, batch_indices)
            loss.backward(retain_graph=True)
            batch_start += batch_size_per_size
            batch_end += batch_size_per_size
        self.dist_net.toggle_compute_grads()
        return losses

    def clear(self):
        self.dist_net.clear()
        self.forward_encodings = None
        self.inputs = None
        for encoder in self.encoders:
            encoder.set_delta(None, None)
