import torch


class DistributedTrainer:
    def __init__(
        self,
        distributed_model,
        loaders,
        optimizer_class=None,
        lr=1e-3,
        loss_class=None,
        device="cpu",
    ):
        self.models = distributed_model.networks
        self.distributed_model = distributed_model
        self.lr = lr
        self.optimizers = [
            optimizer_class(model.parameters(), lr=lr) for model in self.models
        ]
        self.num_batches = min([len(loader) for loader in loaders])
        self.loaders = loaders
        self.loss_class = loss_class
        self.device = device

    def run(self, epoch, verbose=False):
        for model in self.models:
            model.train()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        total_loss = [0 for i in self.models]
        totals = [0 for i in self.models]
        correct = [0 for i in self.models]
        for i, batches in enumerate(zip(*self.loaders)):
            xs = [b[0].to(self.device) for b in batches]
            ys = [b[1].to(self.device) for b in batches]
            yhat = self.distributed_model.forward(*xs)
            loss = self.distributed_model.backward(ys, yhat, self.loss_class)
            for li, (m, x, y) in enumerate(zip(self.models, xs, ys)):
                yhat_local = m(x)
                pred = yhat_local.argmax(dim=1, keepdim=True)
                correct[li] += pred.eq(y.view_as(pred)).sum().item()
                totals[li] += len(yhat_local)
            self.distributed_model.aggregate()
            self.distributed_model.recompute_gradients()
            for optimizer in self.optimizers:
                optimizer.step()
            total_loss = [(t + l.item()) / 2 for t, l in zip(total_loss, loss)]
            self.distributed_model.clear()
        acc = [c / t for c, t in zip(correct, totals)]
        return acc, total_loss

