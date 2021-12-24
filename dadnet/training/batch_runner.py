class BatchRunner:
    def __init__(
        self,
        model,
        loader,
        optimizer_class=None,
        lr=1e-3,
        loss_class=None,
        no_grad=False,
        device="cpu",
    ):
        self.model = model
        self.lr = lr
        if optimizer_class:
            self.optimizer = optimizer_class(self.model.parameters(), lr=lr)
        else:
            self.optimizer = None
        self.num_batches = len(loader)
        self.loader = loader
        self.loss_class = loss_class
        self.no_grad = no_grad
        self.device = device

    def run(self, epoch):
        if self.optimizer and not self.no_grad:
            self.optimizer.zero_grad()
        correct = 0
        total = 0
        for i, (x, y) in enumerate(self.loader):
            x = x.to(self.device)
            y = y.to(self.device)
            yhat = self.model(x)
            loss = self.loss_class()(yhat, y)
            if self.optimizer and not self.no_grad:
                loss.backward()
                self.optimizer.step()
            pred = yhat.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
            total += len(yhat)
        acc = 100.0 * correct / total
        loss_val = loss.item()
        return acc, loss_val

