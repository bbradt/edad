from dadnet.training.batch_runner import BatchRunner


class Trainer(BatchRunner):
    def __init__(
        self,
        model,
        loader,
        optimizer_class=None,
        lr=1e-3,
        loss_class=None,
        device="cpu"
        # no_grad=False,
    ):
        super(Trainer, self).__init__(
            model, loader, optimizer_class, lr, loss_class, no_grad=False, device=device
        )
