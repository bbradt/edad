from dadnet.training.batch_runner import BatchRunner


class Tester(BatchRunner):
    def __init__(
        self,
        model,
        loader,
        optimizer_class=None,
        loss_class=None,
        device="cpu"
        # no_grad=False,
    ):
        super(Tester, self).__init__(
            model, loader, None, 0, loss_class, no_grad=True, device=device,
        )
