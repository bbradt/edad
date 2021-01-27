from dadnet.training.batch_runner import BatchRunner
from dadnet.utils import get_average_model
import copy


class DistributedTester(BatchRunner):
    def __init__(self, models, loader, lr=1e-3, loss_class=None, device="cpu"):
        self.models = models
        self.surrogate = copy.deepcopy(models[0])
        model = get_average_model(self.surrogate, *models)
        super(DistributedTester, self).__init__(
            model, loader, None, lr, loss_class, no_grad=True, device=device
        )

    def run(self, epoch):
        self.model = get_average_model(self.surrogate, *self.models)
        return super(DistributedTester, self).run(epoch)
