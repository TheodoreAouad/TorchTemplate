import torch.optim.lr_scheduler as lr_scheduler

from src.utils import log_console


class ReduceLROnPlateau(lr_scheduler.ReduceLROnPlateau):

    def __init__(self, loss_observable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_observable = loss_observable
        self.logger = None


    def step(self, *args, **kwargs):
        super().step(
            self.loss_observable.logs['val_loss_on_epoch'], *args, **kwargs)

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    log_console('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr),
                          logger=self.logger)


    def set_logger(self, logger):
        self.logger = logger

    def remove_logger(self):
        self.logger = None
