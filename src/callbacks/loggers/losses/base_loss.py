import torch

from src.callbacks.loggers.logger import Logger


class BaseLoss(Logger):
    """
    Base class for losses. Losses should inherit this class.
    """
    def __init__(self, criterion):
        super(BaseLoss, self).__init__()
        self.criterion = criterion
        self.logs = {'train_loss': None}
        self.current_loss = None

    def compute(self, outputs, labels, add_to_history=True,):
        self.current_loss = self.criterion(outputs, labels)
        self.logs['train_loss'] = self.current_loss.detach().cpu().item()
        if add_to_history:
            self.add_to_history()
            self.write_tensorboard()

    def backward(self, *args, **kwargs):
        assert type(self.current_loss) == torch.Tensor, 'Loss type should be torch.Tensor'
        self.current_loss.backward(*args, **kwargs)

    def __call__(self, outputs, labels):
        return self.criterion(outputs, labels)
