import pathlib
import os
from collections import defaultdict

import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    This is a parent class to objects used to track information during training.
    """
    logs_history = defaultdict(list)

    def __init__(self):
        """
            self.writer (torch.utils.tensorboard.writer.SummaryWriter): torch object to write tensorboard
            self.tensorboard_idx (int): index of the tensorboards
            self.results_idx (int): Not implemented yet
            self.current_epoch (int): current epoch of training
            self.number_of_epoch (int): number of epochs in training
            self.current_batch_idx (int): current index of the training batch
            self.number_of_batch (int): number of batches for the training
            self.logs (dict): the current state of variables we want to keep logs of
            self.logs_history (dict): all the previous states of variables we want to log
            self.output_dir_results (str): output directory to store results (Not Implemented)
            self.output_dir_tensorboard (str): output directory to store tensorboards
        """
        self.writer = None
        self.tensorboard_idx = {}
        self.results_idx = None
        self.current_epoch = None
        self.number_of_epoch = None
        self.current_batch_idx = None
        self.number_of_batch = None
        self.logs = {}
        self.logs_history = {}
        self.output_dir_results = None
        self.output_dir_tensorboard = None
        self.do_tensorboard = True

    def show(self):
        print_nicely_on_console(self.logs)

    def set_number_of_epoch(self, number_of_epoch):
        self.number_of_epoch = number_of_epoch
        for key in self.logs.keys():
            self.logs_history[key] = []

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch
        for key in self.logs_history.keys():
            self.logs_history[key].append([])

    def set_current_batch_idx(self, batch_idx):
        self.current_batch_idx = batch_idx

    def set_number_of_batch(self, number_of_batch):
        self.number_of_batch = number_of_batch

    def add_to_history(self, specific_keys=None):
        keys_to_add_to_history = specific_keys or self.logs.keys()
        for key in keys_to_add_to_history:
            if type(self.logs[key]) == torch.Tensor:
                self.logs_history[key][self.current_epoch].append(
                    self.logs[key].detach()
                )
            else:
                self.logs_history[key][self.current_epoch].append(
                    self.logs[key]
                )

    def init_results_writer(self, path):
        """
        NOT IMPLEMENTED
        """
        pass

    def init_tensorboard_writer(self, path=None, keys=None):
        if path is None or not self.do_tensorboard:
            return

        iterator = self._get_keys(self.logs, keys=keys)
        self.tensorboard_idx = {k:0 for k in iterator}
        self.output_dir_tensorboard = pathlib.Path(path)
        self.output_dir_tensorboard.mkdir(parents=True, exist_ok=True)
        self.writer = {}
        for key in iterator:
            self.writer[key] = SummaryWriter(log_dir=self.output_dir_tensorboard / key)


    def _get_keys(self, dict, keys=None):
        if keys is None:
            iterator = dict.keys()
        else:
            iterator = set(dict.keys()).intersection(set(keys))
        return iterator

    def write_tensorboard(self, boards_names={}, keys=None,):
        """

        Args:
            **kwargs (dict): tell on which tensorboard to write each key.

        Returns:

        """
        if self.output_dir_tensorboard is None:
            return

        iterator = self._get_keys(self.writer, keys=keys)

        for key in iterator:
            if self.logs[key] is not None:
                if type(self.logs[key]) == torch.Tensor and self.logs[key].shape == torch.Size([]):
                        self.writer[key].add_scalar(
                            boards_names.get(key, key),
                            self.logs[key],
                            self.tensorboard_idx[key],
                        )
                        self.tensorboard_idx[key] += 1

                elif hasattr(self.logs[key], "__getitem__") and np.array(self.logs[key]).shape != ():

                    self.writer[key].add_scalars(
                        boards_names.get(key, key),
                        {str(idx): logs for idx, logs in enumerate(self.logs[key])},
                        self.tensorboard_idx[key],
                    )
                    self.tensorboard_idx[key] += 1
                    
                else:
                    self.writer[key].add_scalar(
                        boards_names.get(key, key),
                        self.logs[key],
                        self.tensorboard_idx[key],
                    )
                    self.tensorboard_idx[key] += 1

    def close_writer(self):
        if self.output_dir_tensorboard is None:
            return

        for key in self.writer.keys():
            self.writer[key].close()

    def results(self):
        """
        NOT IMPLEMENTED
        """
        pass

    def write_results(self):
        """
        NOT IMPLEMENTED
        """
        pass


def print_nicely_on_console(dic):
    """
    Prints nicely a dict on console.
    Args:
        dic (dict): the dictionary we want to print nicely on console.
    """
    to_print = ''
    for key, value in zip(dic.keys(), dic.values()):
        if type(value) == torch.Tensor:
            value_to_print = value.cpu().detach().numpy()
        else:
            value_to_print = value

        if value is not None:
            if 'accuracy' in key:
                value_to_print = str(round(100 * value_to_print, 2)) + ' %'
            elif type(value_to_print) == np.ndarray:
                if value_to_print.shape == ():
                    value_to_print = "{:.2E}".format(value_to_print)
                else:
                    with np.nditer(value_to_print, op_flags=['readwrite']) as it:
                        for x in it:
                            x[...] = "{:.2E}".format(x)
            else:
                value_to_print = "{:.2E}".format(value)
            to_print += '{}: {}, '.format(key, value_to_print)
    print(to_print)