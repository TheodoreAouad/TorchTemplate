from os.path import join
from time import time
import random
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from src.utils import confusion_df
from src.callbacks.loggers.logger import Logger
from src.plotter import plot_img_mask_on_ax
from src.tasks.test import evaluate_loss, evaluate_metrics, evaluate_loss_mean, evaluate_metrics_mean


class Observables(Logger):
    """
    Base class for all observables during training of deep neural networks.
    """
    def compute_train_on_batch(self, inputs, outputs, targets, dataloader=None):
        pass

    def compute_train_on_epoch(self, inputs, outputs, targets, device, dataloader=None):
        pass

    def compute_val_on_batch(self, inputs, outputs, targets, device, dataloader=None):
        pass

    def compute_val_on_epoch(self, inputs, outputs, targets, device, dataloader=None):
        pass


class ObservableImage(Observables):


    def __init__(
        self,
        obs_name=None,
        save_figure_path=None,
        periods={
            'train_batch': 100,
            'val_batch': 100,
            'train_epoch': 2,
            'val_epoch': 2,
        },
        props={
            'train_batch': 1,
            'val_batch': 1,
            'train_epoch': .3,
            'val_epoch': .3,
        },
        background_to_hide=None,
        do_tasks={
            'train_batch': True,
            'val_batch': True,
            'train_epoch': True,
            'val_epoch': True,
        },
    ):
        """

        Args:
            model ([type]): [description]
            idx_features (int): last index + 1 of the features creator. For ResNet_N, it is -3.
            batch_size (int): Batch size to use when we evaluate the images in the net.
            save_figure_path (str, optional): Path to folder to save figures.
                                                 If None, does not save figures.Defaults to None.
            periods (dict, optional): Number of batch to wait before saving figures.
            props (dict, optional): Proportion of data to save.
            background_to_hide (int, optional): Background of the figure to ignore when showing. Defaults to -1.
        """

        super().__init__()

        self.save_figure_path = save_figure_path
        self.periods = periods
        self.props = props
        self.do_tasks = do_tasks
        self.ticks = {
            'train_batch': 0,
            'val_batch': 0,
            'train_epoch': 0,
            'val_epoch': 0,
        }
        self.bg_to_hide = background_to_hide

        self.obs_name = obs_name


    def _update_status(self, key, inputs, outputs, targets, dataloader, device):
        """ To complete
        """
        pass


    def _compute_key(self, key, inputs, outputs, targets, dataloader, device):
        if not self.do_tasks[key]:
            return
        self._update_status(key, inputs, outputs, targets, dataloader, device)

        if self.save_figure_path is None:
            return

        if 'epoch' in key:
            filename = 'End_Epoch-{}'.format(self.current_epoch + 1)
        else:
            filename = 'Epoch-{}_Batch-{}'.format(self.current_epoch + 1, self.current_batch_idx + 1)
        savepath = join(
            self.save_figure_path,
            '{}_{}'.format(key, self.obs_name),
            filename,
        )
        self._save_on_ticks(savepath, key=key)


    def compute_val_on_epoch(self, inputs, outputs, targets, dataloader, device):
        self._compute_key('val_epoch', inputs, outputs, targets, dataloader, device)


    def compute_val_on_batch(self, inputs, outputs, targets, dataloader, device):
        self._compute_key('val_batch', inputs, outputs, targets, dataloader, device)


    def compute_train_on_batch(self, inputs, outputs, targets):
        self._compute_key('train_batch', inputs, outputs, targets, dataloader=None, device=None)


    def compute_train_on_epoch(self, inputs, outputs, targets, dataloader, device):
        self._compute_key('train_epoch', inputs, outputs, targets, dataloader, device)


    def _save_on_ticks(self, savepath, key):
        if self.save_figure_path is None:
            return

        if self.ticks[key] % self.periods[key] != 0:
            # self.log_console('Saving {} {} in {} / {}'.format(
            #     key, self.obs_name, self.ticks[key] % self.periods[key], self.periods[key]))
            self.ticks[key] += 1
            return

        self._save_fig(savepath, self.props[key])
        self.ticks[key] += 1


    def show(self):
        super().show()
        for key in ['train_batch', 'val_batch']:
            if not self.do_tasks[key]:
                continue
            self.log_console('Saving {} {} in {} / {}'.format(
                key, self.obs_name, self.ticks[key] % self.periods[key], self.periods[key]))



    def _save_fig(self, savepath, prop):
        pathlib.Path(savepath).mkdir(exist_ok=True, parents=True)
        self.log_console('Saving {} figures in '.format(self.obs_name), savepath)

        # idexs = random.sample(range(len(self.cur_inputs)), int(len(self.cur_inputs)*prop))

        for idx in tqdm(range(len(self.cur_inputs))):
            fig = self._create_fig(idx)
            fig.savefig(join(savepath, '{}.png'.format(idx)))
            plt.close(fig)
        self.log_console('Saved {} figures in '.format(self.obs_name), savepath)


    def _create_fig(self, idx):
        """ To complete
        """
        fig, axs = plt.subplots(1, 1)
        return fig


class MetricsAndLoss(Observables):
    """
    Logger to store accuracies and uncertainties
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        save_weights_path=None,
        to_save_on='loss',
        use_recomputed_outputs=False,
    ):
        super(MetricsAndLoss, self).__init__()
        self.metrics = {}
        if type(metrics) == list:
            for metric in metrics:
                self.metrics[metric.__name__] = metric
        else:
            self.metrics = metrics

        self.logs = {
            'val_loss_on_batch': None,
            'val_loss_on_epoch': None,
        }

        for key, metric in self.metrics.items():
            self.logs.update({
                'train_{}_on_batch'.format(key): None,
                'train_{}_on_epoch'.format(key): 0,
                'val_{}_on_batch'.format(key): None,
                'val_{}_on_epoch'.format(key): None,
            })


        # self.max_train_metric_on_epoch = 0
        # self.epoch_with_max_train_metric = 0
        # self.validation_logging = False
        self.min_val_loss = np.infty
        self.max_metric = -np.infty
        self.best_weights_batch_epoch = None
        self.save_weights_path = save_weights_path
        self.model = model
        self.criterion = criterion
        self.to_save_on = to_save_on
        self.use_recomputed_outputs = use_recomputed_outputs

    def compute_train_on_batch(self, inputs, outputs, targets):
        """
        Logs we want to compute for each batch on train
        Args:
            outputs (torch.Tensor): size = (batch_size, number_of_classes): output of the model
            targets (torch.Tensor): size = (batch_size): true targets
        """
        for key, metric in self.metrics.items():
            metric_value = metric(outputs, targets)
            if type(metric_value) == np.ndarray and metric_value.ndim > 1:
                metric_value = metric_value.mean(0)
            self.logs['train_{}_on_batch'.format(key)] = metric_value
            self.add_to_history(['train_{}_on_batch'.format(key)])
            self.write_tensorboard(keys=['train_{}_on_batch'.format(key)])

    def compute_train_on_epoch(self, inputs=None, outputs=None, targets=None, dataloader=None, device='cpu'):
        """
        Logs we want to compute for each epoch on train
        Args:
            model (torch.nn.Module Child): model being trained
            trainloader (torch.utils.data.dataloader.DataLoader): dataloader of the train set
            device (torch.device || str): which device to compute on (either on GPU or CPU). Either torch.device type or
                                      specific string 'cpu' or 'gpu'. Will be the same device as the model.
        """
        if self.use_recomputed_outputs and outputs is not None:
            total_metrics = evaluate_metrics(outputs, targets, self.metrics)
        else:
            total_metrics = evaluate_metrics_mean(self.model, dataloader, self.metrics, device=device)


        for key, metric in total_metrics.items():
            if type(metric) == np.ndarray and metric.ndim > 1:
                metric = metric.mean(0)

            self.logs['train_{}_on_epoch'.format(key)] = metric

            self.add_to_history(['train_{}_on_epoch'.format(key)])
            self.write_tensorboard(keys=['train_{}_on_epoch'.format(key)])

    def compute_val(self, key_type, outputs=None, targets=None, dataloader=None, device='cpu', save_best_weights=True,):
        """
        Compute validation loss and metric and adds to history
        Args:
            model (torch.nn child): model to use to compute validation
            valloader (torch.utils.data.dataloader.DataLoader): dataloader of validation set
            device (torch.device): cpu or gpu
            key_type (str): epoch or batch
            save_best_weights (bool): whether we save weights or not
        """

        if self.use_recomputed_outputs and outputs is not None:
            val_loss = evaluate_loss(outputs, targets, self.criterion)
            val_metrics = evaluate_metrics(outputs, targets, self.metrics)
        else:
            val_loss = evaluate_loss_mean(self.model, dataloader, self.criterion, device=device)
            val_metrics = evaluate_metrics_mean(self.model, dataloader, self.metrics, device=device)


        if 'loss' in self.to_save_on:
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.best_weights_batch_epoch = (self.current_batch_idx, self.current_epoch)
                self.log_console('Updated best weights batch epoch:', self.best_weights_batch_epoch)
                if self.save_weights_path is not None and save_best_weights:
                    torch.save(self.model.state_dict(), join(self.save_weights_path, 'best_weights.pt'))
                    self.log_console('Updated best weights on ', self.to_save_on)
        else:
            if val_metrics[self.to_save_on] > self.max_metric:
                self.max_metric = val_metrics[self.to_save_on]
                self.best_weights_batch_epoch = (self.current_batch_idx, self.current_epoch)
                self.log_console('Updated best weights batch epoch:', self.best_weights_batch_epoch)
                if self.save_weights_path is not None and save_best_weights:
                    torch.save(self.model.state_dict(), join(self.save_weights_path, 'best_weights.pt'))
                    self.log_console('Updated best weights on ', self.to_save_on)

        for key in self.metrics.keys():
            metric_value = val_metrics[key]
            if type(metric_value) == np.ndarray and metric_value.ndim > 1:
                metric_value = metric_value.mean(0)
            self.logs['val_{}_on_{}'.format(key, key_type)] = metric_value
            self.add_to_history(['val_{}_on_{}'.format(key, key_type)])
            self.write_tensorboard(keys=['val_{}_on_{}'.format(key, key_type)])

        self.logs['val_loss_on_{}'.format(key_type)] = val_loss
        self.add_to_history([
            'val_loss_on_{}'.format(key_type),
        ])
        self.write_tensorboard(keys=[
            'val_loss_on_{}'.format(key_type),
        ])

    def compute_val_on_batch(self, inputs=None, outputs=None, targets=None,
     dataloader=None, device='cpu', save_best_weights=True,):
        """
        Compute validation loss and metric on one batch and adds to history
        Args:
            model (torch.nn child): model to use to compute validation
            valloader (torch.utils.data.dataloader.DataLoader): dataloader of validation set
            device (torch.device): cpu or gpu
            save_best_weights (bool): whether we save weights or not
        """

        self.compute_val(
            outputs=outputs,
            targets=targets,
            device=device,
            dataloader=dataloader,
            key_type='batch',
            save_best_weights=save_best_weights,
        )

    def compute_val_on_epoch(self, inputs=None, outputs=None, targets=None,
     dataloader=None, device='cpu', save_best_weights=True,):
        """
        Compute validation loss and metric on epoch and adds to history
        Args:
            model (torch.nn child): model to use to compute validation
            valloader (torch.utils.data.dataloader.DataLoader): dataloader of validation set
            device (torch.device): cpu or gpu
        """
        self.compute_val(
            outputs=outputs,
            targets=targets,
            device=device,
            dataloader=dataloader,
            key_type='epoch',
            save_best_weights=save_best_weights,
        )

    def show(self):
        super().show()
        self.log_console('Epochs Batch Best Weights: ', self.best_weights_batch_epoch)


class LRChecker(Observables):
    """
    Logger to check the evolution of learning rate
    """

    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.logs = {
            'lr': self._lr,
        }

    def update_lr(self):
        self.logs['lr'] = self._lr
        self.add_to_history([
            'lr'
        ])
        self.write_tensorboard()


    def compute_train_on_batch(self, *args, **kwargs):
        self.update_lr()


    @property
    def _lr(self):
        for param in self.optimizer.param_groups:
            return param['lr']


class MemoryChecker(Observables):

    def __init__(self):
        super().__init__()
        self.logs = {
            'memory_allocated': None,
            'max_memory_cached': None,
        }

    def check_memory(self):
        self.logs['memory_allocated'] = torch.cuda.memory_allocated()
        self.add_to_history(['memory_allocated'])
        self.write_tensorboard(keys=['memory_allocated'])

        self.logs['max_memory_cached'] = torch.cuda.max_memory_cached()
        self.add_to_history(['max_memory_cached'])
        self.write_tensorboard(keys=['max_memory_cached'])

    def compute_train_on_batch(self, *args, **kwargs):
        self.check_memory()

    def compute_train_on_epoch(self, *args, **kwargs):
        self.check_memory()

    def compute_val_on_batch(self, *args, **kwargs):
        self.check_memory()

    def compute_val_on_epoch(self, *args, **kwargs):
        self.check_memory()


class ConfusionMatrix(Observables):

    def __init__(
        self,
        save_csv_path=None,
        threshold=None,
        n_classes=2,
        metrics={},
    ):
        super().__init__()
        self.logs = {
            'train_on_batch': None,
            'train_on_epoch': None,
            'val_on_batch': None,
            'val_on_epoch': None,
        }
        self.save_csv_path = save_csv_path
        self.threshold = threshold
        self.n_classes = n_classes
        self.labels = range(n_classes)
        self.metrics = metrics


        if self.save_csv_path is not None:
            self.paths = {
                'train_on_batch': join(save_csv_path, 'train_batch'),
                'train_on_epoch': join(save_csv_path, 'train_epoch'),
                'val_on_batch': join(save_csv_path, 'val_batch'),
                'val_on_epoch': join(save_csv_path, 'val_epoch'),
            }
            for path in self.paths.values():
                pathlib.Path(path).mkdir(exist_ok=True, parents=True)

        self.current_targets = {
            'train_on_batch': None,
            'train_on_epoch': None,
            'val_on_batch': None,
            'val_on_epoch': None,
        }

        self.y_true = {
            'train_on_batch': None,
            'train_on_epoch': None,
            'val_on_batch': None,
            'val_on_epoch': None,
        }
        self.y_pred = {
            'train_on_batch': None,
            'train_on_epoch': None,
            'val_on_batch': None,
            'val_on_epoch': None,
        }



    def compute_train_on_batch(self, inputs, outputs, targets):
        self._compute_and_save(
            outputs,
            targets,
            'train_on_batch',
            'cm_batch_train-{}.csv'.format(self.current_batch_idx),
        )

    def compute_train_on_epoch(self, inputs, outputs, targets, device, dataloader=None):
        self._compute_and_save(outputs, targets, 'train_on_epoch', 'cm_epoch_train-{}.csv'.format(self.current_epoch))

    def compute_val_on_batch(self, inputs, outputs, targets, device, dataloader=None):
        self._compute_and_save(outputs, targets, 'val_on_batch', 'cm_batch_val-{}.csv'.format(self.current_batch_idx))

    def compute_val_on_epoch(self, inputs, outputs, targets, device, dataloader=None):
        self._compute_and_save(outputs, targets, 'val_on_epoch', 'cm_epoch_val-{}.csv'.format(self.current_epoch))

    def _compute_and_save(self, outputs, targets, key, filename):
        outputs = outputs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        self.current_targets[key] = targets
        if self.n_classes == 2 and self.threshold is not None:
            if outputs.squeeze().ndim == 2:
                y_pred = outputs[:, 1] > self.threshold
            elif outputs.squeeze().ndim == 1:
                y_pred = (outputs > self.threshold) + 0
        else:
            y_pred = outputs.argmax(1)

        if targets.squeeze().ndim == 2:
            y_true = targets.argmax(1)
        elif targets.squeeze().ndim == 1:
            y_true = targets + 0

        self.y_pred[key] = y_pred
        self.y_true[key] = y_true

        self.logs[key] = confusion_matrix(y_true, y_pred, labels=self.labels)
        self.add_to_history([key])
        self.save_to_csv(key, filename)



    def _get_dataframe(self, key):
        cm = self.logs[key]
        y_pred = self.y_pred[key]
        y_true = self.y_true[key]
        assert cm is not None, self.log_console(key)
        assert y_pred is not None, self.log_console(key)
        assert y_true is not None, self.log_console(key)

        cmdf = confusion_df(y_pred, y_true, metrics=self.metrics)

        return cmdf

    def save_to_csv(self, key, filename):
        if self.save_csv_path is not None:
            self._get_dataframe(key).to_csv(join(self.paths[key], filename))

    def show(self):
        t1 = time()
        self.log_console(
            'train batch confusion computed on: {}\n'.format(time() - t1),
            self._get_dataframe('train_on_batch'),
        )
        if self.logs['val_on_batch'] is not None:
            t2 = time()
            self.log_console(
                'validation batch confusion computed on: {}\n'.format(time() - t2),
                self._get_dataframe('val_on_batch'),
            )
        if self.logs['train_on_epoch'] is not None:
            t3 = time()
            self.log_console(
                'train epoch confusion computed on: {}\n'.format(time() - t3),
                self._get_dataframe('train_on_epoch'),
            )

        if self.logs['val_on_epoch'] is not None:
            t4 = time()
            self.log_console(
                'val epoch confusion computed on: {}\n'.format(time() - t4),
                self._get_dataframe('val_on_epoch'),
            )

    def init_tensorboard_writer(self, path=None):
        pass


class Activations(Observables):

    def __init__(self, show_train_batch=True, do_tensorboard=False):
        super().__init__()
        self.logs = {
            'activation_train_on_batch': None,
            'activation_train_on_epoch': None,
            'activation_val_on_batch': None,
            'activation_val_on_epoch': None,
        }

        log_keys = list(self.logs.keys())
        for key in log_keys:
            self.logs[key + '-mean'] = None
            self.logs[key + '-std'] = None

        self.show_train_batch = show_train_batch
        self.do_tensorboard = do_tensorboard

    def _evaluate_activation(self, outputs, key):
        key = 'activation_' + key
        outputs = outputs.squeeze()
        if outputs.ndim == 2:
            outputs = outputs[:, 1] + 0

        self.logs[key] = outputs.detach().cpu().numpy()
        self.logs[key + '-mean'] = self.logs[key].mean()
        self.logs[key + '-std'] = self.logs[key].std()

        self.add_to_history([
            key,
            key + '-mean',
            key + '-std',
        ])
        self.write_tensorboard(keys=[
            key,
            key + '-mean',
            key + '-std',
        ])

    def compute_train_on_batch(self, inputs, outputs, targets, ):
        self._evaluate_activation(outputs, 'train_on_batch')

    def compute_train_on_epoch(self, inputs, outputs, targets, device, dataloader=None):
        self._evaluate_activation(outputs, 'train_on_epoch')

    def compute_val_on_batch(self, inputs, outputs, targets, device, dataloader=None):
        self._evaluate_activation(outputs, 'val_on_batch')

    def compute_val_on_epoch(self, inputs, outputs, targets, device, dataloader=None):
        self._evaluate_activation(outputs, 'val_on_epoch')

    def show(self):

        if self.show_train_batch:
            self.log_console('Activations train batch:', self.logs['train_on_batch'])
        for key in ['train_on_batch', 'val_on_batch', 'train_on_epoch']:
            if self.logs['activation_' + key] is not None:
                self.log_console('Activations {}: mean {}, std {}'.format(
                    key, self.logs['activation_' + key+'-mean'], self.logs['activation_' + key+'-std'])
                )

    def init_tensorboard_writer(self, path=None, keys=None):
        if self.do_tensorboard:
            keys = set(self.logs.keys()).difference([
                'activation_train_on_batch',
                'activation_train_on_epoch',
                'activation_val_on_batch',
                'activation_val_on_epoch',
            ])
            super().init_tensorboard_writer(path=path, keys=keys)


class CheckLayers(Observables):

    def __init__(self, model, layers_set={}):
        super().__init__()
        self.model = model
        self.layers_set = layers_set

        if type(layers_set) != dict:
            self.layers_set = {k: v for (k, v) in enumerate(layers_set)}


        self.logs = {
            'other_layers-weights_sum': 0,
        }

        for key in self.layers_set.keys():
            self.logs[key + '-weights_sum'] = 0

    def compute_train_on_batch(self, *args, **kwargs):
        res = dict.fromkeys(self.layers_set.keys(), 0)
        res['other_layers'] = 0
        for param in self.model.parameters():
            res['other_layers'] += param.abs().sum().item()

        for key, layers in self.layers_set.items():
            for param in getattr(self.model, layers).parameters():
                sum_params = param.abs().sum().item()
                res['other_layers'] -= sum_params
                res[key] += sum_params

        for key, value in res.items():
            self.logs[key + '-weights_sum'] = value

        self.add_to_history()
        self.write_tensorboard()

    # def show(self):
    #     self.log_console('Sum of layers minus {}:'.format(self.layers_avoid), list(self.logs.values())[0])


class ShowImages(ObservableImage):


    def __init__(self, model=None, *args, **kwargs):
        super().__init__(obs_name='images', *args, **kwargs)

        self.model = model

        self.cur_inputs = None
        self.cur_outputs = None
        self.cur_targets = None


    def _update_status(self, key, inputs, outputs, targets, dataloader, device):

        if inputs is not None:

            len_data = len(inputs)
            idexs = random.sample(range(len_data), int(len_data * self.props[key]))

            self.cur_inputs = list(inputs.detach().cpu()[idexs])
            self.cur_outputs = list(outputs.detach().cpu()[idexs])
            self.cur_targets = list(targets.detach().cpu()[idexs])

        else:

            self.cur_inputs = []

            self.model.to(device)
            len_data = len(dataloader.dataset)
            idexs = random.sample(range(len_data), int(len_data * self.props[key]))
            with torch.no_grad():
                # for inputs, targets in dataloader:
                for idx in idexs:
                    inputs, targets = dataloader.dataset[idx]
                    inputs = inputs.unsqueeze(0).to(device)
                    targets = targets.unsqueeze(0).to(device)

                    outputs = self.model(inputs)

                    self.cur_inputs += list(inputs.detach().cpu())
                    self.cur_outputs += list(outputs.detach().cpu())
                    self.cur_targets += list(targets.detach().cpu())


    def _create_fig(self, idx):

        img = self.cur_inputs[idx]
        output = self.cur_outputs[idx]
        target = self.cur_targets[idx]
        nchans = img.shape[0]

        fig, axs = plt.subplots(1, nchans, figsize=(nchans*5, 5))

        fig.suptitle('Pred: {}. True: {}.'.format(output, target))
        for chan in range(nchans):
            axs[chan].imshow(img[chan], cmap='gray')
        # axs[0].set_title('True')
        # axs[1].set_title('Pred')
        # fig.suptitle('Epoch: {}. Batch: {}'.format(self.current_epoch+1, self.current_batch_idx+1))
        # if len(img.shape) == 2:
        #     plot_img_mask_on_ax(axs[0], img, mask_true)
        #     plot_img_mask_on_ax(axs[1], img, mask_pred)
        # elif len(img.shape) == 3:
        #     plot_img_mask_on_ax(axs[0], img[img.shape[0]//2], mask_true)
        #     plot_img_mask_on_ax(axs[1], img[img.shape[0]//2], mask_pred)

        return fig


class GradientsLossInputs(Observables):

    def __init__(self, background=None, save_figure_path=None, period=10):
        super().__init__()
        self.bg = background
        self.logs = {
            'gradient_input_sum': None,
            'gradient_input_sum_bg': None,
        }

        self.period = period
        self.ticks = 0

        self.save_figure_path = save_figure_path
        self.cur_inputs = None
        self.cur_grads = None


    def compute_train_on_batch(self, inputs, outputs, targets):
        self.cur_grads = inputs.grad
        self.cur_inputs = inputs.detach().cpu().numpy()
        if self.bg is None:
            self.logs['gradient_input_sum'] = self.cur_grads.abs().sum()
        else:
            self.logs['gradient_input_sum'] = self.cur_grads[inputs != self.bg].abs().sum()
            self.logs['gradient_input_sum_bg'] = self.cur_grads[inputs == self.bg].abs().sum()

        self.add_to_history([
            'gradient_input_sum',
            'gradient_input_sum_bg',
        ])
        self.write_tensorboard()


    def show(self):
        super().show()
        if self.save_figure_path is None:
            return

        if self.ticks % self.period != 0:
            self.log_console('Saving gradients in {} / {}'.format(self.ticks % self.period, self.period))
            self.ticks += 1
            return

        savepath = join(
            self.save_figure_path,
            'images',
            'training_batches_grads',
            'Epoch-{}_Batch-{}'.format(self.current_epoch + 1, self.current_batch_idx + 1),
        )
        self._save_fig(savepath)
        self.ticks += 1

    def compute_train_on_epoch(self, inputs, outputs, targets, device):
        if self.save_figure_path is None:
            return

        savepath = join(
            self.save_figure_path,
            'images',
            'training_batches_grads',
            'End_Epoch-{}'.format(self.current_epoch + 1),
        )
        self._save_fig(savepath)

    def _save_fig(self, savepath):


        pathlib.Path(savepath).mkdir(exist_ok=True, parents=True)
        self.log_console('Saving training figures in ', savepath)
        for idx, (img, grad) in enumerate(zip(self.cur_inputs, self.cur_grads)):
            fig, axs = plt.subplots(2, img.shape[0], figsize=(10, 10))
            for chan in range(img.shape[0]):
                axs[0, chan].imshow(img[chan], cmap='gray')
                axs[0, chan].set_title('Input')
                axs[1, chan].imshow(grad[chan].abs().cpu(), cmap='gray')
                axs[1, chan].set_title('Gradient')

            fig.savefig(join(savepath, '{}.png'.format(idx)))
            plt.close(fig)
        self.log_console('Saved training figures in ', savepath)


class SaliencyMaps(Observables):
    """
    Implements Saliency Maps: <https://arxiv.org/abs/1312.6034>

    """

    def __init__(
        self,
        save_figure_path=None,
        periods={
            'train_on_batch': 100,
            'val_on_batch': 100,
            'train_on_epoch': 2,
            'val_on_epoch': 2,
        },
        props={
            'train_on_batch': 1,
            'val_on_batch': 1,
            'train_on_epoch': .3,
            'val_on_epoch': .3,
        },
        background_to_hide=-1,
        do_tasks={
            'train_on_batch': True,
            'val_on_batch': False,
            'train_on_epoch': True,
            'val_on_epoch': True,
        },
        threshold=.5,
    ):

        super().__init__()
        self.cur_inputs = None
        self.cur_outputs = None
        self.cur_preds = None
        self.cur_targets = None
        self.cur_grads = None


        self.threshold = threshold

        self.bg_to_hide = background_to_hide
        self.save_figure_path = save_figure_path

        self.props = props
        self.periods = periods
        self.ticks = {
            'train_on_batch': 0,
            'val_on_batch': 0,
            'train_on_epoch': 0,
            'val_on_epoch': 0,
        }
        self.do_tasks = do_tasks

    # TODO: put this function in another file, independent from the class
    # TODO: remove dependency on original inputs (e.g. recompute outputs)
    def compute_saliency_maps(self, inputs, outputs):
        if outputs.squeeze().ndim > 1:
            self.cur_preds = outputs.argmax(1)
            grad_weights = torch.zeros_like(outputs)
            grad_weights[np.arange(outputs.shape[0]), self.cur_preds] = 1
            outputs.backward(grad_weights)
            self.cur_grads = inputs.grad
        else:
            self.cur_preds = 2 * (outputs > self.threshold) - 1       # preds are either 1 or -1

            outputs.backward(torch.ones_like(outputs))
            self.cur_grads = inputs.grad * (self.cur_preds[..., None, None])

        self.cur_grads = self.cur_grads.detach().cpu()


    def _update_status(self, inputs, outputs, targets):
        self.cur_inputs = inputs.detach().cpu()
        self.cur_outputs = outputs.detach().cpu()
        self.cur_targets = targets.detach().cpu()

        self.compute_saliency_maps(inputs, outputs)


    def compute_train_on_batch(self, inputs, outputs, targets):
        if not self.do_tasks['train_on_batch']:
            return
        self._update_status(inputs, outputs, targets)

        if self.save_figure_path is None:
            return

        savepath = join(
            self.save_figure_path,
            'images',
            'training_batches_saliency',
            'Epoch-{}_Batch-{}'.format(self.current_epoch + 1, self.current_batch_idx + 1),
        )

        self._save_on_ticks(savepath, key='train_on_batch')


    def compute_train_on_epoch(self, inputs, outputs, targets, device):
        if not self.do_tasks['train_on_epoch']:
            return
        self._update_status(inputs, outputs, targets)


        if self.save_figure_path is None:
            return

        savepath = join(
            self.save_figure_path,
            'images',
            'train_saliency',
            'End_Epoch-{}'.format(self.current_epoch + 1),
        )
        self._save_on_ticks(savepath, key='train_on_epoch')


    def compute_val_on_batch(self, inputs, outputs, targets, device):
        if not self.do_tasks['val_on_batch']:
            return
        self._update_status(inputs, outputs, targets)

        if self.save_figure_path is None:
            return

        savepath = join(
            self.save_figure_path,
            'images',
            'val_batch_saliency',
            'Epoch-{}_Batch-{}'.format(self.current_epoch + 1, self.current_batch_idx + 1),
        )
        self._save_on_ticks(savepath, key='val_on_batch')



    def compute_val_on_epoch(self, inputs, outputs, targets, device):
        if not self.do_tasks['val_on_epoch']:
            return
        self._update_status(inputs, outputs, targets)

        if self.save_figure_path is None:
            return

        savepath = join(
            self.save_figure_path,
            'images',
            'val_saliency',
            'End_Epoch-{}'.format(self.current_epoch + 1),
        )
        self._save_on_ticks(savepath, key='val_on_epoch')


    def show(self):
        super().show()


    def _save_on_ticks(self, savepath, key):
        if self.save_figure_path is None:
            return

        if self.ticks[key] % self.periods[key] != 0:
            self.log_console('Saving gradients in {} / {}'.format(self.ticks[key] % self.periods[key], self.periods[key]))
            self.ticks[key] += 1
            return

        self._save_fig(savepath, self.props[key])
        self.ticks[key] += 1


    def _save_fig(self, savepath, prop):


        pathlib.Path(savepath).mkdir(exist_ok=True, parents=True)
        self.log_console('Saving figures in ', savepath)

        idexs = random.sample(range(len(self.cur_inputs)), int(len(self.cur_inputs)*prop))

        for idx in tqdm(idexs):
            img = self.cur_inputs[idx] + 0
            img[img == self.bg_to_hide] = img[img != self.bg_to_hide].min()
            grad = self.cur_grads[idx]
            fig, axs = plt.subplots(3, img.shape[0], figsize=(img.shape[0]*5, 3*5))
            for chan in range(img.shape[0]):
                axs[0, chan].imshow(img[chan], cmap='gray')
                axs[0, chan].set_title('Input')
                axs[1, chan].imshow(grad[chan], cmap='gray')
                axs[1, chan].set_title('Gradient')
                axs[2, chan].imshow(np.maximum(grad[chan], 0), cmap='gray')
                axs[2, chan].set_title('Max(Gradient, 0)')

            fig.suptitle('Outputs: {}. Prediction: {}. Targets: {}'.format(
                self.cur_outputs[idx].squeeze(),
                self.cur_preds[idx].squeeze().item(),
                self.cur_targets[idx].squeeze().item(),
            ))
            fig.savefig(join(savepath, '{}.png'.format(idx)))
            plt.close(fig)
        self.log_console('Saved figures in ', savepath)


class ShowImageSegmentation(ObservableImage):

    def __init__(self, model=None, *args, **kwargs):
        super().__init__(obs_name='segmentation', *args, **kwargs)

        self.model = model

        self.cur_inputs = None
        self.mask_true = None
        self.mask_pred = None

    def _update_status(self, key, inputs, outputs, targets, dataloader, device):

        if inputs is not None:

            len_data = len(inputs)
            idexs = random.sample(range(len_data), int(len_data * self.props[key]))

            self.cur_inputs = list(inputs.detach().cpu()[idexs])
            self.mask_true = list(targets.detach().cpu()[idexs])
            self.mask_pred = list(outputs.argmax(1).detach().cpu()[idexs])




        else:

            self.cur_inputs = []
            self.mask_true = []
            self.mask_pred = []

            self.model.to(device)
            len_data = len(dataloader.dataset)
            idexs = random.sample(range(len_data), int(len_data * self.props[key]))
            with torch.no_grad():
                # for inputs, targets in dataloader:
                for idx in idexs:
                    inputs, targets = dataloader.dataset[idx]
                    inputs, targets = inputs.unsqueeze(0).to(device), targets.unsqueeze(0).to(device)
                    outputs = self.model(inputs)

                    self.cur_inputs += list(inputs.detach().cpu())
                    self.mask_true += list(targets.int().detach().cpu())
                    self.mask_pred += list(outputs.argmax(1).detach().cpu())


    def _create_fig(self, idx):

        img = self.cur_inputs[idx]
        mask_pred = self.mask_pred[idx]
        mask_true = self.mask_true[idx]

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].set_title('True')
        axs[1].set_title('Pred')
        fig.suptitle('Epoch: {}. Batch: {}'.format(self.current_epoch+1, self.current_batch_idx+1))
        if len(img.shape) == 2:
            plot_img_mask_on_ax(axs[0], img, mask_true)
            plot_img_mask_on_ax(axs[1], img, mask_pred)
        elif len(img.shape) == 3:
            plot_img_mask_on_ax(axs[0], img[img.shape[0]//2], mask_true)
            plot_img_mask_on_ax(axs[1], img[img.shape[0]//2], mask_pred)

        return fig


class GradCAM(Observables):
    """
    Implements Grad-CAM: <https://arxiv.org/abs/1610.02391>

    """

    def __init__(
        self,
        model,
        batch_size,
        idx_features=-3,
        save_figure_path=None,
        periods={
            'train_on_batch': 100,
            'val_on_batch': 100,
            'train_on_epoch': 2,
            'val_on_epoch': 2,
        },
        props={
            'train_on_batch': 1,
            'val_on_batch': 1,
            'train_on_epoch': .3,
            'val_on_epoch': .3,
        },
        background_to_hide=-1,
        do_tasks={
            'train_on_batch': True,
            'val_on_batch': False,
            'train_on_epoch': True,
            'val_on_epoch': True,
        },
    ):
        """

        Args:
            model ([type]): [description]
            idx_features (int): last index + 1 of the features creator. For ResNet_N, it is -3.
            batch_size (int): Batch size to use when we evaluate the images in the net.
            save_figure_path (str, optional): Path to folder to save figures.
                                                 If None, does not save figures.Defaults to None.
            periods (dict, optional): Number of batch to wait before saving figures.
            props (dict, optional): Proportion of data to save.
            background_to_hide (int, optional): Background of the figure to ignore when showing. Defaults to -1.
        """

        super().__init__()

        self.save_figure_path=save_figure_path
        self.periods=periods
        self.props=props
        self.do_tasks = do_tasks
        self.ticks = {
            'train_on_batch': 0,
            'val_on_batch': 0,
            'train_on_epoch': 0,
            'val_on_epoch': 0,
        }
        self.bg_to_hide = background_to_hide

        self.model = model
        self.idx_features = idx_features
        children = list(self.model.children())
        self.fsel = nn.Sequential(*children[:self.idx_features])
        self.frest = nn.Sequential(*children[self.idx_features:])
        self.batch_size = batch_size

        self.cur_inputs = None
        self.cur_outputs = None
        self.cur_targets = None
        self.cur_preds = None

        self.gradcam = None
        self.guided_bp = None
        self.hooks_handles = []

    # TODO: put this function in another file, independent from the class
    def compute_gradcam(self, inputs):
        inputs = inputs.clone().detach()
        inputs.requires_grad = True
        inputs.grad = torch.zeros_like(inputs)

        idx_start = 0
        gradcams = []
        guided_bps = []
        all_preds = []

        self._update_relus()
        for _ in range(len(inputs) // self.batch_size + 1):
            if idx_start >= len(inputs):
                break
            imgs = inputs[idx_start:idx_start + self.batch_size]

            # Compute features activations
            features = self.fsel(imgs)
            features.retain_grad()
            outputs = self.frest(features)

            # Guided backprop
            cur_preds = outputs.argmax(1)
            all_preds.append(cur_preds)
            grad_weights = torch.zeros_like(outputs)
            grad_weights[np.arange(outputs.shape[0]), cur_preds] = 1
            outputs.backward(grad_weights)

            # GradCAM
            wg = features.grad.mean((2, 3))
            gradcam = F.relu((features * wg[..., None, None]).sum(1).detach())
            gradcams.append(gradcam[:, None, ...])  # Put one channel to the batch
            guided_bps.append(inputs.grad[idx_start:idx_start + self.batch_size])
            idx_start = idx_start + self.batch_size

        self.gradcam = torch.cat(gradcams, axis=0).detach()
        self.guided_bp = torch.cat(guided_bps, axis=0).detach()
        self.cur_preds = torch.cat(all_preds, axis=0).detach()

        self._reset_hooks()

    def _update_status(self, inputs, outputs, targets):

        self.cur_inputs = inputs.detach().cpu()
        self.cur_outputs = outputs.detach().cpu()
        self.cur_targets = targets.detach().cpu()

        self.compute_gradcam(inputs)


    def _update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function

        for module in self.model.modules():
            if isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU):
                self.hooks_handles.append(module.register_backward_hook(relu_hook_function))


    def _reset_hooks(self):
        for handle in self.hooks_handles:
            handle.remove()
        self.hooks_handles = []


    def _save_fig(self, savepath, prop):


        pathlib.Path(savepath).mkdir(exist_ok=True, parents=True)
        self.log_console('Saving figures in ', savepath)

        idexs = random.sample(range(len(self.cur_inputs)), int(len(self.cur_inputs)*prop))

        for idx in tqdm(idexs):
            img = self.cur_inputs[idx] + 0
            img[img == self.bg_to_hide] = img[img != self.bg_to_hide].min()
            gradcam = F.upsample(self.gradcam[idx][None, ...], img.shape[-1], mode='bicubic')
            guided_bp = self.guided_bp[idx]
            fig, axs = plt.subplots(5, img.shape[0], figsize=(img.shape[0]*5, 5*5))
            for chan in range(img.shape[0]):
                axs[0, chan].imshow(img[chan], cmap='gray')
                axs[0, chan].set_title('Input')
                axs[1, chan].imshow(gradcam[0, 0].cpu(), cmap='jet')
                axs[1, chan].set_title('GradCAM')
                axs[2, chan].imshow(guided_bp[chan].cpu(), cmap='gray')
                axs[2, chan].set_title('Guided Backprop')
                plot_img_mask_on_ax(axs[3, chan], img[chan], (gradcam[0, 0] * guided_bp[chan]).cpu())
                axs[3, chan].set_title('Guided GradCAM on image')
                plot_img_mask_on_ax(axs[4, chan], img[chan], F.relu(gradcam[0, 0] * guided_bp[chan]).cpu())
                axs[4, chan].set_title('Relu Guided GradCAM')



            fig.suptitle('Outputs: {}. Prediction: {}. Targets: {}'.format(
                self.cur_outputs[idx].squeeze(),
                self.cur_preds[idx].squeeze().item(),
                self.cur_targets[idx].squeeze().item(),
            ))
            fig.savefig(join(savepath, '{}.png'.format(idx)))
            plt.close(fig)
        self.log_console('Saved figures in ', savepath)


    def compute_val_on_epoch(self, inputs, outputs, targets, device):
        if not self.do_tasks['val_on_epoch']:
            return
        self._update_status(inputs, outputs, targets)

        if self.save_figure_path is None:
            return

        savepath = join(
            self.save_figure_path,
            'images',
            'val_gradcam',
            'End_Epoch-{}'.format(self.current_epoch + 1),
        )
        self._save_on_ticks(savepath, key='val_on_epoch')


    def compute_val_on_batch(self, inputs, outputs, targets, device):
        if not self.do_tasks['val_on_batch']:
            return
        self._update_status(inputs, outputs, targets)

        if self.save_figure_path is None:
            return

        savepath = join(
            self.save_figure_path,
            'images',
            'val_batch_gradcam',
            'Epoch-{}_Batch-{}'.format(self.current_epoch + 1, self.current_batch_idx + 1),
        )
        self._save_on_ticks(savepath, key='val_on_batch')


    def compute_train_on_batch(self, inputs, outputs, targets):
        if not self.do_tasks['train_on_batch']:
            return
        self._update_status(inputs, outputs, targets)

        if self.save_figure_path is None:
            return

        savepath = join(
            self.save_figure_path,
            'images',
            'training_batches_gradcam',
            'Epoch-{}_Batch-{}'.format(self.current_epoch + 1, self.current_batch_idx + 1),
        )

        self._save_on_ticks(savepath, key='train_on_batch')

    def show(self):
        super().show()


    def _save_on_ticks(self, savepath, key):
        if self.save_figure_path is None:
            return

        if self.ticks[key] % self.periods[key] != 0:
            self.log_console('Saving gradients in {} / {}'.format(self.ticks[key] % self.periods[key], self.periods[key]))
            self.ticks[key] += 1
            return

        self._save_fig(savepath, self.props[key])
        self.ticks[key] += 1
