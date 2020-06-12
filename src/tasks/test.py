import pathlib
from os.path import join

import torch
from tqdm import tqdm
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils import log_console, isnan
from .utils import EmptyContextManager
from src.plotter import plot_img_mask_on_ax


def test(*args, **kwargs):
    return 0, 0


def evaluate_model_by_batch(
    model,
    testloader,
    criterion,
    metrics_orig,
    number_of_batches=None,
    observables=[],
    grad_in_eval=False,
    show_figure=False,
    device='cpu',
    verbose=False,
    logger=None,
    task='classification',
    args_task={},
    savefig_path=None,
    **figure_args
):
    """
    Evaluates a model on a testloader by batch.

    Args:
        model (nn.Module child): model to evaluate
        testloader (torch.utils.data.dataloader.DataLoader): loader of testset
        criterion (function): loss function
        metrics_orig (dict | list): metrics to evaluate
        number_of_batches (int, optional): Numbero of batches to evaluate. If None given, all batches are tested.
        show_figure (bool, optional): If the task is segmentation, put True to see the figures. Defaults to False.
        device (str, optional): Device to compute on. Defaults to 'cpu'.
        verbose (bool, optional): Show progress bar. Defaults to False.

    Returns:
        float, dict, figure, (tensor, tensor): loss, metrics evaluated, figure, outputs, targets
    """
    model.eval()
    if number_of_batches is None:
        number_of_batches = len(testloader)

    metrics = {}
    if type(metrics_orig) == list:
        for metric in metrics_orig:
            metrics[metric.__name__] = metric
    else:
        metrics = metrics_orig.copy()

    res = testloader.dataset.data.copy()
    # res.to_csv('inter_res.csv')
    # assert False
    # res = res.drop(columns=['pixel_array', 't1_array', 'stir_array', ])
    res['output_loss'] = None
    res['targets'] = None
    res['preds'] = None
    for key, metric in metrics.items():
        res['metric_{}'.format(key)] = None
    res['batch_idx'] = 0

    for obs in observables:
        obs.set_number_of_batch(number_of_batches)
        obs.set_number_of_epoch(1)
        obs.set_current_epoch(0)
    cur_idx = 0
    grad_manager = EmptyContextManager if grad_in_eval else torch.no_grad
    with grad_manager():

        for batch_idx, (inputs, targets) in enumerate(testloader):
            if verbose:
                log_console('============', logger=logger)
                log_console('Batch {} / {}'.format(batch_idx + 1, number_of_batches), logger=logger)
            if batch_idx >= number_of_batches:
                log_console('{} >= {} : Stop testing.'.format(batch_idx, number_of_batches), logger=logger)
                break
            inputs = inputs.to(device)
            targets = targets.to(device)

            inputs.requires_grad = grad_in_eval

            outputs = model(inputs)

            for obs in observables:
                obs.set_current_batch_idx(batch_idx)
                obs.compute_val_on_batch(
                    inputs=inputs,
                    outputs=outputs,
                    targets=targets,
                    device=device,
                )



            batch_size = len(inputs)

            if task.lower() == 'classification':
                results_writer = _write_results_classification
            elif task.lower() == 'segmentation':
                results_writer = _write_results_segmentation
            else:
                assert False, 'Task not recognized.'

            res = results_writer(
                res,
                cur_idx,
                batch_idx,
                batch_size,
                criterion,
                metrics,
                targets,
                outputs,
                inputs,
                logger=logger,
                **args_task,
            )

            cur_idx += batch_size

    return res


def _write_results_classification(
    results,
    cur_idx,
    batch_idx,
    batch_size,
    criterion,
    metrics,
    targets,
    outputs,
    inputs,
    logger,
    *args,
    **kwargs
):
    res = results.copy()


    prev_reduc = criterion.reduction
    criterion.reduction = 'none'
    res.output_loss.iloc[cur_idx:cur_idx + batch_size] = criterion(outputs, targets).detach().cpu().numpy()
    criterion.reduction = prev_reduc
    for i in range(outputs.shape[-1]):
        if 'outputs-{}'.format(i) not in res.columns:
            res['outputs-{}'.format(i)] = torch.zeros(len(res))
        res['outputs-{}'.format(i)].iloc[cur_idx:cur_idx + batch_size] = outputs[..., i].detach().cpu().numpy()
    res.targets.iloc[cur_idx:cur_idx + batch_size] = targets.detach().cpu().numpy()
    res.preds.iloc[cur_idx:cur_idx + batch_size] = outputs.argmax(1).detach().cpu().numpy()
    res.batch_idx.iloc[cur_idx:cur_idx + batch_size] = batch_idx

    for key, metric in metrics.items():
        value_metric = metric(outputs, targets)
        # res['metric_{}'.format(key)].iloc[cur_idx:cur_idx + batch_size] = metric(outputs, targets)
        if type(value_metric) == np.ndarray:
            res['metric_{}'.format(key)].iloc[cur_idx:cur_idx + batch_size] = value_metric.mean()
            for idx_met, met in enumerate(value_metric):
                res['metric_{}_{}'.format(key, idx_met)].iloc[cur_idx:cur_idx + batch_size] = met
        else:
            res['metric_{}'.format(key)].iloc[cur_idx:cur_idx + batch_size] = value_metric

    return res


def _write_results_segmentation(
    results,
    cur_idx,
    batch_idx,
    batch_size,
    criterion,
    metrics,
    targets,
    outputs,
    inputs,
    logger,
    savefig_path=None,
    *args,
    **kwargs
    # period=1,
):

    res = results.copy()

    if not isnan(targets):
        prev_reduc = criterion.reduction
        criterion.reduction = 'none'
        output_loss = criterion(outputs, targets).mean((1, 2)).detach().cpu().numpy()
        res.output_loss.iloc[cur_idx:cur_idx + batch_size] = output_loss
        criterion.reduction = prev_reduc

        targets = targets.detach().cpu()

    outputs = outputs.detach().cpu()
    preds = outputs.argmax(1).detach().cpu().numpy()
    inputs = inputs.detach().cpu()



    for idx in tqdm(range(batch_size)):

        res.at[cur_idx + idx, 'batch_idx'] = batch_idx


        figtitle = ''

        if not isnan(targets):
            for key, metric in metrics.items():
                value_metric = metric(outputs[idx].unsqueeze(0), targets[idx].unsqueeze(0))[0]
                # res['metric_{}'.format(key)].iloc[cur_idx + idx] = metric(outputs, targets)
                if type(value_metric) == np.ndarray and value_metric.ndim > 1:
                    for idx_met, met in enumerate(value_metric):
                        res.at[cur_idx + idx, 'metric_{}_{}'.format(key, idx_met)] = met
                        figtitle += '{}-{}: {} | '.format(key, idx_met, met)
                else:
                    res.at[cur_idx + idx, 'metric_{}'.format(key)] = value_metric
                    figtitle += '{}: {} | '.format(key, value_metric)

        if savefig_path is not None:
            img = inputs[idx].numpy()

            patient = res.at[cur_idx + idx, 'patient']
            instance_number = res.at[cur_idx + idx, 'InstanceNumber']

            pathlib.Path(join(savefig_path, patient, 'png')).mkdir(exist_ok=True, parents=True)
            for fold in ['input', 'output']:
                pathlib.Path(join(savefig_path, patient, 'npy', fold)).mkdir(exist_ok=True, parents=True)
            np.save(join(savefig_path, patient, 'npy', 'input', 'inst{}.npy'.format(instance_number)), img)
            np.save(join(savefig_path, patient, 'npy', 'output', 'inst{}.npy'.format(instance_number)), outputs[idx].numpy())

            res.at[cur_idx + idx, 'input_path'] = join(savefig_path, patient, 'npy', 'input', 'inst{}.npy'.format(instance_number))
            res.at[cur_idx + idx, 'output_path'] = join(savefig_path, patient, 'npy', 'output', 'inst{}.npy'.format(instance_number))

            if not isnan(targets):
                pathlib.Path(join(savefig_path, patient, 'npy', 'target')).mkdir(exist_ok=True, parents=True)
                np.save(join(savefig_path, patient, 'npy', 'target', 'inst{}.npy'.format(instance_number)), targets[idx].detach().cpu().numpy())
                res.at[cur_idx + idx, 'target_path'] = join(savefig_path, 'npy', 'target-{}.npy'.format(cur_idx + idx))



            if img.ndim == 3:
                img = img[img.shape[0]//2]

            nfigs = 2 if isnan(targets) else 3
            fig, axs = plt.subplots(1, nfigs, figsize=(5*nfigs, 5))
            axs[0].imshow(img, cmap='gray')
            plot_img_mask_on_ax(axs[1], img, preds[idx])
            axs[1].set_title('Pred Mask')

            if not isnan(targets):
                plot_img_mask_on_ax(axs[2], img, targets[idx])
                axs[2].set_title('True Mask')

            fig.suptitle(figtitle)

            savepath = join(savefig_path, patient, 'png', 'inst{}.png').format(instance_number)
            fig.savefig(savepath)
            plt.close(fig)

            res.at[cur_idx + idx, 'figure_path'] = savepath

    return res


def evaluate_model(
    model,
    testloader,
    criterion,
    metrics_orig,
    number_of_batches=None,
    return_outputs_targets=False,
    device='cpu',
    verbose=False,
    logger=None,
):
    """
    Evaluates the model on testloader using all dataset at once. More
    memory greedy than the by_batch counterpart.

    Args:
        model (nn.Module child): model to test on.
        testloader (torch.utils.data.dataloader.DataLoader): loader of testset
        criterion (function): loss function
        metrics_orig (dict | list): metrics to evaluate
        number_of_batches (int, optional): Numbero of batches to evaluate. If None given, all batches are tested.
        return_outputs_targets (bool, optional): Put True to have all outputs and targets. Defaults to False.
        device (str, optional): Device to compute on. Defaults to 'cpu'.
        verbose (bool, optional): Show Progress bar. Defaults to False.

    Returns:
        float, dict, (tensor, tensor): loss, metrics evaluated, outputs, targets
    """
    all_outputs, _, all_targets = compute_outputs(
        model=model,
        testloader=testloader,
        number_of_batches=number_of_batches,
        return_inputs_targets=True,
        device=device,
        verbose=verbose,
        logger=logger,
    )

    total_metrics = evaluate_metrics(
        all_outputs, all_targets, metrics_orig
    )

    total_loss = evaluate_loss(all_outputs, all_targets, criterion)

    if return_outputs_targets:
        return total_loss, total_metrics, all_outputs, all_targets
    return total_loss, total_metrics



def compute_outputs(
    model,
    testloader,
    number_of_batches=None,
    return_inputs_targets=False,
    device='cpu',
    verbose=False,
    grad_in_eval=False,
    logger=None,
):
    """
    Computes the outputs of a model on a dataloader.

    Args:
        model (nn.Module child): model to test on.
        testloader (torch.utils.data.dataloader.DataLoader): loader of testset
        number_of_batches (int, optional): Numbero of batches to evaluate. If None given, all batches are tested.
        return_inputs_targets (bool, optional): Put True to have all inputs and targets. Defaults to False.
        device (str, optional): Device to compute on. Defaults to 'cpu'.
        verbose (bool, optional): Show Progress bar. Defaults to False.
        grad_in_eval (bool, optional): Retain the gradient in inputs. Defaults to False.

    Returns:
        torch.tensor, (torch.tensor, torch.tensor): outputs, (inputs, targets)
    """

    if number_of_batches is None:
        number_of_batches = len(testloader)
    if number_of_batches == len(testloader):
        nb_images = len(testloader.dataset)
    else:
        nb_images = number_of_batches * testloader.batch_size
    all_outputs = []
    all_targets = []
    all_inputs = torch.zeros(nb_images, *next(iter(testloader))[0][0].shape).to(device)

    all_inputs.requires_grad = grad_in_eval

    grad_manager = EmptyContextManager if grad_in_eval else torch.no_grad


    iterator = enumerate(testloader)
    if verbose:
        iterator = tqdm(iterator)

    with grad_manager():
        start_idx = 0
        for batch_idx, (inputs, targets) in iterator:
            if batch_idx >= number_of_batches:
                log_console('{} >= {} : Stop testing.'.format(batch_idx, number_of_batches), logger=logger)
                break
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = False        # Ensure all_inputs is leaf
            all_inputs.data[start_idx:start_idx+len(inputs)] = inputs.data
            # outputs = model(inputs)
            outputs = model(all_inputs[start_idx:start_idx+len(inputs)])
            all_outputs.append(outputs)
            all_targets.append(targets)
            # all_inputs.append(inputs)
            start_idx = start_idx + len(inputs)

    if return_inputs_targets:
        return torch.cat(all_outputs, 0), all_inputs, torch.cat(all_targets, 0)
    return torch.cat(all_outputs, 0)


def evaluate_metrics_mean(model, dataloader, metrics_orig, device='cpu'):
    """
    Evaluates metrics on the dataloader by batch, and performs an average of all
    metrics.

    Args:
        dataloader (Iterable): Loader of the data.
        metrics_orig (dict | list): metrics to compute.
        device (str | torch.device): cpu or cuda.

    Returns:
        dict: metrics evaluated
    """

    nb_inputs = len(dataloader.dataset)

    metrics = {}
    if type(metrics_orig) == list:
        for metric in metrics_orig:
            metrics[metric.__name__] = metric
    else:
        metrics = metrics_orig.copy()

    total_metrics = {k: 0 for k in metrics.keys()}

    model.to(device)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            for key, metric in metrics.items():
                if type(metrics) == np.ndarray:
                    total_metrics[key] += metric(outputs, targets).mean(0) * len(inputs) / nb_inputs
                else:
                    total_metrics[key] += metric(outputs, targets) * len(inputs) / nb_inputs


    return total_metrics


def evaluate_loss_mean(model, dataloader, criterion, device='cpu'):
    loss_dict = evaluate_metrics_mean(model, dataloader, {'loss': criterion}, device=device)
    return loss_dict['loss']


def evaluate_metrics(outputs, targets, metrics_orig,):
    """
    Evaluates metrics on outputs given targets

    Args:
        outputs (torch.tensor): outputs of a model to evaluate
        targets (torch.tensor): targets to compare the outputs
        metrics_orig (dict | list): metrics to compute

    Returns:
        dict: metrics evaluated
    """

    metrics = {}
    if type(metrics_orig) == list:
        for metric in metrics_orig:
            metrics[metric.__name__] = metric
    else:
        metrics = metrics_orig.copy()

    total_metrics = {}

    for key, metric in metrics.items():
        total_metrics[key] = metric(outputs, targets)

    return total_metrics


def evaluate_loss(outputs, targets, criterion):
    """
    Evaluates loss on outputs given targets

    Args:
        outputs (torch.tensor): outputs of a model to evaluate
        targets (torch.tensor): targets to compare the outputs
        criterion (function): loss function

    Returns:
        float-like: loss value
    """

    loss_dict = evaluate_metrics(outputs, targets, {'loss': criterion})
    return loss_dict['loss']
