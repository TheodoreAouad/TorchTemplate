import torch
import numpy as np
from tqdm import tqdm

from .utils import EmptyContextManager
from src.plotter import show_results

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
    prev_reduc = criterion.reduction
    if number_of_batches is None:
        number_of_batches = len(testloader)
    figs = []

    metrics = {}
    if type(metrics_orig) == list:
        for metric in metrics_orig:
            metrics[metric.__name__] = metric
    else:
        metrics = metrics_orig.copy()

    res = testloader.dataset.data
    res = res.drop(columns=['pixel_array', 't1_array', 'stir_array',])
    res['output_loss'] = torch.zeros(len(res))
    res['targets'] = torch.zeros(len(res))
    for key, metric in metrics.items():
        res['metric_{}'.format(key)] = torch.zeros(len(res))
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
                print('============')
                print('Batch {} / {}'.format(batch_idx + 1, number_of_batches))
            if batch_idx >= number_of_batches:
                print('{} >= {} : Stop testing.'.format(batch_idx, number_of_batches))
                break
            inputs, targets = inputs.to(device), targets.to(device)
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

            batch_size = len(targets)
            
            criterion.reduction = 'none'
            res.output_loss.iloc[cur_idx:cur_idx + batch_size] = criterion(outputs, targets).detach().cpu().numpy()
            criterion.reduction = prev_reduc
            for i in range(outputs.shape[-1]):
                if 'outputs-{}'.format(i) not in res.columns:
                    res['outputs-{}'.format(i)] = torch.zeros(len(res))
                res['outputs-{}'.format(i)].iloc[cur_idx:cur_idx + batch_size] = outputs[..., i].detach().cpu().numpy()
            res.targets.iloc[cur_idx:cur_idx + batch_size] = targets.detach().cpu().numpy()
            res.batch_idx.iloc[cur_idx:cur_idx + batch_size] = batch_idx

            for key, metric in metrics.items():
                res['metric_{}'.format(key)].iloc[cur_idx:cur_idx + batch_size] = metric(outputs, targets).cpu().detach().numpy()

            if show_figure:
                figs.append(show_results(targets, inputs, [outputs], **figure_args))
            cur_idx += batch_size

    return  res, figs


def evaluate_model(
    model,
    testloader,
    criterion,
    metrics_orig,
    number_of_batches=None,
    return_outputs_targets=False,
    device='cpu',
    verbose=False,
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
    all_outputs, _,  all_targets = compute_outputs(
        model=model,
        testloader=testloader,
        number_of_batches=number_of_batches,
        return_inputs_targets=True,
        device=device,
        verbose=verbose,
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
    all_inputs = torch.zeros(nb_images, *testloader.dataset[0][0].shape).to(device)
    
    all_inputs.requires_grad = grad_in_eval

    grad_manager = EmptyContextManager if grad_in_eval else torch.no_grad


    iterator = enumerate(testloader)
    if verbose:
        iterator = tqdm(iterator)

    with grad_manager():
        start_idx = 0
        for batch_idx, (inputs, targets) in iterator:
            if batch_idx >= number_of_batches:
                print('{} >= {} : Stop testing.'.format(batch_idx, number_of_batches))
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


def evaluate_metrics(outputs, targets, metrics_orig):
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
        total_metrics[key] = metric(outputs, targets).mean(0).detach()

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
