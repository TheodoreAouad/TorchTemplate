import torch
from time import time
from tqdm import tqdm

from src.utils import log_console
from .utils import format_time, EmptyContextManager
from src.callbacks.defreezer import Defreezer
from src.tasks.test import compute_outputs


def train(
        model,
        optimizer,
        loss,
        observables,
        number_of_epochs,
        trainloader,
        defreezer=Defreezer(),
        valloader=None,
        scheduler=None,
        do_recompute_outputs=False,
        grad_input=False,
        retain_graph=False,
        grad_in_eval=False,
        interval=None,
        output_dir_tensorboard=None,
        output_dir_results=None,
        device='cpu',
        logger=None,
        verbose=0,
):
    """
    Train with modular arguments
    Args:
        model (torch.nn.Module child): model we want to train
        optimizer (torch.optim optimizer): how do we update the weights
        loss (src.callbacks.loggers.losses.base_loss.BaseLoss child): loss object
        observables (list of src.callbacks.loggers.observables.Observables): list of the observables object
        number_of_epochs (int): how long do we train our model
        trainloader (torch.utils.data.dataloader.DataLoader): dataloader of train set
        defreezer (src.callbacks.defreezer): how to defreeze frozen tensors. Default does nothing.
        scheduler (): learning rate scheduler
        do_recompute_outputs(bool): Recomputes the outputs to give to all observables,
                                    avoiding to recompute them one by one in the observables.
        grad_input (bool): Retains the grad of inputs during evaluations
        retain_graph (bool): Retains graph in the loss.backward()
        grad_in_eval (bool): Retains the grad of inputs during evaluations
        valloader (torch.utils.data.dataloader.DataLoader): dataloader of validation set
        interval (int): which number of batch to print the verbose. Default is number of batches divided by 10.
        output_dir_results (str): output directory in which to save the results (NOT IMPLEMENTED)
        output_dir_tensorboard (str): output directory in which to save the tensorboard
        device (torch.device || str): cpu or gpu
        verbose (int): print training steps or not. 0 for not showing anything. 1 for showing progress bar on epochs.
                       2 for showing details on metrics.
    Returns
        NOT IMPLEMENTED YET
    """
    start_time = time()


    outputs_batch = None
    inputs_batch = None
    targets_batch = None
    train_outputs_epoch = None
    train_inputs_epoch = None
    train_targets_epoch = None
    val_outputs_epoch = None
    val_inputs_epoch = None
    val_targets_epoch = None

    number_of_batch = len(trainloader)
    if interval is None:
        interval = max(number_of_batch // 10, 1)

    grad_manager = EmptyContextManager if grad_in_eval else torch.no_grad

    iterator = range(number_of_epochs)
    if verbose == 1:
        iterator = tqdm(iterator)

    train_loggers = [loss] + observables

    for train_logger in train_loggers:
        train_logger.set_number_of_epoch(number_of_epochs)
        train_logger.set_number_of_batch(number_of_batch)
        train_logger.init_tensorboard_writer(output_dir_tensorboard)
        train_logger.init_results_writer(output_dir_results)

    model.train()
    for epoch in iterator:
        defreezer.defreeze_epoch(epoch)
        for objc in train_loggers:
            objc.set_current_epoch(epoch)

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            loss.set_current_batch_idx(batch_idx)
            for obs in observables:
                obs.set_current_batch_idx(batch_idx)

            inputs, targets = inputs.to(device), targets.to(device)
            if grad_input:
                inputs.requires_grad = True
            optimizer.zero_grad()
            outputs = model(inputs)

            loss.compute(outputs, targets)
            loss.backward(retain_graph=retain_graph)
            optimizer.step()



            with grad_manager():
                for obs in observables:
                    obs.compute_train_on_batch(inputs, outputs, targets)

            if batch_idx % interval == interval - 1:
                if valloader is not None:
                    model.eval()
                    if do_recompute_outputs:
                        outputs_batch, inputs_batch, targets_batch = compute_outputs(
                            model,
                            valloader,
                            return_inputs_targets=True,
                            device=device,
                            verbose=False,
                            grad_in_eval=False,     # TODO: handle in general
                        )
                    for obs in observables:
                        obs.compute_val_on_batch(
                            inputs=inputs_batch,
                            outputs=outputs_batch,
                            targets=targets_batch,
                            dataloader=valloader,
                            device=device,
                        )
                    model.train()
                if verbose == 2:
                    log_console('======================================', logger=logger)
                    log_console('Epoch [{}/{}]. Batch [{}/{}].'.format(
                        epoch + 1, number_of_epochs, batch_idx+1, number_of_batch,
                    ), logger=logger,)
                    loss.show()
                    for obs in observables:
                        obs.show()
                    log_console('Saved on {}'.format(output_dir_tensorboard), logger=logger)
                    log_console('Time Elapsed: {} s'.format(format_time(time() - start_time)), logger=logger)

        with grad_manager():
            model.eval()
            if do_recompute_outputs:
                train_outputs_epoch, train_inputs_epoch, train_targets_epoch = compute_outputs(
                    model,
                    trainloader,
                    # number_of_batches=5,
                    return_inputs_targets=True,
                    device=device,
                    verbose=False,
                    grad_in_eval=False,     # TODO: handle in general
                )
                val_outputs_epoch, val_inputs_epoch, val_targets_epoch = compute_outputs(
                    model,
                    valloader,
                    return_inputs_targets=True,
                    device=device,
                    verbose=False,
                    grad_in_eval=grad_in_eval,
                )

            for obs in observables:
                obs.compute_train_on_epoch(
                    inputs=train_inputs_epoch,
                    outputs=train_outputs_epoch,
                    targets=train_targets_epoch,
                    dataloader=trainloader,
                    device=device,
                )
                obs.compute_val_on_epoch(
                    inputs=val_inputs_epoch,
                    outputs=val_outputs_epoch,
                    targets=val_targets_epoch,
                    dataloader=valloader,
                    device=device,
                )
            model.train()

        if scheduler is not None:
            scheduler.step()

    loss.close_writer()
    for obs in observables:
        obs.close_writer()
    if verbose == 2:
        log_console('Finished Training', logger=logger)

    return loss.results(), [obs.results() for obs in observables]
