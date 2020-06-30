import config as cfg

from os.path import join
from time import time

import torch
import torch.optim as optim
import torch.nn as nn

import src.utils as u
import src.models.resnet as resnet
import src.models.vgg as vgg
import src.models.mlp as mlp
import src.metrics.metrics as m
import src.callbacks.loggers.losses.base_loss as bl
import src.callbacks.loggers.observables as o
import src.callbacks.defreezer as df
import src.callbacks.schedulers as s



def main():

    u.log_console('==================', logger=cfg.logger)
    u.log_console('Creating model ...', logger=cfg.logger)
    start_model = time()

    cfg.threshold = .5 if cfg.MODEL_ARGS['do_activation'] else None

    if cfg.MODEL_NAME == 'resnet':
        model_init = resnet.ResNet_N
        freezer = df.FreezeFeaturesResNet
        layers = 'fc'
        cfg.MODEL_ARGS['n_classes'] = cfg.n_classes
        cfg.MODEL_ARGS['in_channels'] = cfg.in_channels

    if cfg.MODEL_NAME == 'vgg11':
        model_init = vgg.VGG11
        freezer = df.FreezeFeaturesVGG
        layers = 'classifier'
        cfg.MODEL_ARGS['n_classes'] = cfg.n_classes
        cfg.MODEL_ARGS['in_channels'] = cfg.in_channels

    if cfg.MODEL_NAME == 'mlp':
        model_init = mlp.MLP

    cfg.model = model_init(**cfg.MODEL_ARGS)
    cfg.model.to(cfg.device)
    if not cfg.DEBUG:
        with open(join(cfg.tensorboard_path, 'model.txt'), 'w') as f:
            f.write(str(cfg.model))

    u.log_console('Number of parameters: ', u.get_nparams(cfg.model), logger=cfg.logger)
    if cfg.res is not None:
        cfg.res['nparams'] = u.get_nparams(cfg.model, trainable=False)
        cfg.res['trainable_nparams'] = u.get_nparams(cfg.model, trainable=True)

    if cfg.MODEL_ARGS['pretrained'] and cfg.args['FREEZE_FEATURES']:
        cfg.defreezer = freezer(model=cfg.model)
    else:
        cfg.defreezer = df.Defreezer()


    if cfg.args['OPTIMIZER']['name'] == 'RMSprop':
        optimizer = optim.RMSprop
    elif cfg.args['OPTIMIZER']['name'].lower() == 'adam':
        optimizer = optim.Adam

    cfg.optimizer = optimizer(cfg.model.parameters(), **cfg.args['OPTIMIZER']['args'])
    if cfg.n_classes == 2:
        def base_fn(metric):
            return lambda *x: m.metric_binary_thresh(x[0], x[1], metric=metric, threshold=cfg.threshold)
    else:
        def base_fn(metric):
            return lambda *x: m.metric_multiclass(x[0], x[1], metric=metric)

    cfg.metrics = {
        'accuracy': base_fn(metric=m.accuracy),
        'f1score': base_fn(metric=m.f1score),
        'mcc': base_fn(metric=m.matthewscc),
    }
    if cfg.n_classes >= 2:
        cfg.criterion = nn.NLLLoss() if cfg.MODEL_ARGS['do_activation'] else nn.CrossEntropyLoss()

    elif cfg.n_classes == 1:
        cfg.criterion = nn.BCELoss() if cfg.MODEL_ARGS['do_activation'] else nn.BCEWithLogitsLoss()
    cfg.loss = bl.BaseLoss(cfg.criterion)

    metrics_and_loss_obs = o.MetricsAndLoss(
        model=cfg.model,
        criterion=cfg.criterion,
        metrics=cfg.metrics,
        save_weights_path=cfg.output_obs_parent if not cfg.evaluate_mode else None,
        to_save_on='loss',
        use_recomputed_outputs=True,
    )

    if cfg.args['UPDATE_LR']:
        cfg.scheduler = s.ReduceLROnPlateau(
            loss_observable=metrics_and_loss_obs,
            optimizer=cfg.optimizer,
            **cfg.args['SCHEDULER_ARGS']
        )

    cfg.observables = [
        metrics_and_loss_obs,
        o.LRChecker(cfg.optimizer),
        o.MemoryChecker() if cfg.device == torch.device('cuda') else o.Observables(),
        o.ConfusionMatrix(
            save_csv_path=join(cfg.output_obs_parent, 'confusion_matrix') if cfg.output_obs_parent is not None else None,
            threshold=cfg.threshold,
            metrics=cfg.metrics,
            n_classes=cfg.n_classes,
        ),
        o.Activations(show_train_batch=False,),
        o.CheckLayers(model=cfg.model, layers_set={'unfrozen': layers}),
        o.ShowImages(
            model=cfg.model,
            save_figure_path=join(cfg.output_obs_parent, 'images') if cfg.output_obs_parent is not None else None,
            periods={'train_batch': 100, 'val_batch': 100, 'val_epoch': cfg.args['EPOCHS'], 'train_epoch': cfg.args['EPOCHS']},
            props={'train_batch': .1, 'val_batch': .05, 'val_epoch': .1, 'train_epoch': .05},
            do_tasks={
                'train_batch': False,
                'val_batch': False,
                'train_epoch': True,
                'val_epoch': True,
            },
        ) if (not cfg.evaluate_mode) and cfg.args['SAVE_FIGURES'] else o.Observables(),
        # o.GradientsLossInputs(background=-1, save_figure_path=cfg.output_obs_parent, period=10),
        # o.SaliencyMaps(
        #     save_figure_path=cfg.output_obs_parent,
        #     periods={'train_on_batch': 100, 'val_on_batch': 1, 'val_on_epoch': 5, 'train_on_epoch': 5,},
        #     props={'train_on_batch': 1, 'val_on_batch': 1, 'val_on_epoch': .1, 'train_on_epoch': .03},
        #     do_tasks={
        #         'train_on_batch': True,
        #         'val_on_batch': True if cfg.evaluate_mode else False,
        #         'train_on_epoch': False,
        #         'val_on_epoch': True
        #         },
        #     threshold=cfg.threshold,
        #     background_to_hide=cfg.background,
        # ),
        # o.GradCAM(
        #     model=cfg.model,
        #     batch_size=cfg.BATCH_SIZE,
        #     save_figure_path=cfg.output_obs_parent,
        #     background_to_hide=cfg.background,
        #     periods={'train_on_batch': 100, 'val_on_batch': 1, 'val_on_epoch': 3, 'train_on_epoch': 5,},
        #     props={'train_on_batch': 1, 'val_on_batch': 1, 'val_on_epoch': .1, 'train_on_epoch': .03},
        #     do_tasks={
        #         'train_on_batch': True,
        #         'val_on_batch': True if cfg.evaluate_mode else False,
        #         'train_on_epoch': False,
        #         'val_on_epoch': True
        #         },
        # ),
    ]



    u.log_console('Model created in ', time() - start_model, logger=cfg.logger)
