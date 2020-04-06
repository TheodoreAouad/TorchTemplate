import config as cfg

from os.path import join
from time import time

import torch
import torch.optim as optim
import torch.nn as nn

import src.utils as u
import src.models.resnet as resnet
import src.metrics.metrics as m
import src.data_manager.utils as du
import src.callbacks.loggers.losses.base_loss as bl
import src.callbacks.loggers.observables as o
import src.callbacks.defreezer as df

def main():

    print('==================')
    print('Creating model ...')
    start_model = time()

    # cfg.threshold = .5 if cfg.MODEL_ARGS['do_activation'] else 0

    if cfg.MODEL_NAME == 'resnet':
        model_init = resnet.ResNet_N
        freezer = df.FreezeFeaturesResNet
        layers = 'fc'
    # if cfg.MODEL_NAME == 'vgg11':
    #     model_init = vgg.VGG11
    #     freezer = df.FreezeFeaturesVGG
    #     layers = 'classifier'

    cfg.model = model_init(**cfg.MODEL_ARGS)
    cfg.model.to(cfg.device)
    print('Number of parameters: ', u.get_nparams(cfg.model))
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
    cfg.metrics = {
        'accuracy': lambda *x: m.metric_argmaxer(x[0], x[1], metric=m.accuracy, ), 
        }

    cfg.criterion = nn.NLLLoss() if cfg.MODEL_ARGS['do_activation'] else nn.CrossEntropyLoss()

    cfg.loss = bl.BaseLoss(cfg.criterion)
    
    cfg.observables = [
        o.MetricsAndLoss(
            model=cfg.model,
            criterion=cfg.criterion,
            metrics=cfg.metrics, 
            save_weights_path=cfg.tensorboard_path if not cfg.evaluate_mode else None,
            to_save_on='f1score',
            ),
        o.MemoryChecker() if cfg.device == torch.device('cuda') else o.Observables(),
        # o.ConfusionMatrix(
        #     save_csv_path=join(cfg.tensorboard_path, 'confusion_matrix') if cfg.tensorboard_path is not None else None,
        #     threshold=cfg.threshold,
        # ),
        o.Activations(show_train_batch=False,),
        o.CheckLayers(model=cfg.model, layers_set={'unfrozen': layers}),
        # o.ShowImages(save_figure_path=cfg.tensorboard_path, period=100),
        # o.GradientsLossInputs(background=-1, save_figure_path=cfg.tensorboard_path, period=10),
        o.SaliencyMaps(
            save_figure_path=cfg.tensorboard_path, 
            periods={'train_on_batch': 100, 'val_on_batch': 1, 'val_on_epoch': 5, 'train_on_epoch': 5,},
            props={'train_on_batch': 1, 'val_on_batch': 1, 'val_on_epoch': .1, 'train_on_epoch': .03},
            do_tasks={
                'train_on_batch': True, 
                'val_on_batch': True if cfg.evaluate_mode else False, 
                'train_on_epoch': False, 
                'val_on_epoch': True
                },
            # threshold=cfg.threshold,
            background_to_hide=cfg.background,
        ),
        o.GradCAM(
            model=cfg.model,
            batch_size=cfg.BATCH_SIZE,
            save_figure_path=cfg.tensorboard_path,
            background_to_hide=cfg.background,
            periods={'train_on_batch': 100, 'val_on_batch': 1, 'val_on_epoch': 3, 'train_on_epoch': 5,},
            props={'train_on_batch': 1, 'val_on_batch': 1, 'val_on_epoch': .1, 'train_on_epoch': .03},
            do_tasks={
                'train_on_batch': True, 
                'val_on_batch': True if cfg.evaluate_mode else False, 
                'train_on_epoch': False, 
                'val_on_epoch': True
                },
        ),
    ]

    print('Model created in ', time() - start_model)
