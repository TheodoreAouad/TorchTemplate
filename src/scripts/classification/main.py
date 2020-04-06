print('Importing native ...')
import config as cfg

import argparse
import os
import sys
import time
from os.path import join
from time import time
from datetime import datetime
from distutils.dir_util import copy_tree
import pathlib

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--all_paths', type=str, help='path to the yaml args file', 
                default='all_paths.yaml')
    parser.add_argument('--path_paths', type=str, help='paths to add to PYTHONPATH',
                default='./python_paths.txt')
    parser.add_argument('-v', '--verbose', type=int, help='increase output verbosity', default=2)
    parser.add_argument('--debug', help='debug mode: nothing will be saved', action='store_true',
        default=False)
    """
    OTHER ARGS
    """
    return parser.parse_args()

if __name__ == '__main__':
    # Avoid interactive plotting
    import matplotlib
    matplotlib.use('Agg')

    # Add to path
    cfg.cli_args = get_args()
    cfg.DEBUG = cfg.cli_args.debug
    with open(cfg.cli_args.path_paths, 'r') as f:
        paths = f.read().splitlines()
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)

print('Importing packages...')
import torch
import pandas as pd

print('Importing modules...')
from all_paths import all_paths
import src.tasks.test as te
import src.tasks.train as tr
import src.utils as u
import src.data_manager.utils as du

print('Importing scripts...')
import do_save_code
import load_preprocessing
import load_data
import load_model



def main(args):

    cfg.args = args
    cfg.BATCH_SIZE = args['BATCH_SIZE']
    cfg.EPOCHS = args['EPOCHS'] if not cfg.DEBUG else 2
    cfg.TRAIN_TEST_SPLIT = args['TRAIN_TEST_SPLIT']
    cfg.MODEL_NAME = args['MODEL_NAME']
    cfg.MODEL_ARGS = args['MODEL_ARGS']

    assert type(cfg.BATCH_SIZE) == int, "Batch size must be int"
    assert type(cfg.EPOCHS) == int, "Epochs must be int"

    cfg.res = pd.DataFrame([args])

    cfg.background = cfg.MODEL_ARGS.get('background', None)
    

    now = datetime.now()
    day, hour = now.strftime("%d/%m/%Y %H:%M:%S").split(' ')
    cfg.res['day'] = [day]
    cfg.res['hour'] = [hour]
    # Get tensorboard path to save results
    if not cfg.DEBUG:
        cfg.tensorboard_path = du.get_save_path(
            path_dirs=all_paths['tensorboard_classification'],
            path_to_manager=all_paths['manager_classification'],
            args=args,
        )
        cfg.res['tensorboard_path'] = cfg.tensorboard_path

        u.save_pickle(args, join(cfg.tensorboard_path, 'cur_args.pkl'))
        u.save_yaml(args, join(cfg.tensorboard_path, 'cur_args.yaml'))
        do_save_code.save_in_final_file()
    else:
        cfg.tensorboard_path = None

    # Load data oriented
    # load_preprocessing.main()
    load_data.main_train()

    # Model creation
    load_model.main()
    
    # Training
    print('==================')
    print('Training ...')
    tr.train(
        cfg.model,
        cfg.optimizer,
        cfg.loss,
        cfg.observables,
        defreezer=cfg.defreezer,
        number_of_epochs=cfg.EPOCHS,
        trainloader=cfg.trainloader,
        valloader=cfg.testloader,
        grad_input=True,
        retain_graph=True,
        grad_in_eval=True,
    #    interval=1,
        output_dir_tensorboard=cfg.tensorboard_path,
        device=cfg.device,
        verbose=VERBOSE_TRAIN,
    )
    print('Done.')

    cfg.res['Batch_Epoch_weights'] = [cfg.observables[0].best_weights_batch_epoch]

    if not cfg.DEBUG:
        cfg.model.load_state_dict(
            torch.load(join(cfg.tensorboard_path, 'best_weights.pt'))
        )
        print('weights loaded from', join(cfg.tensorboard_path, 'best_weights.pt'), 'Epoch Batch: ', cfg.res['Batch_Epoch_weights'])

    print('==================')
    print('Evaluating on train ...')
    # Evaluation on train set
    loss_train, metric_train = te.evaluate_model(
        cfg.model,
        cfg.trainloader_for_test,
        cfg.criterion,
        cfg.metrics,
        # 20,
        device=cfg.device,
    )
    print('Done.')

    title_train = {
        'loss': loss_train.item(),
        'metric': u.round_dict_array(metric_train)
    }
    cfg.res['loss_train'] = [title_train['loss']]
    for key, metric in metric_train.items():
        cfg.res['metric_train_{}'.format(key)] = [metric.cpu().numpy()]

    # Evaluation on test set
    print('==================')
    print('Evaluating on test ...')
    loss_test, metric_test = te.evaluate_model(
        cfg.model,
        cfg.testloader,
        cfg.criterion,
        cfg.metrics,
        # 20,
        device=cfg.device,
    )    
    print('Loss Test: {}'.format(loss_test))
    print('Metric Test: {}'.format(metric_test))
    print('Done.')
    


    title_test = {
        'loss': loss_test.item(),
        'metric': u.round_dict_array(metric_test)
    }
    cfg.res['loss_test'] = [title_test['loss']]
    for key, metric in metric_test.items():
        cur_metric = metric.cpu().numpy()
        if cur_metric.shape == ():
            cfg.res['metric_test_{}'.format(key)] = [cur_metric]
        else:
            for idx_met, met in enumerate(cur_metric):
                cfg.res['metric_test_{}_{}'.format(key, idx_met)] = [met]


    return cfg.res



if __name__ == '__main__':
    start_all = time()

    VERBOSE_TRAIN = cfg.cli_args.verbose
    PATH_RESULTS = all_paths['results_classification']

    do_save_code.save_in_temporary_file()

    cfg.mean_norm = [0.485, 0.456, 0.406] # mean and std of ImageNet
    cfg.std_norm = [0.229, 0.224, 0.225] # mean and std of ImageNet

    # cfg.mean_norm = [0.5054398602192114, 0.5054398602192114, 0.5055198023370465]
    # cfg.std_norm = [0.2852900917900417, 0.2852900917900417, 0.2854451397158079]

    # Set device
    if torch.cuda.is_available():
        cfg.device = torch.device('cuda')
    else:
        cfg.device = torch.device('cpu')
    print(cfg.device)

    # load args
    args_dict = u.load_yaml(all_paths['classification_args'])

    args_list = u.dict_cross(args_dict)
    
    if not cfg.DEBUG:
        res = u.load_or_create_df(PATH_RESULTS)
    else:
        res = pd.DataFrame()

    # Train and test
    for i, args in enumerate(args_list):
        print('==================')
        print('==================')
        print('Args number {} / {}'.format(i+1, len(args_list)))
        print('Time since beginning: {} '.format(u.format_time(time() - start_all)))
        res = res.append(main(args=args), sort=True)
    
        if not cfg.DEBUG:
            print('==================')
            res.to_pickle(PATH_RESULTS)
            print('Saved in {}'.format(PATH_RESULTS))
            res.to_csv(PATH_RESULTS.replace('.pkl', '.csv'))
            print('Saved in {}'.format(PATH_RESULTS.replace('.pkl', '.csv')))

    if not cfg.DEBUG:    
        print('==================')
        print('==================')
        res.to_pickle(PATH_RESULTS.replace('.pkl', '_copy.pkl'))
        print('Saved in {}'.format(PATH_RESULTS.replace('.pkl', '_copy.pkl')))
        res.to_csv(PATH_RESULTS.replace('.pkl', '_copy.csv'))
        print('Saved in {}'.format(PATH_RESULTS.replace('.pkl', '_copy.csv')))

    do_save_code.delete_temporary_file()
    print('Done in {} '.format(u.format_time(time() - start_all)))

#python src/scripts/deep_segmentation/main.py -fig --only_middle_frame --nb_patients 20
