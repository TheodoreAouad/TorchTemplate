if __name__ == "__main__":
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
    parser.add_argument('--random_split', help='chooses random split among the patients', action='store_true',)
    parser.add_argument('-split', '--train_test_split_path', 
                help='path to the yaml containing eval patients. Default to value in all_paths.yaml', type=str)
    parser.add_argument('-out', '--output_path', help='path to save the results', type=str)
    # parser.add_argument('-w', '--weights_path', help='path to the weights', type=str)
    parser.add_argument('-tp', '--tensorboard_path', help='path to the tensorboard of the results', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    # Avoid interactive plotting
    import matplotlib
    matplotlib.use('Agg')

    cfg.cli_args = get_args()
    cfg.DEBUG = cfg.cli_args.debug
    #%% Add to path
    with open(cfg.cli_args.path_paths, 'r') as f:
        paths = f.read().splitlines()
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)

import torch

import src.utils as u
import src.tasks.test as te
import src.data_manager.utils as du

import load_data
import load_model


def main():
    res, _  = te.evaluate_model_by_batch(
        model=cfg.model,
        testloader=cfg.testloader,
        criterion=cfg.criterion,
        metrics_orig=cfg.metrics,
        observables=cfg.observables,
        grad_in_eval=True,
        # number_of_batches=cfg.cli_args.nb_batches,
        return_outputs_targets=False,
        device=cfg.device,
        verbose=cfg.cli_args.verbose,
    )
    
    
    res.to_pickle(join(cfg.tensorboard_path, 'evaluation_res.pkl'))
    res.to_csv(join(cfg.tensorboard_path, 'evaluation_res.csv'))
    print("Saved in ", join(cfg.tensorboard_path, 'evaluation_res.csv'))


if __name__ == '__main__':
    start_all = time()
    cfg.evaluate_mode = True
    # Set device
    if torch.cuda.is_available():
        cfg.device = torch.device('cuda')
    else:
        cfg.device = torch.device('cpu')
    print(cfg.device)

    assert cfg.cli_args.tensorboard_path is not None
    if cfg.cli_args.output_path is not None:
        output_path = cfg.cli_args.output_path  
    else: 
        output_path = join(cfg.cli_args.tensorboard_path, 'evaluations')

    cfg.args = u.load_pickle(join(cfg.cli_args.tensorboard_path, 'cur_args.pkl'))

    cfg.BATCH_SIZE = cfg.args['BATCH_SIZE']
    cfg.MODEL_NAME = cfg.args['MODEL_NAME']
    cfg.MODEL_ARGS = cfg.args['MODEL_ARGS']
    # cfg.OPTIMIZER_ARGS = cfg.args['OPTIMIZER']['args']
    cfg.PIXEL_SPACING = cfg.args['PIXEL_SPACING']
    cfg.n_classes = cfg.args['MODEL_ARGS']['num_classes']
    cfg.in_channels = cfg.args['MODEL_ARGS']['in_channels']
    cfg.background = cfg.MODEL_ARGS.get('background', None)
    
    cfg.tensorboard_path = du.get_next_same_name(output_path, pattern='')

    load_data.main_eval()
    load_model.main()


    cfg.model.load_state_dict(
        torch.load(join(cfg.cli_args.tensorboard_path, 'best_weights.pt'))
    )
    print('weights loaded from', join(cfg.cli_args.tensorboard_path, 'best_weights.pt'))

    main()
    print('Done in {} '.format(u.format_time(time() - start_all)))

