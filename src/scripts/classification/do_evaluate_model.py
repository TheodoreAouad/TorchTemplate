if __name__ == "__main__":
    print('Importing native ...')
import config as cfg

import argparse
import sys
import re
from os.path import join
from time import time
import pathlib
import logging


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
    parser.add_argument('-tp', '--tensorboard_path', nargs='+', help='paths to the tensorboard of the results', default=[])
    parser.add_argument('-nobs', '--no_observables', help='ignore all observables', action='store_true',)
    return parser.parse_args()


if __name__ == '__main__':
    # Avoid interactive plotting
    import matplotlib
    matplotlib.use('Agg')

    cfg.cli_args = get_args()
    assert cfg.cli_args.tensorboard_path != []
    cfg.DEBUG = cfg.cli_args.debug
    # Add to path
    with open(cfg.cli_args.path_paths, 'r') as f:
        paths = f.read().splitlines()
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)


import torch

from all_paths import all_paths
import src.utils as u
import src.tasks.test as te
import src.data_manager.utils as du

import load_data
import load_model


def main():
    cfg.res = te.evaluate_model_by_batch(
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

    cfg.res = cfg.res.drop(columns=['pixel_array', ])

    cfg.res.to_pickle(join(cfg.output_path, 'evaluation_res.pkl'))
    cfg.res.to_csv(join(cfg.output_path, 'evaluation_res.csv'))
    u.log_console("Saved joints results in ",
            join(cfg.output_path, 'evaluation_res.csv'), logger=cfg.logger)


if __name__ == '__main__':
    start_all = time()
    cfg.evaluate_mode = True

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s',
    )

    # Set device
    if torch.cuda.is_available():
        cfg.device = torch.device('cuda')
    else:
        cfg.device = torch.device('cpu')
    u.log_console(cfg.device, logger=cfg.logger)

    cfg.path_to_rois_ini = cfg.cli_args.path_to_rois

    tb_parent = str(pathlib.Path(cfg.cli_args.tensorboard_path[0]).parent)

    for tb_path_idx, tb_path in enumerate(cfg.cli_args.tensorboard_path):

        u.log_console('Tb path nb {} / {}'.format(
            tb_path_idx+1, len(cfg.cli_args.tensorboard_path)), logger=cfg.logger)

        if tb_parent not in tb_path:
            tb_path = join(tb_parent, tb_path)
        cfg.tb_path = tb_path

        if cfg.cli_args.output_path is not None:
            tb_nb = re.findall(r'/\d.+/?$', tb_path)
            output_path = join(cfg.cli_args.output_path, tb_nb)
        else:
            output_path = join(tb_path, 'evaluations')

        cfg.output_path = du.get_next_same_name(output_path, pattern='')
        pathlib.Path(cfg.output_path).mkdir(exist_ok=True, parents=True)
        cfg.tensorboard_path = cfg.output_path


        # Set up logging
        formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        error_handler = logging.FileHandler(join(cfg.tensorboard_path, 'error_logs.log'))
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.WARNING)
        cfg.logger.addHandler(error_handler)

        info_handler = logging.FileHandler(join(cfg.tensorboard_path, 'all_logs.log'))
        info_handler.setFormatter(formatter)
        info_handler.setLevel(logging.DEBUG)
        cfg.logger.addHandler(info_handler)

        if cfg.cli_args.only_true:
            cfg.path_to_rois_ini = join(all_paths['rois_dataset'], 'only_true')

        cfg.args = u.load_pickle(join(tb_path, 'cur_args.pkl'))
        if cfg.args['NO_WINGS']:
            cfg.cli_args.path_to_rois = join(cfg.path_to_rois_ini, 'no_wings')

        cfg.BATCH_SIZE = cfg.args['BATCH_SIZE']
        cfg.MODEL_NAME = cfg.args['MODEL_NAME']
        cfg.MODEL_ARGS = cfg.args['MODEL_ARGS']
        cfg.n_classes = cfg.args['MODEL_ARGS']['n_classes']
        cfg.in_channels = cfg.args['MODEL_ARGS']['in_channels']
        cfg.background = (cfg.MODEL_ARGS.get('bg_in', 0)
                            if cfg.MODEL_ARGS['use_mask'] else 0)


        load_data.main_eval()
        load_model.main()

        if cfg.cli_args.no_observables:
            cfg.observables = []
        else:
            cfg.output_obs_parent = join(cfg.output_path, 'observables')
            pathlib.Path(cfg.output_obs_parent).mkdir(
                exist_ok=True, parents=True)

        cfg.model.load_state_dict(
            torch.load(join(tb_path, 'best_weights.pt'))
        )
        u.log_console('weights loaded from', join(cfg.tb_path, 'best_weights.pt'), logger=cfg.logger)

        main()
    u.log_console('Done in {} '.format(u.format_time(time() - start_all)), logger=cfg.logger)
