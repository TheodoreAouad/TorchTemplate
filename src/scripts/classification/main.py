print('Importing native ...')
import config as cfg

import logging
import argparse
import sys
from os.path import join
from time import time
from datetime import datetime
import re


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
    # parser.add_argument('--no_wings',  help='take no_wings rois in: */no_wings', action='store_true',)
    return parser.parse_args()


if __name__ == '__main__':
    # Avoid interactive plotting
    import matplotlib
    matplotlib.use('Agg')

    cfg.cli_args = get_args()
    cfg.DEBUG = cfg.cli_args.debug
    # Add to path
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
import load_data
import load_model



POSSIBLE_MODELS = [
    'resnet',
    'vgg11',
    'mlp',
]


def main(args):

    cfg.args = args
    cfg.BATCH_SIZE = args['BATCH_SIZE']
    cfg.EPOCHS = args['EPOCHS']
    if cfg.cli_args.random_split:
        cfg.TRAIN_TEST_SPLIT = args['TRAIN_TEST_SPLIT']
    # # PROP_TRAIN_DATA = args['PROP_TRAIN_DATA']
    cfg.MODEL_NAME = args['MODEL_NAME']
    cfg.MODEL_ARGS = args['MODEL_ARGS']
    # cfg.OPTIMIZER_ARGS = args['OPTIMIZER']['args']
    cfg.n_classes = cfg.args['MODEL_ARGS']['n_classes']
    cfg.in_channels = args['MODEL_ARGS']['in_channels']

    assert cfg.MODEL_NAME in POSSIBLE_MODELS, "Model Name not recognized: {}".format(cfg.MODEL_NAME)
    assert type(cfg.BATCH_SIZE) == int, "Batch size must be int"
    assert type(cfg.EPOCHS) == int, "Epochs must be int"

    cfg.res = pd.DataFrame([args])
    cfg.res['tensorboard_path'] = cfg.tensorboard_path
    if not cfg.DEBUG:
        cfg.res.insert(0, 'tensorboard_nb', re.findall(r'/(\d+)/?$', cfg.tensorboard_path)[0])

    cfg.background = cfg.MODEL_ARGS.get('bg_in', 0) if cfg.MODEL_ARGS.get('use_mask', False) else 0


    now = datetime.now()
    day, hour = now.strftime("%d/%m/%Y %H:%M:%S").split(' ')
    cfg.res['day'] = [day]
    cfg.res['hour'] = [hour]


    # Load data oriented
    # load_preprocessing.main()
    load_data.main_train()

    # Model creation
    load_model.main()
    # Update loggers for observables
    for obs in cfg.observables:
        obs.set_logger(cfg.logger)

    if cfg.scheduler is not None:
        cfg.scheduler.set_logger(cfg.logger)

    # Training
    u.log_console('==================', logger=cfg.logger)
    u.log_console('Training ...', logger=cfg.logger)
    tr.train(
        cfg.model,
        cfg.optimizer,
        cfg.loss,
        cfg.observables,
        defreezer=cfg.defreezer,
        scheduler=cfg.scheduler,
        number_of_epochs=cfg.EPOCHS,
        trainloader=cfg.trainloader,
        valloader=cfg.testloader,
        do_recompute_outputs=True,
        grad_input=cfg.args['GRAD_IN_EVAL'],
        retain_graph=cfg.args['GRAD_IN_EVAL'],
        grad_in_eval=cfg.args['GRAD_IN_EVAL'],
    #    interval=1,
        output_dir_tensorboard=cfg.tensorboard_path,
        device=cfg.device,
        verbose=VERBOSE_TRAIN,
        logger=cfg.logger,
    )
    u.log_console('Done.', logger=cfg.logger)

    cfg.res['Batch_Epoch_weights'] = [cfg.observables[0].best_weights_batch_epoch]

    if not cfg.DEBUG:
        cfg.model.load_state_dict(
            torch.load(join(cfg.tensorboard_path, 'best_weights.pt'))
        )
        u.log_console('weights loaded from', join(cfg.tensorboard_path, 'best_weights.pt'), 'Batch, Epoch: ', cfg.res['Batch_Epoch_weights'], logger=cfg.logger)

    u.log_console('==================', logger=cfg.logger)
    u.log_console('Evaluating on train ...', logger=cfg.logger)
    # Evaluation on train set
    loss_train, metric_train = te.evaluate_model(
        cfg.model,
        cfg.trainloader_for_test,
        cfg.criterion,
        cfg.metrics,
        # 20,
        device=cfg.device,
        logger=cfg.logger,
    )
    u.log_console('Done.', logger=cfg.logger)

    title_train = {
        'loss': loss_train.item(),
        'metric': u.round_dict_array(metric_train)
    }
    cfg.res['loss_train'] = [title_train['loss']]
    for key, metric in metric_train.items():
        cfg.res['metric_train_{}'.format(key)] = [metric]

    # Evaluation on test set
    u.log_console('==================', logger=cfg.logger)
    u.log_console('Evaluating on test ...', logger=cfg.logger)
    loss_test, metric_test = te.evaluate_model(
        cfg.model,
        cfg.testloader,
        cfg.criterion,
        cfg.metrics,
        # 20,
        device=cfg.device,
        logger=cfg.logger,
    )
    u.log_console('Loss Test: {}'.format(loss_test), logger=cfg.logger)
    u.log_console('Metric Test: {}'.format(metric_test), logger=cfg.logger)
    u.log_console('Done.', logger=cfg.logger)



    title_test = {
        'loss': loss_test.item(),
        'metric': u.round_dict_array(metric_test)
    }
    cfg.res['loss_test'] = [title_test['loss']]
    for key, metric in metric_test.items():
        # cur_metric = metric
        cfg.res['metric_test_{}'.format(key)] = [metric]
        # if cur_metric.shape == ():
        #     cfg.res['metric_test_{}'.format(key)] = [cur_metric]
        # else:
        #     for idx_met, met in enumerate(cur_metric):
        #         cfg.res['metric_test_{}_{}'.format(key, idx_met)] = [met]


    return cfg.res



if __name__ == '__main__':
    start_all = time()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s',
    )

    VERBOSE_TRAIN = cfg.cli_args.verbose
    PATH_RESULTS = all_paths['results_classification']

    if not cfg.DEBUG:
        do_save_code.save_in_temporary_file()

    cfg.mean_norm = [0.485, 0.456, 0.406]  # mean and std of ImageNet
    cfg.std_norm = [0.229, 0.224, 0.225]  # mean and std of ImageNet

    # Set device
    if torch.cuda.is_available():
        cfg.device = torch.device('cuda')
    else:
        cfg.device = torch.device('cpu')
    print(cfg.device)

    # load args
    args_dict = u.load_yaml(all_paths['classification_args'])
    # if not cfg.cli_args.random_split:
    #     del args_dict['TRAIN_TEST_SPLIT']
    #     args_dict['TRAIN_TEST_SPLIT_YAML'] = [all_paths['train_test_split_yaml_binary_classification']]

    args_list = u.dict_cross(args_dict)

    if not cfg.DEBUG:
        res = u.load_or_create_df(PATH_RESULTS)
    else:
        res = pd.DataFrame()

    # Train and test
    bugged = []
    for i, args in enumerate(args_list):
        # Set up Logs

        cfg.logger = logging.getLogger('args_{}'.format(i+1))

        # Get tensorboard path to save results
        if not cfg.DEBUG:
            cfg.tensorboard_path = du.get_save_path(
                path_dirs=all_paths['tensorboard_classification'],
                path_to_manager=all_paths['manager_classification'],
                args=args,
            )

            u.save_pickle(args, join(cfg.tensorboard_path, 'cur_args.pkl'))
            u.save_yaml(args, join(cfg.tensorboard_path, 'cur_args.yaml'))

            do_save_code.save_in_final_file()

            formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

            error_handler = logging.FileHandler(join(cfg.tensorboard_path, 'error_logs.log'))
            error_handler.setFormatter(formatter)
            error_handler.setLevel(logging.WARNING)
            cfg.logger.addHandler(error_handler)

            info_handler = logging.FileHandler(join(cfg.tensorboard_path, 'all_logs.log'))
            info_handler.setFormatter(formatter)
            info_handler.setLevel(logging.DEBUG)
            cfg.logger.addHandler(info_handler)



        else:
            cfg.tensorboard_path = None


        u.log_console('Device: {}'.format(cfg.device), logger=cfg.logger)
        u.log_console('==================', logger=cfg.logger)
        u.log_console('==================', logger=cfg.logger)
        u.log_console('Args number {} / {}'.format(i+1, len(args_list)), logger=cfg.logger)
        u.log_console('Time since beginning: {} '.format(u.format_time(time() - start_all)), logger=cfg.logger)

        cfg.output_obs_parent = cfg.tensorboard_path
        try:
            res = res.append(main(args=args), sort=True)
        except Exception:
            cfg.logger.exception(
                'Args nb {} / {} failed : '.format(i+1, len(args_list)))
            bugged.append(i+1)

        if not cfg.DEBUG:
            u.log_console('==================', logger=cfg.logger)
            res = u.change_position(res, 0, 'tensorboard_nb')
            res.to_pickle(PATH_RESULTS)
            u.log_console('Saved in {}'.format(PATH_RESULTS), logger=cfg.logger)
            res.to_csv(PATH_RESULTS.replace('.pkl', '.csv'), index=False)
            u.log_console('Saved in {}'.format(PATH_RESULTS.replace('.pkl', '.csv')), logger=cfg.logger)

    if not cfg.DEBUG:
        u.log_console('==================', logger=cfg.logger)
        u.log_console('==================', logger=cfg.logger)
        res.to_pickle(PATH_RESULTS.replace('.pkl', '_copy.pkl'))
        u.log_console('Saved in {}'.format(PATH_RESULTS.replace('.pkl', '_copy.pkl')), logger=cfg.logger)
        res.to_csv(PATH_RESULTS.replace('.pkl', '_copy.csv'), index=False)
        u.log_console('Saved in {}'.format(PATH_RESULTS.replace('.pkl', '_copy.csv')), logger=cfg.logger)

    if not cfg.DEBUG:
        do_save_code.delete_temporary_file()
    u.log_console('All Args Bugged: ', bugged, logger=cfg.logger)
    u.log_console('Done in {} '.format(u.format_time(time() - start_all)), logger=cfg.logger)

# python src/scripts/deep_segmentation/main.py -fig --only_middle_frame --nb_patients 20
