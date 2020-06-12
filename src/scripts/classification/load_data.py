import config as cfg

from time import time
from os.path import join
import random

from torch.utils.data import DataLoader
import numpy as np

import src.utils as u
import src.data_manager.datasets as ds

import load_preprocessing


def main_train():
    load_data_train()


def load_data_train():
    start = time()
    u.log_console('==================', logger=cfg.logger)
    u.log_console('Loading data ...', logger=cfg.logger)

    # Define datasets
    trainset = ds.MNIST('./data/MNIST', train=True, transform=None, target_transform=None, download=True)
    testset = ds.MNIST('./data/MNIST', train=False, transform=None, target_transform=None, download=True)
    val_idexs = random.sample(range(len(testset)), 1000)
    testset.data = testset.data[val_idexs]
    testset.targets = testset.targets[val_idexs]
    cfg.res['size_train_data'] = len(trainset)
    cfg.res['TRAIN_TEST_SPLIT'] = str(len(trainset) / (len(trainset) + len(testset)))


    u.log_console('Trainset: {}. Testset: {}.'.format(len(trainset), len(testset)), logger=cfg.logger)

    u.log_console('==================', logger=cfg.logger)
    u.log_console('Preprocessing data ...', logger=cfg.logger)

    load_preprocessing.load_preprocessing()
    cfg.res['preprocessing'] = [str(cfg.preprocessing)]


    trainset.data = cfg.preprocessing.train(trainset.data)
    testset.data = cfg.preprocessing(testset.data)

    # get mean and std to normalize
    if cfg.args['NORMALIZE']:

        cfg.mean_norm = trainset.data.pixel_array.apply(lambda x: np.ma.masked_where(x==-1, x).mean((0, 1))).mean()
        variance = trainset.data.pixel_array.apply(lambda x: np.ma.masked_where(x==-1, (x - cfg.mean_norm)**2).mean((0, 1))).mean()
        cfg.std_norm = np.sqrt(variance)
        cfg.mean_norm, cfg.std_norm = list(cfg.mean_norm), list(cfg.std_norm)

        u.log_console('Mean: {}, Std: {}'.format(cfg.mean_norm, cfg.std_norm), logger=cfg.logger)

    load_preprocessing.load_transform_image()
    trainset.transform = cfg.transform_image_train
    testset.transform = cfg.transform_image_test
    cfg.res['batch_preprocessing'] = [str(cfg.transform_image_train)]

    cfg.trainloader = DataLoader(trainset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    cfg.trainloader_for_test = DataLoader(trainset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    cfg.testloader = DataLoader(testset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    u.log_console("Dataset process time: ", time() - start, logger=cfg.logger)


def load_data_eval():
    u.log_console('==================', logger=cfg.logger)
    u.log_console('Loading data ...', logger=cfg.logger)

    start = time()
    # Define datasets


    testset = ds.MNIST('./data/MNIST', train=False, transform=None, target_transform=None, download=True)
    u.log_console('Testset: {}.'.format(len(testset)), logger=cfg.logger)

    load_preprocessing.load_transform_image()

    u.log_console('==================', logger=cfg.logger)
    u.log_console('Preprocessing data ...', logger=cfg.logger)

    cfg.preprocessing = u.load_pickle(join(cfg.cli_args.tensorboard_path, 'preprocessing.pkl'))
    # cfg.res['preprocessing'] = [str(cfg.preprocessing)]
    # cfg.res['batch_preprocessing'] = [str(cfg.transform_image_train)]

    testset.data = cfg.preprocessing(testset.data)
    cfg.testloader = DataLoader(testset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    u.log_console("Dataset process time: ", time() - start, logger=cfg.logger)


def main_eval():
    load_data_eval()
