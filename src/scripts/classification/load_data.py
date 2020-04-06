import config as cfg

from time import time
from os.path import join
import random

from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from all_paths import all_paths
import src.utils as u
import src.data_manager.get_data as gd
import src.data_manager.dataloaders as dl
import src.data_manager.datasets as ds
import src.data_processing.processing as proc
import src.data_processing.preprocessing as prep

import load_preprocessing

def main_train():
    load_data_train()



def load_data_train():
    start = time()
    print('==================')
    print('Loading data ...')

    # Define datasets
    trainset = ds.MNIST('./data/MNIST', train=True, transform=None, target_transform=None, download=True)
    testset = ds.MNIST('./data/MNIST', train=False, transform=None, target_transform=None, download=True)
    cfg.res['size_train_data'] = len(trainset)
    cfg.res['TRAIN_TEST_SPLIT'] = str(len(trainset) / (len(trainset) + len(testset)))

    print('==================')
    print('Preprocessing data ...')

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
    
        print('Mean: {}, Std: {}'.format(cfg.mean_norm, cfg.std_norm))
    
    load_preprocessing.load_transform_image()
    trainset.transform = cfg.transform_image_train
    testset.transform = cfg.transform_image_test
    cfg.res['batch_preprocessing'] = [str(cfg.transform_image_train)]

    cfg.trainloader = DataLoader(trainset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    cfg.trainloader_for_test = DataLoader(trainset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    cfg.testloader = DataLoader(testset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    print("Dataset process time: ", time() - start)


def load_data_eval():
    print('==================')
    print('Loading data ...')

    start = time()
    # Define datasets
    data = pd.read_csv(all_paths['rois_{}_dataset_csv'.format(cfg.args['ROI_CROP_TYPE'])])
    

    if cfg.cli_args.random_split:
        all_patients = data.patient_id.unique()
        random.shuffle(all_patients)
        pivot_idx = int(cfg.TRAIN_TEST_SPLIT * len(all_patients))
        test_patients = all_patients[pivot_idx:]
    else:
        patients_split = u.load_yaml(all_paths['train_test_split_yaml'])
        test_patients = patients_split['EVAL']


    test_data = data.loc[data.patient_id.isin(test_patients)]
    print('{} patients and {} slices'.format(len(test_patients), len(test_data)))

    load_preprocessing.load_transform_image()
    _, testset = get_train_test_datasets(test_data=test_data, transform_test=cfg.transform_image_test)

    print('==================')
    print('Preprocessing data ...')

    cfg.preprocessing = u.load_pickle(join(cfg.cli_args.tensorboard_path, 'preprocessing.pkl'))
    # cfg.res['preprocessing'] = [str(cfg.preprocessing)]
    # cfg.res['batch_preprocessing'] = [str(cfg.transform_image_train)]

    testset.data = cfg.preprocessing(testset.data)
    cfg.testloader = DataLoader(testset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    print("Dataset process time: ", time() - start)


def main_eval():
    load_data_eval()


