import config as cfg

import os
from os.path import join, exists
from shutil import rmtree, copytree, ignore_patterns

from all_paths import all_paths


def save_in_temporary_file():
    cfg.temporary_path = join(all_paths['tensorboard_binary_classification'], 'temporary')
    # Save code
    if exists(cfg.temporary_path):
        rmtree(cfg.temporary_path)
    cfg.path_to_code = join(cfg.temporary_path, 'code')
    copytree(join(os.getcwd(), 'src'), cfg.path_to_code, ignore=ignore_patterns('*__pycache__*'))


def delete_temporary_file():
    rmtree(cfg.temporary_path)


def save_in_final_file():

    # Save code

    dst = join(cfg.tensorboard_path, 'code')
    copytree(cfg.path_to_code, dst, ignore=ignore_patterns('*__pycache__*'))
