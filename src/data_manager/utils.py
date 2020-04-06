import os
from os.path import join
import pathlib
from datetime import datetime
from time import time
import re

import pandas as pd
from all_paths import all_paths


def get_next_same_name(parent_dir, pattern=''):
    """
    Scans the folder parent dir for files/folders with name 'pattern{}' with {} being an integer. Returns 
    the path to the file/folder with name 'pattern-{}' with the highest number +1.
    
    Args:
        parent_dir (str): path to the parent directory of the files / folders with the pattern
        pattern (str, optional): pattern of the file to look for. Defaults to ''.
    """
    if not os.path.exists(parent_dir):
        return join(parent_dir, '{}{}'.format(pattern, 0))
    directories = [o for o in os.listdir(parent_dir) if re.search(r'^{}\d+$'.format(pattern), o)]
    if len(directories) == 0:
        max_nb = 0
    else:
        nbrs = [int(re.findall(r'\d+$', o)[0]) for o in directories]
        max_nb = max(nbrs) + 1
    return join(parent_dir, '{}{}'.format(pattern, max_nb))


def get_save_path(
    path_dirs,
    path_to_manager,
    args={}, 
):
    """
    Gets the path of the tensorboards and logs for a training session.
    Creates the folder and saves the arguments in a manager.
    
    Args:
        args (dict, optional): Arguments of the session. Defaults to {}.
        path_dirs (str, optional): Parent of the directories. Defaults to all_paths['path_dirs'].
        path_to_manager (str, optional): Path to the manager of the directories. Defaults to all_paths['path_to_manager'].
    
    Returns:
        str: path to the tensorboards and stuff.
    """

    now = datetime.now()
    day, time = now.strftime("%d/%m/%Y %H:%M:%S").split(' ')

    directories = [int(o) for o in os.listdir(path_dirs) if os.path.isdir(os.path.join(path_dirs, o)) and re.search(r'^\d+$', o)]
    # print(directories)
    # for i, d in enumerate(directories):
    #     try: directories[i] = int(d)
    #     except ValueError:
    #         del directories[i]

    if len(directories) == 0:
        id = '0'
    else:
        id = str(max(directories)+1)

    path = os.path.join(path_dirs, id)

    if not os.path.exists(os.path.join(path_to_manager, 'manager.pkl')):
        print(os.path.join(path_to_manager, 'manager.pkl'))
        # path = os.path.join(path_dirs , '0')
        pathlib.Path(path_to_manager).mkdir(exist_ok=True, parents=True)
        manager = pd.DataFrame()
    else:
        manager = pd.read_pickle(os.path.join(path_to_manager, 'manager.pkl'))
    
    manager = manager.append(pd.DataFrame.from_dict({
        'id': [id],
        'day': [day],
        'time': [time],
        'path': [path],
        'args': [args],
    }))

    pathlib.Path(path).mkdir(exist_ok=True, parents=True)
    manager.to_pickle(os.path.join(path_to_manager, 'manager.pkl'))
    manager.to_csv(os.path.join(path_to_manager, 'manager.csv'))

    return path


def min_max_norm(img):
    return (img - img.min())/(img.max() - img.min())


