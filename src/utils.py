import yaml
import re
import os
from os.path import join
import pathlib
import json
import pickle

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid



def save_pickle(obj, path):
    pickle.dump(obj, open( path, "wb" ) )

def load_pickle(path):
    return pickle.load(open(path, "rb"))


def save_json(dic, path, sort_keys=True, indent=4):
    with open(path, 'w') as fp:
        json.dump(dic, fp, sort_keys=sort_keys, indent=indent)

def load_json(path):
    with open(path, 'r') as fp:
        return json.load(fp)

def multi_replace(string, to_replace, by):
    res = string + ''
    for s, b in zip(to_replace, by):
        res = res.replace(s, b)
    return res


def format_time(s):
    h = s // (3600)
    s %= 3600
    m = s // 60
    s %= 60
    return "%02i:%02i:%02i" % (h, m, s)
    

def fix_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, 1e6)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    return seed


def get_nparams(model, trainable=True):
    model_parameters = model.parameters()
    if trainable:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def size_tensor(tensor):
    with torch.no_grad():
        return tensor.nelement() * tensor.element_size()

def size_nn(net):
    with torch.no_grad():
        total_size = 0

        for param in net.parameters():
            total_size += size_tensor(param)
        return total_size


def save_yaml(dic, path):
    yaml.dump(dic, open(path, 'w'))


def load_yaml(path,):

    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(path, 'r') as f:
        try:
            content = yaml.load(f, Loader=loader)
        except yaml.YAMLError as exc:
            print(exc)
    return content

    
def dict_cross(dic):
    """
    Does a cross product of all the values of the dict.
    
    Args:
        dic (dict): dict to unwrap
    
    Returns:
        list: list of the dict
    """

    return list(ParameterGrid(dic))


def ceil_(x):
    return np.int(np.ceil(x))


def floor_(x):
    return np.int(np.floor(x))



def round_list_array(ars, round_nb=2):
    res = []
    for ar in ars:
        if type(ar) == torch.Tensor:
            ar = ar.cpu().numpy()
        res.append(
            np.array2string(ar, formatter={
                'all':'{0:.2E}'.format,
                })
            )
    return res
    

def round_dict_array(dic, round_nb=2):
    res = {}
    for key in dic.keys():
        ar = dic[key]
        if type(ar) == torch.Tensor:
            ar = ar.cpu().numpy()
        res[key] = (
            np.array2string(ar, formatter={
                'all':'{0:.2E}'.format,
                })
            )
    return res

def load_or_create_df(path):

    if os.path.exists(path):
        return pd.read_pickle(path)
    else:
        path_dir = os.path.dirname(path)
        pathlib.Path(path_dir).mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame()
        df.to_pickle(path)
        print('Created {}'.format(path))
        return df


def change_dir_name(previous, destination, verbose=1):

    destination = pathlib.Path(destination).stem
    root = str(pathlib.Path(previous).parent)
    destination = join(root, destination)
    pathlib.Path(destination).mkdir(exist_ok=True)
    
    for filename in os.listdir(previous):
        os.rename(join(previous, filename), join(destination, filename))
    os.rmdir(previous)
    if verbose:
        print('{} changed to {}'.format(previous, destination))



def find_file_in_dir(string, directory, return_first=True):
    res = []
    for root, _, files in os.walk(directory):
        for filename in files: 
            if string in join(root, filename):
                if return_first:
                    return root
                res.append(root)
    return res


def find_dir_in_dir(string, directory,return_first=True):
    res = []
    for root, dirs, _ in os.walk(directory):
        for dir_ in dirs: 
            if string in join(root, dir_):
                if return_first:
                    return join(root, dir_)
                res.append(join(root, dir_))
    return res
