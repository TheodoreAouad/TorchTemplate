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
from sklearn.metrics import confusion_matrix


from torch._six import container_abcs
from itertools import repeat


def isnan(tensor):
    return ~(tensor == tensor).any()


def log_console(to_print, *args, level='info', logger=None, **kwargs):
    """
    Prints on console and saves to a logger if given.

    Args:
        to_print (object): same argument as the print function
        level (str, optional): Levels of the logger. Defaults to 'info'.
        logger (logging.RootLogger, optional): Logger to use. Defaults to None.
                                               If None, the same as print.
    """
    if logger is None:
        print(to_print, *args, **kwargs)
    else:
        to_print = '{}'.format(to_print)
        for st in args:
            to_print = to_print + ' {} '.format(st)
        getattr(logger, level.lower())(to_print)


def _ntuple(n):
    """Taken from torch github
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py

    Args:
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)


def save_pickle(obj, path):
    pickle.dump(obj, open(path, "wb"))


def load_pickle(path):
    return pickle.load(open(path, "rb"))


def intersection_files(filenames):
    if len(filenames) == 0:
        return []
    res = read_patient_file(filenames[0])
    for filename in filenames:
        res = list(
            set(res).intersection(read_patient_file(filename))
        )
    res.sort()
    return res


def union_intersection_files(files_union, files_intersection):

    res = union_files(files_union + files_intersection)
    res = list(
        set(res).intersection(intersection_files(files_intersection))
    )
    return res


def union_files(filenames):
    if len(filenames) == 0:
        return []
    res = read_patient_file(filenames[0])
    for filename in filenames:
        res = list(
            set(res).union(read_patient_file(filename))
        )
    res.sort()
    return res


def read_patient_file(path):
    with open(path, 'r') as f:
        return f.read().splitlines()


def save_json(dic, path, sort_keys=True, indent=4):
    with open(path, 'w') as fp:
        json.dump(dic, fp, sort_keys=sort_keys, indent=indent)


def load_json(path):
    with open(path, 'r') as fp:
        return json.load(fp)


def multi_replace(string, to_replace, by=None):
    res = string + ''
    if type(to_replace) == dict:
        for key, value in to_replace.items():
            res = res.replace(key, value)
    else:
        for s, b in zip(to_replace, by):
            res = res.replace(s, b)
    return res


def merge_suffix(left, right, suffixes, on, **kwargs):
    left = left.copy()
    right = right.copy()
    left.columns = left.columns.map(lambda x: x+str(suffixes[0]) if x not in on else x)
    right.columns = right.columns.map(lambda x: x+str(suffixes[1]) if x not in on else x)
    return pd.merge(left, right, on=on, **kwargs)


def multi_merge(dfs, on, suffixes, **kwargs):
    res = pd.merge(dfs[0], dfs[1], on=on, suffixes=suffixes[:2], **kwargs)
    for i in range(2, len(dfs)):
        res = merge_suffix(res, dfs[i], on=on, suffixes=('', suffixes[i]), **kwargs)
    return res


def get_col_name(cols, s):
    if type(s) == str:
        return [col for col in cols if re.search('.*{}.*'.format(s), col)]
    if type(s) == list:
        schm = '.*'
        for st in s:
            schm += st + '.*'
        return [col for col in cols if re.search(schm, col)]


def change_position(df, loc, col_name):

    dfvalue = df[col_name]
    df = df.drop(columns=col_name)
    df.insert(loc, col_name, dfvalue)
    return df


def list_to_str(lis):
    return multi_replace(str(lis), {'[': '', ']': '', ',': '', "'": ''})


def compose(*all_functions):
    """
    Returns the composition of all functions

    Args:
        preprocessors (list): list of functions to compose. Each function
        must take as input a dataframe and output another dataframe.
    Returns:
        function: the function to preprocess data.
    """

    def recursive(df, functions):
        if len(functions) == 1:
            return functions[0](df)
        else:
            return recursive(functions[-1](df), functions[:-1])

    return lambda df: recursive(df, all_functions)


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


def are_rows_in_array(rows, ar):
    res = np.zeros(len(rows))
    for i, row in enumerate(rows):
        res[i] = (ar == row).all(1).any()
    return res.astype(bool)


def confusion_df(ypred, ytrue, cm=None, metrics={}):
    if cm is None:
        cm = confusion_matrix(ytrue, ypred)
    cmdf = pd.DataFrame(cm)

    cmdf = cmdf.rename(index={idx: 'True ' + str(idx) for idx in cmdf.index})
    cmdf = cmdf.append(pd.Series(name='Sum', data=cmdf.sum(0)))
    cmdf['Sum'] = cmdf.sum(1)
    if len(metrics) != 0:
        cmdf['metrics'] = 0
        cmdf2 = pd.DataFrame(columns=cmdf.columns)
        cmdf2.loc['___', :] = '__'
        for key, fn in metrics.items():
            cmdf2.loc[key, :] = [int(0) for _ in range(len(cmdf.columns)-1)] + [round(fn(ypred, ytrue), 2)]

        cmdf = cmdf.append(cmdf2)
        cmdf.insert(len(cmdf.columns) - 1, '|', '|')

    return cmdf


def is_inside(patient, path_to_file):
    """
    Check if the patient is inside the file.

    Args:
        patient (str): the string of the patient
        path_to_file (str, optional): The file where the patient is or is not.
    Returns:
        bool: patient is or is not in file.
    """
    with open(path_to_file, 'r') as f:
        file = f.read()
    return patient in file


def size_tensor(tensor):
    with torch.no_grad():
        return tensor.nelement() * tensor.element_size()


def size_nn(net):
    with torch.no_grad():
        total_size = 0

        for param in net.parameters():
            total_size += size_tensor(param)
        return total_size


def get_array_slice(array, ax, slice_idx):
    if ax == 0:
        return array[slice_idx, ...]
    if ax == 1:
        return array[:, slice_idx, :]
    if ax == 2:
        return array[..., slice_idx]


def deg(theta):
    return theta * 360/(2*np.pi)


def rad(theta):
    return (theta * 2*np.pi / 360) % (2*np.pi)


def get_angle(cosine, sine, unit='deg'):

    theta = None
    if sine >= 0:
        theta = np.arccos(cosine)
    else:
        theta = 2*np.pi - np.arccos(cosine)

    if unit == 'deg':
        return deg(theta)
    return theta


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
                'all': '{0:.2E}'.format,
            })
        )
    return res


def round_dict_array(dic, round_nb=2):
    res = {}
    for key in dic.keys():
        ar = dic[key]
        if type(ar) == torch.Tensor:
            ar = ar.cpu().numpy()
        elif type(ar) == float:
            ar = np.array(ar)

        if type(ar) == np.ndarray:
            res[key] = (
                np.array2string(ar, formatter={
                    'all': '{0:.2E}'.format,
                })
            )

    return res


def load_or_create_df(path, columns=[]):

    form = path.split('.')[-1]
    if os.path.exists(path):
        if form == 'pkl':
            return pd.read_pickle(path)
        elif form == 'csv':
            return pd.read_csv(path, engine='python')
        else:
            print('{} format not recognized'.format(form))
            assert False
    else:
        path_dir = os.path.dirname(path)
        pathlib.Path(path_dir).mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame(columns=columns)
        if form == 'pkl':
            df.to_pickle(path)
        elif form == 'csv':
            df.to_csv(path, index=False)
        else:
            print('{} format not recognized'.format(form))
            assert False
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


def extand_cube(cube, ref_shape):
    big_cube = np.zeros(ref_shape)
    big_cube[:cube.shape[0], :cube.shape[1], :cube.shape[2]] = cube
    return big_cube


def check_same(arr1, arr2):
    ar1 = arr1 + 0
    ar2 = arr2 + 0
    ar1 = (ar1 - ar1.min()) / (ar1.max() - ar1.min())
    ar2 = (ar2 - ar2.min()) / (ar2.max() - ar2.min())

    return np.mean(np.abs(ar1-ar2)), np.sum(np.abs(ar1-ar2))


def find_file_in_dir(string, directory, return_first=True):
    res = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if string in join(root, filename):
                if return_first:
                    return root
                res.append(root)
    return res


def find_dir_in_dir(string, directory, return_first=True):
    res = []
    for root, dirs, _ in os.walk(directory):
        for dir_ in dirs:
            if string in join(root, dir_):
                if return_first:
                    return join(root, dir_)
                res.append(join(root, dir_))
    return res
