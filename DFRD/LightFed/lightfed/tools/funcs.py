import hashlib
import os
import pickle
import random
from collections import OrderedDict

import numpy as np
import torch
from sklearn.utils import shuffle as shuffle_func


def mf(file_path):
    _dir = os.path.dirname(file_path)
    _dir and os.makedirs(_dir, exist_ok=True)
    return file_path


def md(dirname): 
    dirname and os.makedirs(dirname, exist_ok=True)
    return dirname


def set_seed(seed, to_numpy=True, to_torch=True, to_torch_cudnn=True):
    if seed is None:
        return
    random.seed(seed)
    if to_numpy:
        np.random.seed(seed)
    if to_torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available() and to_torch_cudnn:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True


def consistent_hash(*objs, code_len=6):
    assert os.environ.get('PYTHONHASHSEED') == '0', "env variable : PYTHONHASHSEED==0 should be specified"
    return hashlib.md5(pickle.dumps(objs)).hexdigest()[:code_len]


def formula(func, *params_args):
    res = OrderedDict()
    for name in params_args[0].keys():
        weight = func(*[params[name] for params in params_args])
        res[name] = weight.detach().clone()
    return res


def model_size(model):
    if isinstance(model, torch.nn.Module):
        params_iter = model.named_parameters()
    elif isinstance(model, dict):
        params_iter = model.items()
    else:
        raise Exception(f"unknow type: {type(model)}, expected is torch.nn.Module or dict")
    res = 0.0
    for _, weight in params_iter:
        res += (weight.element_size() * weight.nelement())
    return res


def save_pkl(obj, file_path):
    with open(mf(file_path), "wb") as _f:
        pickle.dump(obj, _f)


def load_pkl(file_path):
    with open(file_path, "rb") as _f:
        return pickle.load(_f)


def batch_iter(*iters, batch_size=256, shuffle=False, random_state=None):
    _iter_num = len(iters)
    if shuffle:
        iters = shuffle_func(*iters, random_state=random_state)
        if _iter_num == 1:
            iters = (iters,)
    _current_size = 0
    _batch = [[] for _ in range(_iter_num)]
    for _item_tuples in zip(*iters):
        for i in range(_iter_num):
            _batch[i].append(_item_tuples[i])
        _current_size += 1
        if _current_size >= batch_size:
            yield _batch if _iter_num > 1 else _batch[0]
            _current_size = 0
            _batch = [[] for _ in range(_iter_num)]
    if _current_size:
        yield _batch if _iter_num > 1 else _batch[0]

def grad_False(model, select_frozen_layers=None):
    if select_frozen_layers==None:
        for name, param in model.named_parameters():
            param.requires_grad = False
    else:
        i = 0
        for name, param in model.named_parameters():
            if select_frozen_layers in model.named_parameter_layers[i]:
                param.requires_grad = False
            i += 1


def grad_True(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
