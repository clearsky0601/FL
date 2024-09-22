import argparse
import logging
import os

import numpy as np
import torch
from experiments.datasets.data_distributer import DataDistributer
from lightfed.tools.funcs import consistent_hash, set_seed

MODEL_SPLIT_RATE = {'a': 1.0, 'b': 0.5, 'c': 0.25, 'd': 0.125, 'e': 0.0625}
RATE_NAME = ['a', 'b', 'c', 'd', 'e']

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--comm_round', type=int, default=100)

    parser.add_argument('--I', type=int, default=20, help='synchronization interval')

    parser.add_argument('--batch_size', type=int, default=64)  
    parser.add_argument('--eval_step_interval', type=int, default=5)

    parser.add_argument('--eval_batch_size', type=int, default=256)


    parser.add_argument('--lr_lm', type=float, default=0.01, help='learning rate of local models!!!') 

    parser.add_argument('--model_here_level', type=int, default=20) 
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--model_type', type=str, default='ResNet_20', choices=['Lenet', 'ResNet_18', 'ResNet_20', 'ResNet_34', 'ResNet_50'])

    parser.add_argument('--model_norm', type=str, default='bn', choices=['none', 'bn', 'in', 'ln', 'gn'])

    parser.add_argument('--model_split_mode', type=str, default='roll', choices=['roll', 'static', 'random'])

    parser.add_argument('--global_model_rate', type=float, default=MODEL_SPLIT_RATE['a'])

    parser.add_argument('--weighting', default='updates', type=str, choices=['avg', 'width', 'updates', 'updates_width']')

    parser.add_argument('--scale', type=lambda s: s == 'True', default=True)

    parser.add_argument('--mask', type=lambda s: s == 'True', default=True)  

    parser.add_argument('--data_set', type=str, default='FOOD101',
                        choices=['MNIST', 'FMNIST', 'CIFAR-10', 'CIFAR-100', 'SVHN', 'Tiny-Imagenet', 'FOOD101'])

    parser.add_argument('--data_partition_mode', type=str, default='non_iid_dirichlet_balanced',
                        choices=['iid', 'non_iid_dirichlet_unbalanced', 'non_iid_dirichlet_balanced']) 

    parser.add_argument('--non_iid_alpha', type=float, default=0.05)  

    parser.add_argument('--client_num', type=int, default=10)

    parser.add_argument('--selected_client_num', type=int, default=10)

    parser.add_argument('--device', type=torch.device, default='cuda')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--log_level', type=logging.getLevelName, default='INFO')

    parser.add_argument('--app_name', type=str, default='Fedrolex')

    args = parser.parse_args()

    args.rate = _get_model_split_rate(args)
    print(args.rate)

    super_params = args.__dict__.copy()
    del super_params['log_level']
    super_params['device'] = super_params['device'].type
    ff = f"{args.app_name}-{consistent_hash(super_params, code_len=64)}.pkl"
    ff = f"{os.path.dirname(__file__)}/Result/{ff}"
    if os.path.exists(ff):
        print(f"output file existed, skip task")
        exit(0)

    args.data_distributer = _get_data_distributer(args)


    return args

def _get_data_distributer(args):
    set_seed(args.seed + 5363)
    return DataDistributer(args)


def _get_model_split_rate(args):
    reta_list = []
    for i in range(args.client_num):
        reta_list.append(cal_level(args.model_here_level, i+1, args.client_num))
    return reta_list

def cal_level(rho, i, N):
    le = rho * i / N
    return 0.5 ** min([4, np.floor(le)])

