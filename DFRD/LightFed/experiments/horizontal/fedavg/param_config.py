import argparse
import logging
import os

import numpy as np
import torch
from experiments.datasets.data_distributer import DataDistributer
from lightfed.tools.funcs import consistent_hash, set_seed


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--comm_round', type=int, default=100)

    parser.add_argument('--I', type=int, default=20, help='synchronization interval')

    parser.add_argument('--batch_size', type=int, default=64)  
    
    parser.add_argument('--eval_step_interval', type=int, default=5)

   
    parser.add_argument('--eval_batch_size', type=int, default=256)

    parser.add_argument('--lr_lm', type=float, default=0.01)

    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--model_type', type=str, default='Lenet', choices=['Lenet', 'ResNet_18', 'ResNet_34', 'ResNet_50'])

    parser.add_argument('--model_norm', type=str, default='bn', choices=['none', 'bn', 'in', 'ln', 'gn'])

    parser.add_argument('--scale', type=lambda s: s == 'True', default=True)

    parser.add_argument('--mask', type=lambda s: s == 'True', default=True) 

    parser.add_argument('--data_set', type=str, default='FOOD101',
                        choices=['MNIST', 'FMNIST', 'CIFAR-10', 'CIFAR-100', 'SVHN', 'Tiny-Imagenet', 'FOOD101'])

    parser.add_argument('--data_partition_mode', type=str, default='non_iid_dirichlet_unbalanced',
                        choices=['iid', 'non_iid_dirichlet_unbalanced', 'non_iid_dirichlet_balanced'])

    parser.add_argument('--non_iid_alpha', type=float, default=0.01) 

    parser.add_argument('--client_num', type=int, default=10)

    parser.add_argument('--selected_client_num', type=int, default=10)

    parser.add_argument('--device', type=torch.device, default='cuda')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--log_level', type=logging.getLevelName, default='INFO')

    parser.add_argument('--app_name', type=str, default='Fedavg')

    args = parser.parse_args()

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

