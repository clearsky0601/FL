import logging
import math
import os

import numpy as np
import pandas as pd
import torch
import copy
from experiments.models.model import model_pull
from lightfed.core import BaseServer
from lightfed.tools.aggregator import ModelStateAvgAgg, NumericAvgAgg
from lightfed.tools.funcs import (consistent_hash, formula, save_pkl, model_size, set_seed)
from lightfed.tools.model import evaluation, get_buffers, get_parameters, get_cpu_param
from torch import nn

from trainer import ClientTrainer
from collections import OrderedDict
import time



class ServerManager(BaseServer):
    def __init__(self, ct, args):
        super().__init__(ct)
        self.super_params = args.__dict__.copy()
        del self.super_params['data_distributer']
        del self.super_params['log_level']
        self.app_name = args.app_name
        self.device = args.device
        self.client_num = args.client_num                   
        self.selected_client_num = args.selected_client_num
        self.comm_round = args.comm_round                 
        self.I = args.I                                     
        self.eval_step_interval = args.eval_step_interval
        self.data_set = args.data_set

        self.model_split_mode = args.model_split_mode
        self.global_model_rate = args.global_model_rate    

        self.full_train_dataloader = args.data_distributer.get_train_dataloader()  
        self.full_test_dataloader = args.data_distributer.get_test_dataloader()   
        self.label_split = args.data_distributer.client_label_list

        self.local_sample_numbers = [len(args.data_distributer.get_client_train_dataloader(client_id).dataset)
                                     for client_id in range(args.client_num)]
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        set_seed(args.seed + 657)

        self.model = model_pull(self.super_params).to(self.device)  
        self.global_params = get_parameters(self.model.state_dict())
        self.rate = args.rate
        
        self.tmp_counts = {}
        for k, v in self.global_params.items():
            self.tmp_counts[k] = torch.ones_like(v)

        self.selected_clients = None
        self.param_idx = None
        
        self.weighting = args.weighting

        self.model_idxs = {}
        self.roll_idx = {}

        torch.cuda.empty_cache()

        self.client_params_collect_list = []

        self.client_test_acc_aggregator = NumericAvgAgg()

        self.comm_load = {client_id: 0 for client_id in range(args.client_num)}

        self.client_eval_info = []  
        self.global_train_eval_info = [] 

        self.unfinished_client_num = -1

        self.step = -1


    def start(self):
        logging.info("start...")
        self.next_step()

    def end(self):
        logging.info("end...")

        self.super_params['device'] = self.super_params['device'].type

        ff = f"{self.app_name}-{consistent_hash(self.super_params, code_len=64)}.pkl"
        logging.info(f"output to {ff}")

        result = {'super_params': self.super_params,
                  'global_train_eval_info': pd.DataFrame(self.global_train_eval_info),
                  'client_eval_info': pd.DataFrame(self.client_eval_info),
                  'comm_load': self.comm_load}
        save_pkl(result, f"{os.path.dirname(__file__)}/Result/{ff}")

        self._ct_.shutdown_cluster()

    def end_condition(self):
        return self.step > self.comm_round - 1

    def next_step(self):
        self.step += 1
        self.selected_clients = self._new_train_workload_arrage()  
        self.unfinished_client_num = self.selected_client_num
        local_parameters, self.param_idx = self.distribute() 

        for id_ in range(len(self.selected_clients)):
            client_id = self.selected_clients[id_]
            self._ct_.get_node('client', client_id) \
                .fed_client_train_step(step=self.step, global_params=local_parameters[id_])

    def _new_train_workload_arrage(self):
        if self.selected_client_num < self.client_num:
            selected_client = np.random.choice(range(self.client_num), self.selected_client_num, replace=False)
        elif self.selected_client_num == self.client_num:
            selected_client = np.array([i for i in range(self.client_num)])
        return selected_client


    def fed_finish_client_train_step(self,
                                     step,
                                     client_id,
                                     client_model_params,
                                     eval_info):
        logging.debug(f"train comm. round of client_id:{client_id} comm. round:{step} was finished")
        assert self.step == step
        self.client_eval_info.append(eval_info)

        weight = self.local_sample_numbers[client_id]

        if self.data_set in ['FOOD101', 'Tiny-Imagenet']:
            if self.step % 5 == 0:
                self.client_test_acc_aggregator.put(eval_info['test_acc'], weight)
        else:
            self.client_test_acc_aggregator.put(eval_info['test_acc'], weight)

        self.client_params_collect_list.append((client_id, client_model_params))


        if self.comm_load[client_id] == 0:
            self.comm_load[client_id] = model_size(client_model_params) / 1024 / 1024  

        self.unfinished_client_num -= 1
        if not self.unfinished_client_num:
            self.server_train_test_res = {'comm. round': self.step, 'client_id': 'server'}

            self.combine(self.client_params_collect_list, self.param_idx)
            self.model.load_state_dict(self.global_params, strict=True)

            if self.I == 0:
                if self.step % self.eval_step_interval == 0:
                    client_test_acc_avg = self.client_test_acc_aggregator.get_and_clear()
                    print('comm. round: {}, client_test_acc: {}'.format(self.step, client_test_acc_avg))

                    self._set_global_train_eval_info()
                    self.global_train_eval_info.append(self.server_train_test_res)
            else:
                if self.data_set in ['FOOD101', 'Tiny-Imagenet']:
                    if self.step % 5 == 0:
                        client_test_acc_avg = self.client_test_acc_aggregator.get_and_clear()
                        print('comm. round: {}, client_test_acc: {}'.format(self.step, client_test_acc_avg))
                        self._set_global_train_eval_info()
                        self.global_train_eval_info.append(self.server_train_test_res)
                else:
                    client_test_acc_avg = self.client_test_acc_aggregator.get_and_clear()
                    print('comm. round: {}, client_test_acc: {}'.format(self.step, client_test_acc_avg))
                    self._set_global_train_eval_info()
                    self.global_train_eval_info.append(self.server_train_test_res)

            logging.debug(f"train comm. round:{step} is finished")
            self.server_train_test_res = {}

            self.client_params_collect_list = []
            self.next_step()
    
    def distribute(self,):
        self.model_rate = np.array(self.rate)
        if self.model_split_mode == 'roll':
            param_idx = self.split_model_roll()
        elif self.model_split_mode == 'static':
            param_idx = self.split_model_static()
        elif self.model_split_mode == 'random':
            param_idx = self.split_model_random()

        local_parameters = [OrderedDict() for _ in range(len(self.selected_clients))]
        for k, v in self.global_params.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(self.selected_clients)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])])
                        else:
                            local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                    else:
                        local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                else:
                    local_parameters[m][k] = copy.deepcopy(v)
        return local_parameters, param_idx
     
    def split_model_roll(self):
        idx_i = [None for _ in range(len(self.selected_clients))]
        idx = [OrderedDict() for _ in range(len(self.selected_clients))]
        for k, v in self.global_params.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(self.selected_clients)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv' in k:
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[self.selected_clients[m]] / self.global_model_rate
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                
                                roll = self.step % output_size
                                model_idx = torch.arange(output_size, device=v.device)
                                model_idx = torch.roll(model_idx, roll, -1)
                                output_idx_i_m = model_idx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            elif 'shortcut' in k:
                                input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                output_idx_i_m = idx_i[m]
                            elif 'linear' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                                
                            else:
                                raise ValueError('Not valid k')
                            idx[m][k] = (output_idx_i_m, input_idx_i_m) 
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'linear' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass

        return idx
    
    def split_model_static(self):
        idx_i = [None for _ in range(len(self.selected_clients))]
        idx = [OrderedDict() for _ in range(len(self.selected_clients))]
        for k, v in self.global_params.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(self.selected_clients)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv' in k:
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[self.selected_clients[m]] / self.global_model_rate
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                model_idx = torch.arange(output_size, device=v.device)
                                output_idx_i_m = model_idx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            elif 'shortcut' in k:
                                input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                output_idx_i_m = idx_i[m]
                            elif 'linear' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                raise ValueError('Not valid k')
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'linear' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass

        return idx
    
    def split_model_random(self):
        idx_i = [None for _ in range(len(self.selected_clients))]
        idx = [OrderedDict() for _ in range(len(self.selected_clients))]
        for k, v in self.global_params.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(self.selected_clients)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv' in k:
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[self.selected_clients[m]] / self.global_model_rate
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                model_idx = torch.randperm(output_size, device=v.device)
                                output_idx_i_m = model_idx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            elif 'shortcut' in k:
                                input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                output_idx_i_m = idx_i[m]
                            elif 'linear' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                raise ValueError('Not valid k')
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'linear' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass

        return idx

    def combine(self, local_parameters, param_idx):
        count = OrderedDict()
        updated_parameters = copy.deepcopy(self.global_params) 
        tmp_counts_cpy = copy.deepcopy(self.tmp_counts)       
        for k, v in updated_parameters.items():
            parameter_type = k.split('.')[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32, device=self.device)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32, device=self.device)
            for m, local_pars in enumerate(local_parameters):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            if 'linear' in k:
                                label_split = self.label_split[local_pars[0]]
                                param_idx[m][k] = list(param_idx[m][k])
                                param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                tmp_v[torch.meshgrid(param_idx[m][k])] += self.tmp_counts[k][torch.meshgrid(param_idx[m][k])] * local_pars[1][k][label_split]
                                count[k][torch.meshgrid(param_idx[m][k])] += self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                tmp_counts_cpy[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                output_size = v.size(0)
                                scaler_rate = self.model_rate[local_pars[0]] / self.global_model_rate
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                if self.weighting == 'avg':
                                    K = 1
                                elif self.weighting == 'width':
                                    K = local_output_size
                                elif self.weighting == 'updates':
                                    K = self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                elif self.weighting == 'updates_width':
                                    K = local_output_size * self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]

                                tmp_v[torch.meshgrid(param_idx[m][k])] += K * local_pars[1][k]
                                count[k][torch.meshgrid(param_idx[m][k])] += K
                                tmp_counts_cpy[k][torch.meshgrid(param_idx[m][k])] += 1
                        else:
                            tmp_v[param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]] * local_pars[1][k]
                            count[k][param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]]
                            tmp_counts_cpy[k][param_idx[m][k]] += 1
                    else:
                        if 'linear' in k:
                            label_split = self.label_split[local_pars[0]]
                            param_idx[m][k] = param_idx[m][k][label_split]
                            tmp_v[param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]] * local_pars[1][k][label_split]
                            count[k][param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]]
                            tmp_counts_cpy[k][param_idx[m][k]] += 1
                        else:
                            tmp_v[param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]] * local_pars[1][k]
                            count[k][param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]]
                            tmp_counts_cpy[k][param_idx[m][k]] += 1
                else:
                    tmp_v += self.tmp_counts[k] * local_pars[1][k]
                    count[k] += self.tmp_counts[k]
                    tmp_counts_cpy[k] += 1
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
            self.tmp_counts = copy.deepcopy(tmp_counts_cpy)
        self.global_params = updated_parameters
        return
        
    def _set_global_train_eval_info(self):
        loss, acc, num = evaluation(model=self.model,
                                    dataloader=self.full_test_dataloader,
                                    criterion=self.criterion,
                                    device=self.device,
                                    eval_full_data=True)
        torch.cuda.empty_cache()
        self.server_train_test_res.update(test_loss=loss, test_acc=acc, test_sample_size=num)

        logging.info(f"global eval info:{self.server_train_test_res}")

class ClientManager(BaseServer):
    def __init__(self, ct, args):
        super().__init__(ct)
        self.I = args.I
        self.device = args.device
        self.client_id = self._ct_.role_index
        self.model_type = args.model_type
        self.data_set = args.data_set

        self.args = args.__dict__.copy()
        del self.args['data_distributer']

        self.trainer = ClientTrainer(args, self.client_id)
        self.step = 0

    def start(self):
        logging.info("start...")

    def end(self):
        logging.info("end...")

    def end_condition(self):
        return False

    def fed_client_train_step(self, step, global_params):
        self.step = step
        logging.debug(f"training client_id:{self.client_id}, comm. round:{step}")

        self.trainer.res = {'communication round': step, 'client_id': self.client_id}
        
        self.trainer.pull_local_model(self.args, model_rate = self.args['rate'][self.client_id])  
        self.trainer.model.load_state_dict(global_params, strict=True)

        self.timestamp = time.time()
        self.trainer.train_locally_step(self.I, step)
        curr_timestamp = time.time()
        train_time = curr_timestamp - self.timestamp

        model_params = get_parameters(self.trainer.model.state_dict())
        self.finish_train_step(model_params, train_time)
        self.trainer.clear()
        torch.cuda.empty_cache()

    def finish_train_step(self, model_params, train_time):
        self.trainer.get_eval_info(self.step, train_time)
        logging.debug(f"finish_train_step comm. round:{self.step}, client_id:{self.client_id}")

        self._ct_.get_node("server") \
            .set(deepcopy=False) \
            .fed_finish_client_train_step(self.step,
                                          self.client_id,
                                          model_params,
                                          self.trainer.res)