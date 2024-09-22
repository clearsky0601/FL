import logging
from collections import OrderedDict

import torch
from experiments.models.model import model_pull
from lightfed.tools.funcs import set_seed, grad_True
from lightfed.tools.model import evaluation, CycleDataloader, get_parameters
from torch import nn
from collections import Counter


class ClientTrainer:
    def __init__(self, args, client_id):
        self.client_id = client_id
        self.device = args.device
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        self.lr_lm = args.lr_lm
        self.mom = args.mom
        self.data_set = args.data_set

        self.train_dataloader = args.data_distributer.get_client_train_dataloader(client_id)
        self.train_batch_data_iter = CycleDataloader(self.train_dataloader)
        self.train_label_list = args.data_distributer.get_client_label_list(client_id)
        self.test_dataloader = args.data_distributer.get_client_test_dataloader(client_id)

        self.res = {}

        set_seed(args.seed + 657)
        
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def pull_local_model(self, args, model_rate=1.0):
        self.model = model_pull(args, model_rate=model_rate, track=False).to(self.device)

    def clear(self):
        self.res = {}
        self.optimizer = None
        torch.cuda.empty_cache()

    def train_locally_step(self, I, step):
        """算法的第5行
        """
        grad_True(self.model)
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr_lm, weight_decay=self.weight_decay)
        self.model.train()
        LOSS = 0
        self.gen_client_bn_layers_dict = {}
        self.gen_client_bn_layers = []
        for i in range(I):
            self.model.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)
            x, y = next(self.train_batch_data_iter)
            x = x.to(self.device)
            y = y.to(self.device)
            logit, local_bn = self.model(x, label_list=self.train_label_list, bn_or_not=True)
            loss = self.criterion(logit, y)
            loss.backward()
            self.optimizer.step()
            LOSS += loss
            
            for id, bn_i in enumerate(local_bn):
                if i == 0:
                    mean = bn_i.mean([0, 2, 3]).clone().detach()
                    var = bn_i.var([0, 2, 3]).clone().detach()
                else:
                    mean = (1 - self.mom) * self.gen_client_bn_layers_dict[id][0] + self.mom * bn_i.mean([0, 2, 3]).clone().detach() 
                    var = (1 - self.mom) * self.gen_client_bn_layers_dict[id][1] + self.mom * bn_i.var([0, 2, 3]).clone().detach()
                self.gen_client_bn_layers_dict[id] = (mean, var)
        
        for id in self.gen_client_bn_layers_dict:
            self.gen_client_bn_layers.append(self.gen_client_bn_layers_dict[id])

        LOSS = LOSS.detach().cpu().numpy() / I
        self.res.update(m_LOSS=LOSS)


    def get_eval_info(self, step, train_time=None):
        self.res.update(train_time=train_time)
        if self.data_set in ['Tiny-Imagenet', 'FOOD101']:
            if step % 5 == 0:
                loss, acc, num = evaluation(model=self.model,
                                            dataloader=self.test_dataloader,
                                            criterion=self.criterion,
                                            model_params=None,
                                            device=self.device)
                self.res.update(test_loss=loss, test_acc=acc, test_sample_size=num)
        else:
            loss, acc, num = evaluation(model=self.model,
                                            dataloader=self.test_dataloader,
                                            criterion=self.criterion,
                                            model_params=None,
                                            device=self.device)
            self.res.update(test_loss=loss, test_acc=acc, test_sample_size=num)
        
