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
        self.data_set = args.data_set

        self.train_dataloader = args.data_distributer.get_client_train_dataloader(client_id)
        self.train_batch_data_iter = CycleDataloader(self.train_dataloader)
        self.train_label_list = args.data_distributer.get_client_label_list(client_id)
        self.test_dataloader = args.data_distributer.get_client_test_dataloader(client_id)

        self.unique_labels = args.data_distributer.class_num
        self.qualified_labels = [i for i in range(self.unique_labels)]
        self.label_counts = {label: 0 for label in range(self.unique_labels)}

        self.cache_y = []
        self.res = {}

        set_seed(args.seed + 657)
        
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
    def pull_local_model(self, args, model_rate):
        self.model = model_pull(args, model_rate=model_rate).to(self.device)

    def update_label_counts(self, counter_dict):
        for label in counter_dict:
            self.label_counts[int(label)] += counter_dict[label]

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:1 for label in range(self.unique_labels)}

    def clear(self):
        self.res = {}
        self.optimizer = None
        torch.cuda.empty_cache()

    def train_locally_step(self, I, step):
        grad_True(self.model)
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr_lm, weight_decay=self.weight_decay)
        self.model.train()
        LOSS = 0
        for _ in range(I):
            self.model.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)
            x, y = next(self.train_batch_data_iter)
            x = x.to(self.device)
            y = y.to(self.device)
            if list(y.cpu().numpy()) not in self.cache_y:
                self.cache_y.append(list(y.cpu().numpy()))
            logit = self.model(x, label_list=self.train_label_list)
            loss = self.criterion(logit, y)
            loss.backward()
            self.optimizer.step()
            LOSS += loss
        LOSS = LOSS.detach().cpu().numpy() / I
        self.res.update(m_LOSS=LOSS)
        self.model.zero_grad(set_to_none=True)

        for y in self.cache_y:
            counter_dict = dict(Counter(y))
            self.update_label_counts(counter_dict)


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
        
