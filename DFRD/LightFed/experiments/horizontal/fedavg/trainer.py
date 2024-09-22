import logging
from collections import OrderedDict

import torch
from experiments.models.model import model_pull
from lightfed.tools.funcs import set_seed
from lightfed.tools.model import evaluation, CycleDataloader, get_parameters
from torch import nn


class ClientTrainer:
    def __init__(self, args, client_id):
        self.client_id = client_id
        self.device = args.device
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        self.lr_lm = args.lr_lm

        self.train_dataloader = args.data_distributer.get_client_train_dataloader(client_id)
        self.train_batch_data_iter = CycleDataloader(self.train_dataloader)
        self.train_label_list = args.data_distributer.get_client_label_list(client_id)
        self.test_dataloader = args.data_distributer.get_client_test_dataloader(client_id)


        self.res = {}

        set_seed(args.seed + 657)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def pull_local_model(self, args):
        self.model = model_pull(args).to(self.device)

    def clear(self):
        self.res = {}
        self.model = {}
        self.optimizer = {}
        torch.cuda.empty_cache()

    def train_locally_step(self, I, step):
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr_lm, weight_decay=self.weight_decay)
        self.model.train()
        LOSS = 0
        for tau in range(I):
            self.model.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)

            x, y = next(self.train_batch_data_iter)
            x = x.to(self.device)
            y = y.to(self.device)
            logit = self.model(x, self.train_label_list)
            loss = self.criterion(logit, y)
            loss.backward()
            self.optimizer.step()
            LOSS += loss
        LOSS = LOSS.detach().cpu().numpy() / I
        self.res.update(m_LOSS=LOSS)
        self.model.zero_grad(set_to_none=True)


    def get_eval_info(self, step, train_time=None):
        self.res.update(train_time=train_time)

        loss, acc, num = evaluation(model=self.model,
                                    dataloader=self.test_dataloader,
                                    criterion=self.criterion,
                                    model_params=None,
                                    device=self.device)
        self.res.update(test_loss=loss, test_acc=acc, test_sample_size=num)
