import copy
import pathlib
import time
from typing import Union

import numpy as np
# from sklearnex import patch_sklearn, config_context
# patch_sklearn()

# from sklearn.cluster import KMeans
# from ATTA.utils.fast_pytorch_kmeans import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from typing import Literal

from torch import nn
import torch
# import models for resnet18
from munch import Munch
from ATTA import register
from ATTA.utils.config_reader import Conf
from ATTA.data.loaders.fast_data_loader import InfiniteDataLoader, FastDataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from .Base import AlgBase
import pandas as pd
from ATTA.definitions import STORAGE_DIR
import random



@register.alg_register
class Random(AlgBase):
    def __init__(self, config: Conf):
        super(Random, self).__init__(config)

        self.teacher = copy.deepcopy(self.model.to('cpu'))
        self.model.cuda()
        self.teacher.cuda()
        self.update_teacher(0)  # copy student to teacher

        self.budgets = 0
        self.anchors = None
        self.source_anchors = None
        self.buffer = []
        self.n_clusters = 10
        # self.nc_increase = self.config.atta.SimATTA.nc_increase
        self.source_n_clusters = 100
        self.batch_num = 0

        # self.cold_start = self.config.atta.SimATTA.cold_start

        self.consistency_weight = 0
        self.alpha_teacher = 0
        self.accumulate_weight = True
        self.weighted_entropy: Union[Literal['low', 'high', 'both'], None] = 'both'
        self.aggressive = True
        # self.beta = self.config.atta.SimATTA.beta
        self.alpha = 0.2

        self.target_cluster = True 
        self.LE = True 
        self.vis_round = 0


    def __call__(self, *args, **kwargs):
        # super(SimATTA, self).__call__()
        self.continue_result_df = pd.DataFrame(
            index=['Current domain', 'Budgets', *(i for i in self.config.dataset.test_envs), 'Frame AVG'],
            columns=[*(i for i in self.config.dataset.test_envs), 'Test AVG'], dtype=float)
        self.random_result_df = pd.DataFrame(
            index=['Current step', 'Budgets', *(i for i in self.config.dataset.test_envs), 'Frame AVG'],
            columns=[*(i for i in range(4)), 'Test AVG'], dtype=float)

        self.enable_bn(self.model)
        if 'ImageNet' not in self.config.dataset.name:
            for env_id in self.config.dataset.test_envs:
                acc = self.test_on_env(env_id)[1]
                self.continue_result_df.loc[env_id, self.config.dataset.test_envs[0]] = acc
                self.random_result_df.loc[env_id, self.config.dataset.test_envs[0]] = acc

        all_batches = 0
        for adapt_id in self.config.dataset.test_envs[1:]:
            all_batches += self.fast_loader[adapt_id]._length
            
        avg_active_num = self.config.atta.budgets
        base_avg_active_num = int(avg_active_num/ int(all_batches)) 
        active_samples_batch_num = torch.zeros(int(all_batches) , dtype=torch.int) + base_avg_active_num
        buff_active_num = avg_active_num - base_avg_active_num * int(all_batches)
        random_numbers = random.sample(range(len(active_samples_batch_num)),buff_active_num)
        for index in range(len(active_samples_batch_num)):
            if index in random_numbers:
                active_samples_batch_num[index] += 1
        
        for adapt_id in self.config.dataset.test_envs[1:]:
            self.continue_result_df.loc['Current domain', adapt_id] = self.adapt_on_env(self.fast_loader, adapt_id, active_samples_batch_num)
            self.continue_result_df.loc['Budgets', adapt_id] = self.budgets
            print(self.budgets)
            if 'ImageNet' not in self.config.dataset.name:
                for env_id in self.config.dataset.test_envs:
                    self.continue_result_df.loc[env_id, adapt_id] = self.test_on_env(env_id)[1]
                    
        all_batches = 0
        for adapt_id in range(len(self.target_loader)):
            all_batches += self.target_loader[adapt_id]._length
        # avg_active_num = 200
        base_avg_active_num = int(avg_active_num/ int(all_batches)) 
        active_samples_batch_num = torch.zeros(int(all_batches) , dtype=torch.int) + base_avg_active_num
        buff_active_num = avg_active_num - base_avg_active_num * int(all_batches)
        random_numbers = random.sample(range(len(active_samples_batch_num)),buff_active_num)
        for index in range(len(active_samples_batch_num)):
            if index in random_numbers:
                active_samples_batch_num[index] += 1
        # self.__init__(self.config)
        # for target_split_id in range(4):
        #     self.random_result_df.loc['Current step', target_split_id] = self.adapt_on_env(self.target_loader, target_split_id, active_samples_batch_num)
        #     self.random_result_df.loc['Budgets', target_split_id] = self.budgets
        #     print(self.budgets)
        #     if 'ImageNet' not in self.config.dataset.name:
        #         for env_id in self.config.dataset.test_envs:
        #             self.random_result_df.loc[env_id, target_split_id] = self.test_on_env(env_id)[1]

        print(f'#IM#\n{self.continue_result_df.round(4).to_markdown()}\n')
        # print(self.random_result_df.round(4).to_markdown(), '\n')
        self.continue_result_df.round(4).to_csv(f'{self.config.log_file}.csv')
        # self.random_result_df.round(4).to_csv(f'{self.config.log_file}.csv', mode='a')



    @torch.no_grad()
    def val_anchor(self, loader):
        self.model.eval()
        val_loss = 0
        val_acc = 0
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            output = self.fc(self.encoder(data))
            val_loss += self.config.metric.loss_func(output, target, reduction='sum').item()
            val_acc += self.config.metric.score_func(target, output) * len(data)
        val_loss /= len(loader.sampler)
        val_acc /= len(loader.sampler)
        del loader
        return val_loss, val_acc

    def update_teacher(self, alpha_teacher):  # , iteration):
        for t_param, s_param in zip(self.teacher.parameters(), self.model.parameters()):
            t_param.data[:] = alpha_teacher * t_param[:].data[:] + (1 - alpha_teacher) * s_param[:].data[:]
        if not self.config.model.freeze_bn:
            for tm, m in zip(self.teacher.modules(), self.model.modules()):
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    tm.running_mean = alpha_teacher * tm.running_mean + (1 - alpha_teacher) * m.running_mean
                    tm.running_var = alpha_teacher * tm.running_var + (1 - alpha_teacher) * m.running_var

    @torch.enable_grad()
    def cluster_train(self, target_anchors, source_anchors):
        self.model.train()

       
        target_loader = FastDataLoader(TensorDataset(target_anchors.data, target_anchors.target), weights=None,
                                             batch_size=self.config.train.train_bs, num_workers=self.config.num_workers)
        alpha = target_anchors.num_elem() / (target_anchors.num_elem() + source_anchors.num_elem())
      
        ST_loader = iter(target_loader)
        # val_loader = FastDataLoader(TensorDataset(target_anchors.data, target_anchors.target), weights=None,
        #                             batch_size=self.config.train.train_bs, num_workers=self.config.num_workers)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.atta.Random.lr, momentum=0.9)
        # print('Cluster train')
        delay_break = False
        loss_window = []
        tol = 0
        lowest_loss = float('inf')
        for i, (T_data, T_targets) in enumerate(ST_loader):
            T_data, T_targets = T_data.cuda(), T_targets.cuda()
            L_T = self.one_step_train( T_data, T_targets, alpha, optimizer)
            # self.update_teacher(self.alpha_teacher)
            
        # del source_loader
        del target_loader
        del ST_loader
        

    def one_step_train(self, T_data, T_targets, alpha, optimizer):
        # print('one step train')
        L_T = self.config.metric.loss_func(self.model(T_data), T_targets)
        loss =  L_T
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return L_T

    def softmax_entropy(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        if y is None:
            if x.shape[1] == 1:
                x = torch.cat([x, -x], dim=1)
            return -(x.softmax(1) * x.log_softmax(1)).sum(1)
        else:
            return - 0.5 * (x.softmax(1) * y.log_softmax(1)).sum(1) - 0.5 * (y.softmax(1) * x.log_softmax(1)).sum(1)

    def update_anchors(self, anchors, data, target, feats, weight):
        if anchors is None:
            anchors = Munch()
            anchors.data = data
            anchors.target = target
            anchors.feats = feats
            anchors.weight = weight
            anchors.num_elem = lambda: len(anchors.data)
        else:
            anchors.data = torch.cat([anchors.data, data])
            anchors.target = torch.cat([anchors.target, target])
            anchors.feats = torch.cat([anchors.feats, feats])
            anchors.weight = torch.cat([anchors.weight, weight])
        return anchors

    def update_anchors_feats(self, anchors):
        # sequential_data = torch.arange(200)[:, None]
        anchors_loader = FastDataLoader(TensorDataset(anchors.data), weights=None,
                                        batch_size=32, num_workers=self.config.num_workers, sequential=True)

        anchors.feats = None
        self.model.eval()
        for data in anchors_loader:
            # print(data)
            data = data[0].cuda()
            if anchors.feats is None:
                anchors.feats = self.model[0](data).cpu().detach()
            else:
                anchors.feats = torch.cat([anchors.feats, self.model[0](data).cpu().detach()])
        del anchors_loader
        return anchors

    @torch.no_grad()
    def adapt_on_env(self, loader, env_id, active_samples_batch_num):
        # beta_func = torch.distributions.beta.Beta(0.8, 0.8)
        acc = 0
        for data, target in tqdm(loader[env_id]):
            data, target = data.cuda(), target.cuda()
            if self.anchors == None:
                outputs, closest, self.anchors = self.sample_select(self.model, data, target, self.anchors,active_samples_batch_num, int(self.n_clusters), 1, ent_bound=self.config.atta.Random.eh, incremental_cluster=self.target_cluster)
            elif self.anchors.num_elem() < self.config.atta.budgets:
                    outputs, closest, self.anchors = self.sample_select(self.model, data, target, self.anchors,active_samples_batch_num, int(self.n_clusters), 1, ent_bound=self.config.atta.Random.eh, incremental_cluster=self.target_cluster)
            else:
                closest = []
            self.source_anchors = self.update_anchors(None, torch.tensor([]), None, None, None)
            self.budgets += len(closest)
            self.batch_num += 1

            print(self.anchors.num_elem(), self.source_anchors.num_elem())
            if self.source_anchors.num_elem() > 0:
                self.cluster_train(self.anchors, self.source_anchors)
            else:
                self.cluster_train(self.anchors, self.anchors)
            # self.anchors = self.update_anchors_feats(self.anchors)
            self.model.eval()
            feats = self.model[0](data)
            outputs = self.model[1](feats)
            acc += self.config.metric.score_func(target, outputs).item() * data.shape[0]
        acc /= len(loader[env_id].sampler)
        print(f'#IN#Env {env_id} real-time Acc.: {acc:.4f}')
        return acc

    @torch.no_grad()
    def sample_select(self, model, data, target, anchors,active_samples_batch_num, n_clusters, ent_beta, use_pseudo_label=False, ent_bound=1e-2, incremental_cluster=False):
        model.eval()
        feats = model[0](data)
        outputs = model[1](feats)
        data = data.cpu().detach()
        feats = feats.cpu().detach()
        target = target.cpu().detach()
        
        if True:
            if anchors == None:
                gt_mask = torch.rand(target.shape[0], device=target.device) < self.config.atta.Random.al_rate
                closest = torch.where(gt_mask == 1)[0]
            else:
                leave_active_buff = self.config.atta.budgets - anchors.num_elem()
                gt_mask = torch.rand(target.shape[0], device=target.device) < self.config.atta.Random.al_rate
                closest = torch.where(gt_mask == 1)[0]
                if len(closest) > leave_active_buff:
                    closest = random.sample(range(len(target)), leave_active_buff)
            
            
        weights = torch.tensor(1.0).unique(return_counts=True)[1]

        anchors = self.update_anchors(anchors, data[closest], target[closest], feats[closest], weights)

        return outputs, closest, anchors

    def enable_bn(self, model):
        if not self.config.model.freeze_bn:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.momentum = 0.1
