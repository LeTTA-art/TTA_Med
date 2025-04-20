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
from torch.nn.functional import cross_entropy

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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, SpectralClustering,MeanShift
import torch.nn.functional as F


@register.alg_register
class ATMO_xary(AlgBase):
    def __init__(self, config: Conf):
        super(ATMO_xary, self).__init__(config)

        self.teacher = copy.deepcopy(self.model.to('cpu'))

        self.model.cuda()
        self.teacher.cuda()
        self.update_teacher(0)  # copy student to teacher

        self.budgets = 0
        self.anchors = None
        self.source_anchors = None
        self.added_anchors = None
        self.buffer = []
        self.n_clusters = 10
        self.nc_increase = self.config.atta.SimATTA.nc_increase
        self.source_n_clusters = 100
        self.batch_num = 0

        self.cold_start = self.config.atta.SimATTA.cold_start

        self.consistency_weight = 0
        self.alpha_teacher = 0
        self.accumulate_weight = True
        self.weighted_entropy: Union[Literal['low', 'high', 'both'], None] = 'both'
        self.aggressive = True
        self.beta = self.config.atta.SimATTA.beta
        self.alpha = 0.2

        self.target_cluster = True if self.config.atta.SimATTA.target_cluster else False
        self.LE = True if self.config.atta.SimATTA.LE else False
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
        # neighbor_idx = [3,5,4]
        # weight_idx = [1.0,1.0,1.0]
        # neighbor_idx = [3,5,4]
        weight_idx = [1.0,1.0,1.0,1.0]
        neighbor_idx = [3,1,1,4]
        relax_weight  = [1.0,1.0,1.0]
        # weight_idx = [0.5,0.5,0.5]
        all_batches = 0
        for adapt_id in self.config.dataset.test_envs[1:]:
            all_batches += self.fast_loader[adapt_id]._length
            
        avg_active_num = self.config.atta.budgets
        base_avg_active_num = int(avg_active_num/ int(all_batches)) 
        active_samples_batch_num = torch.zeros(int(all_batches) , dtype=torch.int) + base_avg_active_num
        buff_active_num = avg_active_num - base_avg_active_num * int(all_batches)
        random_numbers = random.sample(range(len(active_samples_batch_num)), buff_active_num)
        for index in range(len(active_samples_batch_num)):
            if index in random_numbers:
                active_samples_batch_num[index] += 1
        n_idx = 0
        for adapt_id in self.config.dataset.test_envs[1:]:
            self.continue_result_df.loc['Current domain', adapt_id] = self.adapt_on_env(self.fast_loader, adapt_id, active_samples_batch_num,neighbor_idx[n_idx],weight_idx[n_idx],relax_weight[n_idx])
            self.continue_result_df.loc['Budgets', adapt_id] = self.budgets
            print(self.budgets)
            if 'ImageNet' not in self.config.dataset.name:
                for env_id in self.config.dataset.test_envs:
                    self.continue_result_df.loc[env_id, adapt_id] = self.test_on_env(env_id)[1]
            n_idx += 1
                    
        print(f'#IM#\n{self.continue_result_df.round(4).to_markdown()}')
        self.continue_result_df.round(4).to_csv(f'{self.config.log_file}.csv')



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
    def cluster_train(self, target_anchors, source_anchors,flag):
        self.model.train()

        source_loader = FastDataLoader(TensorDataset(source_anchors.data, source_anchors.target), weights=None,
                                           batch_size=self.config.train.train_bs,
                                           num_workers=self.config.num_workers)
        target_loader = FastDataLoader(TensorDataset(target_anchors.data, target_anchors.target,target_anchors.weight), weights=None,
                                             batch_size=self.config.train.train_bs, num_workers=self.config.num_workers)
      
        alpha = target_anchors.num_elem() / (target_anchors.num_elem() + source_anchors.num_elem())
        if source_anchors.num_elem() < self.cold_start:
            alpha = min(0.2, alpha)

        ST_loader = iter(zip(source_loader, target_loader))
        # val_loader = FastDataLoader(TensorDataset(target_anchors.data, target_anchors.target), weights=None,
        #                             batch_size=self.config.train.train_bs, num_workers=self.config.num_workers)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.atta.SimATTA.lr, momentum=0.9)
        # print('Cluster train')
        delay_break = False
        loss_window = []
        tol = 0
        lowest_loss = float('inf')
        for i, ((S_data, S_targets), (T_data, T_targets,T_weight)) in enumerate(ST_loader):
            S_data, S_targets = S_data.cuda(), S_targets.cuda()
            T_data, T_targets,T_weight = T_data.cuda(), T_targets.cuda(),T_weight.cuda()
            L_T = self.one_step_train(S_data, S_targets, T_data, T_targets, T_weight, alpha, flag,optimizer)
            # self.update_teacher(self.alpha_teacher)
            if len(loss_window) < self.config.atta.SimATTA.stop_tol:
                loss_window.append(L_T.item())
            else:
                mean_loss = np.mean(loss_window)
                tol += 1
                if mean_loss < lowest_loss:
                    lowest_loss = mean_loss
                    tol = 0
                if tol > 5:
                    break
                loss_window = []
            if 'ImageNet' in self.config.dataset.name or 'CIFAR' in self.config.dataset.name:
                if i > self.config.atta.SimATTA.steps:
                    break
        del source_loader
        del target_loader
        del ST_loader
    
    import torch.nn.functional as F

    def weighted_cross_entropy_loss(self, input, target, sample_weights, reduction='mean'):
        # 计算未加权的交叉熵损失（shape: [batch_size]）
        loss = F.cross_entropy(input, target,reduction='none')
        
        # 将损失与样本权重相乘
        weighted_loss = loss * sample_weights

        # 根据 reduction 参数，选择如何返回损失
        return weighted_loss.mean()
        
            
    

    def one_step_train(self, S_data, S_targets, T_data, T_targets, weights, alpha, flag, optimizer):
        # print('one step train')
        output = self.model(T_data)
        # L_T = self.config.metric.loss_func(output, T_targets)
        L_T = self.weighted_cross_entropy_loss(output, T_targets,weights)
        # print(alpha)
        if flag == 1:
            loss = L_T
        else:
            L_S = self.config.metric.loss_func(self.model(S_data), S_targets)
            loss = 0.1* L_S + L_T
       
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

    def update_anchors(self, anchors, data, target, feats, weight,active_f):
        if anchors is None:
            anchors = Munch()
            anchors.data = data
            anchors.target = target
            anchors.feats = feats
            anchors.weight = weight
            anchors.active_f = active_f
            anchors.num_elem = lambda: len(anchors.data)
        else:
            anchors.data = torch.cat([anchors.data, data])
            anchors.target = torch.cat([anchors.target, target])
            anchors.feats = torch.cat([anchors.feats, feats])
            anchors.weight = torch.cat([anchors.weight, weight])
            anchors.active_f = torch.cat([anchors.active_f, active_f])
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
    def adapt_on_env(self, loader, env_id, active_samples_batch_num,k_neighbor,weight_domain,relax_weight):
        # beta_func = torch.distributions.beta.Beta(0.8, 0.8)
        acc = 0
        for data, target in tqdm(loader[env_id]):
            data, target = data.cuda(), target.cuda()
            outputs, closest, self.anchors = self.sample_select(self.model, data, target, self.anchors,active_samples_batch_num, k_neighbor,weight_domain,relax_weight, int(self.n_clusters), 1, ent_bound=self.config.atta.SimATTA.eh, incremental_cluster=self.target_cluster)
            
            if self.LE:
                _, _, self.source_anchors = self.sample_select(self.teacher, data, target, self.source_anchors, self.source_n_clusters,active_samples_batch_num, k_neighbor,weight_domain, relax_weight, 0,
                                                               use_pseudo_label=True, ent_bound=self.config.atta.SimATTA.el, incremental_cluster=False)
            else:
                self.source_anchors = self.update_anchors(None, torch.tensor([]), None, None, None,None)
            if not self.target_cluster:
                self.n_clusters = 0
            self.source_n_clusters = 100

            self.budgets += len(closest)
            self.n_clusters += self.nc_increase
            self.source_n_clusters += 1
            self.batch_num += 1
            flag = 1.0

            print(self.anchors.num_elem(), self.source_anchors.num_elem())
            if int(torch.sum(self.anchors.active_f))/self.config.atta.budgets > self.config.atta.SimATTA.begin:
                flag = 0
                self.cluster_train(self.anchors, self.source_anchors,flag)
            else:
                self.cluster_train(self.anchors, self.anchors,flag)
            # 挑选的主动标签的数量超过了一半，就开始使用source anchors 不然只使用anchors.
            #前面先不使用保存的历史样本，防止模型对于当前domain的学习
            if self.source_anchors.num_elem() > 0:
                self.cluster_train(self.anchors, self.source_anchors,flag)
            else:
                self.cluster_train(self.anchors, self.anchors,flag)
            self.anchors = self.update_anchors_feats(self.anchors)
            self.model.eval()
            feats = self.model[0](data)
            outputs = self.model[1](feats)
            acc += self.config.metric.score_func(target, outputs).item() * data.shape[0]
        acc /= len(loader[env_id].sampler)
        print(f'#IN#Env {env_id} real-time Acc.: {acc:.4f}')
        return acc

    @torch.no_grad()
    def sample_select(self, model, data, target, anchors,active_samples_batch_num,k_neighbor, weight_domain,relax_weight, n_clusters, ent_beta, use_pseudo_label=False, ent_bound=1e-2, incremental_cluster=False):
        model.eval()
        feats = model[0](data)
        outputs = model[1](feats)
        pseudo_label = outputs.argmax(1).cpu().detach()
        data = data.cpu().detach()
        feats = feats.cpu().detach()
        target = target.cpu().detach()
        entropy = self.softmax_entropy(outputs).cpu()
        added_label = []
        if not incremental_cluster:
            feats4cluster = feats
            sample_weight = torch.ones(len(feats), dtype=torch.float)
            #Relax
            from ATTA.utils.fast_pytorch_kmeans import KMeans
            from joblib import parallel_backend
            kmeans = KMeans(n_clusters=self.config.atta.SimATTA.s_k, n_init=10, device=self.config.device).fit(
                feats4cluster.cuda(),
                sample_weight=sample_weight.cuda())
            with parallel_backend('threading', n_jobs=8):
                centers, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feats4cluster)
            closest = []
            for index in range(self.config.atta.SimATTA.s_k):
                cls_idx = np.where(kmeans.labels_ == index)[0]
                cls_entropy = entropy[cls_idx]
                if len(cls_entropy) <=3:
                    _,can_idx = torch.topk(cls_entropy,dim=-1,largest=False,k=len(cls_entropy))
                else:
                    _,can_idx = torch.topk(cls_entropy,dim=-1,largest=False,k=self.config.atta.SimATTA.beli_k)
                if len(can_idx) == 1:
                    closest.append(cls_idx[can_idx])
                else:
                    can_idx = cls_idx[can_idx]
                    for e_idx in range(len(can_idx)):
                        closest.append(can_idx[e_idx])
                
            closest = torch.tensor(closest)
            
        else:
            feats4cluster = feats
            sample_weight = torch.ones(len(feats), dtype=torch.float)
            feats_normalized = F.normalize(feats, p=2, dim=1)
            
            #Relax
            from ATTA.utils.fast_pytorch_kmeans import KMeans
            from joblib import parallel_backend
            kmeans = KMeans(n_clusters=active_samples_batch_num[self.batch_num], n_init=10, device=self.config.device).fit(
                feats4cluster.cuda(),
                sample_weight=sample_weight.cuda())
            # with parallel_backend('threading', n_jobs=8):
            #     centers, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feats4cluster)
            stored_active_samples = copy.deepcopy(self.anchors)
            idx_sim_sto = []
            if stored_active_samples == None:
                idx_sim_sto = []
            else:
                stored_active_features = model[0](stored_active_samples.data[stored_active_samples.active_f == 1.0].cuda())
                stored_feats_normalized = F.normalize(stored_active_features, p=2, dim=1)
                cosine_similarity_matrix_sto = torch.mm(stored_feats_normalized.detach().cpu(), feats_normalized.t())
                #最近的样本距离所有样本的距离，选择互惠样本，将其存在一个batch中
                sim_values,idx_sim = torch.topk(cosine_similarity_matrix_sto, dim=-1,largest=True,k=10)
                sim_idx = torch.where(sim_values > relax_weight)
                for index in range(len(sim_idx[0])):
                    idx_sim_sto.append(int(idx_sim[int(sim_idx[0][index]),int(sim_idx[1][index])]))
                            
            
            closest = []
            for index in range(active_samples_batch_num[self.batch_num]):
                cls_idx = np.where(kmeans.labels_ == index)[0]
                cls_entropy = entropy[cls_idx]
                _,can_idx = torch.topk(cls_entropy,dim=-1,largest=True,k=len(cls_entropy))
                if len(can_idx) == 1:
                    closest.append(cls_idx[can_idx])
                else:  
                    can_idx = cls_idx[can_idx]
                    for e_idx in range(len(can_idx)):
                        if can_idx[e_idx] not in idx_sim_sto:
                            closest.append(can_idx[e_idx])
                            break
            closest = torch.tensor(closest)
            
            data_added = []
            added_label = []
            all_added_idx = []
            #继续求距离,然后进行扩充:
            # 计算余弦相似度矩阵 (100x100)
            cosine_similarity_matrix = torch.mm(feats_normalized, feats_normalized.t())
            _,idx_sim = torch.topk(cosine_similarity_matrix, dim=-1,largest=True,k=k_neighbor)
            idx_sim_can = idx_sim[closest]
            for index in range(len(idx_sim_can)):
                can_i_s = idx_sim_can[index]
                for s_i in range(len(can_i_s)):
                    if s_i == 0:
                        continue
                    if int(closest[index]) in idx_sim[can_i_s[s_i]] and can_i_s[s_i] not in all_added_idx:
                        if len(data_added) == 0:
                            data_added = data[can_i_s[s_i]]
                            data_added = data_added.unsqueeze(0)
                        else:
                            data_added = torch.cat([data_added, data[can_i_s[s_i]].unsqueeze(0)])
                        added_label.append(int(target[closest[index]]))
                        all_added_idx.append(int(can_i_s[s_i]))
            
        weights_t = []
        active_f_t = []
        active_f_a = []
        for index in range(len(closest)):
            weights_t.append(1.0)
            active_f_t.append(1.0)
        weights_a = []
        for index in range(len(added_label)):
            weights_a.append(weight_domain)
            active_f_a.append(0.0)
        weights_t = torch.tensor(weights_t)
        weights_a = torch.tensor(weights_a)
        active_f_t = torch.tensor(active_f_t)
        active_f_a = torch.tensor(active_f_a) 
        # weights = torch.tensor(1.0).unique(return_counts=True)[1]

        if use_pseudo_label:
            anchors = self.update_anchors(anchors, data[closest], pseudo_label[closest], feats[closest], weights_t, active_f_t)
            print(np.sum(pseudo_label[closest].numpy() == target[closest].numpy())/len(pseudo_label[closest]))
        else:
            anchors = self.update_anchors(anchors, data[closest], target[closest], feats[closest], weights_t,active_f_t)
            if len(all_added_idx) != 0:
                anchors = self.update_anchors(anchors, data_added, torch.tensor(added_label), feats[all_added_idx], weights_a,active_f_a)

        return outputs, closest, anchors

    def enable_bn(self, model):
        if not self.config.model.freeze_bn:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.momentum = 0.1
