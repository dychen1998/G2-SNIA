import os
from typing import Optional, Union, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.sparse as tsp

from config import cfg
from default import models_path
from algorithms.utils.load import load_model, load_optim
from algorithms.utils.functional import get_filter


@torch.no_grad()
def reset_parameters(m):
    """
    初始化参数

    :param m: nn.Module
    :return: None
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


class Classifier:
    def __init__(self, model_name: str, input_dim: int, output_dim: int, dropout: float, role: str = 'victim',
                 device: Optional[torch.device] = None):
        # 加载模型
        self.model = load_model(
            model=model_name,
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
            **cfg.net[model_name]
        )
        # 加载优化器
        self.optimizer = load_optim(
            opt=cfg.train.optimal.name,
            model=self.model,
            lr=cfg.train.optimal.lr,
            weight_decay=cfg.train.optimal.weight_decay
        )
        self.model_name = model_name
        self.dropout = dropout
        # role in ['victim', 'surrogate']
        self.role = role
        self.device = device
        self.output = None
        self.best_model = None

        # 训练参数
        self.train_iters = cfg.train.max_epoch
        self.patience = cfg.train.patience

    def fit(self, features: Union[Tensor, tsp.Tensor], adj: tsp.Tensor, labels: Tensor, idx_train: Tensor,
            idx_val: Optional[Tensor] = None, verbose=False):

        adj_norm = get_filter(adj, model=self.model_name)
        early_stopping = self.patience
        best_loss_val = 100

        i = 0
        for i in range(self.train_iters):
            self.model.train()
            self.optimizer.zero_grad()
            output = F.log_softmax(self.model(features, adj_norm, dropout=self.dropout), dim=1)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                output = F.log_softmax(self.model(features, adj_norm), dim=1)
                loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if verbose and i % 50 == 0:
                output = torch.argmax(output, dim=1)
                correct_train = torch.eq(output[idx_train], labels[idx_train]).sum()
                train_acc = correct_train.item() * 1.0 / idx_train.shape[0]
                correct_val = torch.eq(output[idx_val], labels[idx_val]).sum()
                val_acc = correct_val.item() * 1.0 / idx_val.shape[0]
                print('Epoch {}, training loss: {} val loss: {} train acc:{} val acc: {}'.format(i, loss_train.item(),
                                                                                                 loss_val.item(),
                                                                                                 train_acc, val_acc))
            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.best_model = deepcopy(self.model.state_dict())
                early_stopping = self.patience
            else:
                early_stopping -= 1
            if early_stopping <= 0:
                break
        if verbose:
            print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val))
        self.model.load_state_dict(self.best_model)

    def save(self, dataset_name: str):
        """
        保存GNN模型参数

        :param dataset_name: 模型对应的数据集名称
        :return: None
        """
        save_path = os.path.join(models_path, self.role)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(
            {'state_dict': {k: v.to('cpu') for k, v in self.best_model.items()}},
            os.path.join(save_path, f'{dataset_name}_{self.model_name}.pt')
        )

    @torch.no_grad()
    def predict(self, features: Union[Tensor, tsp.Tensor], adj: tsp.Tensor) -> Tensor:
        """
        计算给定输入下GNN模型的输出

        :param features: 特征
        :param adj: 邻接矩阵
        :return: GNN输出
        """
        adj_norm = get_filter(adj, model=self.model_name)
        self.model.eval()
        output = self.model(features, adj_norm)
        return torch.argmax(output, dim=1)

    @torch.no_grad()
    def get_acc(self, output: Tensor, labels: torch.LongTensor, index: torch.LongTensor) -> Tuple[float, int]:
        """
        计算给定样本的分类精度

        :param output: GNN模型的输出
        :param labels: 标签集
        :param index: 样本
        :return: （精度, 正确样本数量)
        """
        correct = torch.eq(output[index], labels[index]).sum()
        acc = correct.item() * 1.0 / index.shape[0]
        return acc, correct.item()

    def initialize(self):
        """
        初始化参数

        :return: None
        """
        self.model.apply(reset_parameters)

    def load(self, dataset_name: str) -> bool:
        """
        加载模型

        :return: bool
        """
        load_path = os.path.join(models_path, self.role, f'{dataset_name}_{self.model_name}.pt')
        if os.path.exists(load_path):
            model_dict = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(model_dict['state_dict'])
            return True
        else:
            return False

    def inference(self):
        """
        推理模式 不计算梯度 不使用Drop等优化训练过程的方法

        :return: None
        """
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def train(self):
        """
        训练模式 计算梯度 使用Drop等优化训练过程的方法

        :return: None
        """
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

    def __call__(self, features: Union[Tensor, tsp.Tensor], adj: tsp.Tensor):
        adj_norm = get_filter(adj, model=self.model_name)
        return self.model(features, adj_norm)
