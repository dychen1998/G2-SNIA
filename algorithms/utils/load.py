from typing import Optional
import argparse

import torch
import torch.optim as optim
from torch.nn import Module
from torch.optim import Optimizer
import torch.nn.functional as F

from default import _SET
from config import cfg
from algorithms.datasets.base_dataset import BaseDataset
from algorithms.datasets.dblp import DBLP
from algorithms.datasets.cora import Cora
from algorithms.datasets.citeseer import Citeseer
from algorithms.models.gcn import gcn
from algorithms.models.sgcn import sgcn
from algorithms.models.gcnii import GCNII
from algorithms.models.tagcn import TAGCN

_DATASET = {
    'cora': Cora,
    'citeseer': Citeseer,
    'dblp': DBLP
}

_MODEL = {
    'gcn': gcn,
    'sgcn': sgcn,
    'gcnii': GCNII,
    'tagcn': TAGCN,
}

_ACTIVATION = {
    'relu': F.relu,
    'elu': F.elu,
    'leak_relu': F.leaky_relu,
    'tanh': torch.tanh,
    '': None
}

_OPTIMIZER = {
    'adam': optim.Adam,
    'sgd': optim.SGD
}


def load_dataset(dataset: str, device: Optional[torch.device] = None) -> BaseDataset:
    """
    加载一个数据集

    :param dataset: 数据集名称
    :param device: 将数据集加载到对应的GPU设备中
    :return: 返回一个改数据集的类
    """
    assert dataset in _DATASET, f"Not found datasets {dataset}"
    return _DATASET[dataset](device=device)


def load_model(model: str, input_dim: int, output_dim: int,
               device: Optional[torch.device] = None, **kwargs) -> Module:
    """
    加载模型

    :param model: 模型名称
    :param input_dim: 输入维度
    :param output_dim: 输出维度
    :param device: 设备
    :return: 模型
    """
    assert model in _MODEL, f"Not found model {model}"
    assert kwargs['activation'] in _ACTIVATION, f"Not found activation {kwargs['activation']}"
    kwargs['activation'] = _ACTIVATION[kwargs['activation']]

    return _MODEL[model](input_dim, output_dim, **kwargs).to(device)


def load_optim(opt: str, model: Module, lr: float, weight_decay: float) -> Optimizer:
    """
    加载优化器

    :param opt: 优化器名称
    :param model: 优化模型
    :param lr: 学习率
    :param weight_decay: 权重衰减
    :return: 优化器
    """
    assert opt in _OPTIMIZER, f"Not found optimizer {opt}"
    return _OPTIMIZER[opt](model.parameters(), lr=lr, weight_decay=weight_decay)


def set_seed(seed: int):
    """
    设置随机数种子

    :param seed: 随机种子
    :return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_set(dataset: str, model: str, set_net: bool = True, set_train: bool = True):
    """
    设置模型结构和训练参数

    :param dataset: 数据集名称
    :param model: 模型名称
    :param set_net: 是否设置网络结构
    :param set_train: 是否设置训练参数
    :return: None
    """
    assert model in _SET[dataset], f'{model} not support {dataset}'
    net_set = _SET[dataset][model]['net_set']
    train_set = _SET[dataset][model]['train_set']
    if set_net:
        cfg.net[model] = net_set
    if set_train:
        cfg.train.max_epoch = train_set['max_epoch']
        cfg.train.patience = train_set['patience']
        cfg.train.dropout = train_set['dropout']
        cfg.train.seed = train_set['seed']

        cfg.train.optimal.name = train_set['opt']
        cfg.train.optimal.lr = train_set['lr']
        cfg.train.optimal.weight_decay = train_set['weight_decay']

        set_seed(cfg.train.seed)
