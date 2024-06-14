import os
import random
import sys
import pickle as pkl
from typing import Optional

import scipy.sparse as sp
import torch
import numpy as np

from default import datasets_path, _ALL_DATASET


class BaseDataset:
    def __init__(self, name: str, split_seed: int = 42, device: Optional[torch.device] = None):
        self.device = device
        self.split_seed = split_seed
        self.root = os.path.join(datasets_path, name)
        self.attack_num = 1000
        assert os.path.exists(self.root), f"Not found {name} dataset"

        if name in _ALL_DATASET['pt']:
            self._pt_init(name)
        else:
            self._other_init(name)

    def _pt_init(self, name):
        adj_path = os.path.join(self.root, 'adj.pt')
        feats_path = os.path.join(self.root, 'features.pt')
        labels_path = os.path.join(self.root, 'labels.pt')
        train_index_path = os.path.join(self.root, 'train_index.pt')
        val_index_path = os.path.join(self.root, 'val_index.pt')
        test_index_path = os.path.join(self.root, 'test_index.pt')
        attack_index_path = os.path.join(self.root, 'attack_index.pt')

        self.adj = torch.load(adj_path, map_location=self.device)
        self.feats = torch.load(feats_path, map_location=self.device).to_dense()
        self.labels = torch.load(labels_path, map_location=self.device)
        if os.path.exists(train_index_path):
            self.train_index = torch.load(train_index_path, map_location=self.device)
            self.val_index = torch.load(val_index_path, map_location=self.device)
            self.test_index = torch.load(test_index_path, map_location=self.device)
            self.attack_index = torch.load(attack_index_path, map_location=self.device)
        else:
            self._split_dataset()

        if self.adj.is_sparse:
            self.adj = self.adj.coalesce()

        # 计算节点数量 标签类别数 特征维度
        self.num_nodes = self.adj.shape[0]
        self.num_classes = torch.max(self.labels).item() + 1
        self.num_feats = self.feats.shape[1]

    def _other_init(self, name):
        pass

    def _split_dataset(self):
        # 划分数据集
        _all = [i for i in range(self.labels.shape[0])]
        test_split = 0.8
        val_split = 0.1
        train_split = 1 - test_split - val_split
        random.seed(self.split_seed)
        random.shuffle(_all)
        self.train_index = torch.tensor(
            data=_all[0:int(len(_all) * train_split)],
            dtype=torch.long,
            device=self.device
        )
        self.val_index = torch.tensor(
            data=_all[int(len(_all) * train_split): int(len(_all) * (train_split + val_split))]
        )
        self.test_index = torch.tensor(
            data=_all[int(len(_all) * (train_split + val_split)):],
            dtype=torch.long,
            device=self.device
        )
        self.attack_index = torch.tensor(
            data=random.sample(_all[int(len(_all) * (train_split + val_split)):], self.attack_num),
            dtype=torch.long,
            device=self.device
        )

        train_index_path = os.path.join(self.root, 'train_index.pt')
        val_index_path = os.path.join(self.root, 'val_index.pt')
        test_index_path = os.path.join(self.root, 'test_index.pt')
        attack_index_path = os.path.join(self.root, 'attack_index.pt')

        torch.save(self.train_index.to('cpu'), train_index_path)
        torch.save(self.val_index.to('cpu'), val_index_path)
        torch.save(self.test_index.to('cpu'), test_index_path)
        torch.save(self.attack_index.to('cpu'), attack_index_path)