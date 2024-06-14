import random
import time
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from config import cfg
from algorithms.utils.load import load_dataset, load_set
from algorithms.utils.classifier import Classifier
from algorithms.utils.graph import GraphObj
from algorithms.utils.my_deque import MyDeque


class Environment:
    def __init__(self, dataset_name: str, model_name: str,
                 device: Optional[torch.device] = None, render: bool = False):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.seed = cfg.attack.seed
        self.device = device
        # 展示信息
        self.render = render
        # 原始图
        self.dataset = load_dataset(self.dataset_name, device=self.device)
        # 注入孤立节点图
        self.graph = GraphObj(self.dataset.adj, self.dataset.feats, self.device)
        # 实际交互中使用的图
        self.num_envs = None
        self.vec_graph = None
        self.inject_node = None
        self.neighbor_nodes = None
        self.feat_mask = None
        self.added_edges = None
        self.added_feats = None
        self.labels_state = None
        # 受害者模型分类概率分布
        self.victim_dists = None
        # 受害者模型输出的标签
        self.victim_labels = None
        # 所有节点
        self.total_index = None
        # 训练节点
        self.train_index = None
        # 验证节点
        self.val_index = None
        # 测试节点
        self.test_index = None

        # 当前攻击的目标节点的index
        self.target_index = None
        # 当前攻击的目标节点
        self.target_node = None
        # 目标节点原始类别
        self.ori_label = None
        # 目标节点攻击类别
        self.target_label = None
        # 统计智能体与环境的交互次数
        self.edge_step = 0
        self.feat_step = 0
        self.old_entropy = None
        '''可视化'''
        # 初始概率分布
        self.ori_logist = None
        # 记录概率分布的变化
        self.change_logist = None
        # 记录奖励的变化
        self.change_reward = []

        # self.feat_budget = self.graph.feats[:-1].sum(dim=1).mean().long().item()
        self.feat_budget = self.graph.feats.sum(dim=1).max().long().item()
        # self.feat_budget = 50
        self.edge_budget = cfg.attack.edge_budget

        # 保存攻击过程的文件
        self.f = None

        # 候选节点和候选类别的概率分布 均匀分布
        self.attack_index = self.dataset.attack_index
        self.train_categorical = torch.ones(self.attack_index.shape[0], dtype=torch.float, device=self.device)
        self.label_categorical = torch.ones(self.dataset.num_classes, dtype=torch.float, device=self.device)

        self._load()

        # 设计结构扰动
        self.graph.add_nodes(1)

    def _load(self):
        role = 'victim' if self.model_name != 'surrogate' else 'surrogate'
        # 受害者模型
        load_set(self.dataset_name, self.model_name)
        self.victim = Classifier(
            model_name=self.model_name,
            input_dim=self.dataset.num_feats,
            output_dim=self.dataset.num_classes,
            dropout=cfg.train.dropout,
            role=role,
            device=self.device
        )
        self._load_model(self.victim)
        self.victim.inference()
        # 获取victim分类标签 (N,C)
        self.victim_dists = torch.softmax(self.victim(self.dataset.feats, self.dataset.adj), dim=1)
        self.victim_labels = torch.argmax(self.victim_dists, dim=1)
        victim_acc, _ = self.victim.get_acc(self.victim_labels, self.dataset.labels, self.dataset.test_index)

        self.labels_state = torch.empty((self.dataset.num_classes, self.dataset.num_feats),
                                        dtype=torch.float,
                                        device=self.device)
        adj_norm = self.graph.get_filter('gcn')
        nodes_state = torch.spmm(adj_norm, self.graph.feats)
        for c in range(self.dataset.num_classes):
            # (num of label c nodes, F)
            label_nodes_state = nodes_state[self.victim_labels == c]
            # (F,)
            self.labels_state[c] = torch.mean(label_nodes_state, dim=0, keepdim=False)

    def reset(self, target_node: Optional[Tensor] = None, target_label: Optional[Tensor] = None, num_envs: int = 1) -> \
            Tuple[Tensor, Tensor]:
        """
        初始化化环境
        :param num_envs:
        :param target_node: 目标节点 (m,)
        :param target_label: 目标类别 (m,)
        :return: state, action_mask
        """
        self.num_envs = num_envs
        # Vec环境交互中使用的图
        if self.vec_graph is not None:
            self.vec_graph = None
            torch.cuda.empty_cache()
        self.vec_graph = [(self.graph.dense_feats, self.graph.sparse_adj, self.graph.degrees) for _ in
                          range(self.num_envs)]
        self.vec_graph = map(list, zip(*self.vec_graph))
        # [(num_envs,N,F), (num_envs,N,N), (num_envs,N,1)]
        self.vec_graph = [torch.stack(part, dim=0) for part in self.vec_graph]
        # 注入节点
        self.inject_node = torch.ones(self.num_envs, dtype=torch.long, device=self.device) * (self.graph.num_nodes - 1)
        # 目标节点邻居节点
        self.neighbor_nodes = torch.zeros((self.num_envs, self.graph.num_nodes), dtype=torch.bool,
                                          device=self.device)
        # 特征选择空间 (F,)
        self.feat_mask = torch.zeros((self.num_envs, self.graph.num_feats), dtype=torch.bool, device=self.device)
        # 记录每个step的动作
        self.added_edges = MyDeque(num_envs=self.num_envs, max_len=self.edge_budget, device=self.device)
        self.added_feats = MyDeque(num_envs=self.num_envs, max_len=self.feat_budget, device=self.device)
        # 采样目标节点
        if target_node is None:
            self.target_index = torch.multinomial(self.train_categorical, num_samples=self.num_envs, replacement=True)
            self.target_node = self.attack_index[self.target_index]
        else:
            self.target_node = target_node
        # 目标标签
        if target_label is None:
            self.target_label = torch.multinomial(self.label_categorical, num_samples=self.num_envs, replacement=True)
        else:
            self.target_label = target_label
        self.ori_label = self.victim_labels[self.target_node]
        '''添加结构扰动'''
        self._struc_perturbation()
        # self._random_struc_perturbation()
        # d^{-1/2} (num_envs, N, 1)
        vec_d_sqrt = torch.pow(self.vec_graph[2], -0.5)
        # (2e, 2e, 2e)
        _indices = self.vec_graph[1]._indices()
        _values = self.vec_graph[1]._values()
        size = self.vec_graph[1].size()
        # (2e,)
        row_factor = vec_d_sqrt[(_indices[0], _indices[1])].squeeze(dim=-1)
        col_factor = vec_d_sqrt[(_indices[0], _indices[2])].squeeze(dim=-1)
        new_values = _values * row_factor * col_factor
        vec_adj_norm = torch.sparse_coo_tensor(
            indices=_indices,
            values=new_values,
            dtype=torch.float,
            size=size,
            device=self.device
        )
        self.vec_graph.append(vec_adj_norm)
        self.feat_mask[:] = True
        s_0 = self._get_state()

        self.feat_step = 0

        logist = self._get_logist()
        p_target = torch.gather(logist, 1, self.target_label.unsqueeze(dim=1))
        old_entropy = torch.add(p_target, 1e-2)
        self.old_entropy = torch.log(old_entropy)

        if self.render:
            self.change_logist = torch.zeros((self.num_envs, self.feat_budget+1), dtype=torch.float,
                                             device=self.device)
            self.change_logist[:, self.feat_step] += self.victim_dists[(self.target_node, self.target_label)]
        return s_0, self.feat_mask

    def step(self, action: Tensor, action_prob: Optional[Tensor] = None) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Union[None, Tensor]]:
        """
        智能体与环境交互
        :param action: 注入节点增加的属性
        :param action_prob: 动作概率
        :return ((state_embed,action_mask), reward, done, info)
        """
        # (num_envs,)
        select_feature = torch.squeeze(action, dim=1)
        '''add perturbation'''
        # (num_envs,)
        extend_g = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        # (num_envs,)
        extend_row = self.inject_node
        # (num_envs,)
        extend_col = select_feature
        # (num_envs,)
        self.vec_graph[0][(extend_g, extend_row, extend_col)] = 1
        self.added_feats.append(self.feat_step, select_feature)
        mask_indices = (extend_g, extend_col)
        mask_value = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.feat_mask[mask_indices] = mask_value
        self.feat_step += 1
        '''get new state'''
        # (3F,)
        nx_s = self._get_state()
        '''计算reward'''
        new_logist = self._get_logist()
        p_target = torch.gather(new_logist, 1, self.target_label.unsqueeze(dim=1))
        new_entropy = torch.add(p_target, 1e-2)
        new_entropy = torch.log(new_entropy)
        reward = (new_entropy - self.old_entropy) * 10
        self.old_entropy = new_entropy

        if self.feat_step < self.feat_budget:
            done = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
            info = None
        else:
            done = torch.ones((self.num_envs, 1), dtype=torch.bool, device=self.device)
            info = torch.eq(new_logist.argmax(dim=1), self.target_label).unsqueeze(dim=-1)

        if self.render:
            self.change_logist[:, self.feat_step] += p_target.squeeze(dim=-1)
        return nx_s, self.feat_mask, reward, done, info

    def _get_logist(self):
        output = self.victim.model(self.vec_graph[0], self.vec_graph[-1])
        # (num_envs,1)
        index = self.target_node.unsqueeze(dim=-1)
        # (num_envs,1,C)
        gather_index = index.repeat(1, output.shape[-1]).unsqueeze(dim=1)
        # (num_envs, C)
        logist = torch.gather(output, 1, gather_index).squeeze(dim=1)
        logist = torch.softmax(logist, dim=-1)
        return logist

    def _get_state(self) -> Tensor:
        """
        获取当前时刻状态S_t
        :return:
        """
        '''subgraph state'''
        # (num_envs,N,F)
        aggregate = torch.bmm(self.vec_graph[-1], self.vec_graph[0])
        # (num_envs,1)
        target_index = self.target_node.unsqueeze(dim=-1)
        # (num_envs,1,F)
        gather_index = target_index.repeat(1, aggregate.shape[-1]).unsqueeze(dim=1)
        # (num_envs, F)
        subgraph_state = torch.gather(aggregate, 1, gather_index).squeeze(dim=1)
        '''inj state'''
        # (num_envs,1)
        inj_index = self.inject_node.unsqueeze(dim=-1)
        # (num_envs,1,F)
        gather_index = inj_index.repeat(1, aggregate.shape[-1]).unsqueeze(dim=1)
        # (num_envs, F)
        inj_state = torch.gather(aggregate, 1, gather_index).squeeze(dim=1)
        '''label state'''
        # (num_envs, F)
        label_state = self.labels_state[self.target_label]
        # (num_envs, 2F)
        mdp_state = torch.cat([subgraph_state, inj_state, label_state], dim=1)

        return mdp_state

    def is_misclassify(self, record=False) -> float:
        """
        判断对抗样本是否攻击成功

        :return: 是否攻击成功
        """
        if not record:
            adv_logist = self._get_logist()
            adv_label = torch.argmax(adv_logist, dim=1)
            mis_num = torch.eq(adv_label, self.target_label).sum().item()
            return mis_num
        else:
            if self.render:
                pass

    def _struc_perturbation(self):
        """
        :return: None
        """
        '''directly connect target'''
        # (num_envs,)
        extend_g = torch.arange(self.target_node.shape[0], dtype=torch.long, device=self.device)
        # (2*num_envs,)
        extend_g = extend_g.repeat(1, 2).squeeze(dim=0)
        # (2*num_envs,)
        extend_row = torch.cat([self.target_node, self.inject_node], dim=0)
        # (2*num_envs,)
        extend_col = torch.cat([self.inject_node, self.target_node], dim=0)
        add_per = torch.sparse_coo_tensor(
            indices=torch.stack([extend_g, extend_row, extend_col], dim=0),
            values=torch.ones(extend_col.shape[0], dtype=torch.float, device=self.device),
            dtype=torch.float,
            device=self.device
        )
        self.vec_graph[1] += add_per
        self.vec_graph[2][(extend_g, extend_row)] += 1
        self.added_edges.append(0, self.target_node)
        self.vec_graph.append(add_per)

    def _load_model(self, model: Classifier):
        """
        加载GNN模型

        :param model: GNN模型
        :return: None
        """
        if not model.load(self.dataset_name):
            model.initialize()
            model.fit(self.dataset.feats, self.dataset.adj, self.dataset.labels, self.dataset.train_index,
                      self.dataset.val_index, verbose=True)
            model.save(self.dataset_name)
            output = model.predict(self.dataset.feats, self.dataset.adj)
            test_acc, test_correct = model.get_acc(output, self.dataset.labels, self.dataset.test_index)
            result = {
                'numbers': self.dataset.test_index.shape[0],
                'corrects': test_correct,
                'accuracy': round(test_acc * 100, 2)
            }
            print('test_acc: ', round(test_acc * 100, 2))

    @property
    def input_dims(self) -> int:
        # return 5 * self.dataset.num_classes + 2 * self.dataset.num_feats
        # return 7 * self.H + 5 * self.dataset.num_classes
        return 3 * self.dataset.num_feats

    @property
    def output_dims(self) -> int:
        return self.graph.num_feats

    @property
    def action_dim(self) -> int:
        return self.graph.num_feats

    @property
    def state_dim(self) -> int:
        # return 7 * self.H + 5 * self.dataset.num_classes
        return 3 * self.dataset.num_feats


if __name__ == '__main__':
    # env = Environment(dataset_name='cora', model_name='gcn', device=torch.device('cuda:0'))
    pass
