from typing import Union, Optional, List

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.sparse as tsp
from algorithms.utils.functional import get_filter


class GraphObj:
    adj: tsp.Tensor
    feats: Tensor
    degrees: Tensor
    neighbor: List
    device: Optional[torch.device]
    num_nodes: int
    num_feats: int

    def __init__(self, adj: tsp.Tensor, feats: Union[Tensor, tsp.Tensor], device: Optional[torch.device] = None):
        self.reset(adj, feats, device)

    @property
    def dense_adj(self) -> Tensor:
        """
        返回稠密邻接矩阵

        :return: 稠密邻接矩阵
        """
        return self.adj.to_dense()

    @property
    def dense_feats(self) -> Tensor:
        """
        返回稠密特征矩阵

        :return: 稠密特征矩阵
        """
        return self.feats.clone()

    @property
    def sparse_adj(self) -> tsp.Tensor:
        """
        返回稀疏邻接矩阵

        :return: 稀疏邻接矩阵
        """
        return self.adj.clone()

    @property
    def sparse_feats(self) -> tsp.Tensor:
        """
        返回稀疏特征矩阵

        :return: 稀疏特征矩阵
        """
        return self.feats.to_sparse(2)

    def get_filter(self, model_name: Optional[str] = None) -> tsp.Tensor:
        """
        返回图滤波器

        :return: 归一化后的拉普斯拉矩阵
        """
        return get_filter(self.adj, model_name)

    def reset(self, adj: tsp.Tensor, feats: Union[tsp.Tensor, Tensor], device: Optional[torch.device] = None):
        """
        初始化图

        :param adj: 邻接矩阵
        :param feats: 特征矩阵
        :param device: 设备
        :return: None
        """
        self.adj = adj
        self.feats = feats
        self.device = device

        if self.feats.is_sparse:
            self.feats = feats.to_dense()
        if self.adj.device != self.device:
            self.adj = self.adj.to(self.device)
        if self.feats.device != self.device:
            self.feats = self.feats.to(self.device)

        self.num_nodes = self.feats.shape[0]
        self.num_feats = self.feats.shape[1]

        self.degrees = tsp.sum(self.adj, dim=1).to_dense().unsqueeze(dim=-1).long()
        neighbor_indices = self.adj.indices()[1]
        self.neighbor = []
        cur_index = 0
        for i in range(self.num_nodes):
            nx_index = cur_index + self.degrees[i, 0]
            self.neighbor.append(neighbor_indices[cur_index: nx_index].tolist())
            cur_index = nx_index

    def add_edge(self, u: Union[int, Tensor], v: Union[int, Tensor]):
        """
        添加连边

        :param u: 节点1
        :param v: 节点2
        :return: None
        """
        assert 0 <= u < self.num_nodes and 0 <= v < self.num_nodes, 'Out of range of adjacency len.'
        if self.adj[u, v] == 0:
            if u == v:
                add_indices = [[u], [u]]
                add_values = [1.]
                self.degrees[u] += 1
            else:
                add_indices = [[u, v], [v, u]]
                add_values = [1., 1.]
                self.degrees[u] += 1
                self.degrees[v] += 1
            add_matrix = torch.sparse_coo_tensor(
                indices=torch.tensor(add_indices),
                values=add_values,
                dtype=torch.float,
                size=self.adj_shape,
                device=self.device
            )
            self.adj += add_matrix
            self.adj = self.adj.coalesce()

        else:
            print(u, v)
            raise ValueError

    def add_mul_edges(self, u_lst: Union[List, Tensor], v_lst: Union[List, Tensor], value:float=1):
        """

        :param u_lst:
        :param v_lst:
        :param value:
        :return:
        """

        if isinstance(u_lst, List):
            e_0 = torch.tensor(u_lst, dtype=torch.long)
        else:
            e_0 = u_lst
        if isinstance(v_lst, List):
            e_1 = torch.tensor(v_lst, dtype=torch.long)
        else:
            e_1 = v_lst
        assert torch.all(e_0 >= 0) and torch.all(e_0 < self.num_nodes), 'Out of range of adjacency len. u_lst'
        assert torch.all(e_1 >= 0) and torch.all(e_1 < self.num_nodes), 'Out of range of adjacency len. v_lst'

        num_e = e_0.shape[0]
        if not all((self.adj[e_0[i], e_1[i]] == 0 for i in range(num_e))):
            # print(e_0)
            # print(e_1)
            # print([self.adj[e_0[i], e_1[i]] == 0 for i in range(num_e)])
            raise ValueError
        extend_row = torch.cat([e_0, e_1], dim=0)
        extend_col = torch.cat([e_1, e_0], dim=0)
        add_indices = torch.stack([extend_row, extend_col], dim=0)
        add_values = torch.zeros(extend_row.shape[0], dtype=torch.float)+value
        self._manipulate_adj(add_indices, add_values)
        for i in range(num_e):
            self.degrees[e_0[i]] += 1
            self.degrees[e_1[i]] += 1

    def delete_mul_inj_edges(self, u_lst: Union[List, Tensor]):
        """

        :param u_lst:
        :return:
        """
        assert all((u >= 0 for u in u_lst)) and all(
            (u < self.num_nodes - 1 for u in u_lst)), 'Out of range of adjacency len.'
        if all((self.adj[u, -1] == 1 for u in u_lst)):
            n_inj = torch.ones(len(u_lst), dtype=torch.long) * (self.num_nodes - 1)
            if isinstance(u_lst, List):
                n_connect = torch.tensor(u_lst, dtype=torch.long)
            else:
                n_connect = u_lst
            extend_row = torch.cat([n_connect, n_inj], dim=0)
            extend_col = torch.cat([n_inj, n_connect], dim=0)
            add_indices = torch.stack([extend_row, extend_col], dim=0)
            add_values = - torch.ones(extend_row.shape[0], dtype=torch.float)
            add_matrix = torch.sparse_coo_tensor(
                indices=add_indices,
                values=add_values,
                dtype=torch.float,
                size=self.adj_shape,
                device=self.device
            )
            self.adj += add_matrix
            self.adj = self.adj.coalesce()
            for n in u_lst:
                self.degrees[n] -= 1
            self.degrees[-1] -= 1
        else:
            raise ValueError

    def delete_edge(self, u: Union[int, Tensor], v: Union[int, Tensor]):
        """
        删除连边

        :param u: 节点1
        :param v: 节点2
        :return: None
        """
        assert 0 <= u < self.num_nodes and 0 <= v < self.num_nodes, 'Out of range of adjacency len.'
        if self.adj[u, v] == 1:
            add_indices = [[u, v], [v, u]]
            add_values = [-1., -1.]
            add_matrix = torch.sparse_coo_tensor(
                indices=torch.tensor(add_indices),
                values=add_values,
                dtype=torch.float,
                size=self.adj_shape,
                device=self.device
            )
            self.adj += add_matrix
            self.adj = self.adj.coalesce()
            self.degrees[u] -= 1
            self.degrees[v] -= 1
        else:
            print(u, v)
            raise ValueError

    def add_feat(self, u: Tensor, x: Tensor, normalize: bool = True):
        """
        添加节点属性(离散)

        :param u: 节点
        :param x: 属性
        :param normalize: 是否归一化属性
        :return: None
        """
        assert 0 <= u < self.num_nodes and 0 <= x < self.num_feats, 'Out of range of feature len'
        if abs(self.feats[u, x] - 0) < 1e-4:
            if normalize:
                exit_feats = torch.nonzero(self.feats[u]).squeeze(dim=-1)
                exit_feats_num = exit_feats.shape[0]
                if exit_feats_num > 0:
                    self.feats[u][exit_feats] -= 1. / (exit_feats_num * (exit_feats_num + 1))
                self.feats[u, x] += 1. / (exit_feats_num + 1)
            else:
                self.feats[u, x] = 1.
        else:
            print(u, x)
            raise ValueError

    def add_nodes(self, num: int) -> List:
        """
        增加节点

        :param num: 增加节点数量
        :return: added_nodes: 新增节点编号
        """
        self.num_nodes += num
        self.adj = torch.sparse_coo_tensor(
            indices=self.adj.indices(),
            values=self.adj.values(),
            dtype=torch.float,
            size=self.adj_shape,
            device=self.device
        )
        # feat
        self.feats = F.pad(self.feats, (0, 0, 0, num), 'constant', 0)
        # degree
        self.degrees = F.pad(self.degrees, (0, 0, 0, num), 'constant', 0)
        # adj
        u_lst = torch.arange(start=self.num_nodes-num, end=self.num_nodes, step=1, dtype=torch.long)
        self.add_self_loop(u_lst)
        return [v for v in range(self.num_nodes-num, self.num_nodes)]


    def add_self_loop(self, u_lst: Union[List, Tensor]):
        if isinstance(u_lst, List):
            e_0 = torch.tensor(u_lst, dtype=torch.long)
        else:
            e_0 = u_lst
        e_1 = e_0
        num_e = e_0.shape[0]
        assert torch.all(e_0 >= 0) and torch.all(e_0 < self.num_nodes), 'Out of range of adjacency len. u_lst'
        if not all((self.adj[e_0[i], e_1[i]] == 0 for i in range(num_e))):
            raise ValueError
        add_indices = torch.stack([e_0, e_1], dim=0)
        add_values = torch.ones(num_e, dtype=torch.float)
        self._manipulate_adj(add_indices, add_values)
        for i in range(num_e):
            self.degrees[e_0[i]] += 1

    def _manipulate_adj(self, indices: Tensor, values: Tensor):
        """
        修改邻接矩阵

        :param indices:
        :param values:
        :return:
        """
        add_matrix = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            dtype=torch.float,
            size=self.adj_shape,
            device=self.device
        )
        self.adj += add_matrix
        self.adj = self.adj.coalesce()

    @property
    def adj_shape(self):
        """
        获取邻接矩阵大小

        :return: 邻接矩阵大小
        """
        return torch.Size((self.num_nodes, self.num_nodes))

    @property
    def feats_shape(self):
        """
        获取特征矩阵大小

        :return: 特征矩阵大小
        """
        return torch.Size((self.num_nodes, self.num_feats))


if __name__ == '__main__':
    pass
