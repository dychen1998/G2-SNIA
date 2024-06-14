from typing import Tuple, Union, Optional

from scipy.sparse import coo_matrix, lil_matrix
import torch
from torch import Tensor
import torch.sparse as tsp


def eye(m: int, dtype: Union[torch.dtype, None], device=Union[torch.device, None]) -> tsp.Tensor:
    """
    Returns a sparse matrix with ones on the diagonal and zeros elsewhere.

    :param m: The first dimension of the sparse matrix.
    :param dtype: The data type of the sparse matrix.

    :return: layout=torch.sparse_coo
    """

    row = torch.arange(m, dtype=torch.long)
    indices = torch.stack([row, row], dim=0)
    value = torch.ones(m)
    Identity = torch.sparse_coo_tensor(
        indices=indices,
        values=value,
        dtype=dtype,
        size=torch.Size((m, m)),
        device=device
    )
    del row, indices, value
    return Identity


def gcn_filter(adj: Union[Tensor, tsp.Tensor], power=-0.5) -> Union[Tensor, tsp.Tensor]:
    """
    计算重归一化拉普拉斯矩阵

    :param adj:  邻接矩阵
    :param power: 度矩阵乘的幂数
    :return: 重归一化拉普拉斯矩阵
    """
    if adj.is_sparse:
        # 合并filter
        adj_ = adj.coalesce()
        filter_indices = adj_.indices()
        row, col = filter_indices
        adj_values = adj_.values()
        """计算D^{-1/2}"""
        d = tsp.sum(adj_, dim=1).to_dense()
        d = torch.pow(d, power)
        d[d == float('inf')] = 0
        """计算D^{-1/2} @ A @ D^{-1/2}"""
        filter_values = adj_values * d[row] * d[col]
        _filter = torch.sparse_coo_tensor(
            indices=filter_indices,
            values=filter_values,
            size=adj_.shape,
            device=adj.device
        )
    else:
        # (N,1)
        d = adj.sum(dim=1, keepdim=True)
        d = torch.pow(d, power)
        # (N,1)
        d_row = d
        # (1,N)
        d_col = d.transpose(0, 1)
        # (N,N) broadcast
        _filter = d_row * adj * d_col
    return _filter


def get_filter(adj: Union[Tensor, tsp.Tensor], model: Optional[str] = None) -> Union[Tuple[tsp.Tensor, tsp.Tensor], tsp.Tensor]:
    """
    该函数在测试模型分类精度时使用 无需重复加载Dataset类
    计算重归一化拉普拉斯矩阵等提取网络拓扑信息的矩阵

    :param: adj 邻接矩阵
    :return: filter or (filter, filter)
    """
    return gcn_filter(adj)


def scipy_sparse_to_tensor(sp_matrix: Union[coo_matrix, lil_matrix], dtype: Optional[torch.dtype] = None,
                           device: Optional[torch.device] = None) -> tsp.Tensor:
    """
    将scipy.sparse.coo_matrix或scipy.sparse.lil_matrix 转换为torch的稀疏矩阵

    :param sp_matrix: scipy的稀疏矩阵
    :param dtype: 转换后数据类型
    :param device: 数据存放的设备
    :return: torch的稀疏矩阵
    """
    if isinstance(sp_matrix, lil_matrix):
        sp_matrix = sp_matrix.tocoo()
    _indices = torch.tensor([sp_matrix.row.tolist(), sp_matrix.col.tolist()])
    _values = sp_matrix.data
    return torch.sparse_coo_tensor(
        indices=_indices,
        values=_values,
        dtype=dtype,
        size=sp_matrix.shape,
        device=device
    )