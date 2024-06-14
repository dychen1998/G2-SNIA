from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, activation=None, dropout=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ll = nn.Linear(in_features, out_features, bias=False)

        self.activation = activation
        self.dropout = dropout
        # 初始化参数
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """初始化参数"""
        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x, adj, dropout=0):
        x = self.ll(x)
        if adj.is_sparse:
            x = torch.sparse.mm(adj, x) if adj.sparse_dim() == 2 else torch.bmm(adj, x)
        else:
            x = torch.matmul(adj, x)
        if not (self.activation is None):
            x = self.activation(x)
        if self.dropout:
            x = F.dropout(x, dropout)
        return x


class GCN(nn.Module):
    def __init__(self, num_layers, num_features, activation=F.relu):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(
                    GraphConvolution(num_features[i], num_features[i + 1], activation=activation, dropout=True))
            else:
                self.layers.append(GraphConvolution(num_features[i], num_features[i + 1]))

    def forward(self, x, adj, dropout=0):

        for layer in self.layers:
            x = layer(x, adj, dropout=dropout)
        return x


class gcn(nn.Module):
    def __init__(self, in_feat, out_feat, hidden_layers: list, activation: Optional[Callable] = None):
        super(gcn, self).__init__()
        layers = [in_feat]
        layers.extend(hidden_layers)
        layers.append(out_feat)
        self.gcn = GCN(len(layers) - 1, layers, activation)

    def forward(self, x, adj, dropout=0):
        x = self.gcn(x, adj, dropout=dropout)
        return x


if __name__ == '__main__':
    pass

