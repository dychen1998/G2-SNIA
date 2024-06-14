from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class sglayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(sglayer, self).__init__()
        self.lin = nn.Linear(in_feat, out_feat)
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x, adj, k=2):
        if adj.is_sparse:
            if adj.sparse_dim() == 2:
                multi_func = torch.spmm
            else:
                multi_func = torch.bmm
        else:
            multi_func = torch.matmul
        for i in range(k):
            x = multi_func(adj, x)
        x = self.lin(x)
        return x


class sgcn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers: list, activation: Optional[Callable] = None):
        super(sgcn, self).__init__()
        self.out_conv = nn.Linear(hidden_layers[-1], output_dim)
        self.act = activation
        self.layers = nn.ModuleList()
        layers = [input_dim]
        layers.extend(hidden_layers)
        layers.append(output_dim)
        num_sglayer = len(layers) - 2
        for i in range(num_sglayer):
            self.layers.append(sglayer(layers[i], layers[i + 1]))

    def forward(self, x, adj, dropout=0):
        if x.is_sparse:
            x = x.to_dense()
        x = F.dropout(x, dropout)
        for i in range(len(self.layers)):
            x = self.layers[i](x, adj)
            x = self.act(x)
        x = self.out_conv(x)

        return x


if __name__ == '__main__':
    pass
