from typing import Optional, Callable

import torch.nn as nn
import torch
import math

import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)

        hi = torch.spmm(adj, input) if adj.sparse_dim() == 2 else torch.bmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.matmul(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class GCNII(nn.Module):
    def __init__(self, in_feat, out_feat, hidden_layers, lamda, alpha, activation: Optional[Callable] = None,
                 variant=False):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        nlayers = len(hidden_layers)
        nhidden = hidden_layers[0]
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_feat, nhidden))
        self.fcs.append(nn.Linear(nhidden, out_feat))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = activation
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj, dropout=0):
        _layers = []
        x = F.dropout(x, dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner


if __name__ == '__main__':
    pass
