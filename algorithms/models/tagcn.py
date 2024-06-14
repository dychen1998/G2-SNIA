from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class rep(nn.Module):
    def __init__(self, num_features):
        super(rep, self).__init__()
        mid = num_features
        # mid=int(np.sqrt(num_features))+1
        # self.ln=nn.LayerNorm(100).cuda()
        self.num_features = num_features
        self.lin1 = nn.Linear(num_features, mid)
        self.lin2 = nn.Linear(mid, num_features)
        self.ln = nn.LayerNorm(mid)
        # self.att=nn.Linear(num_features,1)
        self.activation1 = F.relu
        self.activation2 = F.sigmoid
        # gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_normal_(self.lin.weight, gain=gain)
        # print(num_layers)

        # print(self.layers)

    def forward(self, x, adj):
        '''
        att=self.att(x)
        att=self.activation1(att)
        att=F.softmax(att,dim=0)
        #print(att.size())
        '''
        sumlines = torch.sparse.sum(adj, [0]).to_dense()
        allsum = torch.sum(sumlines)
        avg = allsum / x.size()[0]
        att = sumlines / allsum
        att = att.unsqueeze(1)
        # print(att.size())

        normx = x / sumlines.unsqueeze(1)
        avg = torch.mm(att.t(), normx)
        # avg=torch.mean(x,dim=0,keepdim=True)

        y = self.lin1(avg)
        y = self.activation1(y)
        y = self.ln(y)
        y = self.lin2(y)
        y = self.activation2(y)
        y = 0.25 + y * 2

        # print(x.size(),avg.size())
        dimmin = normx - avg  # n*100
        dimmin = torch.sqrt(att) * dimmin
        rep = torch.mm(dimmin.t(), dimmin)  # 100*100
        # ones=torch.ones(x.size()).cuda() #n*100
        # ones=torch.sum(ones,dim=0,keepdim=True)
        covariance = rep
        # conv=covariance.unsqueeze(0)
        q = torch.squeeze(y)
        qq = torch.norm(q) ** 2
        ls = covariance * q
        ls = ls.t() * q
        diag = torch.diag(ls)
        sumdiag = torch.sum(diag)
        sumnondiag = torch.sum(ls) - sumdiag
        loss = sumdiag - sumnondiag / self.num_features
        diagcov = torch.diag(covariance)
        sumdiagcov = torch.sum(diagcov)
        sumnondiagcov = torch.sum(covariance) - sumdiagcov
        lscov = sumdiagcov - sumnondiagcov / self.num_features
        k = loss / lscov
        k = k * self.num_features / qq
        if not (self.training):
            # print(ls)
            print(k)
        # print(y.shape)
        x = x * y

        # print((z-x).norm())
        return x, k


class TAGraph(nn.Module):
    def __init__(self, in_features, out_features, k=2, activation=None, dropout=False, norm=False):
        super(TAGraph, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin = nn.Linear(in_features * (k + 1), out_features)
        self.norm = norm
        self.norm_func = nn.BatchNorm1d(out_features, affine=False)
        self.activation = activation
        self.dropout = dropout
        self.k = k
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.lin.weight, gain=gain)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.lin.weight, gain=gain)

    def forward(self, x, adj, dropout=0):

        fstack = [x]
        multi_func = torch.spmm if adj.sparse_dim() == 2 else torch.bmm
        for i in range(self.k):
            y = multi_func(adj, fstack[-1])
            fstack.append(y)
        x = torch.cat(fstack, dim=-1)
        x = self.lin(x)
        if self.norm:
            x = self.norm_func(x)
        if not (self.activation is None):
            x = self.activation(x)
        if self.dropout:
            x = F.dropout(x, dropout)
        return x


class TArep(nn.Module):
    def __init__(self, num_layers, num_features, k):
        super(TArep, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.layers = nn.ModuleList()
        # print(num_layers)

        for i in range(num_layers):
            self.layers.append(rep(num_features[i]))
            if i != num_layers - 1:
                self.layers.append(
                    TAGraph(num_features[i], num_features[i + 1], k, activation=F.leaky_relu, dropout=True))
            else:
                self.layers.append(TAGraph(num_features[i], num_features[i + 1], k))
            # print(self.layers)

    def forward(self, x, adj, dropout=0, min=-1000, max=1000):
        # x=torch.clamp(x,min,max)
        # x=torch.atan(x)*2/np.pi
        kk = 0
        for i in range(len(self.layers)):

            if i % 2 == 0:
                # x=self.layers[i](x)
                x, k = self.layers[i](x, adj)
                kk = k + kk
            else:
                # print(i,self.layers[i].lin.weight.norm(),x.shape)
                x = self.layers[i](x, adj, dropout=dropout)

        return x, kk


class TAGCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers: list, activation: Optional[Callable] = None,
                 with_rep=False):
        super(TAGCN, self).__init__()
        k = 3
        num_features = [input_dim]
        num_features.extend(hidden_layers)
        num_features.append(output_dim)
        self.num_layers = num_layers = len(hidden_layers) + 1
        self.layers = nn.ModuleList()
        # print(num_layers)
        self.with_rep = with_rep
        if with_rep:
            self.rep_layers = nn.ModuleList()
        for i in range(num_layers):
            if with_rep:
                self.rep_layers.append(rep(num_features[i]))
            if i != num_layers - 1:
                self.layers.append(
                    TAGraph(num_features[i], num_features[i + 1], k, activation=activation, dropout=True))
            else:
                self.layers.append(TAGraph(num_features[i], num_features[i + 1], k))

    def forward(self, x, adj, dropout=0):
        if x.is_sparse:
            x = x.to_dense()
        for i in range(len(self.layers)):
            if self.with_rep:
                x = self.rep_layers[i](x, adj)
            x = self.layers[i](x, adj, dropout=dropout)
        return x
