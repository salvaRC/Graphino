import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    This GCN layer was adapted from the PyTorch version by T. Kipf, see README.
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, residual=False, batch_norm=False,
                 activation=F.relu, dropout=0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.residual = residual

        if self.in_features != self.out_features:
            self.residual = False

        self.batchnorm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self._norm = False

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)  # (batch-size, #nodes, #out-dim)
        node_repr = torch.matmul(adj, support)  # (batch-size, #nodes, #out-dim)

        if self.bias is not None:
            node_repr = node_repr + self.bias

        if self.batchnorm is not None:
            node_repr = node_repr.transpose(1, 2)  # --> (batch-size, #out-dim, #nodes)
            node_repr = self.batchnorm(node_repr)  # batch normalization over feature/node embedding dim.
            node_repr = node_repr.transpose(1, 2)

        node_repr = self.activation(node_repr)

        if self.residual:
            node_repr = input + node_repr  # residual connection

        node_repr = self.dropout(node_repr)
        return node_repr

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
