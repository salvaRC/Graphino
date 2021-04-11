"""
Author: Salva Rühling Cachay
"""

import torch.nn as nn
from utilities.utils import get_activation_function


class ONI_MLP(nn.Module):
    """
    Fully connected MLP on top of node embeddings
    L - number of hidden layers, each of those having half the number of neurons than the previous one.
    """

    def __init__(self, input_dim, output_dim, dropout=0, L=2, batch_norm=True, act_func='elu', device='cuda'):
        super().__init__()
        FC_layers = []
        for l in range(L):
            out_dim_l = input_dim // 2 ** (l + 1)
            FC_layers.append(
                nn.Linear(input_dim // 2 ** l, out_dim_l, bias=True)
            )
            if batch_norm:
                FC_layers.append(nn.BatchNorm1d(out_dim_l))
            FC_layers.append(get_activation_function(act_func, device=device))
            if dropout > 0:
                FC_layers.append(nn.Dropout(dropout))

        self.FC_layers = nn.ModuleList(FC_layers)
        self.out_dim_last_L = input_dim // 2 ** L
        self.out_layer = nn.Linear(self.out_dim_last_L, output_dim, bias=True)
        self.L = L

    def forward(self, x):
        for module in self.FC_layers:
            x = module(x)
        y = self.out_layer(x)
        return y
