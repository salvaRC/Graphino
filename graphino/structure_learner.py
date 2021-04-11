"""
Author: Salva Rühling Cachay
"""

import torch
import torch.nn as nn


class EdgeStructureLearner(nn.Module):
    def __init__(self, num_nodes, max_num_edges, dim, static_feat, device='cuda', alpha1=0.1, alpha2=2.0,
                 self_loops=True):
        super().__init__()
        if static_feat is None:
            raise ValueError("Please give static node features (e.g. part of the timeseries)")
        self.num_nodes = num_nodes
        xd = static_feat.shape[1]
        self.lin1 = nn.Linear(xd, dim)
        self.lin2 = nn.Linear(xd, dim)

        self.static_feat = static_feat if isinstance(static_feat, torch.Tensor) else torch.from_numpy(static_feat)
        self.static_feat = self.static_feat.float().to(device)

        self.device = device
        self.dim = dim
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.num_edges = max_num_edges
        self.self_loops = self_loops
        self.diag = torch.eye(self.num_nodes).bool().to(device)

    def forward(self):
        nodevec1 = torch.tanh(self.alpha1 * self.lin1(self.static_feat))
        nodevec2 = torch.tanh(self.alpha1 * self.lin2(self.static_feat))

        adj = torch.sigmoid(self.alpha2 * nodevec1 @ nodevec2.T)
        adj = adj.flatten()
        mask = torch.zeros(self.num_nodes * self.num_nodes).to(self.device)
        _, strongest_idxs = torch.topk(adj, self.num_edges)  # Adj to get the strongest weight value indices
        mask[strongest_idxs] = 1
        adj = adj * mask
        adj = adj.reshape((self.num_nodes, self.num_nodes))
        if self.self_loops:
            adj[self.diag] = adj[self.diag].clamp(min=0.5)

        return adj
