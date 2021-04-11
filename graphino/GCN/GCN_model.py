import torch
import torch.nn as nn
from graphino.readout_MLP import ONI_MLP
from graphino.GCN.graph_conv_layer import GraphConvolution
from graphino.structure_learner import EdgeStructureLearner
from utilities.utils import get_activation_function


class GCN(nn.Module):
    def __init__(self, net_params, static_feat=None, adj=None, device='cuda', outsize=1, verbose=True):
        super().__init__()
        self.L = net_params['L']
        assert self.L > 1
        self.act = net_params['activation']
        self.out_dim = self.mlp_input_dim = net_params['out_dim']
        self.batch_norm = net_params['batch_norm']
        self.graph_pooling = net_params['readout'].lower()
        self.jumping_knowledge = net_params['jumping_knowledge']
        dropout = net_params['dropout']
        hid_dim = net_params['hidden_dim']
        num_nodes = net_params['num_nodes']
        activation = get_activation_function(self.act, functional=True, num=1, device=device)
        conv_kwargs = {'activation': activation, 'batch_norm': self.batch_norm,
                       'residual': net_params['residual'], 'dropout': dropout}
        layers = [GraphConvolution(net_params['in_dim'], hid_dim, **conv_kwargs)]
        layers += [GraphConvolution(hid_dim, hid_dim, **conv_kwargs) for _ in range(self.L - 2)]
        layers.append(GraphConvolution(hid_dim, self.out_dim, **conv_kwargs))
        self.layers = nn.ModuleList(layers)
        if self.jumping_knowledge:
            self.mlp_input_dim = self.mlp_input_dim + hid_dim * (self.L - 1)
        if self.graph_pooling in ['sum+mean', 'mean+sum', 'mean&sum', 'sum&mean']:
            self.mlp_input_dim = self.mlp_input_dim * 2
        self.MLP_layer = ONI_MLP(self.mlp_input_dim, outsize, act_func=self.act, batch_norm=net_params['mlp_batch_norm'],
                                 dropout=dropout, device=device)
        if adj is None:
            self.adj, self.learn_adj = None, True
            max_num_edges = int(net_params['avg_edges_per_node'] * num_nodes)
            self.graph_learner = EdgeStructureLearner(
                num_nodes, max_num_edges, dim=net_params['adj_dim'], device=device, static_feat=static_feat,
                alpha1=net_params['tanh_alpha'], alpha2=net_params['sig_alpha'], self_loops=net_params['self_loop']
            )
        else:
            print('Using a static connectivity structure !!!')
            self.adj, self.learn_adj = torch.from_numpy(adj).float().to(device), False

        if verbose:
            print([x for x in self.layers])

    def forward(self, input, readout=True):
        if self.learn_adj:
            # Generate an adjacency matrix/connectivity structure for the graph convolutional forward pass
            self.adj = self.graph_learner.forward()

        # GCN forward pass --> Generate node embeddings
        node_embs = self.layers[0](input, self.adj)  # shape (batch-size, #nodes, #features)
        X_all_embeddings = node_embs.clone()
        for conv in self.layers[1:]:
            node_embs = conv(node_embs, self.adj)
            if self.jumping_knowledge:
                X_all_embeddings = torch.cat((X_all_embeddings, node_embs), dim=2)
        final_embs = X_all_embeddings if self.jumping_knowledge else node_embs

        # Graph pooling, e.g. take the mean over all node embeddings (dimension=1)
        if self.graph_pooling == 'sum':
            g_emb = torch.sum(final_embs, dim=1)
        elif self.graph_pooling == 'mean':
            g_emb = torch.mean(final_embs, dim=1)
        elif self.graph_pooling == 'max':
            g_emb, _ = torch.max(final_embs, dim=1)  # returns (values, indices)
        elif self.graph_pooling in ['sum+mean', 'mean+sum', 'mean&sum', 'sum&mean']:
            xmean = torch.mean(final_embs, dim=1)
            xsum = torch.sum(final_embs, dim=1)  # (batch-size, out-dim)
            g_emb = torch.cat((xmean, xsum), dim=1)  # (batch-size 2*out-dim)
        else:
            raise ValueError('Unsupported readout operation')

        # After graph pooling: (batch-size, out-dim)
        out = self.graph_embedding_to_pred(g_emb=g_emb) if readout else g_emb
        return out

    def graph_embedding_to_pred(self, g_emb):
        out = self.MLP_layer.forward(g_emb).squeeze(1)
        return out
