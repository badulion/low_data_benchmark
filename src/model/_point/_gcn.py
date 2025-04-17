# The MIT License

# Copyright (c) 2017 Thomas Kipf

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torch.nn.parameter import Parameter
# from torch.nn.modules.module import Module


# class GraphConvolution(Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """

#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, input, adj):
#         support = torch.matmul(input, self.weight)
#         output = torch.matmul(adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'


    
# class GCN(nn.Module):
#     def __init__(self, num_convs, in_features, out_features, lookback, hidden_features, dropout, knn: int = 10):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(in_features*lookback, hidden_features)
#         self.gcs = nn.ModuleList([GraphConvolution(hidden_features, hidden_features) for _ in range(num_convs-2)])
#         self.gc2 = GraphConvolution(hidden_features, out_features)
#         self.dropout = dropout
#         self.knn = knn

#     def _knn(self, query_points, data_points, k=5):
#         # Calculate pairwise distances
#         distances = torch.cdist(query_points, data_points)
#         # Sort distances to find nearest neighbors
#         _, indices = torch.topk(distances, k+1, largest=False)
#         return indices[:,:,1:] ####

#     def forward(self, x, p):
#         batched = True
#         if x.dim() == 2:
#             x = x.unsqueeze(0)
#             batched = False
#         if p.dim() == 2:
#             p = p.unsqueeze(0)

#         # adjacency matrix with padded nearest neighbors
#         p_padded = torch.cat((p,
#                                 p + torch.tensor([0, 1], device=p.device),
#                                 p + torch.tensor([1, 0], device=p.device),
#                                 p + torch.tensor([1, 1], device=p.device),
#                                 p + torch.tensor([0, -1], device=p.device),
#                                 p + torch.tensor([-1, 0], device=p.device),
#                                 p + torch.tensor([-1, -1], device=p.device),
#                                 p + torch.tensor([1, -1], device=p.device),
#                                 p + torch.tensor([-1, 1], device=p.device)
#                                 ), dim=1)
#         p_knn = self._knn(p, p_padded, self.knn)
#         p_knn = p_knn%p.shape[1]
#         batch_size = p.size(0)
#         adj = torch.zeros(batch_size, p.size(1), p.size(1), device=p.device)
#         for i in range(batch_size):
#             adj[i, torch.arange(p.size(1)).unsqueeze(-1), p_knn[i]] = 1
#             adj[i] = adj[i] + adj[i].T
#             adj[i][adj[i] > 1] = 1
#         norm = adj.sum(dim=-1, keepdim=True)
#         adj = adj / norm

#         x = self.gc1(x, adj)
#         x = F.relu(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#         for gc in self.gcs:
#             x = gc(x, adj)
#             x = F.relu(x)
#             x = F.dropout(x, self.dropout, training=self.training)
#         y = self.gc2(x, adj)
        
#         if not batched: y = y.squeeze(0)
#         return y


from torch_geometric.nn import GCNConv
from torch import nn
import torch

class GCN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_layers=1, spatial_dimensions=2) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.spatial_dimensions = spatial_dimensions

        self.input_layer = GCNConv(input_size+spatial_dimensions, hidden_size)
        self.hidden_layers = nn.ModuleList([GCNConv(hidden_size+spatial_dimensions, hidden_size) for _ in range(hidden_layers-1)])
        self.output_layer = GCNConv(hidden_size+spatial_dimensions, output_size)
        self.activation = nn.ReLU()

    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index

        x = self.input_layer(torch.hstack([x, pos]), edge_index)
        x = self.activation(x)

        for layer in self.hidden_layers:
            x = layer(torch.hstack([x, pos]), edge_index)
            x = self.activation(x)

        x = self.output_layer(torch.hstack([x, pos]), edge_index)
        data.x = x
        return data