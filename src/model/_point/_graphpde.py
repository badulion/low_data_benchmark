from torch import nn
from torch_geometric.nn import MessagePassing
import torch
from .components import MLP

class Dynamics(MessagePassing):
    def __init__(self, gamma, phi):
        super(Dynamics, self).__init__(aggr='mean', flow='target_to_source')
        self.gamma = gamma
        self.phi = phi

    def forward(self, u, edge_index, pos):
        return self.propagate(edge_index, u=u, pos=pos)

    def message(self, u_i, u_j, pos_i, pos_j):
        rel_pos = pos_i - pos_j
        phi_input = torch.cat([u_i, u_j-u_i, rel_pos], dim=1)
        return self.phi(phi_input)

    def update(self, aggr, u):
        gamma_input = torch.cat([u, aggr], dim=1)
        dudt = self.gamma(gamma_input)
        return dudt
    


class GraphPDE(nn.Module):
    def __init__(self, 
                 input_size, 
                 lookback, 
                 phi_hidden_size, 
                 phi_hidden_layers, 
                 gamma_hidden_size, 
                 gamma_hidden_layers, 
                 spatial_dimensions=2) -> None:
        super().__init__()
        self.input_size = input_size
        self.lookback = lookback
        self.phi_hidden_size = phi_hidden_size
        self.phi_hidden_layers = phi_hidden_layers
        self.gamma_hidden_size = gamma_hidden_size
        self.gamma_hidden_layers = gamma_hidden_layers
        self.spatial_dimensions = spatial_dimensions

        message_dimension = 1
        self.phi = MLP(input_size=2*input_size*lookback+spatial_dimensions, 
                       output_size=message_dimension, 
                       hidden_size=phi_hidden_size, 
                       hidden_layers=phi_hidden_layers)

        self.gamma = MLP(input_size=input_size*lookback+message_dimension, 
                         output_size=input_size, 
                         hidden_size=gamma_hidden_size, 
                         hidden_layers=gamma_hidden_layers)
        
        self.dudt = Dynamics(self.gamma, self.phi)

    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index

        dudt = self.dudt(x, edge_index, pos)

        
        data.x = x[:,-self.input_size:] + dudt
        
        return data