from torch_geometric.transforms import KNNGraph
from torch_geometric.data import Data, Batch
import torch
from typing import List

class GraphIterativeWrapper(torch.nn.Module):
    """
    Wrapper class for iterative geometric-based model evaluation.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be wrapped and iteratively evaluated.
    batch_first : bool, default True
        If True, the first dimension of the input tensor is considered as the batch dimension.

    Attributes
    ----------
    model : torch.nn.Module
        The wrapped model.
    batch_first : bool
        Indicates if the first dimension of the input tensor is the batch dimension.

    Methods
    -------
    forward(x: torch.Tensor, t_eval: List[float] = [1]) -> torch.Tensor
        Perform iterative evaluation of the model at specified time points.
    """
    def __init__(self, 
                 model,
                 batch_first: bool = True,
                 num_neighbors: int = 8):
        super().__init__()
        self.model = model
        self.batch_first = batch_first
        self.k = num_neighbors

    def _make_data(self, x, points):

        points_padded = torch.cat((points,
                        points + torch.tensor([0, 1], device=points.device),
                        points + torch.tensor([1, 0], device=points.device),
                        points + torch.tensor([1, 1], device=points.device),
                        points + torch.tensor([0, -1], device=points.device),
                        points + torch.tensor([-1, 0], device=points.device),
                        points + torch.tensor([-1, -1], device=points.device),
                        points + torch.tensor([1, -1], device=points.device),
                        points + torch.tensor([-1, 1], device=points.device)
                        ), dim=1)
        
        transformation = KNNGraph(k=self.k)
        data_list = []
        for i in range(x.shape[0]):
            x_graph = Data(x=torch.cat((x[i],)*9, dim=0), pos=points_padded[i])
            x_graph = transformation(x_graph)
            x_graph.x = x_graph.x[:x.shape[1]]
            x_graph.pos = x_graph.pos[:x.shape[1]]
            x_graph.edge_index = x_graph.edge_index[:,:x.shape[1]*self.k] % x.shape[1]
            data_list.append(x_graph)
        
        return Batch.from_data_list(data_list)


    def forward(self, 
                x: torch.Tensor, # features
                p: torch.Tensor, # point coordinates
                t_eval: List[float] = [1]):
        
        rollout = []

        x_shape = x.shape

        x = x.view(x_shape[0], x_shape[2], x_shape[1]*x_shape[3])

        data = self._make_data(x, p)

        for t in t_eval:
            data = self.model(data)
            dl = Batch.to_data_list(data)

            x_list = []
            x = x.view(x_shape[0], x_shape[1], x_shape[2], x_shape[3])
            for i in range(len(dl)):
                tmp = torch.cat((x[i,1:], torch.unsqueeze(dl[i].x,dim=0)), dim=0)
                dl[i].x = tmp.view(x_shape[2], x_shape[1]*x_shape[3])
                x_list.append(tmp.view(x_shape[2], x_shape[1]*x_shape[3]))

            x = torch.stack(x_list, dim=0)
            data = Batch.from_data_list(dl)
            x_ = x.view(x_shape[0], x_shape[1], x_shape[2], x_shape[3])[:,-1]
            rollout.append(x_)

        dim = 1 if self.batch_first else 0
        return torch.stack(rollout, dim=dim)
  