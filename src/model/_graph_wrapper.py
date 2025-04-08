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

    def _pad_points(self, points):

        points_padded = torch.cat((points,
                        points + torch.tensor([0, 1]),
                        points + torch.tensor([1, 0]),
                        points + torch.tensor([1, 1]),
                        points + torch.tensor([0, -1]),
                        points + torch.tensor([-1, 0]),
                        points + torch.tensor([-1, -1]),
                        points + torch.tensor([1, -1]),
                        points + torch.tensor([-1, 1])
                        ), dim=1)
        
        return points_padded


    def forward(self, 
                x: torch.Tensor, # features
                p: torch.Tensor, # point coordinates
                t_eval: List[float] = [1]):
        
        rollout = []

        # data [x, pos, edge_index]
        # create pyg graphs -> knn with padding??
        transformation = KNNGraph(k=self.k)
        data_list = []
        for i in range(x.shape[0]):
            x_graph = Data(x=x[i], pos=p[i])
            x_graph = transformation(x_graph)
            data_list.append(x_graph)

        data = Batch.from_data_list(data_list)

        for t in t_eval:
            data = self.model(data)
            #x = x.squeeze(dim=1)
            # get batches back
            x = data.x.view(x.shape)
            rollout.append(x)

        dim = 1 if self.batch_first else 0
        return torch.stack(rollout, dim=dim)