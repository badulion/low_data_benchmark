import torch

from typing import List

class PointIterativeWrapper(torch.nn.Module):
    """
    Wrapper class for iterative point-based model evaluation.

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
    forward(x: torch.Tensor, p: torch.Tensor, t_eval: List[float] = [1]) -> torch.Tensor
        Perform iterative evaluation of the model at specified time points.
    """
    def __init__(self, 
                 model,
                 batch_first: bool = True):
        super().__init__()
        self.model = model
        self.batch_first = batch_first

    def forward(self, 
                x: torch.Tensor, # features
                p: torch.Tensor, # point coordinates
                t_eval: List[float] = [1]):
        
        rollout = []

        for t in t_eval:
            x_ = x.view(x.shape[0], x.shape[2], x.shape[1]*x.shape[3])
            x_ = self.model(x_, p)
            x = torch.cat((x[:,1:],x_.view(x.shape[0], 1, x.shape[2], x_.shape[-1])), dim=1)
            rollout.append(x_)

        dim = 1 if self.batch_first else 0
        return torch.stack(rollout, dim=dim)
    

class GridIterativeWrapper(torch.nn.Module):
    """
    Wrapper class for iterative grid-based model evaluation.

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
                 batch_first: bool = True):
        super().__init__()
        self.model = model
        self.batch_first = batch_first

    def forward(self, 
                x: torch.Tensor, # features
                t_eval: List[float] = [1]):
        
        rollout = []

        for t in t_eval:
            x_ = x.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])
            x_ = self.model(x_)
            x = torch.cat((x[:,1:],x_.view(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4])), dim=1)
            rollout.append(x_)

        dim = 1 if self.batch_first else 0
        return torch.stack(rollout, dim=dim)