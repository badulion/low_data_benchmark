from torch import nn

class BaselinePersistence_grid(nn.Module):
    def __init__(self, input_size: int, lookback: int, spatial_dimensions: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.lookback = lookback
        self.spatial_dimensions = spatial_dimensions

    def forward(self, x):
        return x
    
class BaselinePersistence_cloud(nn.Module):
    def __init__(self, input_size: int, lookback: int, spatial_dimensions: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.lookback = lookback
        self.spatial_dimensions = spatial_dimensions

    def forward(self, x, p):
        return x