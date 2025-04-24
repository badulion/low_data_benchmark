from torch import nn
import einops

class BaselinePersistence_grid(nn.Module):
    def __init__(self, input_size: int, output_size: int, lookback: int, spatial_dimensions: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lookback = lookback
        self.spatial_dimensions = spatial_dimensions

    def forward(self, x):
        return x[:,-self.output_size:]
    
class BaselinePersistence_cloud(nn.Module):
    def __init__(self, input_size: int, output_size: int, lookback: int, spatial_dimensions: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lookback = lookback
        self.spatial_dimensions = spatial_dimensions

    def forward(self, x, p):
        return x[:,:,-self.output_size:]