"""
Module containing the models used in the DynaBench benchmark.
"""

from ._grid._cnn import CNN
from ._grid._resnet import ResNet
from ._grid._neuralpde import NeuralPDE

from ._point.point_transformer import PointTransformerV1


__all__ = ['CNN', 'ResNet', 'NeuralPDE', 'PointTransformerV1']