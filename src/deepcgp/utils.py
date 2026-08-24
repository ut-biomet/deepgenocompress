"""Public utility functions for deepcgp."""

from ._core.compression import LayerSizeOption, possible_first_layer_sizes
from ._core.utils import encoding_size

__all__ = [
    "LayerSizeOption",
    "encoding_size",
    "possible_first_layer_sizes",
]
