"""Public utility functions for deepgenocompress."""

from typing import get_args

from ._core.compression import LayerSizeOption, possible_first_layer_sizes
from ._core.utils import ExampleFiles, encoding_size
from ._core.vcf import _MARKER_ID_FORMAT, build_marker_ids, infer_marker_id_format

MARKER_ID_FORMATS: tuple[_MARKER_ID_FORMAT] = get_args(_MARKER_ID_FORMAT)
"""Tuple of accepted marker id formats"""

__all__ = [
    "MARKER_ID_FORMATS",
    "ExampleFiles",
    "LayerSizeOption",
    "build_marker_ids",
    "encoding_size",
    "infer_marker_id_format",
    "possible_first_layer_sizes",
]
