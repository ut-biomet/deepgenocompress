"""Deep learning toolkit for compressed genomic prediction.

Provides tools to encode raw genotype data, compress it using autoencoders, and build
predictive models from their compressed representations.
"""

import logging

from . import exceptions, warnings
from .compression import AutoencoderModels, CompressionModel
from .data_processing import build_one_hot_encoding_map, encode_snp_array
from .genomic_prediction import create_model

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AutoencoderModels",
    "CompressionModel",
    "build_one_hot_encoding_map",
    "create_model",
    "encode_snp_array",
    "exceptions",
    "warnings",
]
