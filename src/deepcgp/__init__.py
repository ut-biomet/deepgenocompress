"""Deep learning toolkit for compressed genomic prediction.

Provides tools to encode raw genotype data, compress it using autoencoders, and build
predictive models from their compressed representations.
"""

import logging

from . import exceptions, utils, warnings
from ._core.compression import AutoencoderModels, CompressionModel
from ._core.data_processing import build_one_hot_encoding_map, encode_snp_array
from ._core.genomic_prediction import create_model
from ._core.vcf import (
    build_vcf_encoding_map,
    read_vcf,
    reindex_vcf_data,
    validate_vcf_data,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AutoencoderModels",
    "CompressionModel",
    "build_one_hot_encoding_map",
    "build_vcf_encoding_map",
    "create_model",
    "encode_snp_array",
    "exceptions",
    "read_vcf",
    "reindex_vcf_data",
    "utils",
    "validate_vcf_data",
    "warnings",
]
