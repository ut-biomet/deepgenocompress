"""Deep learning toolkit for compressed genomic prediction.

Provides tools to encode raw genotype data, compress it using autoencoders,
and build predictive models from the compressed representations.

Submodules
----------
data_processing
    Components for genotype data processing for usage with Kera models.
compression
    Components for genotype data compression using autoencoder-based
    dimensionality reduction.
genomic_prediction
    Components for related to genomic prediction.

Classes
-------
AutoencoderModels
    Builds a symmetric autoencoder and its corresponding encoder from a list
    of layer sizes.
CompressionModel
    Orchestrates data chunking, autoencoder construction, training, and
    compression.

Functions
---------
build_one_hot_encoding_map(geno_array, exclude={"N"})
    Build a one-hot encoding map from the unique alleles in a genotype array.
encode_snp_array(geno_array, missing_values={"N"}, encoding_map=None)
    Encode a genotype array into a float32 numerical matrix.
"""

import logging

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
]
