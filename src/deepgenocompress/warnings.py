"""Public warnings hierarchy for deepgenocompress.

All warnings issued by deepgenocompress inherit from :class:`DeepgenocompressWarning`,
allowing callers to catch any package-specific warnings.
"""

from ._core.compression import (
    ColumnPaddingWarning,
    IncompleteEncodingChunkWarning,
    LessThanOneAlleleChunksWarning,
    NonCompressiveAutoencoderWarning,
)
from ._core.data_processing import AllZerosEncodedWarning, UnmappedValuesWarning
from ._core.warnings import DeepgenocompressWarning

__all__ = [
    "AllZerosEncodedWarning",
    "ColumnPaddingWarning",
    "DeepgenocompressWarning",
    "IncompleteEncodingChunkWarning",
    "LessThanOneAlleleChunksWarning",
    "NonCompressiveAutoencoderWarning",
    "UnmappedValuesWarning",
]
