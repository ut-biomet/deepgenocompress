"""Public warnings hierarchy for deepcgp.

All warnings issued by deepcgp inherit from :class:`DeepcgpWarning`, allowing callers to
catch any package-specific warnings.
"""

from deepcgp._base_warnings import DeepcgpWarning
from deepcgp.compression import (
    ColumnPaddingWarning,
    IncompleteEncodingChunkWarning,
    LessThanOneAlleleChunks,
    NonCompressiveAutoencoderWarning,
)
from deepcgp.data_processing import AllZerosEncodedWarning

__all__ = [
    "AllZerosEncodedWarning",
    "ColumnPaddingWarning",
    "DeepcgpWarning",
    "IncompleteEncodingChunkWarning",
    "LessThanOneAlleleChunks",
    "NonCompressiveAutoencoderWarning",
]
