"""Public warnings hierarchy for deepcgp.

All warnings issued by deepcgp inherit from :class:`DeepcgpWarning`, allowing callers to
catch any package-specific warnings.
"""

from ._core.compression import (
    ColumnPaddingWarning,
    IncompleteEncodingChunkWarning,
    LessThanOneAlleleChunks,
    NonCompressiveAutoencoderWarning,
)
from ._core.data_processing import AllZerosEncodedWarning
from ._core.warnings import DeepcgpWarning

__all__ = [
    "AllZerosEncodedWarning",
    "ColumnPaddingWarning",
    "DeepcgpWarning",
    "IncompleteEncodingChunkWarning",
    "LessThanOneAlleleChunks",
    "NonCompressiveAutoencoderWarning",
]
