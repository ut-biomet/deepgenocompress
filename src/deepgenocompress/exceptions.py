"""Public exception hierarchy for deepgenocompress.

All exceptions raised by deepgenocompress inherit from :class:`DeepgenocompressError`,
allowing callers to catch any package-specific error with a single ``except`` clause.
Specific exceptions are defined alongside the code that raises them (see each
class's source module), and re-exported here to provide a single, stable
import location.

Examples
--------
.. jupyter-execute::

    from deepgenocompress.exceptions import DeepgenocompressError

    try:
        ...
    except DeepgenocompressError as e:
        print(f"deepgenocompress failed: {e}")
"""

from ._core.compression import (
    CompressionModelConfigurationError,
    IncompatibleDataError,
    LayerSizesConfigurationError,
    ModelStateError,
)
from ._core.data_processing import InvalidEncodingMapError
from ._core.exceptions import DeepgenocompressError, DuplicatedMarkerIDsError
from ._core.vcf import InvalidVcfDataError, UnexpectedMarkerIdFormatError

__all__ = [
    "CompressionModelConfigurationError",
    "DeepgenocompressError",
    "DuplicatedMarkerIDsError",
    "IncompatibleDataError",
    "InvalidEncodingMapError",
    "InvalidVcfDataError",
    "LayerSizesConfigurationError",
    "ModelStateError",
    "UnexpectedMarkerIdFormatError",
]
