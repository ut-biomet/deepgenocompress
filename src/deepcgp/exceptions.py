"""Public exception hierarchy for deepcgp.

All exceptions raised by deepcgp inherit from :class:`DeepcgpError`, allowing
callers to catch any package-specific error with a single ``except`` clause.
Specific exceptions are defined alongside the code that raises them (see each
class's source module), and re-exported here to provide a single, stable
import location.

Examples
--------
.. jupyter-execute::

    from deepcgp.exceptions import DeepcgpError

    try:
        ...
    except DeepcgpError as e:
        print(f"deepcgp failed: {e}")
"""

from ._core.compression import (
    CompressionModelConfigurationError,
    IncompatibleDataError,
    LayerSizesConfigurationError,
    ModelStateError,
)
from ._core.data_processing import InvalidEncodingMapError
from ._core.exceptions import DeepcgpError, DuplicatedMarkerIDsError
from ._core.vcf import InvalidVcfDataError, UnexpectedMarkerIdFormatError

__all__ = [
    "CompressionModelConfigurationError",
    "DeepcgpError",
    "DuplicatedMarkerIDsError",
    "IncompatibleDataError",
    "InvalidEncodingMapError",
    "InvalidVcfDataError",
    "LayerSizesConfigurationError",
    "ModelStateError",
    "UnexpectedMarkerIdFormatError",
]
