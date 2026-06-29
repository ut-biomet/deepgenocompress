"""Utilities for encoding genotype arrays into numerical matrices.

Provides functions to build one-hot encoding maps from allele data and
encode SNP arrays into float32 matrices suitable for use with Keras models.

Functions
---------
build_one_hot_encoding_map(geno_array, exclude={"N"})
    Build a one-hot encoding map based on alleles found in the genotype array.

encode_snp_array(geno_array, missing_values={"N"}, encoding_map=None)
    Encode a genotype array into a numerical matrix.
"""

import warnings
from collections.abc import Collection, Container, Mapping
from enum import StrEnum, auto
from numbers import Real
from typing import Any, ClassVar, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from deepcgp._base_exceptions import DeepcgpError, _type_fullname


def build_one_hot_encoding_map(
    geno_array: ArrayLike,
    exclude: Collection = {"N"},
) -> dict[Any, list[float]]:
    """Build a one-hot encoding map based on alleles found in the genotype array.

    The encoding map is built by extracting all unique values from the array,
    excluding specified missing or ambiguous values, and assigning each valid
    allele a one-hot vector of length equal to the number of unique valid alleles.

    Parameters
    ----------
    geno_array :
        2D array of genotype data.
    exclude :
        Any object supporting the ``in`` operator (e.g. set, list, tuple).
        Collection of values to exclude from the one-hot encoding map.
        Typically the values of ``geno_array`` representing missing or ambiguous
        genotype calls that should not be assigned an encoding vector.
        Defaults to ``{"N"}``. Note that any value for which :func:`pandas.isna`
        returns ``True`` (e.g. ``np.nan``, ``None``, ``pd.NA``...)
        is always implicitly excluded regardless of this container's contents.

    Returns
    -------
        A dictionary mapping each unique allele to its one-hot encoded vector.
        The vectors are of length n, where n is the number of unique, non-excluded
        alleles.

    Notes
    -----
        Any value for which  :func:`pandas.isna` returns ``True`` (ie. ``np.nan``,
        ``None``, ``pd.NA``, and ``pd.NaT``) is always excluded from the encoding map.

    Examples
    --------
    .. jupyter-execute::

        import numpy as np
        from deepcgp import build_one_hot_encoding_map

        # Array containing valid alleles, an excluded "N", and `None`
        geno_array = [
            ["G", "A", "T", "N", "A", "C", "A"],
            ["A", "C", "A", "T", "T", "A", None],
        ]
        build_one_hot_encoding_map(geno_array)

    """
    # Note: to make the code simpler all "NA like" (ie. for which pd.isna returns True,
    # np.nan, pd.NA, None, etc...) values are always excluded. If not we would have to
    # handle each manually:
    #   - np.nan != np.nan so checking `np.nan in exclude` would fail
    #   - ("A" == pd.NA) equals pd.NA
    #   - etc...

    unique_values = set(np.asarray(geno_array).ravel())

    unique_non_missing_values = {u for u in unique_values if not pd.isna(u)}
    safe_exclude = [ex for ex in exclude if not pd.isna(ex)]

    # Note:
    alleles = [
        a
        for a in sorted(unique_non_missing_values, key=lambda x: str(x))
        if a not in safe_exclude
    ]
    n = len(alleles)
    return {
        allele: [1.0 if i == j else 0.0 for j in range(n)]
        for i, allele in enumerate(alleles)
    }


class InvalidEncodingMapError(DeepcgpError):
    """Raised when an encoding map is not valid.

    Instances are constructed with a :class:`ReasonCode` identifying which validation
    failed, plus an ``extra`` mapping of contextual values
    """

    class ReasonCode(StrEnum):
        """Possible invalid reasons."""

        def __repr__(self) -> str:
            return self.name

        INVALID_TYPE = auto()
        """Encoding map is not a :class:`~collections.abc.Mapping`."""
        EMPTY_ENCODING_MAP = auto()
        """Encoding map has no entries."""
        INVALID_ENCODING = auto()
        """An allele's encoding is not a list of numerical values."""
        LENGTH_MISMATCH = auto()
        """An allele's encoding has a different length than the reference allele's
        encoding."""
        MISSING_VALUE_NOT_ZERO = auto()
        """An allele listed as a missing value is not encoded as a vector of all
        zeros."""
    _MESSAGES: ClassVar[dict["InvalidEncodingMapError.ReasonCode", str]] = {
        ReasonCode.INVALID_TYPE: (
            f"`encoding_map` must be a {_type_fullname(Mapping)} "
            "got a {provided_type_str}."
        ),
        ReasonCode.EMPTY_ENCODING_MAP: "Encoding map is empty.",
        ReasonCode.INVALID_ENCODING: (
            "Encoding for allele '{offending_allele}' must be a list of numerical "
            "values, got '{provided_encoding}'."
        ),
        ReasonCode.LENGTH_MISMATCH: (
            "All encodings must have the same length, got a length of "
            "{provided_encoding_length} for allele '{offending_allele}' but "
            "{reference_encoding_length} for allele '{reference_allele}'"
        ),
        ReasonCode.MISSING_VALUE_NOT_ZERO: (
            "Missing value '{offending_missing_value}' must be encoded with a "
            "vector of 0, got '{provided_encoding}'."
        ),
    }

    class _Extra(TypedDict, total=False):
        reason: "InvalidEncodingMapError.ReasonCode"
        provided_type: type
        provided_encoding: Any
        offending_allele: Any
        reference_encoding: Any
        reference_allele: Any
        offending_missing_value: Any

    extra: "_Extra | dict[str, Any]"
    """Extra information related to the error.

    :class:`dict` with possible keys depending on the :attr:`reason`:
        - ``reason``: :class:`ReasonCode`
        - ``provided_type``: :class:`type`
        - ``provided_encoding``: :class:`Any`
        - ``offending_allele``: :class:`Any`
        - ``reference_encoding``: :class:`Any`
        - ``reference_allele``: :class:`Any`
        - ``offending_missing_value``: :class:`Any`
    """

    reason: ReasonCode
    """Which validation rule the encoding map failed."""

    def __init__(
        self,
        reason: "InvalidEncodingMapError.ReasonCode",
        extra: "InvalidEncodingMapError. _Extra | None" = None,
    ):
        self.reason = reason
        extra = extra or {}
        str_extra = {
            "provided_type_str": (
                _type_fullname(extra["provided_type"])
                if "provided_type" in extra
                else None
            ),
            "provided_encoding_length": (
                len(extra["provided_encoding"])
                if "provided_encoding" in extra
                else None
            ),
            "reference_encoding_length": (
                len(extra["reference_encoding"])
                if "reference_encoding" in extra
                else None
            ),
        }
        str_extra = {k: v for k, v in str_extra.items() if v is not None}
        message = self._MESSAGES[reason].format(**extra, **str_extra)
        super().__init__(message=message, extra={"reason": reason, **extra})


def _validate_encoding_map(
    encoding_map: Mapping[Any, list[int] | list[float]],
    missing_values: Container = [],
) -> None:
    """Ensure the encoding map is valid.

    Raises an :class:`InvalidEncodingMapError` if invalid.
    """
    if not isinstance(encoding_map, Mapping):
        raise InvalidEncodingMapError(
            reason=InvalidEncodingMapError.ReasonCode.INVALID_TYPE,
            extra={"provided_type": type(encoding_map)},
        )

    if len(encoding_map) == 0:
        raise InvalidEncodingMapError(
            reason=InvalidEncodingMapError.ReasonCode.EMPTY_ENCODING_MAP
        )

    reference_allele = next(iter(encoding_map.keys()))  # first key
    reference_encoding = encoding_map[reference_allele]

    for allele, encoding in encoding_map.items():
        if not isinstance(encoding, list) or not all(
            isinstance(v, Real) for v in encoding
        ):
            raise InvalidEncodingMapError(
                reason=InvalidEncodingMapError.ReasonCode.INVALID_ENCODING,
                extra={"offending_allele": allele, "provided_encoding": encoding},
            )
        if len(encoding) != len(reference_encoding):
            raise InvalidEncodingMapError(
                reason=InvalidEncodingMapError.ReasonCode.LENGTH_MISMATCH,
                extra={
                    "offending_allele": allele,
                    "provided_encoding": encoding,
                    "reference_allele": reference_allele,
                    "reference_encoding": encoding_map[reference_allele],
                },
            )
        if allele in missing_values and any(v != 0 for v in encoding):
            raise InvalidEncodingMapError(
                reason=InvalidEncodingMapError.ReasonCode.MISSING_VALUE_NOT_ZERO,
                extra={
                    "offending_missing_value": allele,
                    "provided_encoding": encoding,
                },
            )


def encode_snp_array(
    geno_array: NDArray,
    missing_values: Collection = {"N"},
    encoding_map: Mapping[Any, list[int] | list[float]] | None = None,
) -> NDArray[np.float32]:
    """Encode a genotype array into a numerical matrix.

    Each allele in the input array is replaced by its corresponding vector
    from the encoding map. Unknown alleles are encoded as a zero vector.
    The output is cast to float32 as this is the expected input dtype for Keras models.

    Parameters
    ----------
    geno_array :
        2D array of genotype data. Each element is typically a string
        representing an allele (e.g. "A", "C", "G", "T"), but other types
        (e.g. integers) are also supported.
    missing_values :
        Any object supporting the ``in`` operator (e.g. set, list, tuple).
        Collection of value of geno_array representing missing genotype calls
        that should not be assigned a default one hot encoding vector.
        They will be encoded as a zero vector. Defaults to ``{"N"}``.
    encoding_map :
        A dictionary mapping each allele to its encoding vector.
        If None, a one-hot encoding map is built automatically from the unique
        alleles found in ``geno_array`` using :func:`build_one_hot_encoding_map`
        excluding ``missing_values``

    Returns
    -------
        2D array of shape (n_samples, n_alleles * encoding_length) where each
        row is the flattened encoding of the corresponding input row.

    Raises
    ------
    InvalidEncodingMapError
        If ``encoding_map`` is provided but:
          - is not a Mapping (e.g. a dict)
          - is empty
          - encodings are not lists of numerical values
          - encodings have inconsistent lengths
          - ``missing_values`` are encoded with a value other than a vector of 0

    Examples
    --------
    .. jupyter-execute::

        from deepcgp import encode_snp_array
        import numpy as np

        geno_array = np.array([["A", "C"], ["G", "T"]])
        encode_snp_array(geno_array)

    .. jupyter-execute::

        geno_array = np.array([["A", "C"], ["G", "T"]])
        custom_map = {
            "A": [0, 0],
            "C": [0, 1],
            "G": [1, 0],
            "T": [1, 1]
        }
        encode_snp_array(geno_array, encoding_map=custom_map)

    """
    if encoding_map is not None:
        _validate_encoding_map(encoding_map, missing_values)
    else:
        encoding_map = build_one_hot_encoding_map(geno_array, missing_values)

    encoding_length = len(next(iter(encoding_map.values())))

    encoded_rows = []
    for row in geno_array:
        encoded_row = []
        for allele in row:
            encoded_row.extend(encoding_map.get(allele, [0] * encoding_length))
        encoded_rows.append(encoded_row)
    encoded_geno = np.array(encoded_rows, dtype=np.float32)

    if not np.any(encoded_geno):
        warnings.warn(
            "The encoded array contains only zeros. This may indicate that all values "
            "in geno_array are missing or not present in encoding_map.",
            UserWarning,
            stacklevel=2,
        )

    return encoded_geno
