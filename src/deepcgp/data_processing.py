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
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


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


def _validate_encoding_map(
    encoding_map: Mapping[Any, list[int] | list[float]],
    missing_values: Container = [],
) -> None:
    """Ensure the encoding map is valid.

    Raises an exception if invalid.
    """
    if not isinstance(encoding_map, Mapping):
        raise TypeError(  # pyright: ignore [reportUnreachable]
            f"`encoding_map` must be a dict, got {type(encoding_map)}"
        )

    if len(encoding_map) == 0:
        raise ValueError("`encoding_map` is empty.")

    expected_length = len(next(iter(encoding_map.values())))

    for allele, encoding in encoding_map.items():
        if not isinstance(encoding, list) or not all(
            isinstance(v, Real) for v in encoding
        ):
            raise ValueError(
                f"Encoding for '{allele}' must be a list of numerical values ",
                f"got {encoding}",
            )
        if len(encoding) != expected_length:
            raise ValueError(
                f"All encodings must have the same length, got {len(encoding)} "
                f"for '{allele}' but {expected_length}"
                f"for '{next(iter(encoding_map.keys()))}'"
            )
        if allele in missing_values and any(v != 0 for v in encoding):
            raise ValueError(
                "Missing values not encoded with a vector of 0, "
                f"for '{allele}' got {encoding}."
            )


def encode_snp_array(
    geno_array: NDArray,
    # geno_array: NDArray[np.str_],
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
    TypeError
        If ``encoding_map`` is provided but is not a dict.
    ValueError
        If ``encoding_map`` contains encodings that are not lists of numerical
        values, or if the encodings have inconsistent lengths.
    ValueError
        If ``missing_values`` are encoded with a value other than a vector of 0 in
        ``encoding_map``.

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
