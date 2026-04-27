from collections.abc import Container, Mapping
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def build_one_hot_encoding_map(
    geno_array: ArrayLike,
    exclude: Container = {"N"},
) -> dict[Any, list[float]]:
    """Build a one-hot encoding map based on alleles found in the genotype array.

    The encoding map is built by extracting all values from the array,
    and assigning each a one-hot vector of length equal to the number of unique values.

    Parameters
    ----------
    geno_array : ArrayLike
        2D array of genotype data
    exclude : Container, optional
        Any object supporting the ``in`` operator (e.g. set, list, tuple).
        Collection of values to exclude from the one-hot encoding map.
        These typically the values of geno_array representing missing or ambiguous
        genotype calls that should not be assigned an encoding vector.
        Defaults to ``{"N"}``.


    Returns
    -------
    dict[list[int]]
        A dictionary mapping each unique allele to its one-hot encoded vector.
        The vectors are of length n, where n is the number of unique alleles.

    Examples
    --------
    .. jupyter-execute::

        from deepcgp import build_one_hot_encoding_map
        geno_array = [
            ["G", "A", "T", "T", "A", "C", "A"],
            ["A", "C", "A", "T", "T", "A", "G"],
        ]
        build_one_hot_encoding_map(geno_array)

    """
    alleles = [a for a in sorted(np.unique(geno_array)) if a not in exclude]
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

    Return None if valid raise if not.
    """
    if not isinstance(encoding_map, dict):
        # pyright: ignore [reportUnreachable]
        raise TypeError(f"`encoding_map` must be a dict, got {type(encoding_map)}")

    expected_length = len(list(encoding_map.values())[0])

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
                f"for '{list(encoding_map.keys())[0]}'"
            )
        if allele in missing_values and sum(encoding) != 0:
            raise ValueError(
                "Missing values not encoded with a vector of 0, "
                f"for '{allele}' got {encoding}."
            )


def encode_snp_array(
    geno_array: NDArray[np.str_],
    missing_values: Container = {"N"},
    encoding_map: Mapping[Any, list[int] | list[float]] | None = None,
) -> NDArray[np.float32]:
    """Encode a genotype array into a numerical matrix.

    Each allele in the input array is replaced by its corresponding vector
    from the encoding map. Unknown alleles are encoded as a zero vector.
    The output is cast to float32 as this is the expected input dtype for Keras models.

    Parameters
    ----------
    geno_array : NDArray[np.str_]
        2D array of genotype data where each element is a string representing
        an allele (e.g. "A", "C", "G", "T").
    missing_values : Container, optional
        Any object supporting the ``in`` operator (e.g. set, list, tuple).
        Collection of value of geno_array representing missing genotype calls
        that should not be assigned a default one hot encoding vector.
        They will be encoded as a zero vector. Defaults to ``{"N"}``.
    encoding_map : Mapping[Any, list[int] | list[float]] | None, optional
        A dictionary mapping each allele to its encoding vector.
        If None, a one-hot encoding map is built automatically from the unique
        alleles found in ``geno_array`` using :func:`build_one_hot_encoding_map`
        excluding ``missing_values``

    Returns
    -------
    NDArray[np.float32]
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
        geno_array = np.array([["A", "C"], ["G", "T"]])
        encode_snp_array(geno_array)

    .. jupyter-execute::

        from deepcgp import encode_snp_array
        geno_array = np.array([["A", "C"], ["G", "T"]])
        custom_map = {"A": [0.9, 0.1], "C": [0.1, 0.9], "G": [0.5, 0.5], "T": [0.2, 0.8]}
        encode_snp_array(geno_array, encoding_map=custom_map)

    """
    if encoding_map is not None:
        _validate_encoding_map(encoding_map, missing_values)
    else:
        encoding_map = build_one_hot_encoding_map(geno_array, missing_values)

    encoding_length = len(list(encoding_map.values())[0])

    encoded_rows = []
    for row in geno_array:
        encoded_row = []
        for allele in row:
            encoded_row.extend(encoding_map.get(allele, [0] * encoding_length))
        encoded_rows.append(encoded_row)
    return np.array(encoded_rows, dtype=np.float32)
