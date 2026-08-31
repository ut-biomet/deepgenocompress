from collections.abc import Mapping
from enum import Enum
from typing import Any


class _Missing(Enum):
    missing = object()


_MISSING = _Missing.missing


def encoding_size(encoding_map: Mapping[Any, list[int] | list[float]]):
    """Return the length of the encoding vectors in an encoding map.

    Minimal function to determined from the first value in ``encoding_map`` from
    the first value. For simplicity this function does not validate the provided
    ``encoding_map`` is correct; use :func:`_validate_encoding_map` first
    if necessary.

    Parameters
    ----------
    encoding_map :
        Mapping from allele values to their encoding vectors.

    Returns
    -------
        Length of the encoding vector associated with the first key in
        ``encoding_map``.

    Raises
    ------
    ValueError
        If ``encoding_map`` is empty.
    """
    try:
        return len(next(iter(encoding_map.values())))
    except StopIteration as e:
        raise ValueError("encoding_map must not be empty") from e
