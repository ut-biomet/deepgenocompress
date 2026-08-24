"""Utilities for autoencoder-based compression of genotype data.

Provides :class:`CompressionModel`, which splits encoded genotype arrays
into fixed-size chunks and trains one autoencoder per chunk. The learned
encoders can then compress new data into a lower-dimensional representation.

Classes
-------
AutoencoderModels
    Builds a symmetric autoencoder and its corresponding encoder from a list of
    layer sizes. (Also available as ``deepcgp.AutoencoderModels``)

CompressionModel
    Orchestrates data chunking, autoencoder construction, training, and
    compression. Splits the genotype data into chunks matching the first
    layer size, and fits one :class:`AutoencoderModels` instance per chunk.
    (Also available as ``deepcgp.CompressionModel``)
"""

import logging
import random
from collections.abc import Collection, Mapping
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, TypedDict

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.losses import Loss
from keras.optimizers import Adam

from .utils import encoding_size
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split

from deepcgp._core.exceptions import DeepcgpError, _type_fullname
from deepcgp._core.warnings import DeepcgpWarning, _deepcgp_warn

from .data_processing import (
    _validate_encoding_map,
    build_one_hot_encoding_map,
    encode_snp_array,
)

logger = logging.getLogger(__name__)


def _default_encoder_activation_functions(
    encoder_layers_sizes: list[int] | tuple[int],
) -> list[str]:
    """Return default encoder activations functions.

    - "relu" for encoding layers
    - "sigmoid" for latent layer
    """
    n = len(encoder_layers_sizes) - 1
    return ["relu"] * (n - 1) + ["sigmoid"]


def _default_decoder_activation_functions(
    encoder_layers_sizes: list[int] | tuple[int],
) -> list[str]:
    """Return default decoder activations functions.

    Because of symmetry this is the same as ``_get_encoder_activation_functions``:
    """
    return _default_encoder_activation_functions(encoder_layers_sizes)


class LayerSizesConfigurationError(DeepcgpError):
    """Raised when layer sizes configuration is not valid.

    Instances are constructed with a :class:`ReasonCode` identifying which validation
    failed, plus an ``extra`` mapping of contextual values.
    """

    class ReasonCode(StrEnum):
        """Possible invalid reasons."""

        def __repr__(self) -> str:
            """Return the string representation."""
            return self.name

        INVALID_TYPE = auto()
        """Provided layer sizes is not a :class:`list` or :class:`tuple`."""
        TOO_FEW_LAYERS = auto()
        """Provided layer sizes have less than 2 elements."""
        INVALID_LAYER_TYPE = auto()
        """Provided layer sizes values are not :class:`int`."""

    _MESSAGES: ClassVar[dict["LayerSizesConfigurationError.ReasonCode", str]] = {
        ReasonCode.INVALID_TYPE: (
            "`layer_sizes` must be a list or tuple, got a {provided_type_str}."
        ),
        ReasonCode.TOO_FEW_LAYERS: (
            "`layer_sizes` length must be at least 2, got {layer_sizes_length}."
        ),
        ReasonCode.INVALID_LAYER_TYPE: (
            "`layer_sizes` must be a list or tuple of int, got "
            "`{provided_element_type_str}` at index {offending_index}."
        ),
    }

    class _Extra(TypedDict, total=False):
        reason: "LayerSizesConfigurationError.ReasonCode"
        provided_type: type
        provided_layer_sizes: list | tuple
        provided_element_type: type
        offending_index: int

    extra: "_Extra | dict[str, Any]"
    """Extra information related to the error.

    :class:`dict` with possible keys depending on the :attr:`reason`:
        - ``reason``: :class:`ReasonCode`
        - ``provided_type``: :class:`type`
        - ``provided_layer_sizes``: :class:`list` or  :class:`tuple`
        - ``provided_element_type``: :class:`type`
        - ``offending_index``: :class:`int`
    """

    reason: ReasonCode
    """Which validation rule layer sizes failed."""

    def __init__(
        self,
        reason: "LayerSizesConfigurationError.ReasonCode",
        extra: "LayerSizesConfigurationError._Extra | None" = None,
    ):
        self.reason = reason
        extra = extra or {}
        str_extra = {
            "provided_type_str": (
                _type_fullname(extra["provided_type"])
                if "provided_type" in extra
                else None
            ),
            "provided_element_type_str": (
                _type_fullname(extra["provided_element_type"])
                if "provided_element_type" in extra
                else None
            ),
            "layer_sizes_length": (
                len(extra["provided_layer_sizes"])
                if "provided_layer_sizes" in extra
                else None
            ),
        }
        str_extra = {k: v for k, v in str_extra.items() if v is not None}
        message = self._MESSAGES[reason].format(**extra, **str_extra)
        super().__init__(message=message, extra={"reason": reason, **extra})


def _check_layer_sizes(layer_sizes):
    """Check layers_sizes's type and value.

    Warns
    -----
    UserWarning
        If latent layer size (i.e. ``layer_sizes[-1]``) is larger or equal to
        input layer size (i.e. ``layer_sizes[0]``).

    Raises
    ------
    LayerSizesConfigurationError
        If ``layer_sizes`` is invalid:
        - not a list or tuple.
        - has a length lower than 2.
        - values are not integers.
    """
    if not isinstance(layer_sizes, (list, tuple)):
        raise LayerSizesConfigurationError(
            reason=LayerSizesConfigurationError.ReasonCode.INVALID_TYPE,
            extra={"provided_type": type(layer_sizes)},
        )

    if len(layer_sizes) < 2:
        raise LayerSizesConfigurationError(
            reason=LayerSizesConfigurationError.ReasonCode.TOO_FEW_LAYERS,
            extra={"provided_layer_sizes": layer_sizes},
        )
    for i, ls in enumerate(layer_sizes):
        if not isinstance(ls, int):
            raise LayerSizesConfigurationError(
                reason=LayerSizesConfigurationError.ReasonCode.INVALID_LAYER_TYPE,
                extra={
                    "provided_element_type": type(layer_sizes[i]),
                    "offending_index": i,
                },
            )
    if layer_sizes[-1] >= layer_sizes[0]:
        _deepcgp_warn(NonCompressiveAutoencoderWarning(layer_sizes))


class NonCompressiveAutoencoderWarning(DeepcgpWarning):
    """Issued when the autoencoder will increase data size.

    Due to the latent layer size being larger than the input size.
    """

    class _Extra(TypedDict, total=True):
        layer_sizes: list[int]

    extra: "_Extra | dict[str, Any]"

    def __init__(self, layer_sizes):
        super().__init__(
            message=(
                f"Latent layer size ({layer_sizes[-1]}) is equal or larger than "
                f"the input layer size ({layer_sizes[0]}). This will expand rather "
                "than compress the data."
            ),
            extra={"layer_sizes": layer_sizes},
        )


class AutoencoderModels:
    """Data structure representing an autoencoder and its corresponding encoder.

    The architecture is determined by ``layer_sizes``: the first element is the
    input/output dimension, the last is the latent dimension, and the elements
    in between define the encoding layers. The decoder mirrors the encoder in
    reverse.

    Encoding and decoding layers use ReLU activation, the latent layer and output layer
    use sigmoid activation.

    Parameters
    ----------
    layer_sizes :
        Ordered layer dimensions, Must have at least 2 elements
        e.g. ``[28, 14, 7, 3]`` produces:

        input(28) -> encoding(14, relu) -> encoding(7, relu) -> latent(3, sigmoid)
        -> decoding(7, relu) -> decoding(14, relu) -> output(28, sigmoid).

    Warns
    -----
    UserWarning
        If latent layer size (i.e. ``layer_sizes[-1]``) is larger or equal to
        input layer size (i.e. ``layer_sizes[0]``).


    Raises
    ------
    LayerSizesConfigurationError
        If ``layer_sizes`` is invalid:
        - not a list or tuple.
        - has a length lower than 2.
        - values are not integers.

    Examples
    --------
    .. jupyter-kernel::
       :id: AutoencoderModels-example

    .. jupyter-execute::

        from deepcgp import AutoencoderModels

        aem = AutoencoderModels([28, 14, 7, 3])
        aem.autoencoder.summary(print_fn=print)

    .. jupyter-execute::

        aem.encoder.summary(print_fn=print)

    """

    autoencoder: Model
    """The full autoencoder keras model."""

    encoder: Model
    """The encoder keras model.

    (Shares same layers with the autoencoder)
    """

    is_fitted: bool = False
    """Is the model fitted?

    Note: this value is not synchronized with the actual model state, but updated
    by this module when calling :meth:`deepcgp.CompressionModel.fit`.
    Fitting the model manually will not update this value.
    """

    def __init__(self, layer_sizes: list[int] | tuple[int]):
        """Initialise from layers_sizes."""
        _check_layer_sizes(layer_sizes)

        encoder_activation_functions = _default_encoder_activation_functions(
            layer_sizes
        )
        decoder_activation_functions = _default_decoder_activation_functions(
            layer_sizes
        )

        input_size = layer_sizes[0]
        encoding_sizes = layer_sizes[1:-1]
        latent_size = layer_sizes[-1]
        decoding_sizes = encoding_sizes[::-1]
        output_size = layer_sizes[0]

        encoding_activations = encoder_activation_functions[:-1]
        latent_activations = encoder_activation_functions[-1]
        decoding_activations = decoder_activation_functions[:-1]
        output_activation = decoder_activation_functions[-1]

        input_layer = Input(shape=(input_size,))

        encoder_layers = input_layer
        for i, (layer_size, act_fun) in enumerate(
            zip(encoding_sizes, encoding_activations, strict=True)
        ):
            encoder_layers = Dense(
                layer_size, activation=act_fun, name=f"encoding_{i}"
            )(encoder_layers)
        encoder_layers = Dense(
            latent_size, activation=latent_activations, name="latent"
        )(encoder_layers)

        decoder_layers = encoder_layers
        for i, (layer_size, act_fun) in enumerate(
            zip(decoding_sizes, decoding_activations, strict=True)
        ):
            decoder_layers = Dense(
                layer_size, activation=act_fun, name=f"decoding_{i}"
            )(decoder_layers)
        decoder_layers = Dense(
            output_size, activation=output_activation, name="output"
        )(decoder_layers)

        self.encoder = Model(inputs=input_layer, outputs=encoder_layers, name="encoder")
        self.autoencoder = Model(
            inputs=input_layer, outputs=decoder_layers, name="autoencoder"
        )


class ModelStateError(DeepcgpError):
    """Raised when model is not correctly prepared for the requested operation.

    Instances are constructed with a :class:`ReasonCode` identifying which state
    prediction was violated.
    """

    class ReasonCode(StrEnum):
        """Possible invalid reasons."""

        def __repr__(self) -> str:
            """Return the string representation."""
            return self.name

        NO_TRAINING_DATA = auto()
        """:attr:`CompressionModel.training_encoded_geno_array` is ``None``."""
        NO_LAYER_SIZES = auto()
        """:attr:`CompressionModel.layer_sizes` is not set, so the autoencoder
        architecture and chunking cannot be determined."""
        MODEL_NOT_FITTED = auto()
        """:attr:`CompressionModel.is_fitted` is ``False``, so the autoencoders are not
        ready to compress data."""

    _MESSAGES: ClassVar[dict["ModelStateError.ReasonCode", str]] = {
        ReasonCode.NO_TRAINING_DATA: "No training data available.",
        ReasonCode.NO_LAYER_SIZES: "`layer_sizes` is not set.",
        ReasonCode.MODEL_NOT_FITTED: "Model is not fitted.",
    }

    class _Extra(TypedDict, total=False):
        reason: "ModelStateError.ReasonCode"

    extra: "_Extra | dict[str, Any]"
    """Extra information related to the error.

    :class:`dict` with possible keys depending on the :attr:`reason`:
        - ``reason``: :class:`ReasonCode`
    """

    reason: ReasonCode
    """Which state precondition was violated."""

    def __init__(
        self,
        reason: "ModelStateError.ReasonCode",
        extra: "ModelStateError._Extra | None" = None,
    ):
        self.reason = reason
        extra = extra or {}
        message = self._MESSAGES[reason]
        super().__init__(message=message, extra={"reason": reason, **extra})


class IncompatibleDataError(DeepcgpError):
    """Raised when data to compress are incompatible with the model.

    Instances are constructed with a :class:`ReasonCode` identifying which compatibility
    check failed, plus an ``extra`` mapping of contextual values
    """

    class ReasonCode(StrEnum):
        """Possible invalid reasons."""

        def __repr__(self) -> str:
            """Return the string representation."""
            return self.name

        INVALID_NUMBER_OF_COLUMNS = auto()
        """The encoded data to compress does not have the same number of columns as the
        encoded training data."""
        INVALID_COLUMN_INDEX = auto()
        """The provided DataFrame's columns do not match the training markers index."""

    _MESSAGES: ClassVar[dict["IncompatibleDataError.ReasonCode", str]] = {
        ReasonCode.INVALID_NUMBER_OF_COLUMNS: (
            "Incompatible data. Expected {encoded_training_data_ncols} columns "
            "(from encoded training data) but provided encoded data have "
            "{encoded_provided_data_ncols} columns. Ensure the input is encoded with "
            "the same encoding map and contains the same markers as the training data."
        ),
        ReasonCode.INVALID_COLUMN_INDEX: (
            "Incompatible data. Provided DataFrame columns do not match training "
            "markers index."
        ),
    }

    class _Extra(TypedDict, total=False):
        encoded_provided_data_ncols: int
        encoded_training_data_ncols: int
        training_data_index: pd.Index
        provided_data_index: pd.Index

    extra: "_Extra | dict[str, Any]"
    """Extra information related to the error.

    :class:`dict` with possible keys depending on the :attr:`reason`:
        - ``reason``: :class:`ReasonCode`
        - ``encoded_provided_data_ncols``: :class:`int`
        - ``encoded_training_data_ncols``: :class:`int`
        - ``training_data_index``: :class:`pd.Index`
        - ``provided_data_index``: :class:`pd.Index`
    """
    reason: ReasonCode
    """Which compatibility check failed."""

    def __init__(
        self,
        reason: "IncompatibleDataError.ReasonCode",
        extra: "IncompatibleDataError._Extra | None" = None,
    ):
        self.reason = reason
        extra = extra or {}
        message = self._MESSAGES[reason].format(**extra)
        super().__init__(message=message, extra={"reason": reason, **extra})


class CompressionModelConfigurationError(DeepcgpError):
    """Raised when a :class:`CompressionModel` is configured with incompatible inputs.

    Instances are constructed with a :class:`ReasonCode` identifying which configuration
    check failed, plus an ``extra`` mapping of contextual values.
    """

    class ReasonCode(StrEnum):
        """Possible invalid reasons."""

        def __repr__(self) -> str:
            """Return the string representation."""
            return self.name

        EMPTY_DATAFRAME = auto()
        """The training DataFrame is empty."""
        MARKER_INDEX_SIZE_MISMATCH = auto()
        """Marker index length missmatch training data size."""

    _MESSAGES: ClassVar[dict["CompressionModelConfigurationError.ReasonCode", str]] = {
        ReasonCode.EMPTY_DATAFRAME: (
            "`training_dataframe` is empty. Provide a DataFrame compatible with "
            "at least one row and one column."
        ),
        ReasonCode.MARKER_INDEX_SIZE_MISMATCH: (
            "Marker index length missmatch training data size. "
            "Marker index has {markers_index_length} markers, but "
            "the encoded training genotype array implies {expected_n_markers} markers "
            "({encoded_training_data_ncols} columns / encoding size {encoding_size})."
        ),
    }

    class _Extra(TypedDict, total=False):
        reason: "CompressionModelConfigurationError.ReasonCode"
        encoded_training_data_ncols: int
        encoding_size: int
        markers_index: pd.Index

    extra: "_Extra | dict[str, Any]"
    """Extra information related to the error.

    :class:`dict` with possible keys depending on the :attr:`reason`:
        - ``reason``: :class:`ReasonCode`
        - ``encoded_training_data_ncols``: :class:`int`
        - ``encoding_size``: :class:`int`
        - ``markers_index``: :class:`pd.Index`
    """
    reason: ReasonCode
    """Which configuration check failed."""

    def __init__(
        self,
        reason: "CompressionModelConfigurationError.ReasonCode",
        extra: "CompressionModelConfigurationError._Extra | None" = None,
    ):
        self.reason = reason
        extra = extra or {}
        str_extra = {
            "markers_index_length": (
                len(extra["markers_index"]) if "markers_index" in extra else None
            ),
            "expected_n_markers": (
                int(extra["encoded_training_data_ncols"] / extra["encoding_size"])
                if ("encoded_training_data_ncols" in extra and "encoding_size" in extra)
                else None
            ),
        }
        str_extra = {k: v for k, v in str_extra.items() if v is not None}
        message = self._MESSAGES[reason].format(**extra, **str_extra)
        super().__init__(message=message, extra={"reason": reason, **extra})


def _check_marker_index_size_compatibility(
    markers_index: pd.Index,
    n_encoded_cols: int,
    encoding_size: int,
):
    """Validate compatibility between encoded data dimensions and markers_index.

    Raise error if the number of columns of the encoded data, the encoding size
    and the length of the markers index doesnt match.

    It is expected that: ``len(markers_index) == n_encoded_cols / encoding_size``


    Parameters
    ----------
    markers_index :
        Index of the geno markers.
    n_encoded_cols :
        Total number of columns in the encoded genotype array.
    encoding_size :
        Number of columns used to encode a single marker.

    Raises
    ------
    CompressionModelConfigurationError
        If ``len(markers_index) != n_encoded_cols / encoding_size``.
    """
    # Note: could be slightly improve if encoding_size is not available
    # by checking n_encoded_cols is divisible by len(markers_index)
    expected_n_markers = n_encoded_cols / encoding_size
    if len(markers_index) != expected_n_markers:
        raise CompressionModelConfigurationError(
            reason=CompressionModelConfigurationError.ReasonCode.MARKER_INDEX_SIZE_MISMATCH,
            extra={
                "encoded_training_data_ncols": n_encoded_cols,
                "encoding_size": encoding_size,
                "markers_index": markers_index,
            },
        )


def _check_layer_size_and_encoded_data_size(n_cols, first_layer_size):
    """Validate compatibility between data dimensions and layer configuration.

    Emits warnings for configurations that may produce zero-padding.

    Parameters
    ----------
    n_cols :
        Number of columns in the encoded genotype array.
    first_layer_size :
        Size of the first layer (i.e. chunk size). Should ideally be a
        divisor of ``n_cols`` and a multiple of ``encoding_size``.

    Returns
    -------
        None

    Warns
    -----
    UserWarning
        If ``first_layer_size`` is not a divisor of ``n_cols`` — zero-padding
        will be required to fit the data into equal-sized chunks.
    """
    if n_cols % first_layer_size != 0:
        _deepcgp_warn(
            ColumnPaddingWarning(
                first_layer_size=first_layer_size, n_encoded_cols=n_cols
            )
        )


class ColumnPaddingWarning(DeepcgpWarning):
    """Issued when padding will be added to data.

    Due to missalignment between input layer size and number of encoded columns in data.

    Example
    -------
    With Input layer size of 4 and an encoded data with 6 columns.

    .. code-block:: text

        Allele A -> [0 1]
        Allele T -> [1 0]

        Raw sequence:     <start> |  A  |  A  |  T  | <end>
        Encoded sequence:         | 0 1 | 0 1 | 1 0 |
        expected chunks:          └───────────┴───────────┘
                                                      ▲ ▲ missing data
                                                      ▼ ▼ add zero padding
        actual chunks:            └ 0 1   0 1 ┴ 1 0   0 0 ┘

    """

    class _Extra(TypedDict, total=True):
        first_layer_size: int
        n_encoded_cols: int

    extra: "_Extra | dict[str, Any]"

    def __init__(self, first_layer_size, n_encoded_cols):
        super().__init__(
            message=(
                f"layer_sizes[0]={first_layer_size} is not a divisor of the number of "
                f"encoded columns ({n_encoded_cols}). Column padding (filled with 0) "
                "will be added to the end of the data to fit requested input layer "
                "size."
            ),
            extra={
                "first_layer_size": first_layer_size,
                "n_encoded_cols": n_encoded_cols,
            },
        )


def _check_layer_size_and_encoding_compatibility(first_layer_size, encoding_size):
    """Validate compatibility between data encoding size and layer configuration.

    Emits warnings for configurations where chunks will cut through encoded alleles
    or will consist of only 1 encoded allele.

    Parameters
    ----------
    first_layer_size :
        Size of the first layer (i.e. chunk size). Should ideally be a
        divisor of ``n_cols`` and a multiple of ``encoding_size``.
    encoding_size :
        Number of columns used to encode a single allele. If ``None``,
        encoding-alignment warnings are skipped.

    Returns
    -------
        None

    Warns
    -----
    UserWarning
        If ``encoding_size`` is provided and ``first_layer_size`` is not a
        multiple of it — chunk boundaries will cut across encoded alleles,
        leaving incomplete encodings at chunk edges.
    UserWarning
        If ``encoding_size`` is provided and equals ``first_layer_size`` —
        each chunk will consist of exactly one encoded allele.
    """
    if first_layer_size % encoding_size != 0:
        _deepcgp_warn(IncompleteEncodingChunkWarning(first_layer_size, encoding_size))
    if encoding_size >= first_layer_size:
        _deepcgp_warn(
            LessThanOneAlleleChunks(
                first_layer_size=first_layer_size, encoding_size=encoding_size
            )
        )


class IncompleteEncodingChunkWarning(DeepcgpWarning):
    """Issued when chuncks will consist of incomplete encoded alleles.

    Due to missalignment between input layer size encoding size.

    Example
    -------
    With encoding size of 2 and an input layer size of 5.


    .. code-block:: text

        Allele A -> [0 1]
        Allele T -> [1 0]

        Raw sequence:     |  A  |  A  |  T  |  A  |  A  | ...
        Encoded sequence: | 0 1 | 0 1 | 1 0 | 0 1 | 0 1 | ...
        chunks:           ├──────────────┼──────────────┼ ...
                               chunk 1       chunk 2

        Encoded allele T is splited between 2 chunks

    """

    class _Extra(TypedDict, total=True):
        first_layer_size: int
        encoding_size: int

    extra: "_Extra | dict[str, Any]"

    def __init__(self, first_layer_size, encoding_size):
        super().__init__(
            message=(
                f"layer_sizes[0]={first_layer_size} is not a multiple of "
                f"{encoding_size=}. (ie. each chunk will cut through encoded alleles, "
                "leaving incomplete encodings at chunk edges)"
            ),
            extra={
                "first_layer_size": first_layer_size,
                "encoding_size": encoding_size,
            },
        )


class LessThanOneAlleleChunks(DeepcgpWarning):
    """Issued when chuncks consist of only 1 alleles.

    Due to input layer size being smaler than the encoding size.


    Example
    -------
    With encoding size of 4 and an input layer size of 2:


    .. code-block:: text

        Allele A -> [0 0 0 1]
        Allele T -> [0 0 1 0]

        Raw sequence:     |    A    |    T    | ...
        Encoded sequence: | 0 0 0 1 | 0 0 1 0 | ...
        chunks:           ├────┼────┼────┼────┼ ...
                            c1   c2   c3   c4   ...

        Each chunks represent less than 1 allele.
    """

    class _Extra(TypedDict, total=True):
        first_layer_size: int
        encoding_size: int

    extra: "_Extra | dict[str, Any]"

    def __init__(self, first_layer_size, encoding_size):
        super().__init__(
            message=(
                f"layer_sizes[0]={first_layer_size} is lower or equal to "
                f"`encoding_size` ({encoding_size}) (ie. each chunk will consist of 1 "
                "or less encoded allele)."
            ),
            extra={
                "first_layer_size": first_layer_size,
                "encoding_size": encoding_size,
            },
        )


def _split_data(
    encoded_geno_array: NDArray[np.float32],
    chunk_size: int,
    encoding_size=None,
):
    """Split an encoded genotype array into list of equal-sized horizontal chunks.

    If the number of columns is not evenly divisible by ``chunk_size``, the
    array is zero-padded on the right before splitting.

    Parameters
    ----------
    encoded_geno_array :
        2-D array of encoded genotype data with shape ``(n_samples, n_cols)``.
    chunk_size :
        Number of columns per chunk. Determines how the array is divided along
        the column axis.
    encoding_size :
        Expected encoding dimensionality. Passed to
        ``_check_layer_sizes_and_data_compatibility`` for validation. If
        ``None``, the check is performed without an encoding size constraint.


    Returns
    -------
        List of 2-D arrays each with shape ``(n_samples, chunk_size)``.
        The final chunk may contain trailing zero-padding columns if the
        original column count was not divisible by ``chunk_size``.

    Warns
    -----
    UserWarning
        If ``chunk_size`` is not a divisor of ``n_cols`` — zero-padding will
        be applied to the right of the array.
    UserWarning
        If ``encoding_size`` is provided and ``chunk_size`` is not a multiple
        of it — chunks will cut across encoded allele boundaries, leaving
        incomplete encodings at chunk edges.
    UserWarning
        If ``encoding_size`` is provided and equals ``chunk_size`` — each
        chunk will contain exactly one encoded allele.
    """
    n_cols: int = encoded_geno_array.shape[1]

    # TODO: may be remove the check. Could be redundant with
    # instance initialisation.
    _check_layer_size_and_encoded_data_size(n_cols=n_cols, first_layer_size=chunk_size)
    if encoding_size is not None:
        _check_layer_size_and_encoding_compatibility(
            first_layer_size=chunk_size, encoding_size=encoding_size
        )

    remainder = n_cols % chunk_size
    if remainder != 0:
        padding_width = chunk_size - remainder
        encoded_geno_array = np.pad(
            encoded_geno_array,
            pad_width=((0, 0), (0, padding_width)),
            mode="constant",
            constant_values=0,
        )

    n_chunks = encoded_geno_array.shape[1] // chunk_size
    return np.hsplit(encoded_geno_array, n_chunks)


class CompressionModel:
    """Utility class for genomic data compression based on autoencoders.

    Splits genotype data into fixed-size chunks of alleles and trains one
    :class:`AutoencoderModels` instance per chunk. After fitting, can pass
    genotype data through the encoders to produce a compressed representation
    of these data.

    Parameters
    ----------
    training_encoded_geno_array :
        Encoded genotype array used for training. Can be set after initialisation.
    encoding_map :
        Mapping from allele values to their encoding vectors. See
        :attr:`CompressionModel.encoding_map`
    layer_sizes :
        Dimensions of each autoencoder layer. See :attr:`CompressionModel.layer_sizes`.
    batch_size :
        Number of samples per batch during training and compression.
        See :attr:`CompressionModel.batch_size`.
    epochs :
        Maximum number of training epochs per autoencoder. See
        :attr:`CompressionModel.epochs`.
    fitting_callbacks :
        Keras callbacks applied during training. See
        :attr:`CompressionModel.fitting_callbacks`.
    learning_rate :
        Learning rate for the Adam optimiser. See
        :attr:`CompressionModel.learning_rate`.
    loss :
        Loss function used to compile each autoencoder. See
        :attr:`CompressionModel.loss`.
    seed :
        Random seed for the train/validation/evaluation split. See
        :attr:`CompressionModel.seed`.
    training_size :
        Proportion or absolute number of samples used for training.
        See :attr:`CompressionModel.training_size`.
    validation_size :
        Proportion or absolute number of the remaining samples used for
        validation; the rest form the evaluation set. See
        :attr:`CompressionModel.validation_size`.

    Raises
    ------
    LayerSizesConfigurationError
    InvalidEncodingMapError
    CompressionModelConfigurationError
    """

    # TODO:
    #
    # simpler alternative for layer_sizes:
    #    chunk_size, desired chunk size (first value of layers_sizes)
    #    compression_ratio, desired compression ratio (last value of layers_sizes)
    #    n_encoding_layers
    #    ??? -- need a way to calculate the encoders layers sizes (linear decrease?)
    #
    # add optimizer as parameter ?

    @property
    def training_encoded_geno_array(self) -> NDArray[np.float32] | None:
        """Encoded genotype array used for training."""
        return self._training_encoded_geno_array

    @training_encoded_geno_array.setter
    def training_encoded_geno_array(
        self, training_encoded_geno_array: NDArray[np.float32] | None
    ):
        if training_encoded_geno_array is None:
            self._training_encoded_geno_array = None
            return

        if self.layer_sizes:
            _check_layer_size_and_encoded_data_size(
                n_cols=training_encoded_geno_array.shape[1],
                first_layer_size=self.layer_sizes[0],
            )

        if self.training_markers_index is not None and self.encoding_size is not None:
            _check_marker_index_size_compatibility(
                self.training_markers_index,
                training_encoded_geno_array.shape[1],
                self.encoding_size,
            )

        self._training_encoded_geno_array = training_encoded_geno_array

    @property
    def _n_col_train(self):
        """Number of columns in the encoded training data."""
        if self.training_encoded_geno_array is None:
            return 0
        return self.training_encoded_geno_array.shape[1]

    @property
    def training_markers_index(self) -> pd.Index | None:
        """List of training markers IDs.

        Used to verify compatibility with training data before compression.
        """
        return self._training_markers_index

    @training_markers_index.setter
    def training_markers_index(self, training_markers_index: ArrayLike | None):
        if training_markers_index is None:
            self._training_markers_index = None
            return
        if self.training_encoded_geno_array is None or self.encoding_size is None:
            self._training_markers_index = pd.Index(training_markers_index)
            return

        markers_index = pd.Index(training_markers_index)

        _check_marker_index_size_compatibility(
            markers_index, self._n_col_train, self.encoding_size
        )

        self._training_markers_index = markers_index

    @property
    def encoding_map(self) -> Mapping[Any, list[float]] | None:
        """Mapping from allele values to their encoding vectors.

        Used to determine the :attr:`CompressionModel.encoding_size` for some internal
        validation.

        See Also
        --------
        :func:`build_one_hot_encoding_map` : Build a one-hot encoding map from a
            genotype array.
        :attr:`CompressionModel.encoding_size` : Length of the encoding vectors,
            derived from this map.
        """
        return self._encoding_map

    @encoding_map.setter
    def encoding_map(self, encoding_map: Mapping[Any, list[float]] | None):
        if encoding_map is None:
            self._encoding_map = None
            return

        _validate_encoding_map(encoding_map)

        enc_size = encoding_size(encoding_map)

        if self.training_encoded_geno_array is not None:
            if self.layer_sizes:
                _check_layer_size_and_encoding_compatibility(
                    first_layer_size=self.layer_sizes[0],
                    encoding_size=enc_size,
                )
            if self.training_markers_index is not None:
                _check_marker_index_size_compatibility(
                    self.training_markers_index,
                    self._n_col_train,
                    enc_size,
                )

        self._encoding_map = encoding_map

    @property
    def encoding_size(self) -> int | None:
        """Length of the alleles encoding vectors.

        Used to validate chunk alignment if ``chunk_size`` is not a multiple of
        ``encoding_size``, chunk boundaries will cut across encoded alleles.

        See Also
        --------
        :attr:`CompressionModel.encoding_map` : The allele-to-vector mapping this
            is derived from.
        :attr:`CompressionModel.chunk_size` : The chunk size validated against
            this value.
        """
        if self.encoding_map is None:
            return None
        return encoding_size(self.encoding_map)

    @property
    def _train_data_chunks(self) -> list[NDArray[np.float32]]:
        """List of training data chunks."""
        if self.training_encoded_geno_array is None:
            return []
        if not self.layer_sizes:
            return []
        return _split_data(
            encoded_geno_array=self.training_encoded_geno_array,
            chunk_size=self.chunk_size,
            encoding_size=self.encoding_size,
        )

    @property
    def n_chunks(self) -> int:
        """Number of chunks the data is split into.

        Each chunk is fitted with its own :class:`AutoencoderModels` instance,
        so this also corresponds to the number of autoencoders trained during
        :meth:`~CompressionModel.fit`. Returns ``0`` if no training data or
        layer sizes are set.

        See Also
        --------
        :attr:`CompressionModel.chunk_size` : The number of columns of each chunk.
        """
        return len(self._train_data_chunks)

    @property
    def chunk_size(self):
        """Number of columns in each data chunk.

        Corresponds to ``layer_sizes[0]``, which is the input dimension of each
        autoencoder. The data is split into chunks of this size before
        fitting or compression. Returns ``0`` if ``layer_sizes`` is not set.

        See Also
        --------
        :attr:`CompressionModel.layer_sizes` : Autoencoder layer dimensions.
        :attr:`CompressionModel.n_chunks` : Number of chunks the data is split into.
        """
        return self.layer_sizes[0] if self.layer_sizes else 0

    batch_size: int
    """Number of samples per batch during training and compression.

    Passed to Keras's model ``fit``, ``evaluate``, and ``predict`` methods. Can be
    overridden locally in :meth:`~CompressionModel.compress` via its
    ``batch_size`` parameter.

    See Also
    --------
    `keras.Model.fit <https://keras.io/api/models/model_training_apis/#fit-method>`_
    :meth:`CompressionModel.fit`
    :meth:`CompressionModel.compress`
    """
    epochs: int
    """Number of epochs to train the model.

    Passed to Keras's model ``fit`` methods.

    See Also
    --------
    `keras.Model.fit <https://keras.io/api/models/model_training_apis/#fit-method>`_
    :meth:`CompressionModel.fit`
    """
    fitting_callbacks: list
    """List of callbacks to apply during training.

    Passed to Keras's model ``fit`` methods.

    See Also
    --------
    `keras.Model.fit <https://keras.io/api/models/model_training_apis/#fit-method>`_
    `keras.callbacks <https://keras.io/api/callbacks/>`_
    :meth:`CompressionModel.fit`
    """
    learning_rate: float
    """Learning rate for the Adam optimizer used to compile each autoencoder.

    See Also
    --------
    `keras.optimizers.Adam <https://keras.io/api/optimizers/adam/>`_
    :meth:`CompressionModel.fit`
    """
    loss: str | Loss = "mse"
    """Loss function used to compile each autoencoder.

    Accepts either a string identifier (e.g. ``"mse"``) or a
    :class:`keras.losses.Loss` instance. Defaults to ``"mse"``.

    See Also
    --------
    `keras.losses <https://keras.io/api/losses/>`_
    :meth:`CompressionModel.fit`
    """
    seed: int | None
    """Random seed used for the train/test split during fitting.

    Passed as ``random_state`` to :func:`sklearn.model_selection.train_test_split`
    for each chunk. If ``None`` at fit time, a random seed is generated and stored
    so that the same split is used across all chunks.

    Important
    ---------
    The seed only controls the data split, not the model weight initialisation or
    the training process. Same seed can produce models with different weights.

    See Also
    --------
    :meth:`CompressionModel.fit`
    """
    training_size: int | float
    """Proportion or absolute number of samples used for training each autoencoder.

    Passed as ``train_size`` to :func:`sklearn.model_selection.train_test_split`.
    The remaining samples are further split into validation and evaluation sets
    according to :attr:`CompressionModel.validation_size`.

    Important
    ---------
    The type matters: a :class:`float` is interpreted as a proportion of the
    dataset, while an :class:`int` is interpreted as an absolute sample count.
    See :func:`sklearn.model_selection.train_test_split` for details.

    See Also
    --------
    :attr:`CompressionModel.validation_size` : Controls the split of the
        remaining data into validation and evaluation sets.
    :meth:`CompressionModel.fit` : Where the split is performed.
    """
    validation_size: int | float
    """Proportion or absolute number of the non-training samples used for validation.

    After the training split (controlled by :attr:`CompressionModel.training_size`), the
    remaining samples are further split into a validation set and an evaluation set with
    a second call to :func:`sklearn.model_selection.train_test_split` on the remaining
    data. The validation set is passed to Keras's ``fit`` as ``validation_data``,
    while the evaluation set is passed to ``evaluate``.

    If set to ``1.0``, all remaining samples go to validation and no evaluation
    is performed (i.e. :attr:`CompressionModel.autoencoder_evaluations` entries will be
    ``None``).

    Important
    ---------
    The type matters: a :class:`float` is interpreted as a proportion of the
    remaining (non-training) data, while an :class:`int` is interpreted as an
    absolute sample count. See :func:`sklearn.model_selection.train_test_split`
    for details.

    See Also
    --------
    :attr:`CompressionModel.CompressionModel.training_size` : Controls the initial
    train split.
    """
    autoencoder_models: list[AutoencoderModels]
    """List of autoencoder populated during :meth:`~CompressionModel.fit`.

    Each :class:`AutoencoderModels` instance is built from
    :attr:`CompressionModel.layer_sizes` and trained on the corresponding chunk of
    the training data. Empty list before fitting.

    See Also
    --------
    :attr:`CompressionModel.n_chunks` : Number of autoencoders to be trained.
    :meth:`CompressionModel.fit` : Where the models are instantiated and trained.
    """
    autoencoder_evaluations: list
    """List of evaluation results for each autoencoder.

    Each entry is a dict mapping the loss name to its evaluation score (e.g.
    ``{"mse": 0.042}``), computed on the evaluation set after training. Entries
    are ``None`` for each chunks if :attr:`CompressionModel.validation_size` is ``1.0``.
    Empty list before fitting.

    See Also
    --------
    :attr:`CompressionModel.validation_size` : Controls whether an evaluation set
        is available.
    :attr:`CompressionModel.loss` : The loss function used for evaluation.
    :meth:`CompressionModel.fit` : Where evaluation is performed.
    """
    autoencoder_fits: list
    """Training histories for each autoencoder.

    Each entry is a :class:`keras.callbacks.History` object returned by
    ``autoencoder.fit``, containing per-epoch training and validation loss.
    Empty list before fitting.

    See Also
    --------
    `keras.Model.fit <https://keras.io/api/models/model_training_apis/>`_
    :meth:`CompressionModel.fit` : Where the histories are collected.
    """

    @property
    def layer_sizes(self) -> list[int]:
        """Layer dimensions (shared across all autoencoders).

        The first element defines the input/output dimension of each autoencoder
        and determines how the data is chunked (see
        :attr:`CompressionModel.chunk_size`). The last element is the latent dimension.
        Elements in between define the encoding layers; the decoder mirrors them in
        reverse.

        For example, ``[28, 14, 7, 3]`` produces:

        .. code-block:: none

            input(28) → encoding(14, relu) → encoding(7, relu) → latent(3, sigmoid)
            → decoding(7, relu) → decoding(14, relu) → output(28, sigmoid)

        Must have at least 2 elements and contain only integers. Setting this
        attribute triggers compatibility validation against the training data
        if it is already set.

        Raises
        ------
        LayerSizesConfigurationError
            If ``layer_sizes`` is not valid:
              - a list or tuple.
              - has fewer than 2 elements or contains non-integer values.

        Warns
        -----
        UserWarning
            If ``layer_sizes[0]`` is not a divisor of ``n_cols`` — zero-padding
            will be required to fit the data into equal-sized chunks.
        UserWarning
            If ``encoding_size`` is provided and ``layer_sizes[0]`` is not a
            multiple of it — chunk boundaries will cut across encoded alleles,
            leaving incomplete encodings at chunk edges.
        UserWarning
            If ``encoding_size`` is provided and equals ``layer_sizes[0]`` —
            each chunk will consist of exactly one encoded allele.

        See Also
        --------
        :attr:`CompressionModel.chunk_size` : Derived from ``layer_sizes[0]``.
        :class:`AutoencoderModels` : Consumes ``layer_sizes`` to build each autoencoder.
        :func:`utils.possible_first_layer_sizes` : Utility function to find
            first layer perferctly compatible with the given encoded data structure
        """
        return self._layer_sizes

    @layer_sizes.setter
    def layer_sizes(self, layers_sizes: list[int] | tuple[int, ...]):
        _check_layer_sizes(layers_sizes)
        if self.training_encoded_geno_array is not None:
            _check_layer_size_and_encoded_data_size(
                n_cols=self._n_col_train, first_layer_size=layers_sizes[0]
            )
        if self.encoding_map is not None:
            _check_layer_size_and_encoding_compatibility(
                first_layer_size=layers_sizes[0], encoding_size=self.encoding_size
            )

        self._layer_sizes = list(layers_sizes)

    @property
    def is_fitted(self) -> bool:
        """Whether all autoencoders have been fitted."""
        return (
            all(m.is_fitted for m in self.autoencoder_models)
            if self.autoencoder_models and len(self.autoencoder_models) == self.n_chunks
            else False
        )

    def __init__(
        self,
        # -- data --
        training_encoded_geno_array: NDArray[np.float32] | None = None,
        encoding_map: Mapping[Any, list[float]] | None = None,
        training_markers_index: ArrayLike | None = None,
        # -- autoencoders --
        layer_sizes: list[int] | tuple[int, ...] | None = None,
        # -- fitting --
        batch_size: int = 50,
        epochs: int = 200,
        fitting_callbacks: list | None = None,
        learning_rate: float = 0.001,
        loss: str | Loss = "mse",
        seed: int | None = None,
        training_size: int | float = 0.4,
        validation_size: int | float = 0.5,
    ):
        """Initialise compression model."""
        self._training_encoded_geno_array = None
        self._layer_sizes = []
        self._training_markers_index = None
        self._encoding_map = None

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.seed = seed
        self.training_size = training_size
        self.validation_size = validation_size

        self.autoencoder_models = []
        self.autoencoder_evaluations = []
        self.autoencoder_fits = []

        if layer_sizes is not None:
            self.layer_sizes = layer_sizes

        self.fitting_callbacks = (
            fitting_callbacks
            if fitting_callbacks is not None
            else [
                EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            ]
        )

        self.training_encoded_geno_array = training_encoded_geno_array
        self.training_markers_index = training_markers_index
        self.encoding_map = encoding_map

    @classmethod
    def from_dataframe(
        cls,
        training_dataframe: pd.DataFrame,
        encoding_map: Mapping[Any, list[float]] | None = None,
        missing_values: Collection = {"N"},
        **kwargs,
    ) -> "CompressionModel":
        """Initialise a CompressionModel from a genotype DataFrame.

        Encodes the DataFrame into a genotype array and constructs a
        CompressionModel with the encoded data, encoding map, and marker index
        derived from the DataFrame's columns.

        Parameters
        ----------
        training_dataframe :
            Raw genotype DataFrame where rows are samples and columns are markers.
            Must not be empty.
        encoding_map :
            Mapping from allele values to their encoding vectors. If ``None``,
            a one-hot encoding map is built automatically from the unique alleles
            found in ``training_dataframe``, excluding ``missing_values``.
            See :func:`build_one_hot_encoding_map`.
        missing_values :
            Collection of values representing missing genotype calls. These are
            encoded as zero vectors rather than assigned a one-hot encoding.
            Defaults to ``{"N"}``.
        **kwargs :
            Additional keyword arguments passed to :class:`CompressionModel`.
            ``training_encoded_geno_array`` cannot be passed, as it is derived
            from ``training_dataframe``.

        Returns
        -------
        CompressionModel
            A new instance initialised with the encoded training data, encoding
            map, and marker index from ``training_dataframe``.

        Raises
        ------
        CompressionModelConfigurationError
            If ``training_dataframe`` is empty.
        TypeError
            If ``training_encoded_geno_array`` is passed as a keyword argument.
        InvalidEncodingMapError
            If the ``encoding_map`` is invalid (see :func:`encode_snp_array`)
        LayerSizesConfigurationError
            If the layer sizes are invalid

        See Also
        --------
        :func:`build_one_hot_encoding_map` : Builds the default encoding map.
        :func:`encode_snp_array` : Encodes the raw genotype array.
        """
        if "training_encoded_geno_array" in kwargs:
            raise TypeError(
                "`training_encoded_geno_array` cannot be passed as a keyword argument "
                "it is derived from `training_dataframe`."
            )
        if training_dataframe.empty:
            raise CompressionModelConfigurationError(
                reason=CompressionModelConfigurationError.ReasonCode.EMPTY_DATAFRAME
            )

        geno_array = training_dataframe.to_numpy()

        if encoding_map is None:
            # even if encode_snp_array can fallback to build_one_hot_encoding_map
            # we need to keep the actual `encoding_map` in the instance.
            encoding_map = build_one_hot_encoding_map(geno_array, missing_values)

        training_encoded_geno_array = encode_snp_array(
            geno_array,
            missing_values,
            encoding_map,
        )

        return cls(
            training_encoded_geno_array=training_encoded_geno_array,
            encoding_map=encoding_map,
            training_markers_index=training_dataframe.columns,
            **kwargs,
        )

    def fit(
        self,
        seed: int | None = None,
    ):
        """Fit the autoencoders (one per data chunk).

        Splits the training data into chunks of size
        :attr:`CompressionModel.chunk_size`, builds one :class:`AutoencoderModels` per
        chunk, and trains each autoencoder on its corresponding chunk.
        Training and validation splits are drawn from the same rows across all chunks.

        Populates :attr:`CompressionModel.autoencoder_models`,
        :attr:`CompressionModel.autoencoder_fits`, and
        :attr:`CompressionModel.autoencoder_evaluations`.

        Parameters
        ----------
        seed :
            Random seed for the train/test split. Overrides
            :attr:`CompressionModel.seed` if provided. If neither is set, a random seed
            is generated and stored in :attr:`CompressionModel.seed`. Only affects data
            splitting, not weight initialisation.

        Important
        ---------
        The seed only controls the data split, not the model weight initialisation or
        the training process. Same seed can produce models with different weights.

        Raises
        ------
        ModelStateError
            - If :attr:`CompressionModel.training_encoded_geno_array` is ``None``.
            - If :attr:`CompressionModel.layer_sizes` is not set.
        """
        if self.training_encoded_geno_array is None:
            raise ModelStateError(reason=ModelStateError.ReasonCode.NO_TRAINING_DATA)

        if not self.layer_sizes:
            raise ModelStateError(reason=ModelStateError.ReasonCode.NO_LAYER_SIZES)

        self.autoencoder_models = [
            AutoencoderModels(self.layer_sizes) for _ in range(self.n_chunks)
        ]

        if seed is not None:
            self.seed = seed
        if self.seed is None:
            self.seed = random.randint(0, 2**32 - 1)  # noqa: S311 no crypto purpose

        self.autoencoder_fits = []
        self.autoencoder_evaluations = []

        for i, (chunk, aem) in enumerate(
            zip(self._train_data_chunks, self.autoencoder_models, strict=True)
        ):
            logger.info("Fitting autoencoders: chunk %d / %d", i + 1, self.n_chunks)

            x_train, x_mid = train_test_split(
                chunk, train_size=self.training_size, random_state=self.seed
            )
            if isinstance(self.validation_size, float) and self.validation_size == 1.0:
                x_evaluation, x_valid = ([], x_mid)
            else:
                x_evaluation, x_valid = train_test_split(
                    x_mid, test_size=self.validation_size, random_state=self.seed
                )

            aem.autoencoder.compile(
                optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss
            )

            self.autoencoder_fits.append(
                aem.autoencoder.fit(
                    x_train,
                    x_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_data=(x_valid, x_valid),
                    callbacks=self.fitting_callbacks,
                    verbose=0,
                )
            )

            if len(x_evaluation) == 0:
                self.autoencoder_evaluations.append(None)
            else:
                evaluation = aem.autoencoder.evaluate(
                    x_evaluation,
                    x_evaluation,
                    batch_size=self.batch_size,
                    verbose=0,
                )
                self.autoencoder_evaluations.append({self.loss: evaluation})

            aem.is_fitted = True

    def compress(
        self,
        encoded_geno_array: NDArray[np.float32],
        batch_size: int | None = None,
    ) -> NDArray[np.float32]:
        """Compress genotype data using the fitted autoencoders.

        Splits the input array into chunks matching the training chunk size,
        runs each chunk through its corresponding encoder, and returns the
        horizontally stacked compressed representations.

        Parameters
        ----------
        encoded_geno_array :
            Encoded genotype array to compress. Must have the same number of
            columns as the training data.
        batch_size :
            Batch size for encoder prediction. Defaults to ``self.batch_size``
            if not provided.

        Returns
        -------
            Array of shape ``(n_samples, n_chunks * latent_dim)``,
            where ``latent_dim`` is ``self.layer_sizes[-1]`` representing
            the compressed genomic data.

        Raises
        ------
        ModelStateError
            If the model has not been fitted yet (i.e.
            :attr:`CompressionModel.is_fitted` is ``False``).
        IncompatibleDataError
            If ``encoded_geno_array`` has a different number of columns than
            the training data.
        """
        if not self.is_fitted:
            raise ModelStateError(reason=ModelStateError.ReasonCode.MODEL_NOT_FITTED)

        if encoded_geno_array.shape[1] != self._n_col_train:
            raise IncompatibleDataError(
                reason=IncompatibleDataError.ReasonCode.INVALID_NUMBER_OF_COLUMNS,
                extra={
                    "encoded_provided_data_ncols": encoded_geno_array.shape[1],
                    "encoded_training_data_ncols": self._n_col_train,
                },
            )

        data_splits = _split_data(
            encoded_geno_array=encoded_geno_array,
            chunk_size=self.chunk_size,
            encoding_size=self.encoding_size,
        )

        if not batch_size:
            batch_size = self.batch_size

        encoded_chunks = []
        for i, (chunk, aem) in enumerate(
            zip(data_splits, self.autoencoder_models, strict=True)
        ):
            logger.info("Predict chunk %d / %d", i + 1, self.n_chunks)
            encoded = aem.encoder.predict(chunk, batch_size=batch_size, verbose=0)
            encoded_chunks.append(encoded)

        return np.hstack(encoded_chunks)

    def compress_dataframe(
        self,
        geno_dataframe: pd.DataFrame,
        batch_size: int | None = None,
        encoding_map: Mapping[Any, list[float]] | None = None,
        missing_values: Collection = {"N"},
    ) -> NDArray[np.float32]:
        """Compress genotype data using the fitted autoencoders.

        Encode the input data.frame to an array (using
        :attr:`CompressionModel.encoding_map` or ``encoding_map`` and
        ``missing_values``), splits the array into chunks matching the training chunk
        size, runs each chunk through its corresponding encoder, and returns the
        horizontally stacked compressed representations.

        Parameters
        ----------
        geno_dataframe :
            Raw genotype data.frame to compress.
        batch_size :
            (Optional) Batch size for encoder prediction. Defaults to
            ``self.batch_size`` if not provided.
        encoding_map :
            (Optional) The encoding map to use for the encoding.
            Defaults to ``None``, in which case the encoding will be done with:

            1. :attr:`CompressionModel.encoding_map` if it is set.
            2. Or, :func:`encode_snp_array`'s default behaviour when not provided
               an encoding_map

        missing_values :
            (Optional), Only effective if ``encoding_map`` is ``None``. Any object
            supporting the ``in`` operator (e.g. set, list, tuple).
            Collection of value of ``geno_dataframe`` representing
            missing genotype calls that should not be assigned a default one hot
            encoding vector. They will be encoded as a zero vector.
            Defaults to ``{"N"}``.

        Returns
        -------
            Array of shape ``(n_samples, n_chunks * latent_dim)``,
            where ``latent_dim`` is ``self.layer_sizes[-1]`` representing
            the compressed genomic data.

        Raises
        ------
        ModelStateError
            If the model has not been fitted yet (i.e.
            :attr:`CompressionModel.is_fitted` is ``False``).
        IncompatibleDataError
            - If the set of ``geno_dataframe``'s columns index is different from
              the one from the training data (i.e.
              :attr:`CompressionModel.training_markers_index`)
            If ``geno_dataframe`` has a different number of markers than
            the training data.
        InvalidEncodingMapError
            If the ``encoding_map`` is invalid (see :func:`encode_snp_array`)
        """
        if self.training_markers_index is not None:
            if set(self.training_markers_index) != set(geno_dataframe.columns):
                raise IncompatibleDataError(
                    reason=IncompatibleDataError.ReasonCode.INVALID_COLUMN_INDEX,
                    extra={
                        "provided_data_index": geno_dataframe.columns,
                        "training_data_index": self.training_markers_index,
                    },
                )
            geno_dataframe = geno_dataframe.loc[:, self.training_markers_index]

        geno_array = geno_dataframe.to_numpy()

        encoded_geno_array = encode_snp_array(
            geno_array,
            missing_values=missing_values,
            encoding_map=encoding_map or self.encoding_map,
        )
        return self.compress(encoded_geno_array, batch_size=batch_size)


class LayerSizeOption(NamedTuple):
    """Utility type representing a possible first-layer size and its chunk count.

    The first value is the first layer size, and the second value is the number
    of chunks resulting from that layer size.
    """

    first_layer_size: int
    n_chunks: int


def possible_first_layer_sizes(
    n_encoded_colums: int,
    encoding_size: int,
    desired_n_chunks: int | None = None,
) -> list[LayerSizeOption]:
    """Find first layer sizes perferctly compatible with the given data structure.

    When instantiating a :class:`deepcgp.CompressionModel` both
    :attr:`deepcgp.CompressionModel.layer_sizes` and
    :attr:`deepcgp.CompressionModel.training_encoded_geno_array` are checked to be sure
    the first layer size divide the number of training data's columns and is a multiple
    of the encoding size, so that the data can be split into equally sized chunks
    without padding or splitting encoded alleles across chunk boundaries. This functions
    help find first layer sizes matching those properties for a given the data
    structure.

    Parameters
    ----------
    n_encoded_colums :
        Total number of columns in the encoded genotype array. Candidate first-layer
        sizes must evenly divide this value to avoid
        :class:`deepcgp.warnings.ColumnPaddingWarning`
    encoding_size :
        Number of columns used to encode a single allele. Candidate first-layer
        sizes must be a multiple of this value to avoid
        :class:`deepcgp.warnings.IncompleteEncodingChunkWarning`
    desired_n_chunks :
        If provided, narrow the returned options down to the one(s) whose resulting
        chunk count is closest to this value. If ``None``, all valid options are
        returned.

    Returns
    -------
        List of :class:`LayerSizeOption`, sorted by ascending ``first_layer_size``
        (equivalently, descending ``n_chunks``). Empty if no valid first-layer size
        exists. If ``desired_n_chunks`` is provided and at least one valid option
        exists, contains either a single option (exact match or closest match) or two
        options (the two options closest to ``desired_n_chunks``, the one with a greater
        ``n_chunks`` and the one with a lower ``n_chunks`` than the desired).

    Warns
    -----
    deepcgp.warnings.DeepcgpWarning
        If ``n_encoded_colums`` or ``encoding_size`` are not positive. No valid
        first-layer size can exist in that case, and an empty list is returned.

    See Also
    --------
    :class:`deepcgp.warnings.ColumnPaddingWarning`
    :class:`deepcgp.warnings.IncompleteEncodingChunkWarning`

    Examples
    --------
    .. jupyter-kernel::
       :id: possible_first_layer_sizes-example

    .. jupyter-execute::

        from pprint import pprint
        from deepcgp.utils import possible_first_layer_sizes

        pprint(
            possible_first_layer_sizes(n_encoded_colums=60, encoding_size=5)
        )

    .. jupyter-execute::

        # desired_n_chunks non-exact match
        pprint(possible_first_layer_sizes(
            n_encoded_colums=60,
            encoding_size=5,
            desired_n_chunks=5
        ))

    .. jupyter-execute::

        # desired_n_chunks exact match
        pprint(possible_first_layer_sizes(
            n_encoded_colums=60,
            encoding_size=5,
            desired_n_chunks=6
        ))
    """
    # TODO: refact, doc, should this function returns the number of chunks too ????

    # Return all numbers `possible_first_layer_size` such that:
    #   - possible_first_layer_size is a multiple of enc_size
    #   - possible_first_layer_size is a divisor of n_enc_cols

    # Args:
    #     n_enc_cols (int): the number that possible_first_layer_size must divide
    #     enc_size (int): the number that possible_first_layer_size must be a multiple of

    # Returns:
    #     list[int]: all valid possible_first_layer_size values, sorted ascending

    options: list[LayerSizeOption] = []

    if n_encoded_colums <= 0 or encoding_size <= 0:
        # invalid cases
        # especially encoding_size must be positive for the while loop below
        # to end, and != 0 for the modulo operation
        _deepcgp_warn(
            "possible_first_layer_sizes() called with invalid inputs, "
            f"n_encoded_colums <= 0 ({n_encoded_colums}) or "
            f"encoding_size <= 0 ({encoding_size}). "
            "No valid layer sizes exist, returning an empty list."
        )
        return []

    # possible_first_layer_size must be a multiple of enc_size,
    # so we only need to check multiples of enc_size up to n_enc_cols
    candidate = encoding_size
    while candidate <= n_encoded_colums:
        if n_encoded_colums % candidate == 0:
            options.append(LayerSizeOption(candidate, n_encoded_colums // candidate))
        candidate += encoding_size

    if len(options) == 0 or desired_n_chunks is None:
        return options

    # by construction `options` **is sorted**
    #  - the first elements have the lowest layer size and the hightest n_chunks
    #  - the last elements have the highest layer size and the lowest n_chunks
    # which make the following possible

    if options[0].n_chunks <= desired_n_chunks:
        # the largest achievable n_chunks doesn't reach desired_n_chunks
        # or match exactly
        return [options[0]]

    if options[-1].n_chunks >= desired_n_chunks:
        # the lowest achievable n_chunks doesn't reach desired_n_chunks
        # or match exactly
        # note: here desired_n_chunks would be <=1
        return [options[-1]]

    for opt in options:
        if opt.n_chunks == desired_n_chunks:
            return [opt]
        if opt.n_chunks > desired_n_chunks:
            opt_above = opt
        else:
            opt_below = opt
            break
    return [opt_above, opt_below]
