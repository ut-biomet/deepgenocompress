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
import warnings
from collections.abc import Collection, Mapping
from typing import Any

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.losses import Loss
from keras.optimizers import Adam
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split

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


def _check_layer_sizes(layer_sizes):
    """Check layers_sizes's type and value.

    Warns
    -----
    UserWarning
        If latent layer size (i.e. ``layer_sizes[-1]``) is larger or equal to
        input layer size (i.e. ``layer_sizes[0]``).

    Raises
    ------
    TypeError
        If ``layer_sizes`` is not a list or tuple.
    ValueError
        If ``layer_sizes`` has a length lower than 2.
    ValueError
        If ``layer_sizes``'s values are not integers.

    """
    if not isinstance(layer_sizes, (list, tuple)):
        raise TypeError(
            "layer_sizes must be a list, or tuple of int got " f"`{type(layer_sizes)}`"
        )
    if len(layer_sizes) < 2:
        raise ValueError(
            "layer_sizes length must be greater than 2, got " f"`{len(layer_sizes)=}`"
        )
    for i, ls in enumerate(layer_sizes):
        if not isinstance(ls, int):
            raise ValueError(
                "layer_sizes must be a list, or tuple of int got "
                f"`{type(layer_sizes[i])}` for index layers_sizes[{i}]."
            )
    if layer_sizes[-1] >= layer_sizes[0]:
        warnings.warn(
            f"Latent layer size ({layer_sizes[-1]}) is equal or larger than "
            f"the input layer size ({layer_sizes[0]}). This will expand rather "
            "than compress the data.",
            UserWarning,
            stacklevel=2,
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
    TypeError
        If ``layer_sizes`` is not a list or tuple.
    ValueError
        If ``layer_sizes`` has a length lower than 2.
    ValueError
        If ``layer_sizes``'s values are not integers.

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
    """The encoder keras model. (Shares same layers with the autoencoder)"""

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
    ValueError
        If ``len(markers_index) != n_encoded_cols / encoding_size``.

    """
    # Note: could be slightly improve if encoding_size is not available
    # by checking n_encoded_cols is divisible by len(markers_index)
    expected_n_markers = n_encoded_cols / encoding_size
    if len(markers_index) != expected_n_markers:
        raise ValueError(
            "Marker index length missmatch training data size: "
            f"len(training_markers_index) = {len(markers_index)}, "
            f"encoded geno array have {n_encoded_cols} columns with an "
            f"encoding size of {encoding_size}. "
            f"({len(markers_index)} != {n_encoded_cols} / {encoding_size} "
            f"= {expected_n_markers})."
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
        warnings.warn(
            f"layer_sizes[0]={first_layer_size} is not a divisor of {n_cols=}. "
            "Column padding (filled with 0) will be added to the end of the data "
            "to fit requested layer_sizes[0]",
            UserWarning,
            stacklevel=2,
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
        warnings.warn(
            f"layer_sizes[0]={first_layer_size} is not a multiple of "
            f"{encoding_size=}. (ie. each chunk will cut through encoded alleles, "
            "leaving incomplete encodings at chunk edges)",
            UserWarning,
            stacklevel=2,
        )
    if encoding_size == first_layer_size:
        warnings.warn(
            f"layer_sizes[0]={first_layer_size} is equal to `encoding_size` "
            "(ie. each chunk will consist of only 1 encoded allele).",
            UserWarning,
            stacklevel=2,
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
        Mapping from allele values to their encoding vectors. See :attr:`encoding_map`
    layer_sizes :
        Dimensions of each autoencoder layer. See :attr:`layer_sizes`.
    batch_size :
        Number of samples per batch during training and compression.
        See :attr:`batch_size`.
    epochs :
        Maximum number of training epochs per autoencoder. See :attr:`epochs`.
    fitting_callbacks :
        Keras callbacks applied during training. See :attr:`fitting_callbacks`.
    learning_rate :
        Learning rate for the Adam optimiser. See :attr:`learning_rate`.
    loss :
        Loss function used to compile each autoencoder. See :attr:`loss`.
    seed :
        Random seed for the train/validation/evaluation split. See :attr:`seed`.
    training_size :
        Proportion or absolute number of samples used for training.
        See :attr:`training_size`.
    validation_size :
        Proportion or absolute number of the remaining samples used for
        validation; the rest form the evaluation set. See :attr:`validation_size`.

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

        Used to determine the :attr:`encoding_size` for some internal validation.

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

        encoding_size = len(next(iter(encoding_map.values())))

        if self.training_encoded_geno_array is not None:
            if self.layer_sizes:
                _check_layer_size_and_encoding_compatibility(
                    first_layer_size=self.layer_sizes[0],
                    encoding_size=encoding_size,
                )
            if self.training_markers_index is not None:
                _check_marker_index_size_compatibility(
                    self.training_markers_index,
                    self._n_col_train,
                    encoding_size,
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
        return len(next(iter(self.encoding_map.values())))

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
    """Number of epochs to train the model

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
    according to :attr:`validation_size`.

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

    After the training split (controlled by :attr:`training_size`), the remaining
    samples are further split into a validation set and an evaluation set with a
    second call to :func:`sklearn.model_selection.train_test_split` on the remaining
    data. The validation set is passed to Keras's ``fit`` as ``validation_data``,
    while the evaluation set is passed to ``evaluate``.

    If set to ``1.0``, all remaining samples go to validation and no evaluation
    is performed (i.e. :attr:`autoencoder_evaluations` entries will be ``None``).

    Important
    ---------
    The type matters: a :class:`float` is interpreted as a proportion of the
    remaining (non-training) data, while an :class:`int` is interpreted as an
    absolute sample count. See :func:`sklearn.model_selection.train_test_split`
    for details.

    See Also
    --------
    :attr:`CompressionModel.training_size` : Controls the initial train split.
    """

    autoencoder_models: list[AutoencoderModels]
    """List of autoencoder populated during :meth:`~CompressionModel.fit`.

    Each :class:`AutoencoderModels` instance is built from :attr:`layer_sizes`
    and trained on the corresponding chunk of the training data. Empty list
    before fitting.

    See Also
    --------
    :attr:`CompressionModel.n_chunks` : Number of autoencoders to be trained.
    :meth:`CompressionModel.fit` : Where the models are instantiated and trained.
    """

    autoencoder_evaluations: list
    """List of evaluation results for each autoencoder.

    Each entry is a dict mapping the loss name to its evaluation score (e.g.
    ``{"mse": 0.042}``), computed on the evaluation set after training. Entries
    are ``None`` for each chunks if :attr:`validation_size` is ``1.0``. Empty
    list before fitting.

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
        and determines how the data is chunked (see :attr:`chunk_size`). The last
        element is the latent dimension. Elements in between define the encoding
        layers; the decoder mirrors them in reverse.

        For example, ``[28, 14, 7, 3]`` produces:

        .. code-block:: none

            input(28) → encoding(14, relu) → encoding(7, relu) → latent(3, sigmoid)
            → decoding(7, relu) → decoding(14, relu) → output(28, sigmoid)

        Must have at least 2 elements and contain only integers. Setting this
        attribute triggers compatibility validation against the training data
        if it is already set.

        Raises
        ------
        TypeError
            If ``layer_sizes`` is not a list or tuple.
        ValueError
            If ``layer_sizes`` has fewer than 2 elements or contains non-integer values.

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
        """Initialise from a training data frame.

        TODO
        """

        if training_dataframe.empty:
            raise ValueError(
                "`training_dataframe` is empty. Provide a DataFrame with at least "
                "one row and one column."
            )
        if "training_encoded_geno_array" in kwargs:
            raise ValueError(
                "`training_encoded_geno_array` cannot be passed as a keyword argument "
                "to from_dataframe(); it is derived from `training_dataframe`."
            )

        geno_array = training_dataframe.to_numpy()

        if encoding_map is None:
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

        Splits the training data into chunks of size :attr:`chunk_size`, builds
        one :class:`AutoencoderModels` per chunk, and trains each autoencoder on
        its corresponding chunk. Training and validation splits are drawn from the
        same rows across all chunks.

        Populates :attr:`autoencoder_models`, :attr:`autoencoder_fits`, and
        :attr:`autoencoder_evaluations`.

        Parameters
        ----------
        seed :
            Random seed for the train/test split. Overrides :attr:`seed` if
            provided. If neither is set, a random seed is generated and stored
            in :attr:`seed`. Only affects data splitting, not weight initialisation.

        Important
        ---------
        The seed only controls the data split, not the model weight initialisation or
        the training process. Same seed can produce models with different weights.

        Raises
        ------
        RuntimeError
            If :attr:`training_encoded_geno_array` is ``None``.
        RuntimeError
            If :attr:`layer_sizes` is not set.

        """
        if self.training_encoded_geno_array is None:
            raise RuntimeError("No training data available.")

        if not self.layer_sizes:
            raise RuntimeError("layer_sizes is not set.")

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
        RuntimeError
            If the model has not been fitted yet (i.e. ``self.is_fitted`` is ``False``).
        ValueError
            If ``encoded_geno_array`` has a different number of columns than
            the training data.

        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        if encoded_geno_array.shape[1] != self._n_col_train:
            raise ValueError(
                f"Incompatible data. Provided data have a different number of columns "
                f"({encoded_geno_array.shape[1]}) than the training data "
                f"({self._n_col_train})"
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
        RuntimeError
            If the model has not been fitted yet (i.e.
            :attr:`CompressionModel.is_fitted` is ``False``).
        ValueError
            If the set of ``geno_dataframe``'s columns index is different from
            the one from the training data (i.e.
            :attr:`CompressionModel.training_markers_index`)
        ValueError
            If ``geno_dataframe`` has a different number of markers than
            the training data.
        """
        if self.training_markers_index is not None:
            if set(self.training_markers_index) != set(geno_dataframe.columns):
                raise ValueError(
                    "Incompatible data. Provided data have different column index "
                    "than the training data."
                )
            geno_dataframe = geno_dataframe.loc[:, self.training_markers_index]

        geno_array = geno_dataframe.to_numpy()

        encoded_geno_array = encode_snp_array(
            geno_array,
            missing_values=missing_values,
            encoding_map=encoding_map or self.encoding_map,
        )
        return self.compress(encoded_geno_array, batch_size=batch_size)
