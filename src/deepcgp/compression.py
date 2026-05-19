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
from typing import Any

import numpy as np
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.losses import Loss
from keras.optimizers import Adam
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

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


_default_decoder_activation_functions = _default_encoder_activation_functions
"""Return default decoder activations functions.

Because of symetry this is the same as ``_get_encoder_activation_functions``:
"""


def _check_layer_sizes(layer_sizes):
    """Check layers_sizes's type and value."""
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
    layer_sizes : list[int] or tuple[int]
        Ordered layer dimensions, e.g. ``[28, 14, 7, 3]`` produces:
        input(28) -> encoding(14, relu) -> encoding(7, relu) -> latent(3, sigmoid)
        -> decoding(7, relu) -> decoding(14, relu) -> output(28, sigmoid).
        Must have at least 2 elements.

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
    .. jupyter-execute::
        from deepcgp import AutoencoderModels
        aem = AutoencoderModels([28, 14, 7, 3])
        aem.autoencoder.summary()
        aem.encoder.summary()

    """

    autoencoder: Model
    """The full autoencoder model."""
    encoder: Model
    """The encoder model. (Shares same layers with the autoencoder)"""

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
        laten_size = layer_sizes[-1]
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
            laten_size, activation=latent_activations, name="latent"
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

        self.encoder = Model(inputs=input_layer, outputs=encoder_layers)
        self.autoencoder = Model(inputs=input_layer, outputs=decoder_layers)


def _check_layer_sizes_and_data_compatibility(
    n_cols, first_layer_size, encoding_size=None
):
    if n_cols % first_layer_size != 0:
        warnings.warn(
            f"layer_sizes[0]={first_layer_size} is not a divisor of {n_cols=}. "
            "Column padding (filled with 0) will be added to the end of the data "
            "to fit requested layer_sizes[0]",
            UserWarning,
            stacklevel=2,
        )

    if encoding_size is not None:
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

    return True


def _split_data(
    encoded_geno_array: NDArray[np.float32],
    chunk_size: int,
    encoding_size=None,
):

    n_cols: int = encoded_geno_array.shape[1]

    _check_layer_sizes_and_data_compatibility(
        n_cols=n_cols, first_layer_size=chunk_size, encoding_size=encoding_size
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
    """Model to compress genomic data."""

    # TODO:
    # use raw data as input directly:
    #    geno_df
    #    missing_values
    #    encoding_map
    # we should keep the markers "IDs" to validate later compressions
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
        """Training data."""
        return self._training_encoded_geno_array

    @training_encoded_geno_array.setter
    def training_encoded_geno_array(self, training_encoded_geno_array):
        if self.layer_sizes and training_encoded_geno_array is not None:
            _check_layer_sizes_and_data_compatibility(
                n_cols=training_encoded_geno_array.shape[1],
                first_layer_size=self.layer_sizes[0],
            )
        self._training_encoded_geno_array = training_encoded_geno_array

    @property
    def _n_col_train(self):
        """Number of columns in the encoded training data."""
        if self.training_encoded_geno_array is None:
            return 0
        return self.training_encoded_geno_array.shape[1]

    encoding_map: dict[Any, list[float]] | None
    """"""

    @property
    def encoding_size(self) -> int | None:
        """Length of the encoding vectors."""
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
        """Number of data chunks."""
        return len(self._train_data_chunks)

    @property
    def chunk_size(self):
        """Number of columns in each data chunk.

        Corresponds to ``layer_sizes[0]``, which is the input dimension of each
        autoencoder. The training data is split into chunks of this size before
        fitting or compression. Returns ``0`` if ``layer_sizes`` is not set.

        See Also
        --------
        :attr:`CompressionModel.layer_sizes` : Autoencoder layer dimensions.
        :attr:`CompressionModel.n_chunks` : Number of chunks the data is split into.

        """
        return self.layer_sizes[0] if self.layer_sizes else 0

    batch_size: int
    """"""
    epochs: int
    """"""
    fitting_callbacks: list
    """"""
    learning_rate: float
    """"""
    loss: str | Loss = "mse"
    """"""
    seed: int | None
    """Rng seed used for training."""
    training_size: int | float  # type is importnant
    """"""
    validation_size: int | float  # based on remaining 1-training_size data !!!
    """"""

    autoencoder_models: list[AutoencoderModels]
    """"""
    autoencoder_evaluations: list
    """"""
    autoencoder_fits: list
    """"""

    @property
    def layer_sizes(self) -> list[int]:
        """Autoencoders's layer sizes.

        TODO: explain about the first element being the "chunk size"
        The first element is the input/output dimension, the last is the latent
        dimension, and the elements in between define the encoding layers.
        The decoder mirrors the encoder in reverse. Apply to all autoencoders.
        (cf. :class:`AutoencoderModels`)

        e.g. ``[28, 14, 7, 3]`` for autoencoders:
        input(28) -> encoding(14, relu) -> encoding(7, relu) -> latent(3, sigmoid)
        -> decoding(7, relu) -> decoding(14, relu) -> output(28, sigmoid).
        """
        return self._layer_sizes

    @layer_sizes.setter
    def layer_sizes(self, layers_sizes: list[int] | tuple[int, ...]):
        _check_layer_sizes(layers_sizes)
        if self.training_encoded_geno_array is not None:
            _check_layer_sizes_and_data_compatibility(
                n_cols=self._n_col_train, first_layer_size=layers_sizes[0]
            )
        self._layer_sizes = list(layers_sizes)

    @property
    def is_fitted(self) -> bool:
        """Indicates whether the model has been fitted."""
        return (
            all(m.is_fitted for m in self.autoencoder_models)
            if self.autoencoder_models
            else False
        )

    def __init__(
        self,
        # -- data --
        training_encoded_geno_array: NDArray[np.float32] | None = None,
        encoding_map: dict[Any, list[float]] | None = None,
        # -- autoencoders --
        layer_sizes: list[int] | tuple[int, ...] | None = None,
        # -- fitting --
        batch_size: int = 50,
        epochs: int = 200,
        fitting_callbacks: list | None = None,
        learning_rate: float = 0.001,
        loss: str | Loss = "mse",
        seed: int | None = None,
        training_size: int | float = 0.4,  # type is important cf train_test_split
        validation_size: int | float = 0.5,  # type is important cf train_test_split
    ):
        """Initialise compression model."""
        self._training_encoded_geno_array = None
        self._layer_sizes = []

        self.encoding_map = encoding_map

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

    def fit(
        self,
        seed: int | None = None,
    ):
        """Fit the autoencoders.

        TODO: be clear that `seed` attribute will be updated.
              be clear that `seed` only affect train/test split not model parameters
              mentioned that the same lines are used train/test for each autoencoder
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
        encoded_geno_array : NDArray[np.float32]
            Encoded genotype array to compress. Must have the same number of
            columns as the training data.
        batch_size : int, optional
            Batch size for encoder prediction. Defaults to ``self.batch_size``
            if not provided.

        Returns
        -------
        NDArray[np.float32]
            Compressed array of shape ``(n_samples, n_chunks * latent_dim)``,
            where ``latent_dim`` is ``self.layer_sizes[-1]``.

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
        self._check_compatibility_with_training_data(encoded_geno_array)
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

    def _check_compatibility_with_training_data(self, data):
        # TODO: validation with marker IDs (if possible)
        if data.shape[1] != self._n_col_train:
            raise ValueError(
                f"Provided data have a different number of columns "
                f"({data.shape[1]}) than the training data ({self._n_col_train})"
            )
