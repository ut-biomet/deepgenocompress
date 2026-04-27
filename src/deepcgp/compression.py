import numpy as np
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


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


class AutoencoderModels:
    """Data structure representing an autoencoder and its corresponding encoder.

    The architecture is determined by ``layers_sizes``: the first element is the
    input/output dimension, the last is the latent dimension, and the elements
    in between define the encoding layers. The decoder mirrors the encoder in
    reverse.

    Encoding and decoding layers use ReLU activation, the latent layer and output layer
    use sigmoid activation.

    Parameters
    ----------
    layers_sizes : list[int] or tuple[int]
        Ordered layer dimensions, e.g. ``[28, 14, 7, 3]`` produces:
        input(28) -> encoding(14, relu) -> encoding(7, relu) -> latent(3, sigmoid)
        -> decoding(7, relu) -> decoding(14, relu) -> output(28, sigmoid).
        Must have at least 2 elements.

    Raises
    ------
    TypeError
        If ``layers_sizes`` is not a list or tuple.
    ValueError
        If ``layers_sizes`` has a length lower than 2.
    ValueError
        If ``layers_sizes``'s values are not integers.

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

    def __init__(self, layers_sizes: list[int] | tuple[int]):
        """Initialise from layers_sizes."""
        if not isinstance(layers_sizes, (list, tuple)):
            # pyright: ignore [reportUnreachable]
            raise TypeError(
                "layers_sizes must be a list, or tuple of int got "
                f"`{type(layers_sizes)}`"
            )
        if len(layers_sizes) < 2:
            raise ValueError(
                "layers_sizes length must be greater than 2, got "
                f"`{len(layers_sizes)=}`"
            )
        for i, ls in enumerate(layers_sizes):
            if not isinstance(ls, int):
                raise ValueError(
                    "layers_sizes must be a list, or tuple of int got "
                    f"`{type(layers_sizes[i])}` for index layers_sizes[{i}]."
                )

        encoder_activation_functions = _default_encoder_activation_functions(
            layers_sizes
        )
        decoder_activation_functions = _default_decoder_activation_functions(
            layers_sizes
        )

        input_size = layers_sizes[0]
        encoding_sizes = layers_sizes[1:-1]
        laten_size = layers_sizes[-1]
        decoding_sizes = encoding_sizes[::-1]
        output_size = layers_sizes[0]

        encoding_activations = encoder_activation_functions[:-1]
        latent_activations = encoder_activation_functions[-1]
        decoding_activations = decoder_activation_functions[:-1]
        output_activation = decoder_activation_functions[-1]

        input_layer = Input(shape=(input_size,))

        encoder_layers = input_layer
        for i, (layer_size, act_fun) in enumerate(
            zip(encoding_sizes, encoding_activations)
        ):
            encoder_layers = Dense(
                layer_size, activation=act_fun, name=f"encoding_{i}"
            )(encoder_layers)
        encoder_layers = Dense(
            laten_size, activation=latent_activations, name="latent"
        )(encoder_layers)

        decoder_layers = encoder_layers
        for i, (layer_size, act_fun) in enumerate(
            zip(decoding_sizes, decoding_activations)
        ):
            decoder_layers = Dense(
                layer_size, activation=act_fun, name=f"decoding_{i}"
            )(decoder_layers)
        decoder_layers = Dense(
            output_size, activation=output_activation, name="output"
        )(decoder_layers)

        self.encoder = Model(inputs=input_layer, outputs=encoder_layers)
        self.autoencoder = Model(inputs=input_layer, outputs=decoder_layers)


def compress_data(X, best_config, seed=42, verbose=True):

    input_dim = best_config["InputDim"]
    compress = best_config["Compress"]
    batch_size = best_config["BatchSize"]
    lr = best_config["LearningRate"]
    epochs = best_config["Epochs"]

    n_chunks = X.shape[1] // input_dim
    if n_chunks == 0:
        raise ValueError(f"X has {X.shape[1]} features < InputDim={input_dim}")

    usable = n_chunks * input_dim
    if usable != X.shape[1] and verbose:
        print(
            f"Dropping last {X.shape[1] - usable} feature(s) to fit InputDim={input_dim}."
        )

    split_chunks = np.hsplit(X[:, :usable], n_chunks)

    encoded_chunks = []
    for i, chunk in enumerate(split_chunks):
        if verbose:
            print(f"→ Chunk {i+1}/{len(split_chunks)} | chunk shape={chunk.shape}")

        x_train, x_mid = train_test_split(chunk, test_size=0.4, random_state=seed)
        x_test, x_valid = train_test_split(x_mid, test_size=0.5, random_state=seed)

        aem = AutoencoderModels([input_dim] + compress)
        autoencoder, encoder = aem.autoencoder, aem.encoder

        autoencoder.compile(optimizer=Adam(learning_rate=lr), loss="mse")
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        autoencoder.fit(
            x_train,
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_valid, x_valid),
            callbacks=[early_stop],
            verbose=0,
        )

        encoded = encoder.predict(chunk, batch_size=batch_size, verbose=0)
        encoded_chunks.append(encoded)

    compressed_array = np.hstack(encoded_chunks)
    if verbose:
        print("Final compressed shape:", compressed_array.shape)

    return compressed_array
