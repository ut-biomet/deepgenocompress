# ==============================
# Tanzila Islam
# Email: tanzilamohita@gmail.com
# ===============================

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# for gpu use
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU is available: {gpus}")
else:
    print("GPU is NOT available. Running on CPU.")


# ============ Autoencoder Model ============
def build_autoencoder(input_dim, compress):
    h1, h2, bottleneck = compress
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(h1, activation='relu')(input_layer)
    encoded = Dense(h2, activation='relu')(encoded)
    encoded = Dense(bottleneck, activation='sigmoid')(encoded)
    decoded = Dense(h2, activation='relu')(encoded)
    decoded = Dense(h1, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    return autoencoder, encoder


def compress_single_chromosome(path, best_config):
    input_dim = best_config['InputDim']
    compress = best_config['Compress']
    batch_size = best_config['BatchSize']
    lr = best_config['LearningRate']
    epochs = best_config['Epochs']

    chr_name = os.path.basename(path).replace(f'_C2.npy', '')
    print(f"\nProcessing {chr_name} ...")
    chr_data = np.load(path, allow_pickle=True).astype(np.float32)

    n_chunks = chr_data.shape[1] // input_dim
    usable = n_chunks * input_dim
    split_chunks = np.hsplit(chr_data[:, :usable], n_chunks)

    encoded_chunks = []

    for i, chunk in enumerate(split_chunks):
        print(f"→ Chunk {i + 1}/{len(split_chunks)}")
        x_train, x_mid = train_test_split(chunk, test_size=0.4, random_state=42)
        x_test, x_valid = train_test_split(x_mid, test_size=0.5, random_state=42)

        autoencoder, encoder = build_autoencoder(input_dim, compress)
        autoencoder.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        autoencoder.fit(
            x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_valid, x_valid),
            callbacks=[early_stop],
            verbose=0
        )

        # Encode the current chunk
        encoded = encoder.predict(chunk, batch_size=batch_size, verbose=0)
        encoded_chunks.append(encoded)

    # Combine all encoded chunks and save
    compressed_array = np.hstack(encoded_chunks)

    return compressed_array


def compress_data(X, best_config, seed=42, verbose=True):

    input_dim  = best_config["InputDim"]
    compress   = best_config["Compress"]
    batch_size = best_config["BatchSize"]
    lr         = best_config["LearningRate"]
    epochs     = best_config["Epochs"]

    X = X.astype(np.float32)

    n_chunks = X.shape[1] // input_dim
    if n_chunks == 0:
        raise ValueError(f"X has {X.shape[1]} features < InputDim={input_dim}")

    usable = n_chunks * input_dim
    if usable != X.shape[1] and verbose:
        print(f"Dropping last {X.shape[1] - usable} feature(s) to fit InputDim={input_dim}.")

    split_chunks = np.hsplit(X[:, :usable], n_chunks)

    encoded_chunks = []
    for i, chunk in enumerate(split_chunks):
        if verbose:
            print(f"→ Chunk {i+1}/{len(split_chunks)} | chunk shape={chunk.shape}")

        x_train, x_mid = train_test_split(chunk, test_size=0.4, random_state=seed)
        x_test, x_valid = train_test_split(x_mid, test_size=0.5, random_state=seed)

        autoencoder, encoder = build_autoencoder(input_dim, compress)
        autoencoder.compile(optimizer=Adam(learning_rate=lr), loss="mse")
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        autoencoder.fit(
            x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_valid, x_valid),
            callbacks=[early_stop],
            verbose=0
        )

        encoded = encoder.predict(chunk, batch_size=batch_size, verbose=0)
        encoded_chunks.append(encoded)

    compressed_array = np.hstack(encoded_chunks)
    if verbose:
        print("Final compressed shape:", compressed_array.shape)

    return compressed_array


