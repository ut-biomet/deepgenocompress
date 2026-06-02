# ==============================
# Tanzila Islam
# Email: tanzilamohita@gmail.com
# Date: 7/29/2024
# ===============================

# ==============================
# Autoencoder Hyperparameter Tuning
# ==============================
# This script tunes a simple symmetric autoencoder.
# The encoder compresses the input step by step, and the decoder reconstructs it.
#
# Example architecture:
#     28 -> 14 -> 7 -> 14 -> 28
#
# The bottleneck is always smaller than the input dimension.
# Hidden layers and bottleneck are not allowed to be the same size as the input.
# ==============================

import datetime
import glob
import json
import os
import pathlib
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ==============================
# Reproducibility
# ==============================
# Keeping the seed fixed helps make the tuning results more repeatable.

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ==============================
# GPU Check
# ==============================

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU is available: {gpus}")
else:
    print("GPU is NOT available. Running on CPU.")


# ==============================
# Configuration
# ==============================

C = 3
DATA = "Maize"

# set project root as parent of src folder
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

BASE_DIR = PROJECT_ROOT / f"data/{DATA}"
META_DIR = PROJECT_ROOT / f"ModelMetaData_{DATA}"

NPY_FOLDER = f"{META_DIR}/{DATA}_Compressed_Separated/C2"
TUNE_OUTPUT = f"{META_DIR}/Tuning_{DATA}_DeepCGP/Separated"

os.makedirs(TUNE_OUTPUT, exist_ok=True)


# ==============================
# Autoencoder Model
# ==============================


def build_autoencoder(input_dim, hidden_dim, bottleneck_dim, learning_rate):
    """
    Build a simple symmetric autoencoder.

    The structure is:

        input_dim -> hidden_dim -> bottleneck_dim -> hidden_dim -> input_dim

    Example:

        28 -> 14 -> 7 -> 14 -> 28

    Parameters
    ----------
    input_dim : int
        Number of features in each input chunk.

    hidden_dim : int
        Size of the hidden layer before and after the bottleneck.

    bottleneck_dim : int
        Size of the compressed representation.

    learning_rate :
        Learning rate for Adam optimizer.
    """

    # The hidden layer and bottleneck should not be the same as the input.
    if hidden_dim >= input_dim:
        raise ValueError(
            f"hidden_dim must be smaller than input_dim. "
            f"Got hidden_dim={hidden_dim}, input_dim={input_dim}."
        )

    if bottleneck_dim >= input_dim:
        raise ValueError(
            f"bottleneck_dim must be smaller than input_dim. "
            f"Got bottleneck_dim={bottleneck_dim}, input_dim={input_dim}."
        )

    if bottleneck_dim >= hidden_dim:
        raise ValueError(
            f"bottleneck_dim must be smaller than hidden_dim. "
            f"Got bottleneck_dim={bottleneck_dim}, hidden_dim={hidden_dim}."
        )

    input_layer = Input(shape=(input_dim,))

    encoded = Dense(hidden_dim, activation="relu")(input_layer)
    bottleneck = Dense(bottleneck_dim, activation="sigmoid", name="bottleneck")(encoded)

    decoded = Dense(hidden_dim, activation="relu")(bottleneck)
    output_layer = Dense(input_dim, activation="sigmoid")(decoded)

    autoencoder = Model(input_layer, output_layer)
    encoder = Model(input_layer, bottleneck)

    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    return autoencoder, encoder


# ==============================
# Architecture Options
# ==============================


def generate_architecture_options(input_dim):
    """
    Create simple architecture options.

    This keeps the autoencoder shape simple:

        input_dim -> hidden_dim -> bottleneck_dim -> hidden_dim -> input_dim

    For example:

        28 -> 14 -> 7 -> 14 -> 28
        56 -> 28 -> 14 -> 28 -> 56
        112 -> 56 -> 28 -> 56 -> 112

    The hidden layer and bottleneck are always smaller than the input.
    The bottleneck is always smaller than the hidden layer.
    """

    options = []

    # Main option: clean half-and-quarter compression.
    # This gives:
    #   28 -> 14 -> 7
    #   56 -> 28 -> 14
    #   112 -> 56 -> 28
    option_1 = {"hidden_dim": input_dim // 2, "bottleneck_dim": input_dim // 4}

    # Slightly stronger compression.
    # This gives:
    #   28 -> 14 -> 3
    #   56 -> 28 -> 7
    #   112 -> 56 -> 14
    option_2 = {"hidden_dim": input_dim // 2, "bottleneck_dim": input_dim // 8}

    # Slightly wider hidden layer, but still smaller than the input.
    # This checks whether the model benefits from more room before compression.
    # Example:
    #   28 -> 21 -> 7
    #   56 -> 42 -> 14
    #   112 -> 84 -> 28
    option_3 = {"hidden_dim": (input_dim * 3) // 4, "bottleneck_dim": input_dim // 4}

    candidate_options = [option_1, option_2, option_3]

    for option in candidate_options:
        hidden_dim = max(2, option["hidden_dim"])
        bottleneck_dim = max(2, option["bottleneck_dim"])

        # Keep only valid compression structures.
        # We do not want input_dim == hidden_dim or input_dim == bottleneck_dim.
        if hidden_dim < input_dim and bottleneck_dim < hidden_dim:
            options.append({"hidden_dim": hidden_dim, "bottleneck_dim": bottleneck_dim})

    return options


# ==============================
# Prepare Data for Tuning
# ==============================


def prepare_chunk_training_data(
    data_array, input_dim, subset_snps=100_000, max_chunks=100, random_state=42
):
    """
    Take a SNP subset, split it into fixed-size chunks, and stack selected chunks.

    Each chunk has input_dim columns.

    If the number of features is not divisible by input_dim, zero columns are
    added at the end. This keeps all original features and makes the last chunk
    complete.
    """

    subset_features = subset_snps * 4
    full = data_array[:, :subset_features]

    n_samples, n_features = full.shape

    if n_features == 0:
        raise ValueError("No features found in the input data.")

    remainder = n_features % input_dim

    if remainder == 0:
        padding_features = 0
    else:
        padding_features = input_dim - remainder

    if padding_features > 0:
        padding = np.zeros((n_samples, padding_features), dtype=full.dtype)

        full_padded = np.hstack([full, padding])
    else:
        full_padded = full

    padded_features = full_padded.shape[1]
    n_chunks = padded_features // input_dim

    chunks = np.hsplit(full_padded, n_chunks)

    rng = np.random.default_rng(random_state)

    selected_chunk_count = min(max_chunks, len(chunks))

    selected_indices = rng.choice(len(chunks), size=selected_chunk_count, replace=False)

    selected_chunks = [chunks[i] for i in selected_indices]

    train_data = np.vstack(selected_chunks)

    chunk_info = {
        "OriginalSamples": n_samples,
        "OriginalFeatures": n_features,
        "InputDim": input_dim,
        "PaddingFeatures": padding_features,
        "PaddedFeatures": padded_features,
        "TotalChunks": n_chunks,
        "SelectedChunks": selected_chunk_count,
    }

    return train_data, chunk_info


# ==============================
# Tuning Pipeline
# ==============================


def run_tuning_pipeline(
    data_array, subset_snps=100_000, input_dims=None, max_chunks=100
):
    """
    Run hyperparameter tuning.

    Validation loss is used to choose the best parameters.
    Test loss is calculated only once at the end for the selected best model.
    """

    if input_dims is None:
        input_dims = [28, 56, 112]

    print("Starting autoencoder tuning...")

    # epoch_list = [50, 100, 200]
    # batch_size_list = [32, 64, 128]
    # learning_rates = [0.001, 0.0001]

    epoch_list = [5]
    batch_size_list = [32]
    learning_rates = [0.001]

    all_results = []

    best_result = None
    best_model = None
    best_x_test = None

    for input_dim in input_dims:
        print(f"\nPreparing data for input_dim={input_dim}...")

        train_data, chunk_info = prepare_chunk_training_data(
            data_array=data_array,
            input_dim=input_dim,
            subset_snps=subset_snps,
            max_chunks=max_chunks,
            random_state=SEED,
        )

        print("Chunk information:")
        for key, value in chunk_info.items():
            print(f"  {key}: {value}")

        print(f"Training data shape after stacking chunks: {train_data.shape}")

        # Split once for this input dimension.
        # The test set is kept aside and not used for tuning.
        x_train, x_temp = train_test_split(
            train_data, test_size=0.4, random_state=SEED, shuffle=True
        )

        x_valid, x_test = train_test_split(
            x_temp, test_size=0.5, random_state=SEED, shuffle=True
        )

        architecture_options = generate_architecture_options(input_dim)

        print("Architecture options:")
        for arch in architecture_options:
            print(
                f"  {input_dim} -> {arch['hidden_dim']} -> "
                f"{arch['bottleneck_dim']} -> {arch['hidden_dim']} -> {input_dim}"
            )

        for arch in architecture_options:
            hidden_dim = arch["hidden_dim"]
            bottleneck_dim = arch["bottleneck_dim"]

            for batch_size in batch_size_list:
                for lr in learning_rates:
                    for max_epochs in epoch_list:

                        print(
                            "\nTraining:"
                            f" architecture={input_dim}-{hidden_dim}-{bottleneck_dim}-{hidden_dim}-{input_dim},"
                            f" batch_size={batch_size},"
                            f" lr={lr},"
                            f" max_epochs={max_epochs}"
                        )

                        autoencoder, encoder = build_autoencoder(
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            bottleneck_dim=bottleneck_dim,
                            learning_rate=lr,
                        )

                        early_stop = EarlyStopping(
                            monitor="val_loss", patience=5, restore_best_weights=True
                        )

                        history = autoencoder.fit(
                            x_train,
                            x_train,
                            epochs=max_epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(x_valid, x_valid),
                            callbacks=[early_stop],
                            verbose=0,
                        )

                        actual_epochs = len(history.history["loss"])
                        best_val_loss = float(min(history.history["val_loss"]))

                        train_loss = float(
                            autoencoder.evaluate(x_train, x_train, verbose=0)
                        )

                        valid_loss = float(
                            autoencoder.evaluate(x_valid, x_valid, verbose=0)
                        )

                        result = {
                            "InputDim": input_dim,
                            "HiddenDim": hidden_dim,
                            "BottleneckDim": bottleneck_dim,
                            "Architecture": f"{input_dim}-{hidden_dim}-{bottleneck_dim}-{hidden_dim}-{input_dim}",
                            "BatchSize": batch_size,
                            "LearningRate": lr,
                            "MaxEpochs": max_epochs,
                            "ActualEpochs": actual_epochs,
                            "BestValLoss": best_val_loss,
                            "FinalTrainLoss": train_loss,
                            "FinalValidLoss": valid_loss,
                            "OriginalSamples": chunk_info["OriginalSamples"],
                            "OriginalFeatures": chunk_info["OriginalFeatures"],
                            "TotalChunks": chunk_info["TotalChunks"],
                            "SelectedChunks": chunk_info["SelectedChunks"],
                            "PaddedFeatures": chunk_info["PaddedFeatures"],
                            "PaddingFeatures": chunk_info["PaddingFeatures"],
                        }

                        all_results.append(result)

                        print(
                            f"Best val loss: {best_val_loss:.6f} | "
                            f"Final val loss: {valid_loss:.6f} | "
                            f"Actual epochs: {actual_epochs}"
                        )

                        if (
                            best_result is None
                            or best_val_loss < best_result["BestValLoss"]
                        ):
                            best_result = result
                            best_model = autoencoder
                            best_x_test = x_test

    # Test loss is calculated only for the best model.
    # This gives a cleaner estimate of how well the selected model generalizes.
    test_loss = float(best_model.evaluate(best_x_test, best_x_test, verbose=0))
    best_result["TestLoss"] = test_loss

    print("\nBest configuration:")
    for key, value in best_result.items():
        print(f"  {key}: {value}")

    # Save all tuning results.
    results_df = pd.DataFrame(all_results)

    results_csv_path = f"{TUNE_OUTPUT}/tuning_results_C{C}.csv"
    results_df.to_csv(results_csv_path, index=False)

    # Save the best configuration separately.
    best_config_path = f"{TUNE_OUTPUT}/best_config_C{C}.json"
    with open(best_config_path, "w") as f:
        json.dump(best_result, f, indent=4)

    print(f"\nAll tuning results saved to: {results_csv_path}")
    print(f"Best configuration saved to: {best_config_path}")

    return best_result


# ==============================
# Run Script
# ==============================

if __name__ == "__main__":
    start_time = time.time()

    file_paths = sorted(glob.glob(os.path.join(NPY_FOLDER, "GSTP004_chr*_C2.npy")))

    if len(file_paths) == 0:
        raise FileNotFoundError(f"No .npy files found in folder: {NPY_FOLDER}")

    all_chr_arrays = []

    for path in file_paths:
        print(f"Loading {path} ...")
        arr = np.load(path, allow_pickle=True)
        all_chr_arrays.append(arr)

    X_combined = np.hstack(all_chr_arrays)

    print("Combined X shape:", X_combined.shape)

    # best_params = run_tuning_pipeline(
    #     data_array=X_combined,
    #     subset_snps=100_000,
    #     input_dims=[28, 56, 112],
    #     max_chunks=100
    # )

    best_params = run_tuning_pipeline(
        data_array=X_combined, subset_snps=1000, input_dims=[28], max_chunks=10
    )

    print("\nBest parameters:")
    print(best_params)

    end_time = time.time()
    elapsed = end_time - start_time
    formatted_time = str(datetime.timedelta(seconds=int(elapsed)))

    print(f"\nTotal time: {formatted_time} (days:hh:mm:ss)")

