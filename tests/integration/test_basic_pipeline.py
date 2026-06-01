import numpy as np
import pandas as pd

from deepcgp import CompressionModel, encode_snp_array


def test_full_compression_pipeline_with_sample_data():
    """Mirrors how a user would call the module.

    Read data → encode → compress.
    """
    data = pd.read_csv("tests/fixtures/X.csv", index_col=0)

    geno_array = data.astype(str).values
    encoded_geno_array = encode_snp_array(geno_array)

    cm = CompressionModel(
        training_encoded_geno_array=encoded_geno_array,
        layer_sizes=[28, 14, 7, 3],
        seed=42,
        batch_size=52,
        learning_rate=0.001,
        epochs=200,
    )
    cm.fit()
    compressed = cm.compress(encoded_geno_array)

    assert compressed.shape[0] == len(data)
    assert compressed.shape[1] == 3 * cm.n_chunks
    assert not np.isnan(compressed).any()
