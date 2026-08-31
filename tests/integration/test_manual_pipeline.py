from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deepcgp import CompressionModel, build_vcf_encoding_map, encode_snp_array, read_vcf
from deepcgp.utils import encoding_size, possible_first_layer_sizes


def test_from_csv_to_compression():
    """Read csv → build encoding map → encode -> compress."""
    csv_file = Path(__file__).parent.parent / "fixtures" / "X.csv"
    data = pd.read_csv(csv_file, index_col=0)

    geno_array = data.to_numpy()
    encoded_geno_array = encode_snp_array(geno_array, missing_values=["N"])

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


@pytest.mark.parametrize(
    "use_bases", [True, False], ids=["use_bases = True", "use_bases = False"]
)
def test_from_vcf_to_compression(use_bases):
    """Read VCF → build encoding map → encode -> compress."""

    vcf_file = (
        Path(__file__).parent.parent
        / "fixtures"
        / "vcf"
        / "basic_14snps_50_samples.vcf"
    )

    data = read_vcf(vcf_file, use_bases=use_bases)
    encoding_map = build_vcf_encoding_map(data)

    geno_array = data.to_numpy()
    encoded_geno_array = encode_snp_array(geno_array, encoding_map=encoding_map)

    # assert correctly encoded ???

    first_layer_size = possible_first_layer_sizes(
        encoded_geno_array.shape[1], encoding_size(encoding_map), 2
    )[0].first_layer_size
    layer_sizes = [
        first_layer_size,
        first_layer_size // 2,
        first_layer_size // 4,
    ]

    cm = CompressionModel(
        training_encoded_geno_array=encoded_geno_array,
        layer_sizes=layer_sizes,
        seed=42,
        batch_size=52,
        learning_rate=0.001,
        epochs=200,
        encoding_map=encoding_map,
    )

    cm.fit()
    compressed = cm.compress(encoded_geno_array)

    assert compressed.shape[0] == len(data)
    assert compressed.shape[1] == layer_sizes[-1] * cm.n_chunks
    assert not np.isnan(compressed).any()
