import copy
import re
from dataclasses import dataclass
from unittest.mock import call

import numpy as np
import pandas as pd
import pytest
from keras import Model
from keras.callbacks import EarlyStopping, History
from pytest_mock import MockerFixture
from sklearn.model_selection import train_test_split

from deepcgp.compression import (
    AutoencoderModels,
    CompressionModel,
    _check_layer_size_and_encoded_data_size,
    _check_layer_size_and_encoding_compatibility,
    _split_data,
)
from deepcgp.data_processing import (
    _validate_encoding_map,
    build_one_hot_encoding_map,
    encode_snp_array,
)

LAYER_SIZES_CASES = [
    pytest.param([32, 16, 8, 4], id="2-encoding-layers"),
    pytest.param([128, 64, 32, 16, 8, 4, 2], id="5-encoding-layers"),
    pytest.param([20, 2], id="0-encoding-layer"),
]


@pytest.fixture(params=LAYER_SIZES_CASES)
def layer_sizes(request):
    return request.param


@pytest.fixture
def basic_autoencoder_models(layer_sizes):
    return AutoencoderModels(layer_sizes)


class TestAutoencoderModels:
    """Tests for `AutoencoderModels` class."""

    def test_both_are_keras_models(self, basic_autoencoder_models):
        assert isinstance(basic_autoencoder_models.autoencoder, Model)
        assert isinstance(basic_autoencoder_models.encoder, Model)

    def test_input_shape_autoencoder(self, basic_autoencoder_models, layer_sizes):
        assert basic_autoencoder_models.autoencoder.input_shape == (
            None,
            layer_sizes[0],
        )

    def test_input_shape_encoder(self, basic_autoencoder_models, layer_sizes):
        assert basic_autoencoder_models.encoder.input_shape == (None, layer_sizes[0])

    def test_output_shape_autoencoder(self, basic_autoencoder_models, layer_sizes):
        assert basic_autoencoder_models.autoencoder.output_shape == (
            None,
            layer_sizes[0],
        )

    def test_output_shape_encoder(self, basic_autoencoder_models, layer_sizes):
        assert basic_autoencoder_models.encoder.output_shape == (None, layer_sizes[-1])

    def test_layer_count_autoencoder(self, basic_autoencoder_models, layer_sizes):
        expected = sum(
            [
                1,  # input layer
                len(layer_sizes) - 2,  # ie. - input layer - latent layer
                1,  # latent layer
                len(layer_sizes) - 2,  # ie. - latent layer - output layer
                1,  # output layer
            ]
        )
        assert len(basic_autoencoder_models.autoencoder.layers) == expected

    def test_layer_count_encoder(self, basic_autoencoder_models, layer_sizes):
        expected = len(layer_sizes)
        assert len(basic_autoencoder_models.encoder.layers) == expected

    def test_encoder_share_same_first_layers_of_autoencoder(
        self, basic_autoencoder_models
    ):
        aem = basic_autoencoder_models
        for enc_layer, ae_layer in zip(
            aem.encoder.layers, aem.autoencoder.layers, strict=False
        ):
            assert enc_layer is ae_layer

    def test_use_sigmoid_for_latent_layer(self, basic_autoencoder_models):
        latent_layer = basic_autoencoder_models.encoder.layers[-1]
        assert latent_layer.activation.__name__ == "sigmoid"

    def test_use_relu_for_encoding_decoging_layers(self, basic_autoencoder_models):
        aem = basic_autoencoder_models
        inner_layers = aem.autoencoder.layers[1:-1]  # all but the input/output layers
        latent_layer = aem.encoder.layers[-1]
        for layer in inner_layers:
            if layer.name != latent_layer.name:
                assert layer.activation.__name__ == "relu"

    def test_use_sigmoid_for_autoencoder_output_layer(self, basic_autoencoder_models):
        output_layer = basic_autoencoder_models.autoencoder.layers[-1]
        assert output_layer.activation.__name__ == "sigmoid"

    def test_latent_layer_with_size_one(self):
        layer_sizes = [20, 16, 8, 1]
        aem = AutoencoderModels(layer_sizes)
        assert aem.encoder.output_shape == (None, 1)
        assert aem.autoencoder.output_shape == (None, layer_sizes[0])

    def test_input_dim_one(self):
        with pytest.warns(
            UserWarning,
            match=(
                r"Latent layer size \(\d+\) is equal or "
                r"larger than the input layer size \(1\)"
            ),
        ):
            aem = AutoencoderModels([1, 16, 8, 4])
        assert aem.autoencoder.input_shape == (None, 1)
        assert aem.autoencoder.output_shape == (None, 1)

    def test_no_compression(self):
        with pytest.warns(
            UserWarning,
            match=(
                r"Latent layer size \(\d+\) is equal or "
                r"larger than the input layer size \(\d+\)"
            ),
        ):
            aem = AutoencoderModels([10, 10, 10, 10])
        assert aem.encoder.output_shape == (None, 10)
        assert aem.autoencoder.output_shape == (None, 10)

    def test_raise_if_layer_sizes_is_string(self):
        with pytest.raises(
            TypeError,
            match="layer_sizes must be a list, or tuple of int got `<class 'str'>`",
        ):
            AutoencoderModels("987")  # pyright: ignore [reportArgumentType]

    def test_raise_if_layer_sizes_lenght_is_lower_than_2(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "layer_sizes length must be greater than 2, got `len(layer_sizes)=1`"
            ),
        ):
            AutoencoderModels([42])

    def test_raise_if_layers_sizes_are_not_integers(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "layer_sizes must be a list, or tuple of int got `<class 'float'>` "
                "for index layers_sizes[1]."
            ),
        ):
            AutoencoderModels([42, 3.5])  # pyright: ignore [reportArgumentType]

    def test_is_fitted_is_false(self):
        assert not AutoencoderModels([4, 3, 2]).is_fitted

    def test_raise_warning_if_latent_size_equal_input_size(self):
        with pytest.warns(
            UserWarning,
            match=r"is equal or larger than",
        ) as warn_info:
            AutoencoderModels([4, 3, 4])

        expected_message = (
            "Latent layer size (4) is equal or larger than "
            "the input layer size (4). This will expand rather "
            "than compress the data."
        )
        assert str(warn_info[0].message) == expected_message

    def test_raise_warning_if_latent_size_larger_than_input_size(self):
        with pytest.warns(
            UserWarning,
            match=r"is equal or larger than",
        ) as warn_info:
            AutoencoderModels([5, 7, 8])

        expected_message = (
            "Latent layer size (8) is equal or larger than "
            "the input layer size (5). This will expand rather "
            "than compress the data."
        )
        assert str(warn_info[0].message) == expected_message

    def test_warning_when_enlarging_data(self):
        with pytest.warns(
            UserWarning,
            match=(
                r"Latent layer size \(\d+\) is equal or "
                r"larger than the input layer size \(\d+\)"
            ),
        ) as warn_info:
            AutoencoderModels([1, 5, 3])

        expected_message = (
            "Latent layer size (3) is equal or larger than "
            "the input layer size (1). This will expand rather "
            "than compress the data."
        )
        assert str(warn_info[0].message) == expected_message

    def test_warning_when_data_size_remains_the_same(self):
        with pytest.warns(
            UserWarning,
            match=(
                r"Latent layer size \(\d+\) is equal or "
                r"larger than the input layer size \(\d+\)"
            ),
        ) as warn_info:
            AutoencoderModels([7, 3, 7])

        expected_message = (
            "Latent layer size (7) is equal or larger than "
            "the input layer size (7). This will expand rather "
            "than compress the data."
        )
        assert str(warn_info[0].message) == expected_message


@dataclass
class TrainingDataFixture:
    array: np.ndarray
    dataframe: pd.DataFrame
    encoded_array: np.ndarray
    encoding_map: dict
    layer_sizes: list


@pytest.fixture(scope="module")
def basic_training_data():
    gd = pd.DataFrame(
        [
            ["G", "A", "T", "T", "A", "C"],
            ["A", "C", "A", "T", "T", "A"],
            ["T", "A", "G", "G", "T", "C"],
        ],
        index=["ind_1", "ind_2", "ind_3"],
        columns=["snp_1", "snp_2", "snp_3", "snp_4", "snp_5", "snp_6"],
    )
    ga = gd.to_numpy()

    encoding_map = build_one_hot_encoding_map(ga)
    return TrainingDataFixture(
        array=ga,
        dataframe=gd,
        encoded_array=encode_snp_array(ga),
        encoding_map=encoding_map,
        layer_sizes=[8, 4, 2],
    )


@pytest.fixture(scope="module")
def training_data_requiring_padding():
    gd = pd.DataFrame(
        [
            ["G", "A", "T", "T", "A", "C"],
            ["A", "C", "A", "T", "T", "A"],
            ["T", "A", "G", "G", "T", "C"],
        ],
        index=["ind_1", "ind_2", "ind_3"],
        columns=["snp_1", "snp_2", "snp_3", "snp_4", "snp_5", "snp_6"],
    )
    ga = gd.to_numpy()

    encoding_map = build_one_hot_encoding_map(ga)
    return TrainingDataFixture(
        array=ga,
        dataframe=gd,
        encoded_array=encode_snp_array(ga),
        encoding_map=encoding_map,
        layer_sizes=[16, 8, 2],  # 16 = 4 alleles * encoding_size (=4)
    )


@pytest.fixture(scope="module")
def _initialised_compression_model(basic_training_data: TrainingDataFixture):
    return CompressionModel.from_dataframe(
        training_dataframe=basic_training_data.dataframe,
        layer_sizes=basic_training_data.layer_sizes,
    )


@pytest.fixture(scope="module")
def _fitted_compression_model(_initialised_compression_model: CompressionModel):
    cm = copy.deepcopy(_initialised_compression_model)
    cm.fit()
    return cm


@pytest.fixture
def initialised_compression_model(_initialised_compression_model: CompressionModel):
    return copy.deepcopy(_initialised_compression_model)


@pytest.fixture
def fitted_compression_model(_fitted_compression_model: CompressionModel):
    with pytest.warns(
        DeprecationWarning,
        match=r"__array__ implementation doesn't accept a copy keyword",
    ):  # Keras doesn't accept NumPy 2.0's convention, to remove when problem solved.
        return copy.deepcopy(_fitted_compression_model)


@pytest.fixture
def default_compression_model():
    return CompressionModel()


class Test_check_layer_size_and_encoded_data_size:
    # valid cases
    def test_no_warn_for_valid_n_cols_first_layer(self):
        assert _check_layer_size_and_encoded_data_size(100, 10) is None

    def test_no_warn_for_n_cols_equals_first_layer_size(self):
        assert _check_layer_size_and_encoded_data_size(8, 8) is None

    def test_warn_when_n_cols_is_not_divisible_by_first_layer_size(self):
        with pytest.warns(UserWarning, match="is not a divisor of n_cols"):
            _check_layer_size_and_encoded_data_size(10, 3)

    def test_warn_when_n_cols_is_lower_than_first_layer_size(self):
        with pytest.warns(UserWarning, match="is not a divisor of n_cols"):
            _check_layer_size_and_encoded_data_size(5, 10)

    def test_warning_message(self):
        with pytest.warns(UserWarning, match="is not a divisor of n_cols") as warn_info:
            _check_layer_size_and_encoded_data_size(13, 7)

        expected_message = (
            "layer_sizes[0]=7 is not a divisor of n_cols=13. "
            "Column padding (filled with 0) will be added to the end of the data "
            "to fit requested layer_sizes[0]"
        )
        assert str(warn_info[0].message) == expected_message


class Test_check_layer_size_and_encoding_compatibility:
    def test_no_warn_for_valid_encoding_size_first_layer(self):
        assert _check_layer_size_and_encoding_compatibility(10, 5) is None

    def test_warn_when_encoding_size_equals_first_layer(self):
        with pytest.warns(
            UserWarning,
            match=r"layer_sizes\[0\]=8 is equal to `encoding_size`.",
        ) as warn_info:
            assert _check_layer_size_and_encoding_compatibility(8, 8) is None

        expected_message = (
            "layer_sizes[0]=8 is equal to `encoding_size` "
            "(ie. each chunk will consist of only 1 encoded allele)."
        )
        assert str(warn_info[0].message) == expected_message

    def test_warns_when_encoding_size_not_multiple_of_n_cols(self):
        with pytest.warns(UserWarning, match="is not a multiple of"):
            _check_layer_size_and_encoding_compatibility(10, 3)

    def test_warns_when_encoding_size_larger_than_first_layer(self):
        with pytest.warns(UserWarning, match="is not a multiple of"):
            _check_layer_size_and_encoding_compatibility(10, 20)

    def test_warning_message(self):
        with pytest.warns(UserWarning, match=r"is not a multiple of") as warn_info:
            _check_layer_size_and_encoding_compatibility(10, 7)

        expected_message = (
            "layer_sizes[0]=10 is not a multiple of encoding_size=7. "
            "(ie. each chunk will cut through encoded alleles, "
            "leaving incomplete encodings at chunk edges)"
        )
        assert str(warn_info[0].message) == expected_message


class Test_split_data:

    def test_return_list(self, basic_training_data: TrainingDataFixture):
        chunk_size = basic_training_data.layer_sizes[0]
        result = _split_data(basic_training_data.encoded_array, chunk_size)
        assert isinstance(result, list)

    def test_correct_number_of_chunks(self, basic_training_data: TrainingDataFixture):
        chunk_size = basic_training_data.layer_sizes[0]
        result = _split_data(basic_training_data.encoded_array, chunk_size)

        n_cols = basic_training_data.encoded_array.shape[1]
        assert len(result) == n_cols / chunk_size

    def test_correct_n_rows_in_chunks(self, basic_training_data: TrainingDataFixture):
        chunk_size = basic_training_data.layer_sizes[0]
        result = _split_data(basic_training_data.encoded_array, chunk_size)

        n_rows = basic_training_data.encoded_array.shape[0]
        assert all(chunk.shape[0] == n_rows for chunk in result)

    def test_correct_n_cols_in_chunks(self, basic_training_data: TrainingDataFixture):
        chunk_size = basic_training_data.layer_sizes[0]
        result = _split_data(basic_training_data.encoded_array, chunk_size)
        assert all(chunk.shape[1] == chunk_size for chunk in result)

    def test_can_reconstruct_original_from_chunks(
        self, basic_training_data: TrainingDataFixture
    ):
        chunk_size = basic_training_data.layer_sizes[0]
        result = _split_data(basic_training_data.encoded_array, chunk_size)

        reconstructed = np.hstack(result)
        assert np.array_equal(reconstructed, basic_training_data.encoded_array)

    def test_dtype_preserved(self, basic_training_data: TrainingDataFixture):
        chunk_size = basic_training_data.layer_sizes[0]
        result = _split_data(basic_training_data.encoded_array, chunk_size)
        assert all(
            chunk.dtype == basic_training_data.encoded_array.dtype for chunk in result
        )

    def test_single_chunks(self, basic_training_data: TrainingDataFixture):
        # ie. chunk_size is equal to the number of columns
        n_cols = basic_training_data.encoded_array.shape[1]
        result = _split_data(basic_training_data.encoded_array, n_cols)

        assert isinstance(result, list)
        assert len(result) == 1

    def test_with_single_row(self):
        arr = np.array([[1, 0, 0, 1, 0, 0]])
        result = _split_data(arr, chunk_size=2)
        assert len(result) == 3
        assert all(chunk.shape == (1, 2) for chunk in result)

    def test_chunk_size_one(self, basic_training_data: TrainingDataFixture):
        result = _split_data(basic_training_data.encoded_array, 1)

        n_cols = basic_training_data.encoded_array.shape[1]
        assert len(result) == n_cols

        n_rows = basic_training_data.encoded_array.shape[0]
        assert all(chunk.shape == (n_rows, 1) for chunk in result)

    def test_add_padding_for_incompatible_sizes(
        self, training_data_requiring_padding: TrainingDataFixture
    ):
        n_cols = training_data_requiring_padding.encoded_array.shape[1]
        chunk_size = training_data_requiring_padding.layer_sizes[0]

        with pytest.warns(UserWarning):
            result = _split_data(
                training_data_requiring_padding.encoded_array,
                training_data_requiring_padding.layer_sizes[0],
            )

        assert len(result) == (n_cols // chunk_size) + 1

        n_rows = training_data_requiring_padding.encoded_array.shape[0]
        assert all(chunk.shape == (n_rows, chunk_size) for chunk in result)

        reconstructed = np.hstack(result)
        assert np.array_equal(
            reconstructed[:, :n_cols], training_data_requiring_padding.encoded_array
        )
        padding_width = reconstructed.shape[1] - n_cols
        assert np.array_equal(
            reconstructed[:, n_cols:], np.zeros((n_rows, padding_width))
        )


class TestCompressionModel_basic_initialisation:
    def test_can_instanciate(self):
        CompressionModel()

    def test_default_parameters(self, default_compression_model):
        assert default_compression_model.training_encoded_geno_array is None
        assert default_compression_model.encoding_map is None
        assert default_compression_model.batch_size == 50
        assert default_compression_model.epochs == 200
        assert default_compression_model.learning_rate == 0.001
        assert default_compression_model.loss == "mse"
        assert default_compression_model.seed is None
        assert default_compression_model.training_size == 0.4
        assert default_compression_model.validation_size == 0.5
        assert default_compression_model.layer_sizes == []
        assert default_compression_model.training_markers_index is None

        assert isinstance(default_compression_model.fitting_callbacks, list)
        assert len(default_compression_model.fitting_callbacks) == 1
        assert isinstance(default_compression_model.fitting_callbacks[0], EarlyStopping)

    def test_default_initial_state(self, default_compression_model):
        assert default_compression_model.is_fitted is False
        assert default_compression_model.autoencoder_models == []
        assert default_compression_model.autoencoder_fits == []
        assert default_compression_model.autoencoder_evaluations == []
        assert default_compression_model.encoding_size is None
        assert default_compression_model.n_chunks == 0
        assert default_compression_model.chunk_size == 0

    def test_custom_parameters(self, basic_training_data: TrainingDataFixture):
        model = CompressionModel(
            training_encoded_geno_array=basic_training_data.encoded_array,
            encoding_map=basic_training_data.encoding_map,
            layer_sizes=basic_training_data.layer_sizes,
            batch_size=100,
            epochs=500,
            fitting_callbacks=[],
            learning_rate=0.01,
            loss="mae",
            seed=42,
            training_size=0.8,
            validation_size=0.2,
            training_markers_index=basic_training_data.dataframe.columns,
        )
        assert model.training_encoded_geno_array is not None
        assert np.array_equal(
            model.training_encoded_geno_array, basic_training_data.encoded_array
        )
        assert model.encoding_map == basic_training_data.encoding_map
        assert model.layer_sizes == basic_training_data.layer_sizes
        assert model.batch_size == 100
        assert model.epochs == 500
        assert model.fitting_callbacks == []
        assert model.learning_rate == 0.01
        assert model.loss == "mae"
        assert model.seed == 42
        assert model.training_size == 0.8
        assert model.validation_size == 0.2
        assert isinstance(model.training_markers_index, pd.Index)
        assert model.training_markers_index.equals(
            basic_training_data.dataframe.columns
        )

        assert model.is_fitted is False
        assert model.autoencoder_models == []
        assert model.autoencoder_fits == []
        assert model.autoencoder_evaluations == []
        assert model.encoding_size is len(
            next(iter(basic_training_data.encoding_map.values()))
        )
        assert model.n_chunks == 3
        assert model.chunk_size == 8

    def test_accept_tuple_layer_sizes_and_stored_as_list(self):
        model = CompressionModel(layer_sizes=(28, 14, 7))
        assert isinstance(model.layer_sizes, list)
        assert model.layer_sizes == [28, 14, 7]

    @pytest.mark.parametrize(
        "bad_layer_sizes",
        [[28], [], ()],
        ids=[
            "layer_sizes length 1",
            "layer_sizes list length 0",
            "layer_sizes tuple length 0",
        ],
    )
    def test_raises_with_invalid_layer_sizes(self, bad_layer_sizes):
        with pytest.raises(ValueError):
            CompressionModel(layer_sizes=bad_layer_sizes)

    def test_warns_with_incompatible_layer_size_and_data(
        self,
        basic_training_data: TrainingDataFixture,
    ):
        with pytest.warns(
            UserWarning,
            match="is not a divisor of n_cols",
        ):
            CompressionModel(
                training_encoded_geno_array=basic_training_data.encoded_array,
                layer_sizes=[7, 3, 1],
            )

    def test_raise_with_incompatible_index_and_training_data(
        self, basic_training_data: TrainingDataFixture
    ):
        with pytest.raises(
            ValueError,
            match=r"Marker index length missmatch training data size",
        ):
            CompressionModel(
                training_encoded_geno_array=basic_training_data.encoded_array,
                encoding_map=basic_training_data.encoding_map,
                training_markers_index=["A", "B"],
            )


class TestCompressionModel_initialisation_from_dataframe:
    def test_can_instanciate(self, basic_training_data: TrainingDataFixture):
        cm = CompressionModel.from_dataframe(
            training_dataframe=basic_training_data.dataframe
        )
        assert isinstance(cm, CompressionModel)

    def test_correct_encoding_with_defaults_extra_params(
        self, basic_training_data: TrainingDataFixture
    ):
        cm = CompressionModel.from_dataframe(
            training_dataframe=basic_training_data.dataframe,
        )

        expected_encoded_array = encode_snp_array(
            basic_training_data.dataframe.to_numpy()
        )
        assert cm.training_encoded_geno_array is not None
        np.testing.assert_array_equal(
            cm.training_encoded_geno_array, expected_encoded_array
        )

    def test_correct_encoding_map_with_defaults_extra_params(
        self, basic_training_data: TrainingDataFixture
    ):
        cm = CompressionModel.from_dataframe(
            training_dataframe=basic_training_data.dataframe,
        )

        assert cm.encoding_map == build_one_hot_encoding_map(
            basic_training_data.dataframe.to_numpy()
        )

    def test_correct_encoding_with_custom_encoding_map(
        self, basic_training_data: TrainingDataFixture
    ):
        encoding_map = {
            "A": [0.0, 1.0],
            "C": [1.0, 0.0],
            "T": [0.0, 1.0],
            "G": [1.0, 0.0],
        }
        cm = CompressionModel.from_dataframe(
            training_dataframe=basic_training_data.dataframe,
            encoding_map=encoding_map,
        )

        expected_encoded_array = encode_snp_array(
            basic_training_data.dataframe.to_numpy(), encoding_map=encoding_map
        )
        assert cm.training_encoded_geno_array is not None
        np.testing.assert_array_equal(
            cm.training_encoded_geno_array, expected_encoded_array
        )

    def test_correct_encoding_map_with_custom_encoding_map(
        self, basic_training_data: TrainingDataFixture
    ):
        encoding_map = {
            "A": [0.0, 1.0],
            "C": [1.0, 0.0],
            "T": [0.0, 1.0],
            "G": [1.0, 0.0],
        }
        cm = CompressionModel.from_dataframe(
            training_dataframe=basic_training_data.dataframe,
            encoding_map=encoding_map,
        )

        assert cm.encoding_map == encoding_map

    def test_correct_encoding_with_custom_missing_values(
        self, basic_training_data: TrainingDataFixture
    ):
        missing_values = ["A", "T"]
        cm = CompressionModel.from_dataframe(
            training_dataframe=basic_training_data.dataframe,
            missing_values=missing_values,
        )

        expected_encoded_array = encode_snp_array(
            basic_training_data.dataframe.to_numpy(),
            missing_values=missing_values,
        )
        assert cm.training_encoded_geno_array is not None
        np.testing.assert_array_equal(
            cm.training_encoded_geno_array, expected_encoded_array
        )

    def test_correct_encoding_map_with_custom_missing_values(
        self, basic_training_data: TrainingDataFixture
    ):
        missing_values = ["A", "T"]
        cm = CompressionModel.from_dataframe(
            training_dataframe=basic_training_data.dataframe,
            missing_values=missing_values,
        )

        assert cm.encoding_map == build_one_hot_encoding_map(
            geno_array=basic_training_data.dataframe.to_numpy(),
            exclude=missing_values,
        )

    def test_parameter_propagation(self, basic_training_data: TrainingDataFixture):
        cm = CompressionModel.from_dataframe(
            training_dataframe=basic_training_data.dataframe,
            epochs=111,
            learning_rate=0.5,
            validation_size=0.3,
        )
        assert cm.epochs == 111
        assert cm.learning_rate == 0.5
        assert cm.validation_size == 0.3

    def test_different_dtypes_handling(self):
        geno_data = pd.DataFrame(
            [
                [0, 1, 2],
                [2, -1, 1],
                [0, np.nan, pd.NA],
            ],
            index=["ind_1", "ind_2", "ind_3"],
            columns=["snp_1", "snp_2", "snp_3"],
        )
        cm = CompressionModel.from_dataframe(
            training_dataframe=geno_data, missing_values=[-1]
        )

        # correct encoding
        expected_encoded_array = encode_snp_array(
            geno_data.to_numpy(), missing_values=[-1]
        )
        assert cm.training_encoded_geno_array is not None
        np.testing.assert_array_equal(
            cm.training_encoded_geno_array, expected_encoded_array
        )

        # correct encoding_map
        assert cm.encoding_map == build_one_hot_encoding_map(
            geno_array=geno_data.to_numpy(),
            exclude=[-1],
        )

    def test_raise_with_empty_dataframe(self):
        geno_data = pd.DataFrame()

        with pytest.raises(
            ValueError,
            match=re.escape(
                "`training_dataframe` is empty. Provide a DataFrame with at least "
                "one row and one column."
            ),
        ):
            CompressionModel.from_dataframe(training_dataframe=geno_data)

    def test_training_markers_index_is_correctly_set(
        self, basic_training_data: TrainingDataFixture
    ):
        cm = CompressionModel.from_dataframe(
            training_dataframe=basic_training_data.dataframe,
        )
        assert cm.training_markers_index is not None
        assert cm.training_markers_index.equals(basic_training_data.dataframe.columns)

    def test_encoding_map_is_validated(
        self, basic_training_data: TrainingDataFixture, mocker: MockerFixture
    ):
        mock_validate_encoding_map = mocker.patch(
            "deepcgp.data_processing._validate_encoding_map",
            wraps=_validate_encoding_map,
        )

        encoding_map = {
            "A": [0.0, 1.0],
            "C": [1.0, 0.0],
            "T": [0.0, 1.0],
            "G": [1.0, 0.0],
        }
        missing_values = {"Z", "Y"}
        CompressionModel.from_dataframe(
            training_dataframe=basic_training_data.dataframe,
            encoding_map=encoding_map,
            missing_values=missing_values,
        )
        mock_validate_encoding_map.assert_called_once_with(encoding_map, missing_values)

    def test_raise_error_if_training_encoded_geno_array_is_provided(
        self, basic_training_data: TrainingDataFixture
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "`training_encoded_geno_array` cannot be passed as a keyword argument "
                "to from_dataframe(); it is derived from `training_dataframe`."
            ),
        ):
            CompressionModel.from_dataframe(
                training_dataframe=basic_training_data.dataframe,
                training_encoded_geno_array=None,
            )

    def test_with_only_missing_values(self, basic_training_data: TrainingDataFixture):
        missing_values = set(basic_training_data.dataframe.to_numpy().ravel())
        encoding_map = {  # dummy encoding_map because it cannot be empty
            "Z": [0.0, 1.0],
        }

        with pytest.warns(
            UserWarning,
            match=(
                "The encoded array contains only zeros. This may indicate that all "
                "values in geno_array are missing or not present in encoding_map."
            ),
        ):
            cm = CompressionModel.from_dataframe(
                training_dataframe=basic_training_data.dataframe,
                encoding_map=encoding_map,
                missing_values=missing_values,
            )

        # correct encoding
        with pytest.warns(
            UserWarning,
            match=(
                "The encoded array contains only zeros. This may indicate that all "
                "values in geno_array are missing or not present in encoding_map."
            ),
        ):
            expected_encoded_array = encode_snp_array(
                basic_training_data.dataframe.to_numpy(),
                missing_values=missing_values,
                encoding_map=encoding_map,
            )
        assert cm.training_encoded_geno_array is not None
        np.testing.assert_array_equal(
            cm.training_encoded_geno_array, expected_encoded_array
        )
        assert set(np.unique(cm.training_encoded_geno_array)) == {0.0}

        # correct encoding_map
        assert cm.encoding_map == encoding_map

    def test_warning_when_incompatible_layer_sizes_passed_in_kwargs(
        self, basic_training_data: TrainingDataFixture
    ):
        # Just a smoke test, full validation done is done in
        # TestCompressionModel_layer_sizes and
        # TestCompressionModel_training_encoded_geno_array
        with pytest.warns(UserWarning) as warnings:
            CompressionModel.from_dataframe(
                training_dataframe=basic_training_data.dataframe,
                layer_sizes=[7, 3, 1],
            )

        messages = [str(w.message) for w in warnings]
        assert any("is not a multiple of encoding_size" in m for m in messages)
        assert any("is not a divisor of n_cols" in m for m in messages)


class TestCompressionModel_layer_sizes:
    def test_correct_value(self, default_compression_model: CompressionModel):
        default_compression_model.layer_sizes = [7, 3, 1]
        assert default_compression_model.layer_sizes == [7, 3, 1]

    def test_setter_warn_when_incompatible_with_encoded_data(
        self,
        basic_training_data: TrainingDataFixture,
    ):
        cm = CompressionModel(
            training_encoded_geno_array=basic_training_data.encoded_array
        )
        with pytest.warns(
            UserWarning,
            match="is not a divisor of n_cols",
        ):
            cm.layer_sizes = [7, 3, 1]


class TestCompressionModel_training_encoded_geno_array:
    def test_setter_warns_when_incompatible_with_layer_size(
        self,
        basic_training_data: TrainingDataFixture,
        default_compression_model: CompressionModel,
    ):
        default_compression_model.layer_sizes = [7, 3, 1]
        with pytest.warns(
            UserWarning,
            match="is not a divisor of n_cols",
        ):
            default_compression_model.training_encoded_geno_array = (
                basic_training_data.encoded_array
            )

    def test_setter_raise_when_incompatible_with_marker_index(
        self, basic_training_data: TrainingDataFixture
    ):
        cm = CompressionModel(
            encoding_map=basic_training_data.encoding_map,
            training_markers_index=["A", "B"],
        )
        with pytest.raises(
            ValueError,
            match=r"Marker index length missmatch training data size",
        ):
            cm.training_encoded_geno_array = basic_training_data.encoded_array

    def test_reset(self, initialised_compression_model: CompressionModel):
        cm = initialised_compression_model
        cm.training_encoded_geno_array = None

        assert cm.training_encoded_geno_array is None
        assert cm.n_chunks == 0
        assert cm._n_col_train == 0


class TestCompressionModel_training_markers_index:
    @pytest.fixture(
        params=[
            pytest.param(["A", "B", "C"], id="list"),
            pytest.param(np.array(["D", "E"]), id="np.array"),
            pytest.param(pd.Index(["F"]), id="pd.Index"),
        ]
    )
    def index_of_various_types(self, request):
        # Note: they do not match with basic_training_data size
        return request.param

    def test_setter_accept_none(self):
        cm = CompressionModel()
        cm.training_markers_index = None
        assert cm.training_markers_index is None

    def test_setter_accept_different_types(self, index_of_various_types):
        cm = CompressionModel()
        cm.training_markers_index = index_of_various_types
        training_markers_index = cm.training_markers_index
        assert isinstance(training_markers_index, pd.Index)
        assert training_markers_index.equals(pd.Index(index_of_various_types))

    def test_setter_raise_if_incompatible_with_training_data(
        self, initialised_compression_model, index_of_various_types
    ):
        cm = initialised_compression_model
        with pytest.raises(
            ValueError,
            match=r"Marker index length missmatch training data size",
        ):
            cm.training_markers_index = index_of_various_types

    def test_reset(self, initialised_compression_model: CompressionModel):
        cm = initialised_compression_model
        assert cm.training_markers_index is not None

        cm.training_markers_index = None
        assert cm.training_markers_index is None


class TestCompressionModel_encoding_map:
    def test_setter_validate_map(self):
        cm = CompressionModel()
        with pytest.raises(ValueError, match=r"`encoding_map` is empty."):
            cm.encoding_map = {}

    def test_setter_warns_with_incompatible_layer_size(
        self, basic_training_data: TrainingDataFixture
    ):
        cm = CompressionModel(
            training_encoded_geno_array=basic_training_data.encoded_array,
            layer_sizes=[8, 4, 2],
        )
        # Specify encoding map witht size 3.
        # n_cols (24) is divisible by 3 (8 markers).
        # but chunk_size (8) is not a multiple of 3.
        bad_encoding_map = {"A": [1.0, 0.0, 0.0]}
        with pytest.warns(
            UserWarning,
            match="is not a multiple of encoding_size",
        ):
            cm.encoding_map = bad_encoding_map

    def test_setter_raise_with_incompatible_marker_index(
        self, basic_training_data: TrainingDataFixture
    ):
        cm = CompressionModel(
            training_encoded_geno_array=basic_training_data.encoded_array,
            encoding_map=basic_training_data.encoding_map,
            training_markers_index=basic_training_data.dataframe.columns,
        )

        # Encoding map with size 2.
        # Expected markers = 24/2 = 12 != 6
        bad_encoding_map = {"A": [0.0, 0.0]}
        with pytest.raises(
            ValueError, match=r"Marker index length missmatch training data size"
        ):
            cm.encoding_map = bad_encoding_map

    def test_reset(self, initialised_compression_model: CompressionModel):
        cm = initialised_compression_model
        assert cm.encoding_map is not None

        cm.encoding_map = None
        assert cm.encoding_map is None


class TestCompressionModel_fit:
    def test_raise_with_no_training_data_and_no_layer_sizes(
        self, default_compression_model: CompressionModel
    ):
        with pytest.raises(
            RuntimeError,
            match=r"No training data available\.",
        ):
            default_compression_model.fit()

    def test_raise_with_no_training_data(
        self, default_compression_model: CompressionModel
    ):
        default_compression_model.layer_sizes = [8, 4, 2]
        with pytest.raises(
            RuntimeError,
            match=r"No training data available\.",
        ):
            default_compression_model.fit()

    def test_raise_with_no_layer_sizes(
        self,
        default_compression_model: CompressionModel,
        basic_training_data: TrainingDataFixture,
    ):
        default_compression_model.training_encoded_geno_array = (
            basic_training_data.encoded_array
        )
        with pytest.raises(
            RuntimeError,
            match=r"layer_sizes is not set\.",
        ):
            default_compression_model.fit()

    def test_is_fitted_becomes_true(self, fitted_compression_model: CompressionModel):
        assert fitted_compression_model.is_fitted

    def test_autoencoders_models_are_set(
        self, fitted_compression_model: CompressionModel
    ):
        assert (
            len(fitted_compression_model.autoencoder_models)
            == fitted_compression_model.n_chunks
        )
        for aem in fitted_compression_model.autoencoder_models:
            assert isinstance(aem, AutoencoderModels)
        assert all(aem.is_fitted for aem in fitted_compression_model.autoencoder_models)

    def test_autoencoder_fits_are_set(self, fitted_compression_model: CompressionModel):
        assert (
            len(fitted_compression_model.autoencoder_fits)
            == fitted_compression_model.n_chunks
        )
        for fit in fitted_compression_model.autoencoder_fits:
            assert isinstance(fit, History)

    def test_autoencoder_evaluations_are_set(
        self, fitted_compression_model: CompressionModel
    ):
        assert (
            len(fitted_compression_model.autoencoder_evaluations)
            == fitted_compression_model.n_chunks
        )
        for evaluation in fitted_compression_model.autoencoder_evaluations:
            assert isinstance(evaluation, dict)
            assert list(evaluation.keys()) == [fitted_compression_model.loss]

    def test_autoencoders_related_lists_do_not_accumulate(
        self, fitted_compression_model: CompressionModel
    ):
        fitted_compression_model.fit()

        assert (
            len(fitted_compression_model.autoencoder_models)
            == fitted_compression_model.n_chunks
        )

        assert (
            len(fitted_compression_model.autoencoder_fits)
            == fitted_compression_model.n_chunks
        )

        assert (
            len(fitted_compression_model.autoencoder_evaluations)
            == fitted_compression_model.n_chunks
        )

    def test_AutoencoderModels_is_correctly_called(
        self, basic_training_data: TrainingDataFixture, mocker: MockerFixture
    ):
        def fake_aem(layer_sizes):
            aem = mocker.MagicMock()
            aem.mocked_layer_sizes = layer_sizes
            return aem

        mock_aem_class = mocker.patch(
            "deepcgp.compression.AutoencoderModels", side_effect=fake_aem
        )

        model = CompressionModel(
            training_encoded_geno_array=basic_training_data.encoded_array,
            layer_sizes=basic_training_data.layer_sizes,
        )
        model.fit()

        assert mock_aem_class.call_count == model.n_chunks
        for c in mock_aem_class.call_args_list:
            assert c == call(basic_training_data.layer_sizes)

        for m in model.autoencoder_models:
            assert (
                m.mocked_layer_sizes  # pyright: ignore [reportAttributeAccessIssue]
                == basic_training_data.layer_sizes
            )

    def test_generate_and_store_seed_if_none(self, basic_training_data):
        model = CompressionModel(
            training_encoded_geno_array=basic_training_data.encoded_array,
            layer_sizes=basic_training_data.layer_sizes,
        )
        assert model.seed is None
        model.fit()
        assert model.seed is not None

    def test_split_test_train_uses_same_seed_for_each_autoencoder(
        self, basic_training_data, mocker: MockerFixture
    ):
        # ideally we should test that the same rows index are used for
        # each autoencoder: eg.
        # x_train is always rows "2, 4, 6, 7" for each chunk
        # but this is difficult to test so here we check train_test_split is called
        # with the same seed for each chunk
        # it could break with new implementation
        model = CompressionModel(
            training_encoded_geno_array=basic_training_data.encoded_array,
            layer_sizes=basic_training_data.layer_sizes,
        )

        mock_test_train_split = mocker.patch(
            "deepcgp.compression.train_test_split", wraps=train_test_split
        )
        model.fit()
        calls = mock_test_train_split.call_args_list
        if len(calls) != 2 * model.n_chunks:
            # be sure the model.fit() calls train_test_split twice,
            # one for autoencoder's fit x_train / x_validate
            # and once for evaluation
            pytest.skip("train_test_split doesn't seems to have been called twice")
        first_split_seeds = [c.kwargs["random_state"] for c in calls[::2]]
        second_split_seeds = [c.kwargs["random_state"] for c in calls[1::2]]

        assert len(set(first_split_seeds)) == 1
        assert len(set(second_split_seeds)) == 1

    def test_split_test_train_uses_provided_seed(
        self, basic_training_data, mocker: MockerFixture
    ):
        rng_seed = 1111
        model = CompressionModel(
            training_encoded_geno_array=basic_training_data.encoded_array,
            layer_sizes=basic_training_data.layer_sizes,
            seed=rng_seed,
        )

        mock_test_train_split = mocker.patch(
            "deepcgp.compression.train_test_split", wraps=train_test_split
        )
        model.fit()
        calls = mock_test_train_split.call_args_list
        if len(calls) != 2 * model.n_chunks:
            # be sure the model.fit() calls train_test_split twice,
            # one for autoencoder's fit x_train / x_validate
            # and once for evaluation
            pytest.skip("train_test_split doesn't seems to have been called twice")
        first_split_seeds = [c.kwargs["random_state"] for c in calls[::2]]
        second_split_seeds = [c.kwargs["random_state"] for c in calls[1::2]]

        assert first_split_seeds[0] == rng_seed
        assert second_split_seeds[0] == rng_seed

    def test_split_test_train_uses_provided_seed_edge_case_with_seed_0(
        self, basic_training_data, mocker: MockerFixture
    ):
        rng_seed = 0
        model = CompressionModel(
            training_encoded_geno_array=basic_training_data.encoded_array,
            layer_sizes=basic_training_data.layer_sizes,
            seed=rng_seed,
        )

        mock_test_train_split = mocker.patch(
            "deepcgp.compression.train_test_split", wraps=train_test_split
        )
        model.fit()
        calls = mock_test_train_split.call_args_list
        if len(calls) != 2 * model.n_chunks:
            # be sure the model.fit() calls train_test_split twice,
            # one for autoencoder's fit x_train / x_validate
            # and once for evaluation
            pytest.skip("train_test_split doesn't seems to have been called twice")
        first_split_seeds = [c.kwargs["random_state"] for c in calls[::2]]
        second_split_seeds = [c.kwargs["random_state"] for c in calls[1::2]]

        assert first_split_seeds[0] == rng_seed
        assert second_split_seeds[0] == rng_seed

    def test_passed_seed_take_precedence(
        self, basic_training_data, mocker: MockerFixture
    ):
        rng_seed = 1111
        model = CompressionModel(
            training_encoded_geno_array=basic_training_data.encoded_array,
            layer_sizes=basic_training_data.layer_sizes,
            seed=2222,
        )

        mock_test_train_split = mocker.patch(
            "deepcgp.compression.train_test_split", wraps=train_test_split
        )
        model.fit(seed=rng_seed)
        calls = mock_test_train_split.call_args_list
        if len(calls) != 2 * model.n_chunks:
            # be sure the model.fit() calls train_test_split twice,
            # one for autoencoder's fit x_train / x_validate
            # and once for evaluation
            pytest.skip("train_test_split doesn't seems to have been called twice")
        first_split_seeds = [c.kwargs["random_state"] for c in calls[::2]]
        second_split_seeds = [c.kwargs["random_state"] for c in calls[1::2]]

        assert first_split_seeds[0] == rng_seed
        assert second_split_seeds[0] == rng_seed

    def test_model_seed_is_updated(self, basic_training_data):
        model = CompressionModel(
            training_encoded_geno_array=basic_training_data.encoded_array,
            layer_sizes=basic_training_data.layer_sizes,
            seed=1,
        )
        model.fit(seed=42)
        assert model.seed == 42

    def test_with_validation_size_set_to_float_1(
        self, basic_training_data: TrainingDataFixture
    ):
        # ie. No evaluation set
        model = CompressionModel(
            training_encoded_geno_array=basic_training_data.encoded_array,
            layer_sizes=basic_training_data.layer_sizes,
            validation_size=1.0,
        )
        model.fit()
        assert len(model.autoencoder_evaluations) == model.n_chunks
        assert all(evaluation is None for evaluation in model.autoencoder_evaluations)


class TestCompressionModel_compress:
    def test_raise_when_model_is_not_fitted(
        self,
        default_compression_model: CompressionModel,
        basic_training_data: TrainingDataFixture,
    ):
        with pytest.raises(
            RuntimeError,
            match=r"Model is not fitted\.",
        ):
            default_compression_model.compress(basic_training_data.encoded_array)

    def test_raise_when_n_cols_do_not_match_training_data(
        self,
        fitted_compression_model: CompressionModel,
    ):
        with pytest.raises(
            ValueError,
            match=(
                r"^Incompatible data. Provided data have a different number of columns "
            ),
        ):
            fitted_compression_model.compress(np.array([[1, 0, 0, 1, 0, 0]]))

    def test_output_shape(
        self,
        fitted_compression_model: CompressionModel,
        basic_training_data: TrainingDataFixture,
    ):
        result = fitted_compression_model.compress(basic_training_data.encoded_array)
        expected_n_cols = (
            fitted_compression_model.layer_sizes[-1] * fitted_compression_model.n_chunks
        )
        assert result.shape == (
            basic_training_data.encoded_array.shape[0],
            expected_n_cols,
        )

    def test_output_dtype(
        self,
        fitted_compression_model: CompressionModel,
        basic_training_data: TrainingDataFixture,
    ):
        result = fitted_compression_model.compress(basic_training_data.encoded_array)
        assert result.dtype == np.float32


class TestCompressionModel_compress_dataframe:

    def test_raise_when_model_is_not_fitted(
        self,
        default_compression_model: CompressionModel,
        basic_training_data: TrainingDataFixture,
    ):
        with pytest.raises(
            RuntimeError,
            match=r"Model is not fitted\.",
        ):
            default_compression_model.compress_dataframe(basic_training_data.dataframe)

    def test_raise_when_column_index_does_not_match_training_data_index(
        self,
        fitted_compression_model: CompressionModel,
        basic_training_data: TrainingDataFixture,
    ):
        """Columns present but with different names should raise."""
        bad_df = basic_training_data.dataframe.copy()
        bad_df.columns = [f"wrong_marker_{i}" for i in range(bad_df.shape[1])]
        with pytest.raises(
            ValueError,
            match=r"^Incompatible data. Provided data have different column index",
        ):
            fitted_compression_model.compress_dataframe(bad_df)

    def test_raise_when_columns_are_subset_of_training_markers(
        self,
        fitted_compression_model: CompressionModel,
        basic_training_data: TrainingDataFixture,
    ):
        """Different number of columns should raise.

        With `training_markers_index` error is about the index.
        """
        partial_df = basic_training_data.dataframe.iloc[:, :-1]
        with pytest.raises(
            ValueError,
            match=r"^Incompatible data. Provided data have different column index",
        ):
            fitted_compression_model.compress_dataframe(partial_df)

    def test_raise_when_columns_are_subset_of_training_markers_even_without_saved_index(
        self,
        fitted_compression_model: CompressionModel,
        basic_training_data: TrainingDataFixture,
    ):
        """Different number of columns should raise.

        Without `training_markers_index` error is about the number of columns.
        """
        partial_df = basic_training_data.dataframe.iloc[:, :-1]

        fitted_compression_model.training_markers_index = None  # remove index
        with pytest.raises(
            ValueError,
            match=(
                r"^Incompatible data. Provided data have a different number of columns "
            ),
        ):
            fitted_compression_model.compress_dataframe(partial_df)

    def test_column_reordering(
        self,
        fitted_compression_model: CompressionModel,
        basic_training_data: TrainingDataFixture,
        mocker: MockerFixture,
    ):
        """Shuffled columns should be reordered internally before
        encoding/compression."""
        shuffled_df = basic_training_data.dataframe.sample(
            frac=1, axis=1, random_state=0
        )
        assert not shuffled_df.columns.equals(basic_training_data.dataframe.columns)

        mock_encode_snp_array = mocker.patch(
            "deepcgp.compression.encode_snp_array",
            wraps=encode_snp_array,
        )
        fitted_compression_model.compress_dataframe(shuffled_df)

        mock_encode_snp_array.assert_called_once()
        np.testing.assert_array_equal(
            mock_encode_snp_array.call_args.args[0],
            basic_training_data.dataframe.to_numpy(),
        )

    def test_result_matches_compress_on_manually_encoded_array(
        self,
        fitted_compression_model: CompressionModel,
        basic_training_data: TrainingDataFixture,
    ):
        """compress_dataframe should be equivalent to manually encoding then calling
        compress."""
        result_dataframe = fitted_compression_model.compress_dataframe(
            basic_training_data.dataframe
        )
        result_array = fitted_compression_model.compress(
            basic_training_data.encoded_array
        )
        np.testing.assert_array_almost_equal(result_dataframe, result_array)

    @pytest.mark.parametrize(
        "encoding_map, expected_encoding_map",
        [
            (None, "USE_MODEL_ENCODING_MAP"),
            (
                {"A": [0, 0, 0, 1], "B": [1, 0, 0, 0]},
                {"A": [0, 0, 0, 1], "B": [1, 0, 0, 0]},
            ),
        ],
        ids=["None", "custom"],
    )
    def test_use_correct_encoding_map_when_CM_have_encoding_map(
        self,
        fitted_compression_model: CompressionModel,
        basic_training_data: TrainingDataFixture,
        encoding_map,
        expected_encoding_map,
        mocker: MockerFixture,
    ):
        """When the CompressionModel instance have the encoding map, it should be used
        when provided `encoding_map` is None, else use provided `encoding_map`."""
        if expected_encoding_map == "USE_MODEL_ENCODING_MAP":
            expected_encoding_map = fitted_compression_model.encoding_map

        mock_encode_snp_array = mocker.patch(
            "deepcgp.compression.encode_snp_array",
            wraps=encode_snp_array,
        )

        fitted_compression_model.compress_dataframe(
            basic_training_data.dataframe, encoding_map=encoding_map
        )
        mock_encode_snp_array.assert_called_once()
        assert (
            mock_encode_snp_array.call_args.kwargs["encoding_map"]
            == expected_encoding_map
        )

    @pytest.mark.parametrize(
        "encoding_map",
        [None, {"A": [0, 0, 0, 1], "B": [1, 0, 0, 0]}],
        ids=["None", "custom"],
    )
    def test_forward_encoding_map_when_CM_do_not_have_encoding_map(
        self,
        fitted_compression_model: CompressionModel,
        basic_training_data: TrainingDataFixture,
        encoding_map,
        mocker: MockerFixture,
    ):
        """When the CompressionModel instance do not have the encoding map, it pass the
        provided `encoding_map` to encode_snp_array."""
        fitted_compression_model.encoding_map = None

        mock_encode_snp_array = mocker.patch(
            "deepcgp.compression.encode_snp_array",
            wraps=encode_snp_array,
        )

        fitted_compression_model.compress_dataframe(
            basic_training_data.dataframe, encoding_map=encoding_map
        )
        mock_encode_snp_array.assert_called_once()
        assert mock_encode_snp_array.call_args.kwargs["encoding_map"] == encoding_map

    def test_default_missing_value(
        self,
        fitted_compression_model: CompressionModel,
        basic_training_data: TrainingDataFixture,
        mocker: MockerFixture,
    ):
        mock_encode_snp_array = mocker.patch(
            "deepcgp.compression.encode_snp_array",
            wraps=encode_snp_array,
        )

        fitted_compression_model.compress_dataframe(basic_training_data.dataframe)
        mock_encode_snp_array.assert_called_once()
        assert mock_encode_snp_array.call_args.kwargs["missing_values"] == {"N"}

    def test_forward_missing_value(
        self,
        fitted_compression_model: CompressionModel,
        basic_training_data: TrainingDataFixture,
        mocker: MockerFixture,
    ):
        mock_encode_snp_array = mocker.patch(
            "deepcgp.compression.encode_snp_array",
            wraps=encode_snp_array,
        )

        fitted_compression_model.compress_dataframe(
            basic_training_data.dataframe, missing_values={"-", "."}
        )
        mock_encode_snp_array.assert_called_once()
        assert mock_encode_snp_array.call_args.kwargs["missing_values"] == {"-", "."}

    def test_raise_if_missing_values_is_inconsitent_with_saved_encoding_map(
        self,
        fitted_compression_model: CompressionModel,
        basic_training_data: TrainingDataFixture,
    ):
        with pytest.raises(
            ValueError,
            match=r"Missing values not encoded with a vector of 0",
        ):
            fitted_compression_model.compress_dataframe(
                basic_training_data.dataframe, missing_values={"A"}
            )


def test_public_api_exports():
    import deepcgp

    assert hasattr(deepcgp, "CompressionModel")
    assert hasattr(deepcgp, "AutoencoderModels")
