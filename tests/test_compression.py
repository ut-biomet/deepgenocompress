import re

import pytest
from keras import Model

from deepcgp.compression import AutoencoderModels, compress_data

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
    """Tests for `AutoencoderModels` function"""

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
        expected = sum(
            [
                1,  # input layer
                len(layer_sizes) - 2,  # ie. - input layer - latent layer
                1,  # latent layer
            ]
        )
        assert len(basic_autoencoder_models.encoder.layers) == expected

    def test_encoder_share_same_first_layers_of_autoencoder(
        self, basic_autoencoder_models
    ):
        aem = basic_autoencoder_models
        for enc_layer, ae_layer in zip(aem.encoder.layers, aem.autoencoder.layers):
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
        layers_sizes = [20, 16, 8, 1]
        aem = AutoencoderModels(layers_sizes)
        assert aem.encoder.output_shape == (None, 1)
        assert aem.autoencoder.output_shape == (None, layers_sizes[0])

    def test_input_dim_one(self):
        aem = AutoencoderModels([1, 16, 8, 4])
        assert aem.autoencoder.input_shape == (None, 1)
        assert aem.autoencoder.output_shape == (None, 1)

    def test_no_compression(self):
        aem = AutoencoderModels([10, 10, 10, 10])
        assert aem.encoder.output_shape == (None, 10)
        assert aem.autoencoder.output_shape == (None, 10)

    def test_raise_if_layers_sizes_is_string(self):
        with pytest.raises(
            TypeError,
            match="layers_sizes must be a list, or tuple of int got `<class 'str'>`",
        ):
            AutoencoderModels("987")  # pyright: ignore [reportArgumentType]

    def test_raise_if_layers_sizes_lenght_is_lower_than_2(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "layers_sizes length must be greater than 2, got `len(layers_sizes)=1`"
            ),
        ):
            AutoencoderModels([42])

    def test_raise_if_layers_sizes_are_not_integers(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "layers_sizes must be a list, or tuple of int got `<class 'float'>` "
                "for index layers_sizes[1]."
            ),
        ):
            AutoencoderModels([42, 3.5])  # pyright: ignore [reportArgumentType]
