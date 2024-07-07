import numpy as np
import pytest

from src.ml_model import MlModel
from src.model_builder import ModelBuilder
from src.model_registry import ModelRegistry
from src.model_serializer import ModelSerializer


class TestEvolution:

    def test_adjust_array_shape(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(AssertionError):
            ModelBuilder.adjust_array_shape(x, -1, 1)
        with pytest.raises(AssertionError):
            ModelBuilder.adjust_array_shape(x, 2, 1)
        with pytest.raises(AssertionError):
            ModelBuilder.adjust_array_shape(x, 0, 0)
        y = ModelBuilder.adjust_array_shape(x, 0, 1)
        assert np.array_equal(y, [[1, 2, 3]])
        y = ModelBuilder.adjust_array_shape(x, 1, 1)
        assert np.array_equal(y, [[1], [4]])
        y = ModelBuilder.adjust_array_shape(x, 0, 2)
        assert np.array_equal(y, [[1, 2, 3], [4, 5, 6]])
        y = ModelBuilder.adjust_array_shape(x, 0, 4)
        assert np.shape(y) == (4, 3)
        assert np.array_equal(y[:2, :], [[1, 2, 3], [4, 5, 6]])
        y = ModelBuilder.adjust_array_shape(x, 1, 5)
        assert np.shape(y) == (2, 5)
        assert np.array_equal(y[:, :3], [[1, 2, 3], [4, 5, 6]])

    @pytest.fixture
    def builder(self):
        return ModelBuilder(10, 11, 12, 13)

    @pytest.fixture
    def complex_model(self, builder):
        model_1 = builder.build_model(asset_dependent=True)
        model_2 = builder.build_model(asset_dependent=False)
        return builder.merge_models(model_1, model_2)

    def test_adjust_weights_shape(self, builder: ModelBuilder, complex_model: MlModel):
        model = complex_model
        weights = model.model.get_weights()[:2]
        input_size, output_size = np.shape(weights[0])
        with pytest.raises(AssertionError):
            builder.adjust_weights_shape(model.model.get_weights(), (input_size, output_size))
        with pytest.raises(AssertionError):
            builder.adjust_weights_shape(weights, (0, output_size))
        with pytest.raises(AssertionError):
            builder.adjust_weights_shape(weights, (input_size, 0))
        new_weights = builder.adjust_weights_shape(weights, (input_size, output_size))
        for w1, w2 in zip(weights, new_weights):
            assert np.array_equal(w1, w2)
        new_weights = builder.adjust_weights_shape(weights, (input_size - 2, output_size))
        assert np.array_equal(weights[0][:-2, :], new_weights[0])
        assert np.array_equal(weights[1], new_weights[1])
        new_weights = builder.adjust_weights_shape(weights, (input_size + 2, output_size))
        assert np.array_equal(weights[0], new_weights[0][:-2, :])
        assert abs(new_weights[0][-2:, :].mean()) < 1
        assert new_weights[0][-2:, :].std() > 0
        assert np.array_equal(weights[1], new_weights[1])
        new_weights = builder.adjust_weights_shape(weights, (input_size, output_size - 2))
        assert np.array_equal(weights[0][:, :-2], new_weights[0])
        assert np.array_equal(weights[1][:-2], new_weights[1])
        new_weights = builder.adjust_weights_shape(weights, (input_size, output_size + 2))
        assert np.array_equal(weights[0], new_weights[0][:, :-2])
        assert abs(new_weights[0][:, -2:].mean()) < 1
        assert new_weights[0][:, -2:].std() > 0
        assert np.array_equal(weights[1], new_weights[1][:-2])
        weights[0] = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        weights[1] = np.zeros(4)
        new_weights = builder.adjust_weights_shape(weights, (4, 3, 2))
        assert np.shape(new_weights[0]) == (4, 3, 2)
        assert np.shape(new_weights[1]) == (2,)

    def test_adjust_n_assets(self, builder: ModelBuilder, complex_model: MlModel):
        model = complex_model
        layers = model.get_layers()
        builder.n_assets = 14
        model2 = builder.adjust_n_assets(model)
        layers2 = model2.get_layers()
        assert layers[0].input_shape == (10, 11, 12)
        assert layers2[0].input_shape == (10, 14, 12)
        assert len(layers) == len(layers2)
        input = np.zeros([*layers2[0].input_shape])
        output = model2.predict(input)
        assert np.shape(output) == (14, 13)

    def test_adjust_n_features(self, builder: ModelBuilder, complex_model: MlModel):
        model = complex_model
        layers = model.get_layers()
        builder.n_features = 14
        model2 = builder.adjust_n_features(model)
        layers2 = model2.get_layers()
        assert layers[0].input_shape == (10, 11, 12)
        assert layers2[0].input_shape == (10, 11, 14)
        assert len(layers) == len(layers2)
        input = np.zeros([*layers2[0].input_shape])
        output = model2.predict(input)
        assert np.shape(output) == (11, 13)

    def test_remove_layer(self, builder: ModelBuilder, complex_model: MlModel):
        model = complex_model
        layers = model.get_layers()
        for index in range(len(layers) - 1):
            for length in range(1, 3):
                if index + length - 1 > len(layers) - 2:
                    with pytest.raises(AssertionError):
                        model2 = builder.remove_layer(model, index, index + length - 1)
                    continue
                model2 = builder.remove_layer(model, index, index + length - 1)
                layers2 = model2.get_layers()
                if index in [2, 3, 4, 5, 6] or index + length > 11:
                    assert len(layers) == len(layers2)
                    assert [x.layer_type for x in layers] == [x.layer_type for x in layers2]
                else:
                    assert len(layers) == len(layers2) + length
                    assert sorted(
                        [x.layer_type for x in layers[:index]] + [x.layer_type for x in layers[index + length :]]
                    ) == sorted([x.layer_type for x in layers2])
                input = np.zeros([*layers2[0].input_shape])
                output = model2.predict(input)
                assert np.shape(output) == (11, 13)

    def test_add_dense_layer(self, builder: ModelBuilder, complex_model: MlModel):
        model = complex_model
        layers = model.get_layers()
        for index in range(len(layers)):
            model2 = builder.add_dense_layer(model, index, 111)
            layers2 = model2.get_layers()
            if index in [3, 11]:
                assert len(layers) == len(layers2)
                assert [x.layer_type for x in layers] == [x.layer_type for x in layers2]
            else:
                assert len(layers) + 1 == len(layers2)
                assert sorted(
                    [x.layer_type for x in layers[:index]] + ["dense"] + [x.layer_type for x in layers[index:]]
                ) == sorted([x.layer_type for x in layers2])
                assert len([1 for l in layers2 if l.layer_type == "dense" and l.shape[-1] == 111]) == 1
            input = np.zeros([*layers2[0].input_shape])
            output = model2.predict(input)
            assert np.shape(output) == (11, 13)

    def test_add_conv_layer(self, builder: ModelBuilder, complex_model: MlModel):
        model = complex_model
        model = builder.add_conv_layer(model, 0)
        layers = model.get_layers()
        for index in range(len(layers)):
            model2 = builder.add_conv_layer(model, index)
            layers2 = model2.get_layers()
            if index in [5, 6, 7, 12]:
                assert len(layers) == len(layers2)
                assert [x.layer_type for x in layers] == [x.layer_type for x in layers2]
            else:
                assert len(layers) + 1 == len(layers2) or len(layers) + 3 == len(layers2)
                assert sorted(
                    [x.layer_type for x in layers[:index]]
                    + ["permute", "conv1d", "permute"]
                    + [x.layer_type for x in layers[index:]]
                ) == sorted([x.layer_type for x in layers2]) or sorted(
                    [x.layer_type for x in layers[:index]] + ["conv2d"] + [x.layer_type for x in layers[index:]]
                ) == sorted(
                    [x.layer_type for x in layers2]
                )
            input = np.zeros([*layers2[0].input_shape])
            output = model2.predict(input)
            assert np.shape(output) == (11, 13)

    def test_resize_layer(self, builder: ModelBuilder, complex_model: MlModel):
        model = complex_model
        layers = model.get_layers()
        for index in range(len(layers) - 1):
            for size in [50, 150]:
                model2 = builder.resize_layer(model, index, size)
                layers2 = model2.get_layers()
                assert len(layers) == len(layers2)
                assert [x.layer_type for x in layers] == [x.layer_type for x in layers2]
                if layers[index].layer_type == "dense" and index not in [2]:
                    assert layers2[index].shape[-1] == size
                input = np.zeros([*layers2[0].input_shape])
                output = model2.predict(input)
                assert np.shape(output) == (11, 13)

    def test_add_relu(self, builder: ModelBuilder, complex_model: MlModel):
        model = complex_model
        model = builder.add_conv_layer(model, 0)
        layers = model.get_layers()
        input = np.zeros([*layers[0].input_shape])
        output = model.predict(input)
        for index in range(len(layers) - 1):
            model2 = builder.add_relu(model, index)
            layers2 = model2.get_layers()
            assert layers[index].activation is None
            if layers[index].layer_type in ["dense", "conv1d", "conv2d"]:
                assert layers2[index].activation == "relu"
            else:
                assert layers2[index].activation is None
            output2 = model2.predict(input)
            assert np.shape(output2) == np.shape(output)

    def test_merge_models(self, builder: ModelBuilder):
        model_1 = builder.build_model(asset_dependent=True)
        model_2 = builder.build_model(asset_dependent=False)
        model_3 = builder.merge_models(model_1, model_2)
        model_4 = builder.merge_models(model_2, model_3)
        assert len(model_4.get_layers()) == len(model_2.get_layers()) + len(model_3.get_layers())
        input = np.zeros([*model_1.get_layers()[0].input_shape])
        output_1 = model_1.predict(input)
        output_4 = model_4.predict(input)
        assert np.shape(output_1) == np.shape(output_4)

    def test_merge_existing_models(self):
        builder = ModelBuilder(10, 309, 23, 4)
        model_name_1 = "Olivia_20240628193521_ea9d7"
        model_name_2 = "Charlotte_20240628193521_d2a50"
        model_registry = ModelRegistry("s3://popiol-crypto-models", 1, 10, 10, 10)
        model_serializer = ModelSerializer()
        model_1 = model_serializer.deserialize(model_registry.get_model(model_name_1))
        model_2 = model_serializer.deserialize(model_registry.get_model(model_name_2))
        model_3 = builder.merge_models(model_1, model_2)
        print(model_3)
