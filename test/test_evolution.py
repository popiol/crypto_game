import numpy as np
import pytest

from src.model_builder import ModelBuilder


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

    def test_adjust_weights_shape(self):
        builder = ModelBuilder(10, 11, 12, 13)
        model = builder.build_model()
        weights = model.model.get_weights()[:2]
        input_size, output_size = np.shape(weights[0])
        with pytest.raises(AssertionError):
            builder.adjust_weights_shape(model.model.get_weights(), input_size, output_size)
        with pytest.raises(AssertionError):
            builder.adjust_weights_shape(weights, 0, output_size)
        with pytest.raises(AssertionError):
            builder.adjust_weights_shape(weights, input_size, 0)
        new_weights = builder.adjust_weights_shape(weights, input_size, output_size)
        for w1, w2 in zip(weights, new_weights):
            assert np.array_equal(w1, w2)
        new_weights = builder.adjust_weights_shape(weights, input_size - 2, output_size)
        assert np.array_equal(weights[0][:-2, :], new_weights[0])
        assert np.array_equal(weights[1], new_weights[1])
        new_weights = builder.adjust_weights_shape(weights, input_size + 2, output_size)
        assert np.array_equal(weights[0], new_weights[0][:-2, :])
        assert abs(new_weights[0][-2:, :].mean()) < 1
        assert new_weights[0][-2:, :].std() > 0
        assert np.array_equal(weights[1], new_weights[1])
        new_weights = builder.adjust_weights_shape(weights, input_size, output_size - 2)
        assert np.array_equal(weights[0][:, :-2], new_weights[0])
        assert np.array_equal(weights[1][:-2], new_weights[1])
        new_weights = builder.adjust_weights_shape(weights, input_size, output_size + 2)
        assert np.array_equal(weights[0], new_weights[0][:, :-2])
        assert abs(new_weights[0][:, -2:].mean()) < 1
        assert new_weights[0][:, -2:].std() > 0
        assert np.array_equal(weights[1], new_weights[1][:-2])

    def test_adjust_n_assets(self):
        builder = ModelBuilder(10, 11, 12, 13)
        model = builder.build_model(asset_dependant=True)
        layers = model.get_layers()
        builder.n_assets = 14
        model2 = builder.adjust_n_assets(model)
        layers2 = model2.get_layers()
        assert layers[0].input_shape == (None, 10, 11, 12)
        assert layers2[0].input_shape == (None, 10, 14, 12)
        assert len(layers) == len(layers2)
        input = np.zeros([*layers2[0].input_shape[1:]])
        output = model2.predict(np.array([input]))[0]
        assert np.shape(output) == (14, 13)
