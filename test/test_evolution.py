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
        assert np.array_equal(y, [[1, 2, 3], [4, 5, 6], [0, 0, 0], [0, 0, 0]])
        y = ModelBuilder.adjust_array_shape(x, 1, 5)
        assert np.array_equal(y, [[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]])

    def test_adjust_n_assets(self):
        builder = ModelBuilder(10, 11, 12, 13)
        model = builder.build_model()
        layers = model.get_layers()
        builder.n_assets = 14
        model2 = builder.adjust_n_assets(model)
        layers2 = model2.get_layers()
        assert layers[0].input_shape == (None, 10, 11, 12)
        assert layers2[0].input_shape == (None, 10, 14, 12)
        assert len(layers) == len(layers2)
        assert False
