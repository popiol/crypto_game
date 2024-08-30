import numpy as np

from src.data_transformer import DataTransformer, InputFeatures


class TestTransformer:

    def test_to_vector(self):
        features = InputFeatures()
        features.set_feature("closing_price", 1.1)
        v = features.to_vector()
        assert v[6] == 1.1

    def test_is_price(self):
        assert InputFeatures.is_price(6)
        assert not InputFeatures.is_price(8)

    def test_join_memory(self):
        transformer = DataTransformer(10, 0)
        x = np.ones((10, 11, 12))
        y = np.ones((10, 11, 1))
        z = transformer.join_memory(x, y)
        assert np.shape(z) == (10, 11, 13)
        x = np.ones((10, 11, 12))
        y = np.ones((10, 10, 1))
        z = transformer.join_memory(x, y)
        assert np.shape(z) == (10, 11, 13)
        x = np.ones((10, 10, 12))
        y = np.ones((10, 11, 1))
        z = transformer.join_memory(x, y)
        assert np.shape(z) == (10, 11, 13)
