from src.data_transformer import InputFeatures


class TestTransformer:

    def test_to_vector(self):
        features = InputFeatures()
        features.set_feature("closing_price", 1.1)
        v = features.to_vector()
        assert v[6] == 1.1

    def test_is_price(self):
        assert InputFeatures.is_price(6)
        assert not InputFeatures.is_price(8)
