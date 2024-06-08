from src.model_builder import ModelBuilder


class TestEvolution:

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
