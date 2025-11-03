import numpy as np
import pytest

from src.environment import Environment
from src.ml_model import MlModel
from src.model_builder import ModelBuilder
from src.model_serializer import ModelSerializer


class TestModel:
    @pytest.fixture
    def builder(self):
        return ModelBuilder(4, 6, 5, 3)

    @pytest.fixture
    def model(self, builder: ModelBuilder):
        model_1 = builder.build_model(builder.ModelVersion.V2DEP)
        model_2 = builder.build_model(builder.ModelVersion.V6IND)
        return builder.merge_models(model_1, model_2, builder.MergeVersion.DOT)

    def test_clone_model(self, model: MlModel):
        model_1 = model
        model_2 = model_1.copy()
        print(model_2)

    def test_model_serialization(self, model: MlModel):
        model_1 = model
        serializer = ModelSerializer()
        serialized = serializer.serialize(model_1)
        model_2 = serializer.deserialize(serialized)
        input = np.zeros([*model_1.get_layers()[0].input_shape])
        output_1 = model_1.predict(input)
        output_2 = model_2.predict(input)
        assert np.array_equal(output_1, output_2)
        print(model_2)

    def test_get_edges(self, model: MlModel):
        edges = model.get_edges()
        print(edges)

    def test_model_to_str(self):
        environment = Environment("config/config.yml")
        model_builder = environment.model_builder
        model = model_builder.build_model_v10dep()
        print(model)
