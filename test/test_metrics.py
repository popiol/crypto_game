from unittest.mock import MagicMock

import numpy as np
import pytest

from src.agent import Agent
from src.environment import Environment
from src.metrics import Metrics
from src.ml_model import MlModel
from src.model_builder import ModelBuilder
from src.training_strategy import TrainingStrategy


class TestMetrics:

    @pytest.fixture
    def builder(self):
        return ModelBuilder(10, 11, 12, 13)

    @pytest.fixture
    def simple_model(self, builder: ModelBuilder):
        model = builder.build_model()
        model.name = "agent_2024_12345"
        return model

    def create_agent(self, model):
        data_transformer = MagicMock()
        data_transformer.get_shared_input_stats = MagicMock()
        data_transformer.get_shared_input_stats.return_value = {}
        return Agent("test", data_transformer, None, TrainingStrategy(model), {})

    @pytest.fixture
    def agent(self, simple_model):
        return self.create_agent(simple_model)

    @pytest.fixture
    def metrics(self, agent: Agent):
        return Metrics(agent)

    def test_get_n_merge_ancestors(self, simple_model):
        agent = self.create_agent(simple_model)
        metrics = Metrics(agent)
        assert metrics.get_n_merge_ancestors() == 0
        metrics.metrics["parents"] = {
            "Isabella_20241205145936_81eae": {
                "Amelia_20241101213524_a96a8": None,
                "Benjamin_20241105180526_7a83d": None,
            },
            "Luna_20241029080525_dd5aa": {
                "Olivia_20241022222526_806af": {"Emma_20241017052524_076e7": None},
                "Benjamin_20241107113528_16a28": {"William_20241101090522_9bb9e": None},
            },
        }
        assert metrics.get_n_merge_ancestors() == 6

    def test_get_n_ancestors(self, simple_model):
        agent = self.create_agent(simple_model)
        metrics = Metrics(agent)
        assert metrics.get_n_ancestors() == 0
        agent.metrics = {"parents": {"asd": None}}
        metrics = Metrics(agent)
        assert metrics.get_n_ancestors() == 1

    def test_get_metrics(self, agent):
        agent.metrics = {"a": 1, "n_merge_ancestors": -1}
        metrics = Metrics(agent)
        metrics2 = metrics.get_metrics()
        assert "reward_stats" in metrics2
        assert "n_merge_ancestors" in metrics2
        assert "a" in metrics2
        assert metrics2["n_merge_ancestors"] >= 0

    def test_n_params(self, simple_model: MlModel, metrics: Metrics):
        assert metrics.get_n_params() == simple_model.get_n_params()

    def test_n_layers(self, simple_model: MlModel, metrics: Metrics):
        assert metrics.get_n_layers() == len(simple_model.get_layers())

    def test_n_layers_per_type(self, simple_model: MlModel, metrics: Metrics):
        assert metrics.get_n_layers_per_type() == {"permute": 1, "reshape": 1, "unit": 1, "dense": 2}

    def test_get_n_trainings(self, agent: Agent):
        agent.metrics["reward_stats"] = {"count": 1}
        metrics = Metrics(agent)
        assert metrics.get_n_trainings() == 1
        agent.metrics = metrics.get_metrics()
        metrics = Metrics(agent)
        assert metrics.get_n_trainings() == 2

    def test_get_trained_ratio(self, agent: Agent):
        agent.metrics["reward_stats"] = {"count": 1}
        metrics = Metrics(agent)
        assert np.isclose(metrics.get_trained_ratio(), 1 / agent.training_strategy.model.get_n_params())

    def test_get_leader_portfolio_value(self):
        environment = Environment("config/config.yml")
        reports = environment.reports
        score = reports.get_leader_portfolio_value()
        print(score)
        assert type(score) == float

    def test_get_model_length(self, builder: ModelBuilder):
        model = builder.build_model(asset_dependent=False, version=builder.ModelVersion.V1)
        metrics = Metrics(self.create_agent(model))
        assert metrics.get_model_length() == 5
        model = builder.build_model(asset_dependent=True, version=builder.ModelVersion.V1)
        metrics = Metrics(self.create_agent(model))
        assert metrics.get_model_length() == 8
        model = builder.build_model(asset_dependent=False, version=builder.ModelVersion.V2)
        metrics = Metrics(self.create_agent(model))
        assert metrics.get_model_length() == 8
        model = builder.build_model(asset_dependent=True, version=builder.ModelVersion.V2)
        metrics = Metrics(self.create_agent(model))
        assert metrics.get_model_length() == 11

    def test_get_model_width(self, builder: ModelBuilder):
        model_1 = builder.build_model(asset_dependent=False, version=builder.ModelVersion.V1)
        metrics = Metrics(self.create_agent(model_1))
        assert metrics.get_model_width() == 1
        model_2 = builder.build_model(asset_dependent=True, version=builder.ModelVersion.V1)
        metrics = Metrics(self.create_agent(model_2))
        assert metrics.get_model_width() == 1
        model_3 = builder.merge_models(model_1, model_2)
        metrics = Metrics(self.create_agent(model_3))
        assert metrics.get_model_width() == 2
