import numpy as np
import pytest

from src.agent import Agent
from src.data_transformer import QuotesSnapshot
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

    @pytest.fixture
    def complex_model(self, builder: ModelBuilder):
        model_1 = builder.build_model(asset_dependent=True)
        model_1.name = "agent_2024_23456"
        model_2 = builder.build_model(asset_dependent=False)
        model_2.name = "agent_2024_34567"
        return builder.merge_models(model_1, model_2)

    @pytest.fixture
    def quotes(self):
        return QuotesSnapshot({"TBTCUSD": {"c": [1.1]}, "WBTCUSD": {"c": [1.3]}})

    def create_agent(self, model):
        return Agent("test", None, None, TrainingStrategy(model), {})

    @pytest.fixture
    def agent(self, simple_model):
        return self.create_agent(simple_model)

    @pytest.fixture
    def metrics(self, agent: Agent, quotes: QuotesSnapshot):
        return Metrics(agent, quotes)

    def test_get_bitcoin_quote(self, metrics: Metrics):
        assert np.isclose(metrics.get_bitcoin_quote(), 1.2)

    def test_get_bitcoin_change(self, agent: Agent, metrics: Metrics):
        quotes = QuotesSnapshot({"TBTCUSD": {"c": [1.2]}, "WBTCUSD": {"c": [1.4]}})
        agent.metrics = metrics.get_metrics()
        metrics = Metrics(agent, quotes=quotes)
        assert np.isclose(metrics.get_bitcoin_change(), 1.3 / 1.2 - 1)

    def test_get_n_merge_ancestors(self, simple_model, complex_model):
        agent = self.create_agent(simple_model)
        metrics = Metrics(agent)
        assert metrics.get_n_merge_ancestors() == 0
        agent = self.create_agent(complex_model)
        metrics = Metrics(agent)
        assert metrics.get_n_merge_ancestors() == 2

    def test_get_n_ancestors(self, simple_model):
        agent = self.create_agent(simple_model)
        metrics = Metrics(agent)
        assert metrics.get_n_ancestors() == 0
        agent.metrics = {"parents": {"asd": None}}
        metrics = Metrics(agent)
        assert metrics.get_n_ancestors() == 1

    def test_get_metrics(self, agent, quotes):
        agent.metrics = {"model_id": "abc123", "a": 1, "n_merge_ancestors": -1, "BTCUSD": -1}
        metrics = Metrics(agent, quotes)
        metrics2 = metrics.get_metrics()
        assert "model_id" in metrics2
        assert "reward_stats" in metrics2
        assert "n_merge_ancestors" in metrics2
        assert "BTCUSD" in metrics2
        assert "a" in metrics2
        assert metrics2["model_id"] == "abc123"
        assert metrics2["n_merge_ancestors"] >= 0
        assert metrics2["BTCUSD"] >= 0

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

    def test_get_trained_ratio(self, agent: Agent):
        agent.metrics["reward_stats"] = {"count": 1}
        metrics = Metrics(agent)
        assert np.isclose(metrics.get_trained_ratio(), 1 / agent.training_strategy.model.get_n_params())
