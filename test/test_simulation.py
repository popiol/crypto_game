from unittest.mock import patch

from src.model_registry import ModelRegistry
from src.model_serializer import ModelSerializer
from src.rl_runner import RlRunner


class TestSimulation:

    @patch("src.model_registry.ModelRegistry.set_metrics")
    @patch("src.model_registry.ModelRegistry.get_metrics")
    @patch("src.model_registry.ModelRegistry.iterate_models")
    @patch("src.model_registry.S3Utils")
    def test_simulation_one_iteration(self, S3Utils, iterate_models, get_metrics, set_metrics):
        get_metrics.return_value = {"a": 1}
        rl_runner = RlRunner()
        rl_runner.load_config("config/config.yml")
        rl_runner.training_time_hours = -1
        # rl_runner.config["agent_builder"]["n_agents"] = 1
        rl_runner.prepare()
        rl_runner.initial_run()
        rl_runner.create_agents()
        rl_runner.main_loop()
        rl_runner.save_models()
        iterate_models.return_value = [
            (agent.model_name, ModelSerializer().serialize(agent.training_strategy.model)) for agent in rl_runner.agents
        ]
        rl_runner.evaluate_models()
        assert S3Utils.call_count == 1
        metrics = set_metrics.call_args.args[1]
        assert set(["a", "evaluation_score"]).issubset(set(metrics))

    def test_evaluate(self):
        rl_runner = RlRunner()
        rl_runner.load_config("config/config.yml")
        rl_runner.prepare()
        rl_runner.initial_run()
        rl_runner.evaluate_models()

    def test_get_weak_models(self):
        rl_runner = RlRunner()
        rl_runner.load_config("config/config.yml")
        model_registry = ModelRegistry(**rl_runner.config["model_registry"])
        model_registry.max_mature_models = 1
        models = model_registry.get_weak_models()
        assert len(models) > 0
        model_registry.max_mature_models = len(models)
        models = model_registry.get_weak_models()
        assert len(models) == 1

    @patch("src.model_registry.ModelRegistry.set_metrics")
    @patch("src.model_registry.ModelRegistry.iterate_models")
    def test_evaluate_existing_model(self, iterate_models, set_metrics):
        model_name = "Lucas_20240709084611_65088"
        rl_runner = RlRunner()
        rl_runner.load_config("config/config.yml")
        model_registry = ModelRegistry(**rl_runner.config["model_registry"])
        model = model_registry.get_model(model_name)
        iterate_models.return_value = [(model_name, model)]
        rl_runner.prepare()
        rl_runner.initial_run()
        rl_runner.evaluate_models()
        print(set_metrics.call_args.args[1])
