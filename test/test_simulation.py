from unittest.mock import patch

from src.model_builder import ModelBuilder
from src.model_serializer import ModelSerializer
from src.rl_runner import RlRunner


class TestSimulation:

    @patch("src.model_registry.ModelRegistry.set_metrics")
    @patch("src.model_registry.ModelRegistry.get_metrics")
    @patch("src.model_registry.ModelRegistry.iterate_models")
    @patch("src.rl_runner.Logger.log_simulation_results")
    @patch("src.model_registry.S3Utils")
    def test_simulation_one_iteration(self, S3Utils, log_simulation_results, iterate_models, get_metrics, set_metrics):
        get_metrics.return_value = {"a": 1}
        rl_runner = RlRunner()
        rl_runner.load_config("config/config.yml")
        rl_runner.training_time_hours = -1
        rl_runner.prepare()
        model = ModelBuilder(
            rl_runner.data_transformer.memory_length,
            len(rl_runner.asset_list),
            rl_runner.data_transformer.n_features,
            rl_runner.data_transformer.n_outputs,
        ).build_model(asset_dependant=False)
        iterate_models.return_value = [("test", ModelSerializer().serialize(model))]
        rl_runner.run()
        assert S3Utils.call_count == 1
        assert set(["a", "evaluation_score"]).issubset(set(set_metrics.call_args.args[1]))
        assert log_simulation_results.call_count == 1
