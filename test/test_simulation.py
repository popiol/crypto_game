from unittest.mock import patch

from src.rl_runner import RlRunner


class TestSimulation:

    @patch("src.rl_runner.Logger.log_simulation_results")
    @patch("src.model_registry.S3Utils")
    def test_simulation_one_iteration(self, S3Utils, log_simulation_results):
        rl_runner = RlRunner()
        rl_runner.load_config("config/config.yml")
        rl_runner.training_time_hours = -1
        rl_runner.run()
        assert S3Utils.call_count == 1
        assert log_simulation_results.call_count == 1
