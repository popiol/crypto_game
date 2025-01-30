from unittest.mock import patch

from src.aggregated_metrics import AggregatedMetrics
from src.custom_metrics import CustomMetrics
from src.environment import Environment
from src.evolution_randomizer import EvolutionRandomizer
from src.model_serializer import ModelSerializer
from src.rl_runner import RlRunner


class TestSimulation:

    @patch("src.data_registry.DataRegistry.add_to_trainset")
    @patch("src.evolution_randomizer.EvolutionRandomizer.model_creation_method")
    @patch("src.model_registry.ModelRegistry.set_metrics")
    @patch("src.model_registry.ModelRegistry.get_metrics")
    @patch("src.model_registry.ModelRegistry.iterate_models")
    @patch("src.model_registry.ModelRegistry.save_model")
    def test_simulation_one_iteration(
        self,
        save_model,
        iterate_models,
        get_metrics,
        set_metrics,
        model_creation_method,
        add_to_trainset,
    ):
        get_metrics.return_value = {"a": 1}
        model_creation_method.return_value = EvolutionRandomizer.ModelCreationMethod.NEW_MODEL
        environment = Environment("config/config.yml")
        environment.config["rl_runner"]["training_time_min"] = -1
        environment.config["agent_builder"]["n_agents"] = 1
        rl_runner = RlRunner(environment)
        rl_runner.prepare()
        rl_runner.initial_run()
        rl_runner.create_agents()
        rl_runner.main_loop()
        rl_runner.save_models()
        iterate_models.return_value = [
            (agent.model_name, ModelSerializer().serialize(agent.training_strategy.model)) for agent in rl_runner.agents
        ]
        environment.eval_mode = True
        rl_runner.evaluate_models()
        assert save_model.call_count == 1
        metrics = set_metrics.call_args.args[1]
        print(metrics)
        assert set(["a", "evaluation_score"]).issubset(set(metrics))

    @patch("src.model_registry.S3Utils")
    @patch("src.data_registry.DataRegistry.sync")
    def test_simulation_5_min(self, sync, S3Utils):
        environment = Environment("config/config.yml")
        environment.config["rl_runner"]["training_time_min"] = 5
        environment.config["agent_builder"]["n_agents"] = 1
        rl_runner = RlRunner(environment)
        rl_runner.prepare()
        rl_runner.initial_run()
        rl_runner.create_agents()
        rl_runner.main_loop()

    def test_get_weak_models(self):
        environment = Environment("config/config.yml")
        model_registry = environment.model_registry
        model_registry.max_mature_models = 1
        models = model_registry.get_weak_models(mature=True)
        print("mature", models)
        assert len(models) > 0
        model_registry.max_mature_models = len(models)
        models = model_registry.get_weak_models(mature=True)
        assert len(models) == 1
        model_registry.max_immature_models = 1
        models = model_registry.get_weak_models(mature=False)
        print("immature", models)
        assert len(models) > 0
        model_registry.max_immature_models = len(models)
        models = model_registry.get_weak_models(mature=False)
        assert len(models) == 1

    @patch("src.reports.Reports.upload_reports")
    @patch("src.model_registry.ModelRegistry.set_aggregated_metrics")
    @patch("src.model_registry.ModelRegistry.set_metrics")
    @patch("src.model_registry.ModelRegistry.iterate_models")
    @patch("src.model_registry.ModelRegistry.archive_models")
    def test_evaluate_existing_model(self, archive_models, iterate_models, set_metrics, set_aggregated_metrics, upload_reports):
        model_name = "Ava_20250129185529_31648"
        environment = Environment("config/config.yml")
        rl_runner = RlRunner(environment)
        model_registry = environment.model_registry
        model = model_registry.get_model(model_name)
        iterate_models.return_value = [(model_name, model)]
        rl_runner.evaluate()
        print("model metrics", set_metrics.call_args.args[1])
        print("aggregated", set_aggregated_metrics.call_args.args[0])

    @patch("src.reports.Reports.upload_reports")
    @patch("src.model_registry.ModelRegistry.set_aggregated_metrics")
    @patch("src.model_registry.ModelRegistry.set_metrics")
    @patch("src.model_registry.ModelRegistry.archive_models")
    def test_evaluate_all_existing_models(self, archive_models, set_metrics, set_aggregated_metrics, upload_reports):
        environment = Environment("config/config.yml")
        rl_runner = RlRunner(environment)
        rl_runner.evaluate()

    def test_aggregated_metrics(self):
        environment = Environment("config/config.yml")
        model_registry = environment.model_registry
        all_metrics = []
        for file in model_registry.s3_utils.list_files(model_registry.metrics_prefix + "/"):
            all_metrics.append(model_registry.s3_utils.download_json(file))
        aggregated = AggregatedMetrics(all_metrics)
        metrics = aggregated.get_metrics()
        print(metrics)
        assert "datetime" in metrics

    def test_custom_metrics(self):
        environment = Environment("config/config.yml")
        model_registry = environment.model_registry
        all_metrics = []
        for file in model_registry.s3_utils.list_files(model_registry.metrics_prefix + "/"):
            all_metrics.append(model_registry.s3_utils.download_json(file))
        aggregated = AggregatedMetrics(all_metrics)
        custom = CustomMetrics(aggregated.df, aggregated.get_metrics())
        metrics = custom.get_metrics()
        print(metrics)
        assert type(metrics) == dict
        assert len(metrics) > 0

    def test_merge_existing_models(self):
        environment = Environment("config/config.yml")
        evolution_handler = environment.evolution_handler
        model, metrics = evolution_handler.merge_existing_models()
        print(metrics)
