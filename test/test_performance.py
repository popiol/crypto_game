from unittest.mock import patch

from src.environment import Environment
from src.model_serializer import ModelSerializer
from src.rl_runner import RlRunner


class TestPerformance:
    @patch("src.data_registry.DataRegistry.get_random_trainset_file")
    @patch("src.model_registry.ModelRegistry.iterate_models")
    @patch("src.training_strategy.StrategyPicker.pick_class")
    @patch("src.evolution_handler.EvolutionHandler.create_model")
    def test_performance(self, create_model, pick_class, iterate_models, get_random_trainset_file):
        from src.training_strategy import LearnOnSuccess

        pick_class.return_value = LearnOnSuccess
        get_random_trainset_file.return_value = "20241225175409.pickle.lzma"
        environment = Environment("config/config.yml")
        environment.config["rl_runner"]["training_time_min"] = -1
        environment.config["agent_builder"]["n_agents"] = 1
        rl_runner = RlRunner(environment)
        rl_runner.prepare()
        rl_runner.initial_run()
        model_builder = environment.model_builder
        model = model_builder.build_model_v6(asset_dependent=False)
        model = model_builder.adjust_dimensions(model)
        model = model_builder.filter_assets(model, environment.asset_list, environment.data_transformer.current_assets)
        model_builder.pretrain(model, environment.asset_list, environment.data_transformer.current_assets)
        create_model.return_value = (model, {})
        rl_runner.create_agents()
        rl_runner2 = RlRunner(environment)
        rl_runner2.prepare()
        rl_runner2.initial_run()
        print("weights", rl_runner.agents[0].training_strategy.model.get_weight_stats())
        iterate_models.return_value = [
            (agent.model_name, ModelSerializer().serialize(agent.training_strategy.model)) for agent in rl_runner.agents
        ]
        environment.eval_mode = True
        rl_runner2.evaluate_models()
        rl_runner.pretrain()
        print("weights", rl_runner.agents[0].training_strategy.model.get_weight_stats())
        iterate_models.return_value = [
            (agent.model_name, ModelSerializer().serialize(agent.training_strategy.model)) for agent in rl_runner.agents
        ]
        environment.eval_mode = True
        rl_runner2.evaluate_models()
        rl_runner.train_on_historical()
        print("weights", rl_runner.agents[0].training_strategy.model.get_weight_stats())
        iterate_models.return_value = [
            (agent.model_name, ModelSerializer().serialize(agent.training_strategy.model)) for agent in rl_runner.agents
        ]
        environment.eval_mode = True
        rl_runner2.evaluate_models()
        for _ in range(2):
            environment.eval_mode = False
            rl_runner.main_loop()
            print("weights", rl_runner.agents[0].training_strategy.model.get_weight_stats())
            iterate_models.return_value = [
                (agent.model_name, ModelSerializer().serialize(agent.training_strategy.model)) for agent in rl_runner.agents
            ]
            environment.eval_mode = True
            rl_runner2.evaluate_models()
