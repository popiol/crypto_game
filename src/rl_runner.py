import src.check_running_master  # isort: skip

import argparse
import sys
import time

import yaml

from src.agent import Agent
from src.agent_builder import AgentBuilder
from src.data_registry import DataRegistry
from src.data_transformer import DataTransformer
from src.evolution_handler import EvolutionHandler
from src.model_builder import ModelBuilder
from src.model_registry import ModelRegistry
from src.model_serializer import ModelSerializer


class RlRunner:

    def load_config(self, file_path: str):
        with open(file_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def prepare(self):
        print("Sync data")
        self.data_registry = DataRegistry(**self.config["data_registry"])
        self.data_registry.sync()
        self.data_transformer = DataTransformer(**self.config["data_transformer"])
        self.asset_list = self.data_registry.get_asset_list()
        self.stats = self.data_registry.get_stats()

    def initial_run(self):
        print("Initial run")
        for quotes in self.data_registry.quotes_iterator():
            features = self.data_transformer.quotes_to_features(quotes, self.asset_list)
            if not self.stats:
                self.data_transformer.add_to_stats(features)
        if not self.stats:
            self.stats = {key: val.tolist() for key, val in self.data_transformer.stats.items()}
            self.data_registry.set_stats(self.stats)
        self.data_registry.set_asset_list(self.asset_list)

    def create_agents(self):
        print("Create agents")
        self.agents = []
        self.model_registry = ModelRegistry(**self.config["model_registry"])
        self.model_serializer = ModelSerializer()
        model_builder = ModelBuilder(
            self.data_transformer.memory_length,
            len(self.asset_list),
            self.data_transformer.n_features,
            self.data_transformer.n_outputs,
        )
        evolution_handler = EvolutionHandler(self.model_registry, self.model_serializer, model_builder)
        agent_builder = AgentBuilder(**self.config["agent_builder"])
        for name in agent_builder.get_names():
            print(name)
            model = evolution_handler.create_model()
            agent = Agent(name, model)
            self.agents.append(agent)

    def save_model(self, agent: Agent):
        print("save model", agent.model_name)
        serialized_model = self.model_serializer.serialize(agent.model)
        self.model_registry.save_model(agent.model_name, serialized_model, agent.metrics)

    def main_loop(self):
        for simulation_index in range(1):
            print("Start simulation", simulation_index)
            for quotes in self.data_registry.quotes_iterator():
                features = self.data_transformer.quotes_to_features(quotes, self.asset_list)
                features = self.data_transformer.scale_features(features, self.stats)
                if features is None:
                    continue
                self.data_transformer.add_to_memory(features)

    def run(self):
        self.prepare()
        self.initial_run()
        self.create_agents()
        self.main_loop()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args, other = parser.parse_known_args(argv)
    rl_runner = RlRunner()
    rl_runner.load_config(args.config)
    rl_runner.run()


if __name__ == "__main__":
    time1 = time.time()
    main(sys.argv)
    time2 = time.time()
    print("Overall execution time:", time2 - time1)
