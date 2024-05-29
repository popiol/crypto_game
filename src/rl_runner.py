import src.check_running_master  # isort: skip

import argparse
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import Pool, get_context

import numpy as np
import yaml

from src.agent import Agent
from src.agent_builder import AgentBuilder
from src.data_registry import DataRegistry
from src.data_transformer import DataTransformer
from src.evolution_handler import EvolutionHandler
from src.model_builder import ModelBuilder
from src.model_registry import ModelRegistry
from src.model_serializer import ModelSerializer
from src.portfolio_manager import PortfolioManager


class RlRunner:

    def __init__(self):
        self.start_dt = datetime.now()

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
        self.model_registry = ModelRegistry(**self.config["model_registry"])
        self.model_serializer = ModelSerializer()
        model_builder = ModelBuilder(
            self.data_transformer.memory_length,
            len(self.asset_list),
            self.data_transformer.n_features,
            self.data_transformer.n_outputs,
        )
        evolution_handler = EvolutionHandler(self.model_registry, self.model_serializer, model_builder)
        agent_builder = AgentBuilder(evolution_handler, **self.config["agent_builder"])
        self.agents = agent_builder.create_agents()
        self.portfolio_managers = [PortfolioManager(**self.config["portfolio_manager"]) for _ in self.agents]

    def save_model(self, agent: Agent):
        print("Save model", agent.model_name)
        serialized_model = self.model_serializer.serialize(agent.model)
        self.model_registry.save_model(agent.model_name, serialized_model, agent.metrics)

    @staticmethod
    def run_agent(inputs: np.array, agent: Agent, portfolio_manager: PortfolioManager):
        print("run agent", agent.agent_name)
        orders = agent.process_quotes(inputs, portfolio_manager.portfolio)
        portfolio_manager.handle_orders()
        portfolio_manager.place_orders(orders)
        return portfolio_manager

    def run_agents(self, inputs: np.array):
        params = [(inputs, agent, portfolio_manager) for agent, portfolio_manager in zip(self.agents, self.portfolio_managers)]
        with get_context("spawn").Pool(1) as pool:
            portfolio_managers = pool.starmap(RlRunner.run_agent, params)
        self.portfolio_managers = portfolio_managers

    def main_loop(self):
        for simulation_index in range(1):
            print("Start simulation", simulation_index)
            for quotes in self.data_registry.quotes_iterator():
                features = self.data_transformer.quotes_to_features(quotes, self.asset_list)
                features = self.data_transformer.scale_features(features, self.stats)
                if features is None:
                    continue
                self.data_transformer.add_to_memory(features)
                self.run_agents(self.data_transformer.memory)
                break
            if datetime.now() - self.start_dt > timedelta(days=1):
                break

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
