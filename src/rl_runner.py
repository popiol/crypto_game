import src.check_running_master  # isort: skip

import argparse
import itertools
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import yaml

from src.agent import Agent
from src.agent_builder import AgentBuilder
from src.data_registry import DataRegistry
from src.data_transformer import DataTransformer, QuotesSnapshot
from src.evolution_handler import EvolutionHandler
from src.logger import Logger
from src.metrics import Metrics
from src.model_builder import ModelBuilder
from src.model_registry import ModelRegistry
from src.model_serializer import ModelSerializer
from src.portfolio_manager import PortfolioManager
from src.training_strategy import TrainingStrategy
from src.trainset import Trainset


class RlRunner:

    def __init__(self):
        self.start_dt = datetime.now()

    def load_config(self, file_path: str):
        with open(file_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.training_time_hours: int = self.config["rl_runner"]["training_time_hours"]

    def prepare(self):
        self.logger = Logger()
        self.logger.log("Sync data")
        self.data_registry = DataRegistry(**self.config["data_registry"])
        self.data_registry.sync()
        self.data_transformer = DataTransformer(**self.config["data_transformer"])
        self.asset_list = self.data_registry.get_asset_list()
        self.stats = self.data_registry.get_stats()
        self.model_registry = ModelRegistry(**self.config["model_registry"])
        self.model_serializer = ModelSerializer()
        self.trainset = Trainset(**self.config["trainset"])

    def initial_run(self):
        self.logger.log("Initial run")
        for timestamp, raw_quotes, bidask in self.data_registry.quotes_iterator():
            quotes = QuotesSnapshot(raw_quotes)
            features = self.data_transformer.quotes_to_features(quotes, self.asset_list)
            if not self.stats:
                self.data_transformer.add_to_stats(features)
        if not self.stats:
            self.stats = {key: val.tolist() for key, val in self.data_transformer.stats.items()}
            self.data_registry.set_stats(self.stats)
        self.data_registry.set_asset_list(self.asset_list)

    def create_agents(self):
        self.logger.log("Create agents")
        model_builder = ModelBuilder(
            self.data_transformer.memory_length,
            len(self.asset_list),
            self.data_transformer.n_features,
            self.data_transformer.n_outputs,
        )
        evolution_handler = EvolutionHandler(
            self.model_registry, self.model_serializer, model_builder, **self.config["evolution_handler"]
        )

        agent_builder = AgentBuilder(evolution_handler, self.data_transformer, self.trainset, **self.config["agent_builder"])
        self.agents = agent_builder.create_agents()
        self.portfolio_managers = [PortfolioManager(**self.config["portfolio_manager"]) for _ in self.agents]
        self.logger.log_agents(self.agents)

    def run_agent(
        self,
        agent: Agent,
        portfolio_manager: PortfolioManager,
        timestamp: datetime,
        quotes: QuotesSnapshot,
        input: np.array,
        eval_mode: bool = False,
    ):
        closed_transactions = portfolio_manager.handle_orders(timestamp, quotes)
        if not eval_mode:
            agent.train(closed_transactions)
        orders = agent.make_decision(timestamp, input, quotes, portfolio_manager.portfolio, self.asset_list)
        portfolio_manager.place_orders(timestamp, orders)
        if not eval_mode:
            self.logger.log_transactions(agent.agent_name, closed_transactions)

    def run_agents(self, timestamp: datetime, quotes: QuotesSnapshot, input: np.array):
        self.trainset.store_input(timestamp, input)
        for agent, portfolio_manager in zip(self.agents, self.portfolio_managers):
            self.run_agent(agent, portfolio_manager, timestamp, quotes, input)

    def train_on_open_positions(self):
        for agent, portfolio_manager in zip(self.agents, self.portfolio_managers):
            agent.train_on_open_positions(portfolio_manager.portfolio.positions)

    def save_models(self):
        for agent in self.agents:
            self.logger.log("Save model", agent.model_name)
            serialized_model = self.model_serializer.serialize(agent.training_strategy.model)
            metrics = Metrics(agent).get_metrics()
            self.model_registry.save_model(agent.model_name, serialized_model, metrics)

    def reset_simulation(self):
        self.data_transformer.reset()
        for agent, portfolio_manager in zip(self.agents, self.portfolio_managers):
            agent.reset()
            portfolio_manager.reset()

    def main_loop(self):
        for simulation_index in itertools.count():
            self.logger.log("Start simulation", simulation_index)
            quotes = QuotesSnapshot()
            self.reset_simulation()
            for timestamp, raw_quotes, bidask in self.data_registry.quotes_iterator():
                quotes.update(raw_quotes)
                features = self.data_transformer.quotes_to_features(quotes, self.asset_list)
                features = self.data_transformer.scale_features(features, self.stats)
                if features is None:
                    continue
                self.data_transformer.add_to_memory(features)
                self.run_agents(timestamp, quotes, self.data_transformer.memory)
            self.train_on_open_positions()
            self.logger.log_simulation_results([p.portfolio for p in self.portfolio_managers])
            if datetime.now() - self.start_dt > timedelta(hours=self.training_time_hours):
                break

    def evaluate_models(self):
        self.logger.log("Evaluate models")
        for model_name, serialized_model in self.model_registry.iterate_models():
            model = self.model_serializer.deserialize(serialized_model)
            agent = Agent("eval", self.data_transformer, self.trainset, TrainingStrategy(model))
            portfolio_manager = PortfolioManager(**self.config["portfolio_manager"])
            quotes = QuotesSnapshot()
            for timestamp, raw_quotes, bidask in self.data_registry.quotes_iterator():
                quotes.update(raw_quotes)
                features = self.data_transformer.quotes_to_features(quotes, self.asset_list)
                features = self.data_transformer.scale_features(features, self.stats)
                if features is None:
                    continue
                self.data_transformer.add_to_memory(features)
                self.run_agent(agent, portfolio_manager, timestamp, quotes, self.data_transformer.memory, eval_mode=True)
            score = portfolio_manager.portfolio.value / portfolio_manager.init_cash - 1
            metrics = Metrics(agent, self.model_registry.get_metrics(model_name))
            metrics.set_evaluation_score(score)
            self.model_registry.set_metrics(model_name, metrics.get_metrics())
            self.logger.log(model_name, score)

    def run(self):
        self.prepare()
        self.initial_run()
        self.create_agents()
        self.main_loop()
        self.save_models()
        self.evaluate_models()
        self.model_registry.archive_models()


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
