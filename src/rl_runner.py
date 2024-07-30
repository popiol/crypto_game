import src.check_running_master  # isort: skip
import argparse
import itertools
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml

from src.agent import Agent
from src.agent_builder import AgentBuilder
from src.aggregated_metrics import AggregatedMetrics
from src.custom_metrics import CustomMetrics
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

    def get_model_registry(self):
        return ModelRegistry(**self.config["model_registry"])

    def prepare(self, eval_mode: bool = False):
        self.logger = Logger()
        self.logger.log("Sync data")
        self.data_registry = DataRegistry(**self.config["data_registry"])
        if not eval_mode:  # TODO remove condition once evaluation is executed on a separate node
            self.data_registry.sync()
        self.data_transformer = DataTransformer(**self.config["data_transformer"])
        self.asset_list = self.data_registry.get_asset_list()
        self.stats = self.data_registry.get_stats()
        if self.stats and len(self.stats["mean"]) != self.data_transformer.n_features:
            self.stats = None
        self.model_registry = self.get_model_registry()
        self.model_serializer = ModelSerializer()
        self.trainset = None
        if not eval_mode:
            self.trainset = Trainset(**self.config["trainset"])

    def quotes_iterator(self):
        quotes = QuotesSnapshot()
        for timestamp, raw_quotes, bidask in self.data_registry.quotes_iterator():
            quotes.update(raw_quotes)
            quotes.update_bid_ask(bidask)
            yield timestamp, quotes

    def initial_run(self):
        self.logger.log("Initial run")
        for timestamp, quotes in self.quotes_iterator():
            features = self.data_transformer.quotes_to_features(quotes, self.asset_list)
            if not self.stats:
                self.data_transformer.add_to_stats(features)
        if not self.stats:
            self.stats = {key: val.tolist() for key, val in self.data_transformer.stats.items()}
            self.data_registry.set_stats(self.stats)
        self.data_registry.set_asset_list(self.asset_list)
        self.model_builder = ModelBuilder(
            self.data_transformer.memory_length,
            len(self.asset_list),
            self.data_transformer.n_features,
            self.data_transformer.n_outputs,
        )

    def create_agents(self):
        self.logger.log("Create agents")
        evolution_handler = EvolutionHandler(
            self.model_registry, self.model_serializer, self.model_builder, **self.config["evolution_handler"]
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
        input: np.ndarray,
        eval_mode: bool = False,
    ):
        closed_transactions = portfolio_manager.handle_orders(timestamp, quotes)
        if not eval_mode:
            agent.train(closed_transactions)
        orders = agent.make_decision(timestamp, input, quotes, portfolio_manager.portfolio, self.asset_list)
        portfolio_manager.place_orders(timestamp, orders)
        self.logger.log_transactions(agent.agent_name, closed_transactions)

    def run_agents(self, timestamp: datetime, quotes: QuotesSnapshot, eval_mode: bool = False):
        if not eval_mode:
            self.trainset.store_shared_input(timestamp, self.data_transformer.get_shared_memory())
        for agent, portfolio_manager in zip(self.agents, self.portfolio_managers):
            self.data_transformer.add_portfolio_to_memory(
                agent.agent_name, [p.asset for p in portfolio_manager.portfolio.positions], self.asset_list
            )
            input = self.data_transformer.get_memory(agent.agent_name)
            if not eval_mode:
                self.trainset.store_agent_input(
                    timestamp, self.data_transformer.get_agent_memory(agent.agent_name), agent.agent_name
                )
            self.run_agent(agent, portfolio_manager, timestamp, quotes, input, eval_mode)

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
            self.reset_simulation()
            for timestamp, quotes in self.quotes_iterator():
                features = self.data_transformer.quotes_to_features(quotes, self.asset_list)
                features = self.data_transformer.scale_features(features, self.stats)
                if features is None:
                    continue
                self.data_transformer.add_to_memory(features)
                self.run_agents(timestamp, quotes)
            self.train_on_open_positions()
            # self.logger.log_simulation_results([p.portfolio for p in self.portfolio_managers])
            if datetime.now() - self.start_dt > timedelta(hours=self.training_time_hours):
                break

    def evaluate_models(self):
        self.logger.log("Evaluate models")
        self.logger.transactions = {}
        self.agents: list[Agent] = []
        self.portfolio_managers: list[PortfolioManager] = []
        self.all_metrics = []
        for model_name, serialized_model in self.model_registry.iterate_models():
            model = self.model_serializer.deserialize(serialized_model)
            model = self.model_builder.adjust_dimensions(model)
            metrics = self.model_registry.get_metrics(model_name)
            agent = Agent(model_name.split("_")[0], self.data_transformer, self.trainset, TrainingStrategy(model), metrics)
            agent.model_name = model_name
            self.agents.append(agent)
            portfolio_manager = PortfolioManager(**self.config["portfolio_manager"])
            self.portfolio_managers.append(portfolio_manager)
        initial_quotes = None
        for timestamp, quotes in self.quotes_iterator():
            if initial_quotes is None and quotes.has_asset("TBTCUSD") and quotes.has_asset("WBTCUSD"):
                initial_quotes = quotes.copy()
            features = self.data_transformer.quotes_to_features(quotes, self.asset_list)
            features = self.data_transformer.scale_features(features, self.stats)
            if features is None:
                continue
            self.data_transformer.add_to_memory(features)
            self.run_agents(timestamp, quotes, eval_mode=True)
        for agent, portfolio_manager in zip(self.agents, self.portfolio_managers):
            score = portfolio_manager.portfolio.value / portfolio_manager.init_cash - 1
            metrics = Metrics(agent, initial_quotes)
            agent.metrics["BTCUSD"] = metrics.get_bitcoin_quote()
            metrics = Metrics(agent, quotes, self.logger.transactions.get(agent.agent_name))
            metrics.set_evaluation_score(score)
            metrics_dict = metrics.get_metrics()
            self.all_metrics.append(metrics_dict)
            self.model_registry.set_metrics(agent.model_name, metrics_dict)
            self.logger.log(agent.model_name, score)

    def aggregate_metrics(self):
        aggregated = AggregatedMetrics(self.all_metrics)
        aggregated_dict = aggregated.get_metrics()
        custom = CustomMetrics(aggregated.df, aggregated_dict)
        return {**aggregated_dict, "custom": custom.get_metrics()}

    def get_quick_stats(self) -> pd.DataFrame:
        df = pd.DataFrame(
            columns=[
                "model",
                "score",
                "n_params",
                "n_layers",
                "n_ancestors",
                "training_strategy",
                "n_transactions",
                "trained_ratio",
            ]
        )
        for agent, metrics in zip(self.agents, self.all_metrics):
            df.loc[len(df)] = [
                agent.model_name,
                metrics["evaluation_score"],
                metrics["n_params"],
                metrics["n_layers"],
                metrics["n_ancestors"],
                metrics["training_strategy"],
                metrics["n_transactions"],
                metrics["trained_ratio"],
            ]
        return df

    def train(self):
        self.prepare()
        self.initial_run()
        self.create_agents()
        self.main_loop()
        self.save_models()

    def evaluate(self):
        self.prepare(eval_mode=True)
        self.initial_run()
        self.evaluate_models()
        self.model_registry.archive_models()
        aggregated = self.aggregate_metrics()
        self.model_registry.set_aggregated_metrics(aggregated)
        stats = self.get_quick_stats()
        stats.to_csv("data/quick_stats.csv", index=False)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--evaluate", action="store_true")
    args, other = parser.parse_known_args(argv)
    rl_runner = RlRunner()
    rl_runner.load_config(args.config)
    if args.evaluate:
        rl_runner.evaluate()
    else:
        rl_runner.train()


if __name__ == "__main__":
    time1 = time.time()
    main(sys.argv)
    time2 = time.time()
    print("Overall execution time:", time2 - time1)
