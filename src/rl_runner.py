import src.check_running_master  # isort: skip
import argparse
import itertools
import sys
import time
from datetime import datetime, timedelta

import numpy as np

from src.agent import Agent
from src.data_transformer import QuotesSnapshot
from src.environment import Environment
from src.logger import Logger
from src.metrics import Metrics
from src.portfolio_manager import PortfolioManager
from src.training_strategy import TrainingStrategy


class RlRunner:

    def __init__(self, environment: Environment):
        self.environment = environment
        self.training_time_hours: int = environment.config["rl_runner"]["training_time_hours"]
        self.start_dt = datetime.now()

    def prepare(self):
        self.logger = Logger()
        self.logger.log("Sync data")
        self.data_registry = self.environment.data_registry
        if not self.environment.eval_mode:
            self.data_registry.sync()
        self.data_transformer = self.environment.data_transformer
        self.asset_list = self.data_registry.get_asset_list()
        self.stats = self.data_registry.get_stats()
        if self.stats and len(self.stats["mean"]) != self.data_transformer.n_features:
            self.stats = None
        self.model_registry = self.environment.model_registry
        self.model_serializer = self.environment.model_serializer
        self.trainset = self.environment.trainset

    def quotes_iterator(self, eval_mode: bool = False):
        quotes = QuotesSnapshot()
        for timestamp, raw_quotes, bidask in self.data_registry.quotes_iterator(eval_mode):
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
        self.environment.n_assets = len(self.asset_list)

    def create_agents(self):
        self.logger.log("Create agents")
        agent_builder = self.environment.agent_builder
        self.agents = agent_builder.create_agents()
        self.portfolio_managers = self.environment.get_portfolio_managers(len(self.agents))
        self.logger.log_agents(self.agents)

    def run_agent(
        self,
        agent: Agent,
        portfolio_manager: PortfolioManager,
        timestamp: datetime,
        quotes: QuotesSnapshot,
        input: np.ndarray,
    ):
        closed_transactions = portfolio_manager.handle_orders(timestamp, quotes)
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
            self.run_agent(agent, portfolio_manager, timestamp, quotes, input)

    def reset_simulation(self):
        self.data_transformer.reset()
        for agent, portfolio_manager in zip(self.agents, self.portfolio_managers):
            agent.reset()
            portfolio_manager.reset()

    def on_simulation_end(self):
        for agent, portfolio_manager in zip(self.agents, self.portfolio_managers):
            self.logger.log_open_positions(agent.agent_name, portfolio_manager.portfolio.positions)
        # self.logger.log_simulation_results([p.portfolio for p in self.portfolio_managers])

    def train_agents(self):
        for agent, portfolio_manager in zip(self.agents, self.portfolio_managers):
            agent.train(self.logger.transactions[agent.agent_name], portfolio_manager.portfolio.positions)

    def save_models(self):
        for agent in self.agents:
            self.logger.log("Save model", agent.model_name)
            serialized_model = self.model_serializer.serialize(agent.training_strategy.model)
            metrics = Metrics(agent).get_metrics()
            self.model_registry.save_model(agent.model_name, serialized_model, metrics)

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
            self.on_simulation_end()
            if datetime.now() - self.start_dt > timedelta(hours=self.training_time_hours):
                break

    def evaluate_models(self):
        self.logger.log("Evaluate models")
        self.logger.transactions = {}
        self.agents: list[Agent] = []
        self.all_metrics = []
        model_builder = self.environment.model_builder
        for model_name, serialized_model in self.model_registry.iterate_models():
            model = self.model_serializer.deserialize(serialized_model)
            model = model_builder.adjust_dimensions(model)
            metrics = self.model_registry.get_metrics(model_name)
            agent = Agent(model_name.split("_")[0], self.data_transformer, self.trainset, TrainingStrategy(model), metrics)
            agent.model_name = model_name
            self.agents.append(agent)
        self.portfolio_managers = self.environment.get_portfolio_managers(len(self.agents))
        bitcoin_init = None
        get_bitcoin_quote = lambda q: (q.closing_price("TBTCUSD") + q.closing_price("WBTCUSD")) / 2
        for timestamp, quotes in self.quotes_iterator(eval_mode=True):
            if quotes.has_asset("TBTCUSD") and quotes.has_asset("WBTCUSD"):
                bitcoin = get_bitcoin_quote(quotes)
                if bitcoin_init is None:
                    bitcoin_init = bitcoin
            features = self.data_transformer.quotes_to_features(quotes, self.asset_list)
            features = self.data_transformer.scale_features(features, self.stats)
            if features is None:
                continue
            self.data_transformer.add_to_memory(features)
            self.run_agents(timestamp, quotes, eval_mode=True)
        bitcoin_change = bitcoin / bitcoin_init - 1
        for agent, portfolio_manager in zip(self.agents, self.portfolio_managers):
            score = portfolio_manager.portfolio.value / portfolio_manager.init_cash - 1
            metrics = Metrics(agent)
            metrics.set_evaluation_score(score)
            metrics.set_bitcoin_quote(bitcoin)
            metrics.set_bitcoin_change(bitcoin_change)
            metrics.set_n_transactions(len(self.logger.transactions.get(agent.agent_name, [])))
            metrics_dict = metrics.get_metrics()
            self.all_metrics.append(metrics_dict)
            self.model_registry.set_metrics(agent.model_name, metrics_dict)
            self.logger.log(agent.model_name, score)

    def prepare_reports(self):
        reports = self.environment.reports
        model_names = [agent.model_name for agent in self.agents]
        reports.run(model_names, self.all_metrics)

    def train(self):
        self.prepare()
        self.initial_run()
        self.create_agents()
        self.main_loop()
        self.train_agents()
        self.save_models()

    def evaluate(self):
        self.environment.eval_mode = True
        self.prepare()
        self.initial_run()
        self.evaluate_models()
        self.model_registry.archive_models()
        self.prepare_reports()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--evaluate", action="store_true")
    args, other = parser.parse_known_args(argv)
    rl_runner = RlRunner(Environment(args.config))
    if args.evaluate:
        rl_runner.evaluate()
    else:
        rl_runner.train()


if __name__ == "__main__":
    time1 = time.time()
    main(sys.argv)
    time2 = time.time()
    print("Overall execution time:", time2 - time1)
