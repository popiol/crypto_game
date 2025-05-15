import argparse
import heapq
import itertools
import pickle
import random
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.agent import Agent
from src.baseline.baseline_agent import BaselineAgent
from src.constants import RlTrainset
from src.data_transformer import QuotesSnapshot
from src.environment import Environment
from src.logger import Logger
from src.metrics import Metrics
from src.portfolio_manager import PortfolioManager
from src.training_strategy import TrainingStrategy


class RlRunner:

    def __init__(self, environment: Environment):
        self.environment = environment
        self.training_time_min: int = environment.config["rl_runner"]["training_time_min"]
        self.start_dt = datetime.now()
        self.rl_trainset = []

    def prepare(self):
        self.logger = Logger()
        self.logger.log("Sync data")
        self.data_registry = self.environment.data_registry
        self.data_registry.sync()
        self.data_transformer = self.environment.data_transformer
        self.asset_list = self.environment.asset_list
        self.stats = self.data_registry.get_stats()
        if self.stats and len(self.stats["mean"]) != self.data_transformer.n_features:
            self.stats = None
        self.model_registry = self.environment.model_registry
        self.model_serializer = self.environment.model_serializer
        self.trainset = self.environment.trainset

    def next_quotes(self, quotes: QuotesSnapshot, file: str):
        raw_quotes, bidask = self.data_registry.get_quotes_and_bidask(file)
        quotes.update(raw_quotes)
        quotes.update_bid_ask(bidask)
        quotes.update_custom()
        return quotes

    def next_features(self, quotes: QuotesSnapshot):
        features = self.data_transformer.quotes_to_features(quotes, self.asset_list)
        raw_features = features.copy()
        features = self.data_transformer.scale_features(features, self.stats)
        return features, raw_features

    def quotes_iterator(self, scale: bool = True):
        quotes = QuotesSnapshot()
        cache = self.environment.cache
        for file, timestamp, preprocess in self.data_registry.files_and_timestamps(self.environment.eval_mode):
            quotes = cache.get(self.next_quotes, timestamp, quotes, file)
            if scale:
                features, raw_features = cache.get(self.next_features, timestamp, quotes)
                self.data_transformer.last_features = raw_features
            else:
                features = self.data_transformer.quotes_to_features(quotes, self.asset_list)
            yield timestamp, quotes, features, preprocess

    def initial_run(self):
        self.logger.log("Initial run")
        for timestamp, quotes, features, preprocess in self.quotes_iterator(scale=False):
            if not self.stats:
                self.data_transformer.add_to_stats(features)
        if not self.stats:
            self.stats = self.data_transformer.stats
            self.data_registry.set_stats(self.stats)
        self.data_registry.set_asset_list(self.asset_list)
        self.data_registry.set_current_assets(self.data_transformer.current_assets)

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
        if not self.environment.eval_mode:
            rl_trainset = agent.train(transactions=closed_transactions)
            self.extend_rl_trainset(rl_trainset)
        if not self.environment.eval_mode:
            portfolio_manager.make_deposit(portfolio_manager.init_cash - portfolio_manager.portfolio.cash)
        orders = agent.make_decision(timestamp, input, quotes, portfolio_manager.portfolio, self.asset_list)
        portfolio_manager.place_orders(timestamp, orders)
        self.logger.log_transactions(agent.model_name, closed_transactions)

    def run_agents(self, timestamp: datetime, quotes: QuotesSnapshot):
        if not self.environment.eval_mode:
            self.trainset.store_shared_input(timestamp, self.data_transformer.get_shared_memory())
        for agent, portfolio_manager in zip(self.agents, self.portfolio_managers):
            self.data_transformer.add_portfolio_to_memory(
                agent.agent_name, [p.asset for p in portfolio_manager.portfolio.positions], self.asset_list
            )
            input = self.data_transformer.get_memory(agent.agent_name)
            if not self.environment.eval_mode:
                self.trainset.store_agent_input(
                    timestamp, self.data_transformer.get_agent_memory(agent.agent_name), agent.agent_name
                )
            self.run_agent(agent, portfolio_manager, timestamp, quotes, input)

    def reset_simulation(self):
        self.data_transformer.reset()
        for agent, portfolio_manager in zip(self.agents, self.portfolio_managers):
            agent.reset()
            portfolio_manager.reset()

    def train_on_historical(self):
        for agent in self.agents:
            try:
                rl_trainset = self.data_registry.get_random_trainset(self.asset_list, self.data_transformer.current_assets)
            except IndexError:
                continue
            agent.train(historical=rl_trainset)
        self.logger.log_reward(self.agents)

    def pretrain(self):
        if any([agent.metrics.get("parents") for agent in self.agents]):
            if random.random() < 0.75:
                return
        self.logger.log("Pretrain as a sequence predictor")
        inputs = []
        start_timestamp = None
        start_with_offset = None
        outputs = []
        agent_memory = np.zeros((self.data_transformer.memory_length, len(self.asset_list), 1))
        for offset in range(3):
            self.reset_simulation()
            inputs_ = None
            for timestamp, quotes, features, preprocess in self.quotes_iterator():
                if features is None:
                    continue
                self.data_transformer.add_to_memory(features)
                if preprocess:
                    continue
                if start_timestamp is None:
                    start_timestamp = timestamp
                if timestamp - start_timestamp < timedelta(days=offset):
                    continue
                if inputs_ is None:
                    inputs_ = self.data_transformer.get_shared_memory()
                    inputs_ = np.concatenate((inputs_, agent_memory), axis=-1)
                    inputs.append(inputs_)
                    start_with_offset = timestamp
                if timestamp - start_with_offset > timedelta(days=3, hours=23):
                    outputs_ = self.data_transformer.get_shared_memory()
                    outputs_ = np.concatenate((outputs_, agent_memory), axis=-1)
                    outputs.append(outputs_)
                    break
            if len(inputs) > len(outputs):
                outputs_ = self.data_transformer.get_shared_memory()
                outputs_ = np.concatenate((outputs_, agent_memory), axis=-1)
                outputs.append(outputs_)
            if inputs is None or outputs is None:
                self.logger.log("Pretrain failed")
                return
        for agent in self.agents:
            self.environment.model_builder.pretrain_with(
                agent.training_strategy.model,
                self.asset_list,
                self.data_transformer.current_assets,
                np.array(inputs),
                np.array(outputs),
            )

    def train_on_open_positions(self):
        for agent, portfolio_manager in zip(self.agents, self.portfolio_managers):
            rl_trainset = agent.train(positions=portfolio_manager.portfolio.positions)
            self.extend_rl_trainset(rl_trainset)

    def should_save_rl_trainset(self):
        return random.random() < 0.05

    def extend_rl_trainset(self, rl_trainset: RlTrainset):
        indices = [index for index, asset in enumerate(self.asset_list) if asset in self.data_transformer.current_assets]
        for input, output, reward in rl_trainset:
            wide_output = np.zeros((output.shape[0], len(self.asset_list), output.shape[2]))
            wide_output[:, indices] = output
            self.rl_trainset.append((input, wide_output, reward))

    def get_best_rl_trainset(self, n_records: int = 100):
        top = heapq.nlargest(n_records // 2, self.rl_trainset, key=lambda x: x[2])
        bottom = heapq.nsmallest(n_records // 2, self.rl_trainset, key=lambda x: x[2])
        return bottom + top

    def save_rl_trainset(self):
        if not self.should_save_rl_trainset():
            return
        self.data_registry.add_to_trainset(self.get_best_rl_trainset())

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
            for timestamp, quotes, features, preprocess in self.quotes_iterator():
                if features is None:
                    continue
                self.data_transformer.add_to_memory(features)
                if not preprocess:
                    self.run_agents(timestamp, quotes)
            self.train_on_open_positions()
            self.logger.log_reward(self.agents)
            if datetime.now() - self.start_dt > timedelta(minutes=self.training_time_min):
                break

    def save_current_input(self, quotes: QuotesSnapshot):
        current_input_path = self.environment.config["predictor"]["current_input_path"]
        with open(current_input_path, "wb") as f:
            memory = self.data_transformer.get_shared_memory()
            pickle.dump((memory, quotes), f)

    def evaluate_models(self):
        self.logger.log("Evaluate models")
        self.logger.transactions = {}
        self.agents: list[Agent] = []
        self.all_metrics = []
        model_builder = self.environment.model_builder
        for model_name, serialized_model in self.model_registry.iterate_models():
            print(model_name)
            try:
                model = self.model_serializer.deserialize(serialized_model)
                model = model_builder.adjust_dimensions(model)
                model = model_builder.filter_assets(model, self.asset_list, self.data_transformer.current_assets)
                metrics = self.model_registry.get_metrics(model_name)
                agent = Agent(model_name.split("_")[0], self.data_transformer, None, TrainingStrategy(model), metrics)
                agent.model_name = model_name
                self.agents.append(agent)
            except:
                self.model_registry.archive_model(model_name)
        serialized_model, metrics = self.model_registry.get_leader()
        model = self.model_serializer.deserialize(serialized_model)
        model = model_builder.adjust_dimensions(model)
        model = model_builder.filter_assets(model, self.asset_list, self.data_transformer.current_assets)
        self.agents.append(Agent("Leader", self.data_transformer, None, TrainingStrategy(model), metrics))
        metrics = self.model_registry.get_baseline_metrics()
        self.agents.append(BaselineAgent("Baseline", self.data_transformer, metrics))
        self.portfolio_managers = self.environment.get_portfolio_managers(len(self.agents))
        bitcoin_init = None
        get_bitcoin_quote = lambda q: (q.closing_price("TBTCUSD") + q.closing_price("WBTCUSD")) / 2
        for timestamp, quotes, features, preprocess in self.quotes_iterator():
            if features is None:
                continue
            self.data_transformer.add_to_memory(features)
            if not preprocess:
                if quotes.has_asset("TBTCUSD") and quotes.has_asset("WBTCUSD"):
                    bitcoin = get_bitcoin_quote(quotes)
                    if bitcoin_init is None:
                        bitcoin_init = bitcoin
                self.run_agents(timestamp, quotes)
        bitcoin_change = bitcoin / bitcoin_init - 1
        self.save_current_input(quotes)
        for agent, portfolio_manager in zip(self.agents, self.portfolio_managers):
            score = portfolio_manager.portfolio.value / portfolio_manager.init_cash - 1
            metrics = Metrics(agent)
            metrics.set_evaluation_score(score)
            metrics.set_bitcoin_quote(bitcoin)
            metrics.set_bitcoin_change(bitcoin_change)
            metrics.set_n_transactions(len(self.logger.transactions.get(agent.model_name, [])))
            metrics_dict = metrics.get_metrics()
            self.all_metrics.append(metrics_dict)
            if agent.agent_name == "Leader":
                self.model_registry.set_leader_metrics(metrics_dict)
            elif agent.agent_name == "Baseline":
                self.model_registry.set_baseline_metrics(metrics_dict)
            else:
                self.model_registry.set_metrics(agent.model_name, metrics_dict)
            self.logger.log(agent.model_name, score)

    def get_model_correlations(self):
        correlations = pd.DataFrame(columns=["model_1", "model_2", "correlation", "score_1", "score_2"])
        for agent_1, metrics_1 in zip(self.agents, self.all_metrics):
            if agent_1.agent_name in ["Leader", "Baseline"]:
                continue
            for agent_2, metrics_2 in zip(self.agents, self.all_metrics):
                if agent_2.agent_name in ["Leader", "Baseline"]:
                    continue
                if agent_1.model_name >= agent_2.model_name:
                    continue
                correlation = 0
                if metrics_1["n_ancestors"] > 0 and metrics_2["n_ancestors"] > 0:
                    ancestors_1 = set(Metrics.parents_as_list(metrics_1["parents"]))
                    ancestors_2 = set(Metrics.parents_as_list(metrics_2["parents"]))
                    correlation = len(ancestors_1 & ancestors_2) / len(ancestors_1.union(ancestors_2))
                correlations.loc[len(correlations)] = [
                    agent_1.model_name,
                    agent_2.model_name,
                    correlation,
                    metrics_1["evaluation_score"],
                    metrics_2["evaluation_score"],
                ]
                correlations.loc[len(correlations)] = [
                    agent_2.model_name,
                    agent_1.model_name,
                    correlation,
                    metrics_2["evaluation_score"],
                    metrics_1["evaluation_score"],
                ]
        return correlations

    def archive_models(self):
        df = self.get_model_correlations()
        min_score = df.score_1.min()
        max_score = df.score_1.max()
        df["score"] = df.apply(lambda x: x.score_1 - x.correlation * (max_score - min_score) / 2, axis=1)
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", None):
            print(df[df.correlation > 0].sort_values("correlation", ascending=False))
        best = df[df.score_1 == max_score]
        best.score = best.score_1
        best = best.set_index("model_1").score.to_dict()
        df = df[df.score_1 < df.score_2]
        df = df[["model_1", "score"]]
        df = df.groupby("model_1").min().score.to_dict()
        scores = {**best, **df}
        self.model_registry.archive_models(scores)

    def prepare_reports(self):
        reports = self.environment.reports
        model_names = [agent.model_name for agent in self.agents]
        reports.run(model_names, self.all_metrics)

    def train(self):
        self.prepare()
        self.initial_run()
        self.create_agents()
        self.pretrain()
        self.train_on_historical()
        self.main_loop()
        self.save_rl_trainset()
        self.save_models()

    def evaluate(self):
        self.environment.eval_mode = True
        self.prepare()
        self.initial_run()
        self.evaluate_models()
        self.archive_models()
        self.prepare_reports()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yml")
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
