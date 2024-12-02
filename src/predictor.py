import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

from src.agent import Agent
from src.environment import Environment
from src.portfolio import Portfolio, PortfolioOrder, PortfolioPosition
from src.training_strategy import TrainingStrategy


class Predictor:

    def __init__(self, environment: Environment):
        self.environment = environment

    def predict(self):
        current_input_path = self.environment.config["predictor"]["current_input_path"]
        with open(current_input_path, "rb") as f:
            shared_memory, quotes = pickle.load(f)
        serialized_model, metrics = self.environment.model_registry.get_leader()
        raw_portfolio = self.environment.model_registry.get_leader_portfolio()
        agent_memory_bytes = self.environment.model_registry.get_leader_memory()
        agent_memory = pickle.loads(agent_memory_bytes) if agent_memory_bytes is not None else None
        asset_list = self.environment.data_registry.get_asset_list()
        model = self.environment.model_serializer.deserialize(serialized_model)
        model = self.environment.model_builder.adjust_dimensions(model)
        agent = Agent("Leader", self.environment.data_transformer, None, TrainingStrategy(model), metrics)
        portfolio_manager = self.environment.get_portfolio_managers(1)[0]
        portfolio_manager.debug = True
        if not raw_portfolio:
            raw_portfolio = {
                "positions": [],
                "cash": portfolio_manager.init_cash,
                "value": portfolio_manager.init_cash,
                "orders": [],
            }
        positions = [PortfolioPosition.from_json(p) for p in raw_portfolio["positions"]]
        portfolio_manager.portfolio = Portfolio(positions, raw_portfolio["cash"], raw_portfolio["value"])
        portfolio_manager.place_orders(datetime.now(), [PortfolioOrder.from_json(o) for o in raw_portfolio["orders"]])
        transactions = portfolio_manager.handle_orders(datetime.now(), quotes)
        self.environment.data_transformer.per_agent_memory[agent.agent_name] = agent_memory
        self.environment.data_transformer.add_portfolio_to_memory(agent.agent_name, [p.asset for p in positions], asset_list)
        agent_memory = self.environment.data_transformer.get_agent_memory(agent.agent_name)
        input = self.environment.data_transformer.join_memory(shared_memory, agent_memory)
        orders = agent.make_decision(datetime.now(), input, quotes, portfolio_manager.portfolio, asset_list)
        raw_portfolio = {
            "positions": [p.to_json() for p in portfolio_manager.portfolio.positions],
            "cash": portfolio_manager.portfolio.cash,
            "value": portfolio_manager.portfolio.value,
            "orders": [o.to_json() for o in orders],
        }
        agent_memory_bytes = pickle.dumps(agent_memory)
        self.environment.model_registry.set_leader_portfolio(raw_portfolio)
        self.environment.model_registry.set_leader_memory(agent_memory_bytes)
        with open(self.environment.reports.portfolio_path, "w") as f:
            json.dump(raw_portfolio, f)
        transactions = [t.to_json() for t in transactions]
        self.environment.model_registry.add_transactions(transactions, self.environment.reports.transactions_path)

    def choose_leader(self):
        self.environment.model_registry.download_report(self.environment.reports.quick_stats_path)
        df = pd.read_csv(self.environment.reports.quick_stats_path)
        df = df[~df.model.str.startswith("Baseline_")]
        leader = df[df.model.str.startswith("Leader_")].iloc[0]
        df = df[df.n_ancestors >= df.n_ancestors.mean()]
        maturity_min_hours = self.environment.model_registry.maturity_min_hours
        maturity_dt = (datetime.now() - timedelta(hours=maturity_min_hours)).strftime("%Y%m%d%H:%M:%S")
        df = df[df.model.apply(lambda x: x.split("_")[1]) < maturity_dt]
        print("Current leader score", leader.score, "with", leader.n_transactions, "transactions")
        best_model = df[df.score == df.score.max()].iloc[0]
        print("Best model score", best_model.score, "with", best_model.n_transactions, "transactions")
        active_models = df[df.n_transactions > 0]
        best_active = active_models[active_models.score == active_models.score.max()].iloc[0]
        print("Best active model score", best_active.score, "with", best_active.n_transactions, "transactions")
        new_model = leader
        if leader.n_transactions == 0 and best_active.score > 0.05:
            new_model = best_active
        elif best_model.score > leader.score:
            new_model = best_model
        if new_model.model != leader.model:
            print("Set new leader", new_model.model)
            self.environment.model_registry.set_leader(new_model.model)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yml")
    parser.add_argument("--choose_leader", action="store_true")
    args, other = parser.parse_known_args(argv)

    environment = Environment(args.config)
    predictor = Predictor(environment)

    if args.choose_leader:
        predictor.choose_leader()
    else:
        predictor.predict()


if __name__ == "__main__":
    time1 = time.time()
    main(sys.argv)
    time2 = time.time()
    print("Overall execution time:", time2 - time1)
