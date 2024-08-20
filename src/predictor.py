import argparse
import datetime
import pickle
import sys
import time

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
        raw_portfolio, agent_memory_bytes = self.environment.model_registry.get_portfolio()
        agent_memory = pickle.loads(agent_memory_bytes)
        model = self.environment.model_serializer.deserialize(serialized_model)
        agent = Agent("Leader", self.environment.data_transformer, None, TrainingStrategy(model), metrics)
        asset_list = self.environment.data_registry.get_asset_list()
        positions = [PortfolioPosition.from_json(p) for p in raw_portfolio["positions"]]
        portfolio_manager = self.environment.get_portfolio_managers(1)[0]
        portfolio_manager.portfolio = Portfolio(positions, raw_portfolio["cash"], raw_portfolio["value"])
        portfolio_manager.orders = [PortfolioOrder.from_json(o) for o in orders["orders"]]
        transactions = portfolio_manager.handle_orders(datetime.now(), quotes)
        self.environment.data_transformer.per_agent_memory[agent.agent_name] = agent_memory
        self.environment.data_transformer.add_portfolio_to_memory(agent.agent_name, [p.asset for p in positions], asset_list)
        agent_memory = self.get_agent_memory(agent.agent_name)
        agent_memory_bytes = pickle.dumps(agent_memory)
        input = self.environment.data_transformer.join_memory(shared_memory, agent_memory_bytes)
        orders = agent.make_decision(None, input, quotes, portfolio_manager.portfolio, asset_list)
        raw_portfolio = {
            "positions": [p.to_json() for p in portfolio_manager.portfolio.positions],
            "cash": portfolio_manager.portfolio.cash,
            "value": portfolio_manager.portfolio.value,
            "orders": [o.to_json() for o in orders],
        }
        self.environment.model_registry.set_portfolio(raw_portfolio, agent_memory)
        transactions = [t.to_json() for t in transactions]
        self.environment.model_registry.add_transactions(transactions)

    def choose_leader(self):
        df = pd.read_csv(self.environment.reports.quick_stats_path)
        row = df[df.score == df.score.max()].iloc[0]
        model_name = row.model
        score = row.score
        print("Best model score", score)
        metrics = self.environment.model_registry.get_leader_metrics()
        leader_score = metrics["evaluation_score"]
        print("Current leader score", leader_score)
        if score >= leader_score:
            print("Set new leader", model_name)
            self.environment.model_registry.set_leader(model_name)


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
