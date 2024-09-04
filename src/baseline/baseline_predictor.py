import argparse
import json
import pickle
import sys
import time
from datetime import datetime

from src.baseline.baseline_agent import BaselineAgent
from src.environment import Environment
from src.portfolio import Portfolio, PortfolioOrder, PortfolioPosition


class BaselinePredictor:

    def __init__(self, environment: Environment):
        self.environment = environment

    def predict(self):
        current_input_path = self.environment.config["predictor"]["current_input_path"]
        with open(current_input_path, "rb") as f:
            shared_memory, quotes = pickle.load(f)
        raw_portfolio = self.environment.model_registry.get_baseline_portfolio()
        metrics = self.environment.model_registry.get_baseline_metrics()
        asset_list = self.environment.data_registry.get_asset_list()
        agent = BaselineAgent("Baseline", metrics)
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
        orders = agent.make_decision(datetime.now(), None, quotes, portfolio_manager.portfolio, asset_list)
        raw_portfolio = {
            "positions": [p.to_json() for p in portfolio_manager.portfolio.positions],
            "cash": portfolio_manager.portfolio.cash,
            "value": portfolio_manager.portfolio.value,
            "orders": [o.to_json() for o in orders],
        }
        self.environment.model_registry.set_baseline_portfolio(raw_portfolio)
        with open(self.environment.reports.baseline_portfolio_path, "w") as f:
            json.dump(raw_portfolio, f)
        transactions = [t.to_json() for t in transactions]
        self.environment.model_registry.add_baseline_transactions(
            transactions, self.environment.reports.baseline_transactions_path
        )


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yml")
    args, other = parser.parse_known_args(argv)

    environment = Environment(args.config)
    predictor = BaselinePredictor(environment)
    predictor.predict()


if __name__ == "__main__":
    time1 = time.time()
    main(sys.argv)
    time2 = time.time()
    print("Overall execution time:", time2 - time1)
