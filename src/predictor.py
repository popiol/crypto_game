import argparse
import json
import pickle
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta

import pandas as pd

from src.agent import Agent
from src.environment import Environment
from src.kraken_api import KrakenApi
from src.portfolio import (
    ClosedTransaction,
    Portfolio,
    PortfolioOrder,
    PortfolioPosition,
)
from src.training_strategy import TrainingStrategy


class Predictor:

    def __init__(self, environment: Environment):
        self.environment = environment

    def update_positions_and_transactions(
        self, positions: list[PortfolioPosition], new_positions: list[PortfolioPosition], transactions: list[ClosedTransaction]
    ):
        for transaction in transactions:
            matched = False
            for position in positions:
                if transaction.asset == position.asset:
                    position.volume = 0
                    matched = True
                    transaction.buy_price = position.buy_price
                    transaction.place_buy_dt = position.place_dt
                    transaction.cost = position.cost
            if not matched:
                raise Exception("Unmatched closing transaction", transaction, "for positions", positions)
        updated = new_positions.copy()
        for position in positions:
            if position.volume == 0:
                continue
            if [p for p in updated if p.asset == position.asset]:
                continue
            updated.append(position)
        return updated

    def place_real_orders(self):
        current_input_path = self.environment.config["predictor"]["current_input_path"]
        with open(current_input_path, "rb") as f:
            shared_memory, quotes = pickle.load(f)
        serialized_model, metrics = self.environment.model_registry.get_leader()
        agent_memory_bytes = self.environment.model_registry.get_real_memory()
        agent_memory = pickle.loads(agent_memory_bytes) if agent_memory_bytes is not None else None
        asset_list = self.environment.data_registry.get_asset_list()
        model = self.environment.model_serializer.deserialize(serialized_model)
        model = self.environment.model_builder.adjust_dimensions(model)
        raw_portfolio = self.environment.model_registry.get_real_portfolio()
        portfolio_api = KrakenApi()
        cash = portfolio_api.get_cash()
        if not raw_portfolio:
            raw_portfolio = {
                "positions": [],
                "cash": cash,
                "value": None,
                "orders": [],
            }
        positions = [PortfolioPosition.from_json(p) for p in raw_portfolio["positions"]]
        print("old positions", positions)
        last_update = self.environment.model_registry.get_real_portfolio_last_update() or datetime.now() - timedelta(hours=1)
        since = last_update - timedelta(minutes=1)
        portfolio_manager = self.environment.get_portfolio_managers(1)[0]
        new_positions = portfolio_api.get_positions(since)
        print("new_positions", new_positions)
        transactions = portfolio_api.get_closed_transactions(since)
        positions = self.update_positions_and_transactions(positions, new_positions, transactions)
        positions = [p for p in positions if p.asset != "HNTUSD"]
        print("updated positions", positions)
        portfolio = Portfolio(positions, cash, None)
        portfolio.update_value(quotes)
        portfolio.positions = [p for p in portfolio.positions if p.value >= portfolio_manager.min_transaction * 0.5]
        data_transformer = self.environment.data_transformer
        agent = Agent("Leader", data_transformer, None, TrainingStrategy(model), metrics)
        data_transformer.per_agent_memory[agent.agent_name] = agent_memory
        data_transformer.add_portfolio_to_memory(agent.agent_name, [p.asset for p in portfolio.positions], asset_list)
        agent_memory = data_transformer.get_agent_memory(agent.agent_name)
        input = data_transformer.join_memory(shared_memory, agent_memory)
        data_transformer.current_assets = self.environment.data_registry.get_current_assets()
        orders = agent.make_decision(datetime.now(), input, quotes, portfolio, asset_list)
        portfolio_manager.debug = True
        portfolio_manager.portfolio = deepcopy(portfolio)
        portfolio_manager.orders = portfolio_api.get_orders()
        timestamp = datetime.now()
        portfolio_manager.place_orders(timestamp, orders)
        portfolio_manager.precision = portfolio_api.get_precision([o.asset for o in portfolio_manager.orders])
        portfolio_manager.adjust_orders(quotes)
        orders = [o for o in portfolio_manager.orders if o.place_dt == timestamp]
        for order in orders:
            portfolio_api.place_order(order)
        placed_orders = portfolio_api.get_orders()
        raw_portfolio = {
            "positions": [p.to_json() for p in portfolio.positions],
            "cash": cash,
            "value": portfolio.value,
            "orders": [o.to_json() for o in placed_orders],
        }
        agent_memory_bytes = pickle.dumps(agent_memory)
        print("raw_portfolio", raw_portfolio)
        self.environment.model_registry.set_real_portfolio(raw_portfolio)
        self.environment.model_registry.set_real_memory(agent_memory_bytes)
        with open(self.environment.reports.real_portfolio_path, "w") as f:
            json.dump(raw_portfolio, f)
        raw_transactions = [t.to_json() for t in transactions]
        self.environment.model_registry.add_real_transactions(raw_transactions, self.environment.reports.real_transactions_path)

    def simulate(self):
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
        data_transformer = self.environment.data_transformer
        data_transformer.per_agent_memory[agent.agent_name] = agent_memory
        data_transformer.add_portfolio_to_memory(agent.agent_name, [p.asset for p in positions], asset_list)
        agent_memory = data_transformer.get_agent_memory(agent.agent_name)
        input = data_transformer.join_memory(shared_memory, agent_memory)
        data_transformer.current_assets = self.environment.data_registry.get_current_assets()
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
    parser.add_argument("--real", action="store_true")
    args, other = parser.parse_known_args(argv)

    environment = Environment(args.config)
    predictor = Predictor(environment)

    if args.choose_leader:
        predictor.choose_leader()
    elif args.real:
        predictor.place_real_orders()
    else:
        predictor.simulate()


if __name__ == "__main__":
    time1 = time.time()
    main(sys.argv)
    time2 = time.time()
    print("Overall execution time:", time2 - time1)
