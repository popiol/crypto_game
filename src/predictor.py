import argparse
import pickle
import sys
import time

import pandas as pd

from src.agent import Agent
from src.environment import Environment
from src.portfolio import Portfolio, PortfolioPosition
from src.training_strategy import TrainingStrategy


class Predictor:

    def __init__(self, environment: Environment):
        self.environment = environment

    def predict(self):
        current_input_path = self.environment.config["predictor"]["current_input_path"]
        with open(current_input_path, "rb") as f:
            shared_memory, quotes = pickle.load(f)
        serialized_model, metrics, raw_portfolio, agent_memory = self.environment.model_registry.get_leader()
        model = self.environment.model_serializer.deserialize(serialized_model)
        agent = Agent("Leader", self.environment.data_transformer, None, TrainingStrategy(model), metrics)
        asset_list = self.environment.data_registry.get_asset_list()
        positions = [PortfolioPosition(**p) for p in raw_portfolio["positions"]]
        portfolio = Portfolio(positions, raw_portfolio["cash"], raw_portfolio["value"])
        # handle orders
        self.environment.data_transformer.per_agent_memory[agent.agent_name] = agent_memory
        self.environment.data_transformer.add_portfolio_to_memory(agent.agent_name, [p.asset for p in positions], asset_list)
        updated_agent_memory = self.get_agent_memory(agent.agent_name)
        input = self.environment.data_transformer.join_memory(shared_memory, updated_agent_memory)
        orders = agent.make_decision(None, input, quotes, portfolio, asset_list)
        # persist portfolio, orders and memory

    def choose_leader(self):
        df = pd.read_csv(self.environment.reports.quick_stats_path)
        row = df[df.score == df.score.max()].iloc[0]
        model_name = row.model
        score = row.score
        print("Set leader", model_name)
        print("Score", score)
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
