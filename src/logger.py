from datetime import datetime

import numpy as np
import pandas as pd

from src.agent import Agent
from src.portfolio import ClosedTransaction, Portfolio, PortfolioPosition


class Logger:

    def log(self, *args):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        print(timestamp, *args)

    def print_table(self, column_data: dict):
        max_len = max(len(val) for key, val in column_data.items())
        data = np.full((max_len, len(column_data)), "", (np.unicode_, 20))
        for col, key in enumerate(column_data):
            for row, value in enumerate(column_data[key]):
                data[row, col] = value
        columns = ["{0: >15}".format(key) for key in column_data]
        df = pd.DataFrame(data, columns=columns)
        print(df.to_string(index=False))
        print()

    def log_agents(self, agents: list[Agent]):
        for agent in agents:
            print(agent.agent_name, agent.training_strategy.__class__.__name__)
            print(agent.training_strategy.model)
        self.transactions = {agent.agent_name: [] for agent in agents}
        self.open_positions = {agent.agent_name: [] for agent in agents}

    def log_transactions(self, agent: str, transactions: list[ClosedTransaction]):
        self.transactions[agent] = self.transactions.get(agent, [])
        self.transactions[agent].extend(transactions)

    def log_open_positions(self, agent: str, positions: list[PortfolioPosition]):
        self.open_positions[agent] = self.open_positions.get(agent, [])
        self.open_positions[agent].extend(positions)

    def display_transaction(self, transaction: ClosedTransaction):
        return f"{transaction.asset}:{round(transaction.profit - transaction.cost, 2)}"

    def log_simulation_results(self, portfolios: list[Portfolio]):
        results = {}
        for agent, portfolio in zip(self.transactions, portfolios):
            transactions = [self.display_transaction(x) for x in self.transactions[agent]]
            results[agent] = [round(portfolio.value, 2), *transactions]
        self.print_table(results)
        for agent in self.transactions:
            self.transactions[agent] = []
        print(flush=True)
