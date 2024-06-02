from datetime import datetime

import numpy as np
import pandas as pd

from src.agent import Agent
from src.portfolio import ClosedTransaction, Portfolio


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
        self.transactions = {agent.agent_name: [] for agent in agents}

    def log_transactions(self, agent: str, transactions: list[ClosedTransaction]):
        for transaction in transactions:
            self.transactions[agent].append(f"{transaction.asset}:{transaction.profit - transaction.cost}")

    def log_simulation_results(self, portfolios: list[Portfolio]):
        results = {agent: [portfolio.value, *self.transactions[agent]] for agent, portfolio in zip(self.transactions, portfolios)}
        self.print_table(results)
