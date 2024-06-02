import numpy as np
import pandas as pd

from src.agent import Agent
from src.portfolio import Portfolio


class Logger:

    def __init__(self, log_portfolio_once_per: int):
        self.log_portfolio_once_per = log_portfolio_once_per
        self.log_portfolio_iter = 0

    def log_portfolios(self, agents: list[Agent], portfolios: list[Portfolio], force: bool = False):
        if force or self.log_portfolio_iter % self.log_portfolio_once_per == 0:
            max_len = max(len(p.positions) for p in portfolios)
            data = np.full((max_len + 1, len(agents)), "", (np.unicode_, 20))
            data[0] = [round(p.value, 2) for p in portfolios]
            for col, portfolio in enumerate(portfolios):
                for row, position in enumerate(portfolio.positions):
                    data[row + 1, col] = f"{position.asset}:{round(position.value,2)}"
            columns = ["{0: >15}".format(a.agent_name) for a in agents]
            df = pd.DataFrame(data, columns=columns)
            print(df.to_string(index=False))
            print()
        self.log_portfolio_iter += 1
