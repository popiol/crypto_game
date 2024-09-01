from datetime import datetime

import numpy as np

from src.agent import Agent
from src.data_transformer import OutputFeatures, QuotesSnapshot
from src.portfolio import ClosedTransaction, Portfolio, PortfolioPosition


class BaselineAgent(Agent):

    def __init__(self, agent_name: str, metrics: dict):
        super().__init__(agent_name, None, None, None, metrics)

    def reset(self):
        pass

    def make_decision(
        self, timestamp: datetime, input: np.ndarray, quotes: QuotesSnapshot, portfolio: Portfolio, asset_list: list[str]
    ):
        output = {}
        for asset in asset_list:
            score = min(-quotes.min_ch(asset), quotes.max_ch(asset))
            relative_buy_volume = 0.5
            buy_price = min(quotes.max_val(asset) + quotes.min_ch(asset) * 0.95, quotes.closing_price(asset))
            relative_buy_price = buy_price / quotes.closing_price(asset)
            sell_price = max(quotes.min_val(asset) + quotes.max_ch(asset) * 0.95, quotes.closing_price(asset))
            relative_sell_price = sell_price / quotes.closing_price(asset)
            output[asset] = OutputFeatures(score, relative_buy_volume, relative_buy_price, relative_sell_price)
        return self._make_decision(timestamp, output, quotes, portfolio, asset_list)

    def train(self, transactions: list[ClosedTransaction] = None, positions: list[PortfolioPosition] = None):
        pass
