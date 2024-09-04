from datetime import datetime, timedelta

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
        max_dt = max([p.place_dt for p in portfolio.positions]) if portfolio.positions else None
        skip_buy = max_dt is not None and timestamp - max_dt < timedelta(days=1)
        positions = [p.asset for p in portfolio.positions]
        for asset in asset_list:
            if not skip_buy and asset not in positions and quotes.has_asset(asset):
                score = min(
                    quotes.max_val(asset) - quotes.daily_low(asset),
                    quotes.daily_high(asset) - quotes.min_val(asset),
                ) / (quotes.min_val(asset) + quotes.max_val(asset))
                relative_buy_volume = 0.5
                buy_price = quotes.daily_low(asset)
                relative_buy_price = buy_price / quotes.closing_price(asset)
            else:
                score = np.nan
                relative_buy_volume = np.nan
                relative_buy_price = np.nan
            if quotes.has_asset(asset):
                sell_price = max(quotes.daily_low(asset) * 0.5 + quotes.max_val(asset) * 0.5, quotes.closing_price(asset))
                relative_sell_price = sell_price / quotes.closing_price(asset)
            else:
                relative_sell_price = np.nan
            output[asset] = OutputFeatures(score, relative_buy_volume, relative_buy_price, relative_sell_price)
        return self._make_decision(timestamp, output, quotes, portfolio, asset_list)

    def train(self, transactions: list[ClosedTransaction] = None, positions: list[PortfolioPosition] = None):
        pass
