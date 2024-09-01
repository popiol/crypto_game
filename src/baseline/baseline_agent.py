from datetime import datetime

import numpy as np

from src.agent import Agent
from src.data_transformer import QuotesSnapshot
from src.portfolio import (
    ClosedTransaction,
    Portfolio,
    PortfolioOrder,
    PortfolioOrderType,
    PortfolioPosition,
)


class BaselineAgent(Agent):

    def __init__(self, agent_name: str, metrics: dict):
        super().__init__(agent_name, None, None, None, metrics)

    def reset(self):
        pass

    def make_decision(
        self, timestamp: datetime, input: np.ndarray, quotes: QuotesSnapshot, portfolio: Portfolio, asset_list: list[str]
    ) -> list[PortfolioOrder]:
        output_matrix = self.training_strategy.predict(input)
        if self.trainset:
            self.trainset.store_output(timestamp, output_matrix, self.agent_name)
        output = self.data_transformer.transform_output(output_matrix, asset_list)
        orders = []
        for position in portfolio.positions:
            sell_order = PortfolioOrder(
                order_type=PortfolioOrderType.sell,
                asset=position.asset,
                volume=position.volume,
                price=quotes.closing_price(position.asset) * output[position.asset].relative_sell_price,
                place_dt=timestamp,
            )
            if sell_order.price > 0 and sell_order.volume > 0:
                orders.append(sell_order)
        scores = [
            (
                features.score
                if asset in quotes.quotes and features.relative_buy_price > 0.9 and features.relative_buy_volume > 0
                else np.nan
            )
            for asset, features in output.items()
        ]
        if not np.isnan(scores).all():
            best_asset_index = np.nanargmax(scores)
            best_asset = asset_list[best_asset_index]
            cost = portfolio.cash * output[best_asset].relative_buy_volume
            buy_price = quotes.closing_price(best_asset) * output[best_asset].relative_buy_price
            buy_order = PortfolioOrder(
                order_type=PortfolioOrderType.buy,
                asset=best_asset,
                volume=cost / buy_price,
                price=buy_price,
                place_dt=timestamp,
            )
            orders.append(buy_order)
        return orders

    def train(self, transactions: list[ClosedTransaction] = None, positions: list[PortfolioPosition] = None):
        pass
