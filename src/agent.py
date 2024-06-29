import uuid
from datetime import datetime

import numpy as np

from src.data_transformer import DataTransformer, QuotesSnapshot
from src.portfolio import (
    ClosedTransaction,
    Portfolio,
    PortfolioOrder,
    PortfolioOrderType,
    PortfolioPosition,
)
from src.training_strategy import TrainingStrategy
from src.trainset import Trainset


class Agent:

    def __init__(
        self,
        agent_name: str,
        data_transformer: DataTransformer,
        trainset: Trainset,
        training_strategy: TrainingStrategy,
        metrics: dict,
    ):
        self.agent_name = agent_name
        self.data_transformer = data_transformer
        self.trainset = trainset
        self.training_strategy = training_strategy
        self.metrics = metrics
        self.model_id_len = 5
        self.model_id = uuid.uuid4().hex[: self.model_id_len]
        model_dt = datetime.now().strftime("%Y%m%d%H%M%S")
        self.model_name = f"{agent_name}_{model_dt}_{self.model_id}"

    def reset(self):
        self.training_strategy.reset()

    def make_decision(
        self, timestamp: datetime, input: np.array, quotes: QuotesSnapshot, portfolio: Portfolio, asset_list: list[str]
    ) -> list[PortfolioOrder]:
        output_matrix = self.training_strategy.predict(input)
        self.trainset.store_output(timestamp, output_matrix, self.agent_name)
        output = self.data_transformer.transform_output(output_matrix, asset_list)
        orders = []
        for position in portfolio.positions:
            sell_order = PortfolioOrder(
                order_type=PortfolioOrderType.sell,
                asset=position.asset,
                volume=position.volume,
                price=quotes.closing_price(position.asset) * output[position.asset].relative_sell_price,
            )
            orders.append(sell_order)
        scores = [features.score if asset in quotes.quotes else np.nan for asset, features in output.items()]
        best_asset_index = np.nanargmax(scores)
        best_asset = asset_list[best_asset_index]
        cost = portfolio.cash * output[best_asset].relative_buy_volume
        buy_price = quotes.closing_price(best_asset) * output[best_asset].relative_buy_price
        buy_order = PortfolioOrder(
            order_type=PortfolioOrderType.buy,
            asset=best_asset,
            volume=cost / buy_price,
            price=buy_price,
        )
        orders.append(buy_order)
        return orders

    def train(self, closed_transactions: list[ClosedTransaction]):
        for transaction in closed_transactions:
            buy_input, buy_output = self.trainset.get_by_timestamp(transaction.place_buy_dt, self.agent_name)
            sell_input, sell_output = self.trainset.get_by_timestamp(transaction.place_sell_dt, self.agent_name)
            input = np.array([buy_input, sell_input])
            output = np.array([buy_output, sell_output])
            reward = (transaction.sell_price - transaction.buy_price) * transaction.volume
            self.training_strategy.train(input, output, reward)

    def train_on_open_positions(self, positions: list[PortfolioPosition]):
        for position in positions:
            buy_input, buy_output = self.trainset.get_by_timestamp(position.place_dt, self.agent_name)
            input = np.array([buy_input])
            output = np.array([buy_output])
            reward = position.value - position.buy_price * position.volume
            self.training_strategy.train(input, output, reward)
