import uuid
from datetime import datetime, timedelta

import numpy as np

from src.constants import RlTrainset
from src.data_transformer import DataTransformer, OutputFeatures, QuotesSnapshot
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

    def _make_decision(
        self,
        timestamp: datetime,
        output: dict[str, OutputFeatures],
        quotes: QuotesSnapshot,
        portfolio: Portfolio,
        asset_list: list[str],
        limit_purchase: bool = True,
    ) -> list[PortfolioOrder]:
        orders = []
        for position in portfolio.positions:
            sell_price = quotes.closing_price(position.asset) * output[position.asset].relative_sell_price
            if position.last_sell_price is not None:
                sell_price = sell_price * 0.1 + position.last_sell_price * 0.9
            position.last_sell_price = sell_price
            sell_order = PortfolioOrder(
                order_type=PortfolioOrderType.sell,
                asset=position.asset,
                volume=position.volume,
                price=sell_price,
                place_dt=timestamp,
            )
            if sell_order.price > 0 and sell_order.volume > 0:
                orders.append(sell_order)
        if limit_purchase and timestamp > datetime.now() - timedelta(hours=1) and portfolio.positions and max(p.place_dt for p in portfolio.positions) > datetime.now() - timedelta(days=2):
            scores = []
        else:
            scores = [
                (
                    features.score
                    if quotes.has_asset(asset)
                    and 0.9 < features.relative_buy_price <= 1
                    and features.relative_buy_volume > 0
                    and 1.06 < features.score
                    and not asset.startswith("USD")
                    and asset not in [p.asset for p in portfolio.positions]
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
                predicted_score=float(np.nanmax(scores)),
            )
            orders.append(buy_order)
        return orders

    def make_decision(
            self, timestamp: datetime, input: np.ndarray, quotes: QuotesSnapshot, portfolio: Portfolio, asset_list: list[str], limit_purchase: bool = True
    ):
        output_matrix = self.training_strategy.predict(input)
        current_asset_list = self.data_transformer.get_current_asset_list(asset_list)
        if self.trainset:
            self.trainset.store_output(timestamp, output_matrix, self.agent_name)
        output = self.data_transformer.transform_output(output_matrix, current_asset_list)
        return self._make_decision(timestamp, output, quotes, portfolio, current_asset_list, limit_purchase)

    def get_input_output(self, timestamp: datetime) -> tuple[np.ndarray, np.ndarray]:
        shared_input, agent_input, output = self.trainset.get_by_timestamp(timestamp, self.agent_name)
        input = self.data_transformer.join_memory(shared_input, agent_input)
        return input, output

    def train(
        self,
        transactions: list[ClosedTransaction] = None,
        positions: list[PortfolioPosition] = None,
        historical: RlTrainset = None,
    ):
        if historical:
            print("train on historical", len(historical))
            for record in historical:
                self.training_strategy.train(*record)
        rl_trainset: RlTrainset = []
        if transactions:
            for transaction in transactions:
                buy_input, buy_output = self.get_input_output(transaction.place_buy_dt)
                sell_input, sell_output = self.get_input_output(transaction.place_sell_dt)
                input = np.array([buy_input, sell_input])
                output = np.array([buy_output, sell_output])
                reward = (transaction.sell_price - transaction.buy_price) * transaction.volume
                self.training_strategy.train(input, output, reward)
                rl_trainset.append((input, output, reward))
        if positions:
            for position in positions:
                buy_input, buy_output = self.get_input_output(position.place_dt)
                input = np.array([buy_input])
                output = np.array([buy_output])
                reward = position.value - position.buy_price * position.volume
                self.training_strategy.train(input, output, reward)
                rl_trainset.append((input, output, reward))
        return rl_trainset
