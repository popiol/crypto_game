import uuid
from datetime import datetime
from socket import gethostname

import numpy as np

from src.data_transformer import DataTransformer, QuotesSnapshot
from src.ml_model import MlModel
from src.portfolio import (
    ClosedTransaction,
    Portfolio,
    PortfolioOrder,
    PortfolioOrderType,
)
from src.training_strategy import StrategyPicker
from src.trainset import Trainset


class Agent:

    def __init__(
        self,
        agent_name: str,
        model: MlModel,
        data_transformer: DataTransformer,
        trainset: Trainset,
        strategy_picker: StrategyPicker,
    ):
        self.agent_name = agent_name
        self.model = model
        self.data_transformer = data_transformer
        self.trainset = trainset
        self.model_id = uuid.uuid4().hex[:5]
        model_dt = datetime.now().strftime("%Y%m%d")
        host_name = gethostname()
        self.model_name = f"{agent_name}_{host_name}_{model_dt}_{self.model_id}"
        self.metrics = {}
        self.training_strategy = strategy_picker.pick()

    def make_decision(
        self, timestamp: datetime, input: np.array, quotes: QuotesSnapshot, portfolio: Portfolio, asset_list: list[str]
    ) -> list[PortfolioOrder]:
        output_matrix = self.training_strategy.predict(self.model, input)
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
            self.training_strategy.train(self.model, input, output, reward)
