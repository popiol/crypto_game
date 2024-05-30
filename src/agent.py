import uuid
from datetime import datetime
from socket import gethostname

import numpy as np

from src.data_transformer import DataTransformer, QuotesSnapshot
from src.ml_model import MlModel
from src.portfolio import Portfolio, PortfolioOrder, PortfolioOrderType


class Agent:

    def __init__(self, agent_name: str, model: MlModel, data_transformer: DataTransformer):
        self.agent_name = agent_name
        self.model = model
        self.data_transformer = data_transformer
        self.model_id = uuid.uuid4().hex[:5]
        model_dt = datetime.now().strftime("%Y%m%d")
        host_name = gethostname()
        self.model_name = f"{agent_name}_{host_name}_{model_dt}_{self.model_id}"
        self.metrics = {}

    def make_decision(
        self, inputs: np.array, quotes: QuotesSnapshot, portfolio: Portfolio, asset_list: list[str]
    ) -> list[PortfolioOrder]:
        output_matrix = self.model.predict(np.array([inputs]))[0]
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
        return []
