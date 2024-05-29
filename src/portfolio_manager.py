from datetime import datetime, timedelta

from src.data_transformer import QuotesSnapshot
from src.portfolio import (
    Portfolio,
    PortfolioOrder,
    PortfolioOrderType,
    PortfolioPosition,
)


class PortfolioManager:

    def __init__(self, init_cash: float, transaction_fee: float, expiration_time_sec: int):
        self.init_cash = init_cash
        self.portfolio = Portfolio([], init_cash, init_cash)
        self.transaction_fee = transaction_fee
        self.expiration_time_sec = expiration_time_sec
        self.orders: list[PortfolioOrder] = []

    def place_orders(self, timestamp: datetime, orders: list[PortfolioOrder]):
        for order in orders:
            order.place_dt = timestamp
        orders = [o for o in orders if o.volume > 0]
        assets = [o.asset for o in orders]
        self.orders = [o for o in self.orders if o.asset not in assets]
        self.orders.extend(orders)

    def find_position(self, asset: str) -> int:
        asset_index = None
        for index, position in enumerate(self.portfolio.positions):
            if position.asset == asset:
                asset_index = index
                break
        return asset_index

    def buy_asset(self, order: PortfolioOrder, quotes: QuotesSnapshot, asset_index: int) -> bool:
        if order.order_type != PortfolioOrderType.buy:
            return False
        if quotes.closing_price(order.asset) >= order.price:
            return False
        cost = order.price * order.volume * (1 + self.transaction_fee)
        if cost > self.portfolio.cash:
            print("Not enough funds to buy", order.asset)
            return False
        self.portfolio.cash -= cost
        if asset_index is None:
            self.portfolio.positions.append(PortfolioPosition(order.asset, order.volume))
        else:
            self.portfolio.positions[asset_index].volume += order.volume
        return True

    def sell_asset(self, order: PortfolioOrder, quotes: QuotesSnapshot, asset_index: int) -> bool:
        if order.order_type != PortfolioOrderType.sell:
            return False
        if quotes.closing_price(order.asset) <= order.price:
            return False
        position: PortfolioPosition = self.portfolio.positions[asset_index]
        prev_volume = position.volume
        position.volume = max(0, prev_volume - order.volume)
        if position.volume * quotes.closing_price(order.asset) < self.init_cash / 1000:
            position.volume = 0
        actual_order_volume = prev_volume - position.volume
        self.portfolio.cash += order.price * actual_order_volume * (1 - self.transaction_fee)
        self.portfolio.positions = [p for p in self.portfolio.positions if p.volume > 0]
        return True

    def handle_orders(self, timestamp: datetime, quotes: QuotesSnapshot):
        new_orders = []
        for order in self.orders:
            if timestamp - order.place_dt > timedelta(seconds=self.expiration_time_sec):
                continue
            asset_index = self.find_position(order.asset)
            if self.buy_asset(order, quotes, asset_index):
                continue
            if self.sell_asset(order, quotes, asset_index):
                continue
            new_orders.append(order)
        self.portfolio.update_value(quotes)
        self.orders = new_orders
