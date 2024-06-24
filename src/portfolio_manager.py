import math
from datetime import datetime, timedelta

from src.data_transformer import QuotesSnapshot
from src.portfolio import (
    ClosedTransaction,
    Portfolio,
    PortfolioOrder,
    PortfolioOrderType,
    PortfolioPosition,
)


class PortfolioManager:

    def __init__(self, init_cash: float, transaction_fee: float, expiration_time_sec: int, min_transaction: float):
        self.init_cash = init_cash
        self.transaction_fee = transaction_fee
        self.expiration_time_sec = expiration_time_sec
        self.min_transaction = min_transaction
        self.reset()

    def reset(self):
        self.portfolio = Portfolio([], self.init_cash, self.init_cash)
        self.orders: list[PortfolioOrder] = []

    def adjust_buy_volume(self, orders: list[PortfolioOrder]):
        cost = sum(o.volume * o.price for o in self.orders if o.order_type == PortfolioOrderType.buy)
        if cost > self.portfolio.cash:
            c = self.portfolio.cash / cost
            for order in orders:
                if order.order_type == PortfolioOrderType.buy:
                    order.volume *= c

    def validate_order(self, order: PortfolioOrder):
        assets = set([o.asset for o in self.orders])
        if order.asset in assets:
            return False
        if order.order_type == PortfolioOrderType.buy and order.volume * order.price < self.min_transaction:
            return False
        if order.order_type == PortfolioOrderType.sell:
            asset_index = self.find_position(order.asset)
            if asset_index is None:
                return False
            if (order.volume - self.portfolio.positions[asset_index].volume) * order.price < self.min_transaction:
                order.volume = self.portfolio.positions[asset_index].volume
        return True

    def fix_orders(self, timestamp: datetime, orders: list[PortfolioOrder]) -> list[PortfolioOrder]:
        for order in orders:
            order.place_dt = timestamp
        self.adjust_buy_volume(orders)
        return [o for o in orders if self.validate_order(o)]

    def place_orders(self, timestamp: datetime, orders: list[PortfolioOrder]):
        orders = self.fix_orders(timestamp, orders)
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
        assert order.price > 0
        assert order.volume > 0
        cost = order.price * order.volume
        if cost > self.portfolio.cash * 1.1:
            return True
        cost = min(cost, self.portfolio.cash)
        if cost < self.min_transaction:
            return True
        order.volume = cost / order.price / (1 + self.transaction_fee)
        precision = math.pow(10, math.floor(math.log10(order.volume)) - 3)
        order.volume = math.floor(order.volume / precision) * precision
        cost = order.price * order.volume * (1 + self.transaction_fee)
        assert cost <= self.portfolio.cash
        self.portfolio.cash -= cost
        if asset_index is None:
            self.portfolio.positions.append(PortfolioPosition(order.asset, order.volume, order.price, cost, order.place_dt))
        else:
            position = self.portfolio.positions[asset_index]
            position.buy_price = (position.buy_price * position.volume + order.price * order.volume) / (
                position.volume + order.volume
            )
            position.volume += order.volume
            position.place_dt = order.place_dt
        return True

    def sell_asset(
        self, order: PortfolioOrder, quotes: QuotesSnapshot, asset_index: int, closed_transactions: list[ClosedTransaction]
    ) -> bool:
        if order.order_type != PortfolioOrderType.sell:
            return False
        if asset_index is None:
            return True
        if quotes.closing_price(order.asset) <= order.price:
            return False
        position: PortfolioPosition = self.portfolio.positions[asset_index]
        prev_volume = position.volume
        position.volume = max(0, prev_volume - order.volume)
        if position.volume * quotes.closing_price(order.asset) < self.min_transaction:
            position.volume = 0
        order.volume = prev_volume - position.volume
        assert order.price > 0
        assert order.volume > 0
        profit = order.price * order.volume * (1 - self.transaction_fee)
        self.portfolio.cash += profit
        self.portfolio.positions = [p for p in self.portfolio.positions if p.volume > 0]
        closed_transactions.append(
            ClosedTransaction(
                order.asset,
                order.volume,
                position.buy_price,
                order.price,
                position.place_dt,
                order.place_dt,
                position.cost,
                profit,
            )
        )
        return True

    def handle_orders(self, timestamp: datetime, quotes: QuotesSnapshot) -> list[ClosedTransaction]:
        closed_transactions = []
        new_orders = []
        for order in self.orders:
            if timestamp - order.place_dt > timedelta(seconds=self.expiration_time_sec):
                continue
            asset_index = self.find_position(order.asset)
            if self.buy_asset(order, quotes, asset_index):
                continue
            if self.sell_asset(order, quotes, asset_index, closed_transactions):
                continue
            new_orders.append(order)
        try:
            self.portfolio.update_value(quotes)
        except AssertionError:
            print("Assertion error")
            print(self.portfolio)
            raise
        self.orders = new_orders
        return closed_transactions
