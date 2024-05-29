from src.portfolio import (
    Portfolio,
    PortfolioOrder,
    PortfolioOrderType,
    PortfolioPosition,
)


class PortfolioManager:

    def __init__(self, init_cash: float, transaction_fee: float):
        self.portfolio = Portfolio([], init_cash, init_cash)
        self.transaction_fee = transaction_fee
        self.orders = []

    def place_orders(self, orders: list[PortfolioOrder]):
        print("Place orders")
        self.orders.extend(orders)

    def handle_orders(self):
        print("Handle orders")
        self.portfolio.value += 1
