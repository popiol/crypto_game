from datetime import datetime

from src.kraken_api import KrakenApi
from src.portfolio import PortfolioOrder, PortfolioOrderType


class TestKrakenApi:

    def test_get_cash(self):
        api = KrakenApi()
        cash = api.get_cash()
        print("cash:", cash)

    def test_place_order(self):
        api = KrakenApi()
        api.place_order(
            PortfolioOrder(
                order_type=PortfolioOrderType.buy,
                asset="SBRUSD",
                volume=7000,
                price=0.003,
                place_dt=datetime.now(),
            )
        )

    def test_get_orders(self):
        api = KrakenApi()
        orders = api.get_orders()
        print(orders)

    def test_get_positions(self):
        api = KrakenApi()
        positions = api.get_positions()
        print(positions)

    def test_get_closed_transactions(self):
        api = KrakenApi()
        transactions = api.get_closed_transactions(datetime.strptime("2020-01-01", "%Y-%m-%d"))
        print(transactions)
