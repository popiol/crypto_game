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
                asset="POLISUSD",
                volume=70,
                price=0.301,
                place_dt=datetime.now(),
            )
        )

    def test_get_orders(self):
        api = KrakenApi()
        orders = api.get_orders()
        print(orders)

    def test_get_positions(self):
        api = KrakenApi()
        positions = api.get_positions(datetime.strptime("2024-12-23", "%Y-%m-%d"))
        print(positions)

    def test_get_closed_transactions(self):
        api = KrakenApi()
        transactions = api.get_closed_transactions(datetime.strptime("2024-12-08 10:00:00", "%Y-%m-%d %H:%M:%S"))
        print(transactions)

    def test_get_closed_orders(self):
        api = KrakenApi()
        orders = api.get_closed_orders(datetime.strptime("2024-12-07 00:00:00", "%Y-%m-%d %H:%M:%S"))
        print(orders)

    def test_get_precision(self):
        api = KrakenApi()
        precision = api.get_precision(["SBRUSD", "BLURUSD", "XXBTZUSD"])
        print(precision)
