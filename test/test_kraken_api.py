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
                order_type=PortfolioOrderType.sell,
                asset="SGBUSD",
                volume=0,
                price=0.00919,
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
        orders = api.get_closed_orders(datetime.strptime("2025-03-22 10:00:00", "%Y-%m-%d %H:%M:%S"))
        print(orders)

    def test_get_precision(self):
        api = KrakenApi()
        precision = api.get_precision(["EURUSD", "HNTUSD", "BLURUSD", "XXBTZUSD", "CHZUSD"])
        print(precision)

    def test_get_base_volume_precision(self):
        api = KrakenApi()
        precision = api.get_base_volume_precision(["HNT", "BLUR", "XXBT", "CHZ", "EUR", "ZEUR"])
        print(precision)

    def test_find_asset_pairs(self):
        api = KrakenApi()
        pairs = api.find_asset_pairs(["HNT", "BLUR", "XXBT", "CHZ", "EUR", "ZEUR"])
        print(pairs)
