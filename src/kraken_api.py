import base64
import hashlib
import hmac
import json
import time
import urllib
from datetime import datetime, timedelta
from functools import cache

import requests

from src.portfolio import (
    AssetPrecision,
    ClosedTransaction,
    PortfolioOrder,
    PortfolioOrderType,
    PortfolioPosition,
)


class KrakenApi:

    def __init__(self):
        self.api_url = "https://api.kraken.com"
        self.api_ver = "0"
        self.prefix = f"/{self.api_ver}/private"
        self.endpoint = f"{self.api_url}{self.prefix}"
        self.public_endpoint = f"{self.api_url}/{self.api_ver}/public"
        self.api_key = ""
        self.secret_key = ""
        self.base_currency_map = {}

    def load_secret(self):
        with open("kraken_api_secret.json") as f:
            secret = json.load(f)
        self.api_key = secret["api_key"]
        self.secret_key = secret["secret_key"]

    def get_nonce(self):
        return str(int(time.time() * 1000))

    def get_headers(self, command: str, params: dict):
        if not self.api_key:
            self.load_secret()
        postdata = urllib.parse.urlencode(params)
        encoded = f"{params['nonce']}{postdata}".encode()
        urlpath = f"{self.prefix}/{command}"
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        signature = base64.b64encode(hmac.new(base64.b64decode(self.secret_key), message, hashlib.sha512).digest()).decode()
        return {
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "API-Key": self.api_key,
            "API-Sign": signature,
        }

    def place_order(self, order: PortfolioOrder):
        print("place real order", order)
        command = "AddOrder"
        if order.price:
            params = {
                "nonce": self.get_nonce(),
                "pair": order.asset,
                "type": order.order_type.name,
                "ordertype": "limit",
                "price": order.price,
                "volume": order.volume,
                "expiretm": "+3540",
            }
        else:
            params = {
                "nonce": self.get_nonce(),
                "pair": order.asset,
                "type": order.order_type.name,
                "ordertype": "market",
                "volume": order.volume,
                "expiretm": "+3540",
            }
        headers = self.get_headers(command, params)
        resp = requests.post(f"{self.endpoint}/{command}", headers=headers, data=params)
        print(resp.text)

    @cache
    def get_balance(self) -> dict:
        print("get real balance")
        command = "Balance"
        params = {
            "nonce": self.get_nonce(),
        }
        headers = self.get_headers(command, params)
        resp = requests.post(f"{self.endpoint}/{command}", headers=headers, data=params)
        print(resp.text)
        return resp.json()["result"]

    @cache
    def get_closed_orders(self, since: datetime):
        print("get closed orders since", since)
        command = "ClosedOrders"
        params = {
            "nonce": self.get_nonce(),
            "trades": False,
            "start": since.timestamp(),
            "closetime": "close",
        }
        headers = self.get_headers(command, params)
        resp = requests.post(f"{self.endpoint}/{command}", headers=headers, data=params)
        orders = resp.json()["result"]["closed"]
        orders = {id: order for id, order in orders.items() if order["status"] == "closed"}
        return orders

    def get_positions(self, since: datetime):
        print("get real positions")
        balance = self.get_balance()
        assets = set([asset for asset, volume in balance.items() if float(volume) > 0.00001 and asset != "ZUSD"])
        print("assets", assets)
        jump = 1
        for _ in range(5):
            orders = self.get_closed_orders(since)
            matched = {}
            for order in orders.values():
                for asset in assets:
                    if order["descr"]["pair"].startswith(asset) and order["descr"]["type"] == "buy":
                        if asset not in matched or matched[asset]["opentm"] < order["opentm"]:
                            matched[asset] = order
            if len(matched) >= len(assets):
                break
            since -= timedelta(days=jump)
            jump *= 2
        print(matched)
        return [
            PortfolioPosition(
                asset=order["descr"]["pair"],
                volume=float(balance[asset]),
                buy_price=float(order["price"]),
                cost=float(order["cost"]) + float(order["fee"]),
                place_dt=datetime.fromtimestamp(order["opentm"]),
                value=None,
            )
            for asset, order in matched.items()
        ]

    def get_cash(self) -> float:
        print("get real cash")
        balance = self.get_balance()
        return float(balance["ZUSD"])

    def get_orders(self):
        print("get real orders")
        command = "OpenOrders"
        params = {
            "nonce": self.get_nonce(),
            "trades": False,
        }
        headers = self.get_headers(command, params)
        resp = requests.post(f"{self.endpoint}/{command}", headers=headers, data=params)
        print(resp.text)
        return [
            PortfolioOrder(
                order_type=PortfolioOrderType[o["descr"]["type"]],
                asset=o["descr"]["pair"],
                volume=float(o["vol"]),
                price=float(o["descr"]["price"]),
                place_dt=datetime.fromtimestamp(o["opentm"]),
            )
            for o in resp.json()["result"]["open"].values()
        ]

    def get_closed_transactions(self, since: datetime):
        print("get real closed transactions")
        orders = self.get_closed_orders(since)
        orders = [order for order in orders.values() if order["descr"]["type"] == "sell"]
        return [
            ClosedTransaction(
                asset=order["descr"]["pair"],
                volume=float(order["vol"]),
                buy_price=None,
                sell_price=float(order["price"]),
                place_buy_dt=None,
                place_sell_dt=datetime.fromtimestamp(order["opentm"]),
                cost=None,
                profit=float(order["cost"]) - float(order["fee"]),
            )
            for order in orders
        ]

    def get_volume_precision(self, assets: list[str]):
        if not assets:
            return {}
        print("get volume precision for", assets)
        command = "Assets"
        assets = [self.base_currency_map[asset] for asset in assets]
        pairs = {base: pair for pair, base in self.base_currency_map.items()}
        params = {"asset": ",".join(assets)}
        resp = requests.get(f"{self.public_endpoint}/{command}", params=params)
        print(resp.text)
        resp_json = resp.json()
        if not resp_json["error"]:
            result = resp_json["result"]
            map_1 = {key: val["decimals"] for key, val in result.items()}
            map_2 = {val["altname"]: val["decimals"] for key, val in result.items()}
            return {pairs[asset]: map_1.get(asset, map_2.get(asset)) for asset in assets}

    def get_price_precision(self, assets: list[str]):
        if not assets:
            return {}
        print("get price precision for", assets)
        command = "AssetPairs"
        params = {"pair": ",".join(assets)}
        resp = requests.get(f"{self.public_endpoint}/{command}", params=params)
        print(resp.text)
        result = resp.json()["result"]
        self.base_currency_map = {asset: result[asset]["base"] for asset in assets}
        return {asset: result[asset]["pair_decimals"] for asset in assets}

    def get_precision(self, assets: list[str]):
        if not assets:
            return {}
        price_precision = self.get_price_precision(assets)
        volume_precision = self.get_volume_precision(assets)
        return {
            asset: AssetPrecision(volume_precision.get(asset), price_precision.get(asset))
            for asset in set(volume_precision.keys()).union(price_precision.keys())
        }
