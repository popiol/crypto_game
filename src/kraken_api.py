import base64
import hashlib
import hmac
import json
import time
import urllib
from datetime import datetime
from functools import cache
from time import sleep

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
        offset = 0
        orders_all = {}
        while True:
            params = {
                "nonce": self.get_nonce(),
                "trades": False,
                "start": since.timestamp(),
                "closetime": "close",
                "ofs": offset,
            }
            headers = self.get_headers(command, params)
            resp = requests.post(f"{self.endpoint}/{command}", headers=headers, data=params)
            resp.raise_for_status()
            resp_json = resp.json()
            if "result" not in resp_json:
                break
            orders = resp_json["result"]["closed"]
            if not orders:
                break
            offset += len(orders)
            orders = {id: order for id, order in orders.items() if order["status"] == "closed"}
            orders_all = {**orders_all, **orders}
            sleep(1)
        return orders_all

    def get_positions(self, since: datetime):
        print("get real positions")
        balance = {asset: float(volume) for asset, volume in self.get_balance().items() if float(volume) > 0 and asset != "ZUSD"}
        precision = self.get_base_volume_precision(list(balance))
        assets = [asset for asset, volume in balance.items() if volume > pow(10, 2 - precision[asset])]
        orders = self.get_closed_orders(since)
        matched = {}
        for order in orders.values():
            for asset in assets:
                if order["descr"]["pair"].startswith(asset) and order["descr"]["type"] == "buy":
                    if asset not in matched or matched[asset]["opentm"] < order["opentm"]:
                        matched[asset] = order
        pairs = self.find_asset_pairs(assets)
        return [
            PortfolioPosition(
                asset=order["descr"]["pair"],
                volume=balance[asset],
                buy_price=float(order["price"]),
                cost=float(order["cost"]) + float(order["fee"]),
                place_dt=datetime.fromtimestamp(order["opentm"]),
                value=None,
            )
            for asset, order in matched.items()
        ] + [
            PortfolioPosition(
                asset=pairs[asset],
                volume=balance[asset],
                buy_price=None,
                cost=None,
                place_dt=None,
                value=None,
            )
            for asset in assets
            if asset not in matched
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

    def get_base_volume_precision(self, assets: list[str]):
        if not assets:
            return {}
        print("get volume precision for", assets)
        command = "Assets"
        params = {"asset": ",".join(assets)}
        resp = requests.get(f"{self.public_endpoint}/{command}", params=params)
        resp_json = resp.json()
        if not resp_json["error"]:
            result = resp_json["result"]
            map_1 = {key: val["decimals"] for key, val in result.items()}
            map_2 = {val["altname"]: val["decimals"] for key, val in result.items()}
            map = {**map_1, **map_2}
            return {asset: map[asset] for asset in assets}

    def get_precision(self, assets: list[str]):
        if not assets:
            return {}
        print("get precision for", assets)
        command = "AssetPairs"
        params = {"pair": ",".join(assets)}
        resp = requests.get(f"{self.public_endpoint}/{command}", params=params)
        map_1: dict = resp.json()["result"]
        map_2 = {val["altname"]: val for _, val in map_1.items()}
        result = {**map_1, **map_2}
        return {asset: AssetPrecision(result[asset]["lot_decimals"], result[asset]["pair_decimals"]) for asset in assets}

    def find_asset_pairs(self, assets: list[str]):
        if not assets:
            return {}
        print("find asset pairs for", assets)
        command = "AssetPairs"
        resp = requests.get(f"{self.public_endpoint}/{command}")
        pairs = [pair for pair in resp.json()["result"] if pair.endswith("USD")]
        return {asset: pair for asset in assets for pair in pairs if pair.startswith(asset)}
