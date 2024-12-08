import base64
import hashlib
import hmac
import json
import time
import urllib
from datetime import datetime

import requests

from src.portfolio import PortfolioOrder, PortfolioOrderType, PortfolioPosition


class KrakenApi:

    def __init__(self):
        self.api_url = "https://api.kraken.com"
        self.api_ver = "0"
        self.prefix = f"/{self.api_ver}/private"
        self.endpoint = f"{self.api_url}{self.prefix}"
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
        print("place real order")
        command = "AddOrder"
        params = {
            "nonce": self.get_nonce(),
            "pair": order.asset,
            "type": order.order_type.name,
            "ordertype": "limit",
            "price": order.price,
            "volume": order.volume,
            "expiretm": "+3540",
            # "validate": True,
        }
        headers = self.get_headers(command, params)
        resp = requests.post(f"{self.endpoint}/{command}", headers=headers, data=params)
        print(resp.text)

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

    def get_closed_orders(self, since: datetime):
        print("get closed orders")
        command = "ClosedOrders"
        params = {
            "nonce": self.get_nonce(),
            "trades": False,
            "start": since.timestamp(),
            "closetime": "close",
        }
        headers = self.get_headers(command, params)
        resp = requests.post(f"{self.endpoint}/{command}", headers=headers, data=params)
        print(resp.text)
        orders = resp.json()["result"]["closed"]
        orders = {id: order for id, order in orders.items() if order["status"] == "closed"}
        return orders

    def get_orders_info(self, txids: list[str]) -> dict:
        print("get real order info for", txids)
        command = "QueryOrders"
        params = {
            "nonce": self.get_nonce(),
            "txid": ",".join(txids),
            "trades": False,
        }
        headers = self.get_headers(command, params)
        resp = requests.post(f"{self.endpoint}/{command}", headers=headers, data=params)
        print(resp.text)
        return resp.json()["result"]

    def convert_position(self, api_position: dict, orders_info: dict) -> PortfolioPosition:
        order_info = orders_info[api_position["ordertxid"]]
        return PortfolioPosition(
            api_position["pair"],
            api_position["vol"],
            order_info["price"],
            api_position["cost"],
            datetime.fromtimestamp(order_info["opentm"]),
            api_position["value"],
        )

    def get_positions(self, since: datetime):
        print("get real positions")
        balance = self.get_balance()
        volumes = {asset + "USD": float(volume) for asset, volume in balance.items() if float(volume) > 0 and asset != "ZUSD"}
        print("volumes", volumes)
        orders = self.get_closed_orders(since)
        print("orders", orders)
        orders = [order for order in orders.values() if order["descr"]["pair"] in volumes]
        return [
            PortfolioPosition(
                asset=order["descr"]["pair"],
                volume=order["vol"],
                buy_price=order["price"],
                cost=order["cost"],
                place_dt=datetime.fromtimestamp(order["opentm"]),
                value=None,
            )
            for order in orders
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
                volume=o["vol"],
                price=o["descr"]["price"],
                place_dt=datetime.fromtimestamp(o["opentm"]),
            )
            for o in resp.json()["result"]["open"].values()
        ]

    def get_closed_transactions(self, since: datetime):
        print("get real orders")
        command = "ClosedOrders"
        params = {
            "nonce": self.get_nonce(),
            "trades": False,
            "start": since.timestamp(),
            "closetime": "close",
        }
        headers = self.get_headers(command, params)
        resp = requests.post(f"{self.endpoint}/{command}", headers=headers, data=params)
        print(resp.text)
        # return [
        #     ClosedTransaction(
        #         asset=o["descr"]["pair"],
        #         volume=o["vol"],
        #         price=o["price"],
        #         place_dt=o["opentm"],
        #     )
        #     for o in resp.json()["result"]["closed"].values()
        # ]
