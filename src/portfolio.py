from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

from src.data_transformer import QuotesSnapshot


class PortfolioOrderType(Enum):
    buy = auto()
    sell = auto()


@dataclass
class PortfolioOrder:
    order_type: PortfolioOrderType
    asset: str
    volume: float
    price: float
    place_dt: datetime

    @staticmethod
    def from_json(obj: dict):
        return PortfolioOrder(
            PortfolioOrderType[obj["order_type"]],
            obj["asset"],
            obj["volume"],
            obj["price"],
            datetime.strptime(obj["place_dt"], "%Y%m%d%H%M%S"),
        )

    def to_json(self):
        return {
            "order_type": self.order_type.name,
            "asset": self.asset,
            "volume": self.volume,
            "price": self.price,
            "place_dt": datetime.strftime(self.place_dt, "%Y%m%d%H%M%S"),
        }


@dataclass
class PortfolioPosition:
    asset: str
    volume: float
    buy_price: float
    cost: float
    place_dt: datetime
    value: float = None
    last_sell_price: float = None

    @staticmethod
    def from_json(obj: dict):
        return PortfolioPosition(
            obj["asset"],
            obj["volume"],
            obj["buy_price"],
            obj["cost"],
            datetime.strptime(obj["place_dt"], "%Y%m%d%H%M%S"),
            obj["value"],
            obj.get("last_sell_price"),
        )

    def to_json(self):
        return {
            "asset": self.asset,
            "volume": self.volume,
            "buy_price": self.buy_price,
            "cost": self.cost,
            "place_dt": datetime.strftime(self.place_dt, "%Y%m%d%H%M%S"),
            "value": self.value,
            "last_sell_price": self.last_sell_price,
        }


@dataclass
class Portfolio:
    positions: list[PortfolioPosition]
    cash: float
    value: float

    def update_value(self, quotes: QuotesSnapshot):
        for p in self.positions:
            assert p.volume > 0
            p.value = p.volume * quotes.closing_price(p.asset)
            assert p.value > 0
        assert self.cash >= 0
        self.value = sum(p.value for p in self.positions) + self.cash


@dataclass
class ClosedTransaction:
    asset: str
    volume: float
    buy_price: float
    sell_price: float
    place_buy_dt: datetime
    place_sell_dt: datetime
    cost: float
    profit: float

    def to_json(self):
        return {
            "asset": self.asset,
            "volume": self.volume,
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
            "place_buy_dt": datetime.strftime(self.place_buy_dt, "%Y%m%d%H%M%S"),
            "place_sell_dt": datetime.strftime(self.place_sell_dt, "%Y%m%d%H%M%S"),
            "cost": self.cost,
            "profit": self.profit,
        }


@dataclass
class AssetPrecision:
    volume_precision: int = None
    price_precision: int = None
