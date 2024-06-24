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
    place_dt: datetime = None


@dataclass
class PortfolioPosition:
    asset: str
    volume: float
    buy_price: float
    cost: float
    place_dt: datetime
    value: float = None


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
