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
    value: float = None


@dataclass
class Portfolio:
    positions: list[PortfolioPosition]
    cash: float
    value: float

    def update_value(self, quotes: QuotesSnapshot):
        for p in self.positions:
            p.value = p.volume * quotes.closing_price(p.asset)
        self.value = sum(p.value for p in self.positions) + self.cash
