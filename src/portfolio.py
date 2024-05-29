from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

from src.data_transformer import QuotesSnapshot


class PortfolioOrderType(Enum):
    buy = auto()
    sell = auto()


@dataclass
class PortfolioOrder:
    place_dt: datetime
    order_type: PortfolioOrderType
    asset: str
    volume: float
    price: float


@dataclass
class PortfolioPosition:
    asset: str
    volume: float


@dataclass
class Portfolio:
    positions: list[PortfolioPosition]
    cash: float
    value: float

    def update_value(self, quotes: QuotesSnapshot):
        self.value = sum(p.volume * quotes.closing_price(p.asset) for p in self.positions) + self.cash
