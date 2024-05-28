from dataclasses import dataclass
from enum import Enum, auto


class PortfolioOrderType(Enum):
    buy = auto()
    sell = auto()


@dataclass
class PortfolioOrder:
    action_type: PortfolioOrderType
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
