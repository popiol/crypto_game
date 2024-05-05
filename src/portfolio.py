from dataclasses import dataclass
from enum import Enum, auto


class PortfolioActionType(Enum):
    buy = auto()
    sell = auto()


@dataclass
class PortfolioAction:
    action_type: PortfolioActionType
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
