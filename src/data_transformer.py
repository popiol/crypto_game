from enum import Enum, auto


class DataTransformer:

    class ModelFeatures(Enum):
        ask_price = auto()
        ask_whole_lot_volume = auto()
        ask_lot_volume = auto()
        bid_price = auto()
        bid_whole_lot_volume = auto()
        bid_lot_volume = auto()
        closing_price = auto()
        closing_lot_volume = auto()
        volume_today = auto()
        volume_24h = auto()
        volume_weighted_price_today = auto()
        volume_weighted_price_24h = auto()
        n_trades_today = auto()
        n_trades_24h = auto()
        low_today = auto()
        low_24h = auto()
        high_today = auto()
        high_24h = auto()
        opening_price = auto()

    class ModelOutputs(Enum):
        score = auto()
        buy_price = auto()
        sell_price = auto()

    KEY_MAP = {
        "a": ["ask_price", "ask_whole_lot_volume", "ask_lot_volume"],
        "o": "open",
        "c": "close",
        "l": "low",
        "h": "high",
        "v": "volume",
        "b": "bid",
        "p": "daily_mean",
        "t": "n_trades",
    }

    def __init__(self, memory_length: int):
        self.memory_length = memory_length

    def quotes_to_features(self, quotes: dict):

