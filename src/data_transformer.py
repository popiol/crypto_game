from dataclasses import dataclass, fields

import numpy as np


class ModelFeatures:

    def set_feature(self, name: str, value: float):
        setattr(self, name, value)

    def to_vector(self) -> np.array:
        return np.array([getattr(self, f.name) for f in fields(self)])


@dataclass
class InputFeatures(ModelFeatures):
    ask_price: float = None
    ask_whole_lot_volume: float = None
    ask_lot_volume: float = None
    bid_price: float = None
    bid_whole_lot_volume: float = None
    bid_lot_volume: float = None
    closing_price: float = None
    closing_lot_volume: float = None
    volume_today: float = None
    volume_24h: float = None
    volume_weighted_price_today: float = None
    volume_weighted_price_24h: float = None
    n_trades_today: float = None
    n_trades_24h: float = None
    low_today: float = None
    low_24h: float = None
    high_today: float = None
    high_24h: float = None
    opening_price: float = None

    @classmethod
    def is_positive(cls, feature_index: int) -> bool:
        fields(cls)[feature_index].name in [
            "ask_price",
            "bid_price",
            "closing_price",
            "low_today",
            "low_24h",
            "high_today",
            "high_24h",
            "opening_price",
        ]


@dataclass
class OutputsFeatures(ModelFeatures):
    score: float = None
    relative_buy_price: float = None
    relative_sell_price: float = None


class DataTransformer:

    KEY_MAP = {
        "a": ["ask_price", "ask_whole_lot_volume", "ask_lot_volume"],
        "b": ["bid_price", "bid_whole_lot_volume", "bid_lot_volume"],
        "c": ["closing_price", "closing_lot_volume"],
        "v": ["volume_today", "volume_24h"],
        "p": ["volume_weighted_price_today", "volume_weighted_price_24h"],
        "t": ["n_trades_today", "n_trades_24h"],
        "l": ["low_today", "low_24h"],
        "h": ["high_today", "high_24h"],
        "o": ["opening_price"],
    }

    def __init__(self, memory_length: int):
        self.memory_length = memory_length
        self.memory = []
        self.last_features = None

    def quotes_to_features(self, quotes: dict, asset_list: list[str]) -> np.array:
        """Returns matrix of shape (n_assets, n_features)"""
        sparse_features = []
        for asset_name, asset in quotes.items():
            try:
                asset_index = asset_list.index(asset_name)
            except ValueError:
                asset_list.append(asset_name)
                asset_index = len(asset_list) - 1
            features = InputFeatures()
            for key, values in asset.items():
                key_map = self.KEY_MAP[key]
                for index, feature_name in enumerate(key_map):
                    value = values[index] if type(values) == list else values
                    value = float(value)
                    features.set_feature(feature_name, value)
            sparse_features.append((asset_index, features.to_vector()))
        n_assets = len(asset_list)
        n_features = len(sparse_features[0][1])
        feature_matrix = np.zeros((n_assets, n_features))
        for index, features in sparse_features:
            feature_matrix[index, :] = features
        return feature_matrix

    def scale_feature(self, feature_index: int, value: float, prev_value: float):
        if InputFeatures.is_positive(feature_index):
            return value / prev_value - 1
        else:
            return value / (prev_value + 1) - 1

    def add_to_memory(self, features: np.array):
        raw_features = features
        features = features / self.last_features - 1
        np.nan_to_num(features, copy=False)
        for index in range(self.memory_length - 1):
            self.memory[index, :] = self.memory[index, :] * 0.1 + self.memory[index + 1, :] * 0.9
        self.memory.append(features)
        self.last_features = raw_features
