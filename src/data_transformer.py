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
    def is_price(cls, feature_index: int) -> bool:
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
        self.memory = None
        self.last_features = None
        self.stats = None
        self.stats_source_size = 0

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

    def add_to_stats(self, features: np.array) -> dict:
        n_assets = len(features)
        n_features = len(features[0])
        if self.stats is None:
            self.stats = {"mean": np.zeros(n_features), "mean_squared": np.zeros(n_features), "std": np.zeros(n_features)}
        self.stats["mean"] = (self.stats["mean"] * self.stats_source_size + features.mean(axis=0) * n_assets) / (
            self.stats_source_size + n_assets
        )
        self.stats["mean_squared"] = (
            self.stats["mean_squared"] * self.stats_source_size + np.power(features, 2).mean(axis=0) * n_assets
        ) / (self.stats_source_size + n_assets)
        self.stats["std"] = np.power(self.stats["mean_squared"] - np.power(self.stats["mean"], 2), 0.5)
        self.stats_source_size += n_assets

    def scale_features(self, features: np.array, stats: list[dict]):
        raw_features = features
        for feature_index in len(features[0]):
            if InputFeatures.is_price(feature_index):
                features[:, feature_index] = features[:, feature_index] / self.last_features[:, feature_index] - 1
            else:
                mean = stats[feature_index]["mean"]
                std = stats[feature_index]["std"]
                features[:, feature_index] = (features[:, feature_index] + mean + std) / (
                    self.last_features[:, feature_index] + mean + std
                ) - 1
        np.nan_to_num(features, copy=False)
        self.last_features = raw_features

    def add_to_memory(self, features: np.array):
        if self.memory is None:
            self.memory = np.zeros((self.memory_length, *np.shape(features)))
        for index in range(self.memory_length - 1):
            self.memory[index] = self.memory[index] * 0.1 + self.memory[index + 1] * 0.9
        self.memory = np.concatenate((self.memory[:-1], np.expand_dims(features, 0)))
