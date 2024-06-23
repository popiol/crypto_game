from dataclasses import dataclass, fields

import numpy as np


class QuotesSnapshot:

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
        "bid2": ["bid_price_2", "bid_volume_2"],
        "ask2": ["ask_price_2", "ask_volume_2"],
    }

    def __init__(self, quotes: dict = None):
        self.quotes = quotes or {}

    def update(self, quotes: dict):
        quotes = {key: val for key, val in quotes.items() if float(val["c"][0]) > 0}
        self.quotes = {**self.quotes, **quotes}

    def update_bid_ask(self, bidask: dict):
        if bidask is None:
            return
        for asset in bidask:
            if asset not in self.quotes:
                continue
            prev_bid = self.quotes[asset]["b"]
            prev_ask = self.quotes[asset]["a"]
            for index in range(1, 2):
                self.quotes[asset][f"bid{index+1}"] = (
                    bidask[asset]["bids"][index] if len(bidask[asset]["bids"]) > index else prev_bid
                )
                self.quotes[asset][f"ask{index+1}"] = (
                    bidask[asset]["asks"][index] if len(bidask[asset]["asks"]) > index else prev_ask
                )
                prev_bid = self.quotes[asset][f"bid{index+1}"]
                prev_ask = self.quotes[asset][f"ask{index+1}"]

    def closing_price(self, asset: str) -> float:
        return float(self.quotes[asset]["c"][0])

    def items(self):
        return ((name, self.features(asset)) for name, asset in self.quotes.items())

    def features(self, asset: dict):
        for key, values in asset.items():
            key_map = self.KEY_MAP[key]
            for index, feature_name in enumerate(key_map):
                value = values[index] if type(values) == list else values
                value = float(value)
                yield feature_name, value


class ModelFeatures:
    def set_feature(self, name: str, value: float):
        setattr(self, name, value)

    def to_vector(self) -> np.array:
        return np.array([getattr(self, f.name) or 0.0 for f in fields(self)])

    @classmethod
    def count(cls) -> int:
        return len(fields(cls))


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
    bid_price_2: float = None
    bid_volume_2: float = None
    ask_price_2: float = None
    ask_volume_2: float = None

    @classmethod
    def is_price(cls, feature_index: int) -> bool:
        name = fields(cls)[feature_index].name
        return "price" in name or name in ["low_today", "low_24h", "high_today", "high_24h"]


@dataclass
class OutputFeatures(ModelFeatures):
    score: float = None
    relative_buy_volume: float = None
    relative_buy_price: float = None
    relative_sell_price: float = None


class DataTransformer:

    def __init__(self, memory_length: int, expected_daily_change: float):
        self.memory_length = memory_length
        self.expected_daily_change = expected_daily_change
        self.stats = None
        self.stats_source_size = 0
        self.reset()

    def reset(self):
        self.memory = None
        self.last_features = None

    @property
    def n_features(self):
        return InputFeatures.count()

    @property
    def n_outputs(self):
        return OutputFeatures.count()

    def quotes_to_features(self, quotes: QuotesSnapshot, asset_list: list[str]) -> np.array:
        """Returns matrix of shape (n_assets, n_features)"""
        sparse_features = []
        for asset_name, asset_features in quotes.items():
            try:
                asset_index = asset_list.index(asset_name)
            except ValueError:
                asset_list.append(asset_name)
                asset_index = len(asset_list) - 1
            features = InputFeatures()
            for feature_name, value in asset_features:
                features.set_feature(feature_name, value)
            sparse_features.append((asset_index, features.to_vector()))
        n_assets = len(asset_list)
        feature_matrix = np.zeros((n_assets, self.n_features))
        for index, features in sparse_features:
            feature_matrix[index, :] = features
        return feature_matrix

    def add_to_stats(self, features: np.array) -> dict:
        n_assets = len(features)
        n_features = self.n_features
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
        if self.last_features is None:
            self.last_features = features
            return None
        raw_features = features
        for feature_index in range(len(features[0])):
            if InputFeatures.is_price(feature_index):
                features[:, feature_index] = features[:, feature_index] / self.last_features[:, feature_index] - 1
            else:
                mean = stats["mean"][feature_index]
                std = stats["std"][feature_index]
                features[:, feature_index] = (features[:, feature_index] + mean + std) / (
                    self.last_features[:, feature_index] + mean + std
                ) - 1
        np.nan_to_num(features, copy=False, posinf=0.0, neginf=0.0)
        self.last_features = raw_features
        return features

    def add_to_memory(self, features: np.array):
        if self.memory is None:
            self.memory = np.zeros((self.memory_length, *np.shape(features)))
        for index in range(self.memory_length - 1):
            self.memory[index] = self.memory[index] * 0.1 + self.memory[index + 1] * 0.9
        self.memory = np.concatenate((self.memory[:-1], np.expand_dims(features, 0)))

    def transform_output(self, output_matrix: np.array, asset_list: list[str]) -> dict[str, OutputFeatures]:
        score = output_matrix[:, 0]
        relative_buy_volume = np.clip(output_matrix[:, 0] / np.max(output_matrix[:, 0]), 0, 1)
        relative_buy_price = (output_matrix[:, 1] - 1) * self.expected_daily_change + 1
        relative_sell_price = output_matrix[:, 2] * self.expected_daily_change + 1
        return {
            row[0]: OutputFeatures(*row[1:])
            for row in zip(asset_list, score, relative_buy_volume, relative_buy_price, relative_sell_price)
        }
