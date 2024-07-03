from typing import Union

import numpy as np


class Stats:

    def __init__(self):
        self._stats = None

    def add_to_stats(self, values: Union[float, np.array]):
        if np.ndim(values) == 0:
            batch_size = 1
            n_vars = 1
            values = np.array([[values]])
        elif np.ndim(values) == 1:
            batch_size = 1
            n_vars = len(values)
            values = np.array([values])
        else:
            batch_size = len(values)
            n_vars = len(values[0])
            values = np.array(values)
        if self._stats is None:
            self._stats = {
                "mean": np.zeros(n_vars),
                "mean_squared": np.zeros(n_vars),
                "std": np.zeros(n_vars),
                "min": np.full(n_vars, np.nan),
                "max": np.full(n_vars, np.nan),
                "count": 0,
            }
        self._stats["mean"] = (self._stats["mean"] * self._stats["count"] + values.mean(axis=0) * batch_size) / (
            self._stats["count"] + batch_size
        )
        self._stats["mean_squared"] = (
            self._stats["mean_squared"] * self._stats["count"] + np.power(values, 2).mean(axis=0) * batch_size
        ) / (self._stats["count"] + batch_size)
        self._stats["std"] = np.power(self._stats["mean_squared"] - np.power(self._stats["mean"], 2), 0.5)
        self._stats["min"] = np.nanmin([self._stats["min"], np.nanmin(values, axis=0)], axis=0)
        self._stats["max"] = np.nanmax([self._stats["max"], np.nanmax(values, axis=0)], axis=0)
        self._stats["count"] += batch_size

    def squeeze(self, x) -> float:
        return np.array(x).item() if np.size(x) == 1 else np.squeeze(x)

    @property
    def mean(self):
        return self.squeeze(self._stats["mean"])

    @property
    def std(self):
        return self.squeeze(self._stats["std"])

    @property
    def min(self):
        return self.squeeze(self._stats["min"])

    @property
    def max(self):
        return self.squeeze(self._stats["max"])

    @property
    def count(self) -> int:
        return self._stats["count"]

    def asdict(self):
        if self._stats is None:
            return None
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "count": self.count,
        }
