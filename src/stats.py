import numpy as np


class Stats:

    def __init__(self):
        self.stats = None

    def add_to_stats(self, values: float | np.ndarray):
        if np.ndim(values) == 0:
            batch_size = 1
            n_vars = 1
            values = np.array([[values]])
            self.orig_shape = tuple()
        elif np.ndim(values) == 1:
            batch_size = 1
            n_vars = len(values)
            values = np.array([values])
            self.orig_shape = (n_vars,)
        else:
            batch_size = len(values)
            n_vars = np.shape(values[0])
            values = np.array(values)
            self.orig_shape = n_vars
        if self.stats is None:
            self.stats = {
                "mean": np.zeros(n_vars),
                "mean_squared": np.zeros(n_vars),
                "std": np.zeros(n_vars),
                "min": np.full(n_vars, np.nan),
                "max": np.full(n_vars, np.nan),
                "count": 0,
            }
        self.stats["mean"] = (self.stats["mean"] * self.stats["count"] + values.mean(axis=0) * batch_size) / (
            self.stats["count"] + batch_size
        )
        self.stats["mean_squared"] = (
            self.stats["mean_squared"] * self.stats["count"] + np.power(values, 2).mean(axis=0) * batch_size
        ) / (self.stats["count"] + batch_size)
        self.stats["std"] = np.power(self.stats["mean_squared"] - np.power(self.stats["mean"], 2), 0.5)
        self.stats["min"] = np.nanmin([self.stats["min"], np.nanmin(values, axis=0)], axis=0)
        self.stats["max"] = np.nanmax([self.stats["max"], np.nanmax(values, axis=0)], axis=0)
        self.stats["count"] += batch_size

    def squeeze(self, x) -> float | np.ndarray:
        squeezed = np.array(x).item() if np.size(x) == 1 else np.squeeze(x)
        if self.orig_shape:
            squeezed = np.reshape(squeezed, self.orig_shape)
        return squeezed

    @property
    def mean(self):
        return self.squeeze(self.stats["mean"])

    @property
    def std(self):
        return self.squeeze(self.stats["std"])

    @property
    def min(self):
        return self.squeeze(self.stats["min"])

    @property
    def max(self):
        return self.squeeze(self.stats["max"])

    @property
    def count(self) -> int:
        return self.stats["count"]

    def asdict(self):
        if self.stats is None:
            return None
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "count": self.count,
        }
