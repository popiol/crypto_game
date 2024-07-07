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
                "samples": [],
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
        self.add_sample(self.stats["samples"], values)

    def add_sample(self, samples: list[np.ndarray], values: np.ndarray):
        if len(samples) > 1:
            return
        for sample in values:
            while np.ndim(sample) > 1:
                sample = np.take(sample, 0, axis=-1)
            sample = np.squeeze(sample)
            if len(samples) == 1:
                if np.ndim(sample) == 0 or len(sample) == 1:
                    if sample * samples[0] >= 0:
                        continue
                elif (sample[-1] - sample[0]) * (samples[0][-1] - samples[0][0]) >= 0:
                    continue
            self.sample_shape = np.shape(sample)
            samples.append(sample)
            break

    def squeeze(self, x, shape=None) -> float | np.ndarray:
        shape = self.orig_shape if shape is None else shape
        squeezed = np.array(x).item() if np.size(x) == 1 else np.squeeze(x)
        if shape:
            squeezed = np.reshape(squeezed, shape)
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

    @property
    def samples(self):
        return np.array([self.squeeze(x, self.sample_shape) for x in self.stats["samples"]])

    def asdict(self):
        if self.stats is None:
            return None
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "count": self.count,
            "samples": self.samples,
        }
