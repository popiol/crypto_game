import random

import numpy as np

from src.ml_model import MlModel
from src.stats import Stats


class TrainingStrategy:

    def __init__(self, model: MlModel):
        self.model = model
        self._stats = Stats()
        self.reset()

    def reset(self):
        pass

    def predict(self, input: np.ndarray) -> np.ndarray:
        return self.model.predict(input)

    def train(self, input: np.ndarray, output: np.ndarray, reward: float):
        raise NotImplementedError()

    def add_to_stats(self, reward: float):
        self._stats.add_to_stats(reward)

    @property
    def stats(self):
        return self._stats.asdict()


class LearnOnMistakes(TrainingStrategy):

    def reset(self):
        self.clone = self.model.copy()
        self.clone.add_noise(0.3)

    def predict(self, input: np.ndarray) -> np.ndarray:
        return self.clone.predict(input)

    def train(self, input: np.ndarray, output: np.ndarray, reward: float):
        self.add_to_stats(reward)
        if reward < 0:
            output = np.round(1 - output)
            n_epochs = 1 if reward > self.stats["mean"] - self.stats["std"] else 2
            self.model.train(input, output, n_epochs=n_epochs)


class LearnOnSuccess(TrainingStrategy):

    def reset(self):
        self.clone = self.model.copy()
        self.clone.add_noise(0.4)

    def predict(self, input: np.ndarray) -> np.ndarray:
        return self.clone.predict(input)

    def train(self, input: np.ndarray, output: np.ndarray, reward: float):
        self.add_to_stats(reward)
        if reward > self.stats["mean"] + self.stats["std"]:
            self.model.train(input, output)


class LearnOnBoth(TrainingStrategy):

    def reset(self):
        self.clone = self.model.copy()
        self.clone.add_noise(0.4)

    def predict(self, input: np.ndarray) -> np.ndarray:
        return self.clone.predict(input)

    def train(self, input: np.ndarray, output: np.ndarray, reward: float):
        self.add_to_stats(reward)
        if reward < self.stats["mean"] - self.stats["std"]:
            self.model.train(input, np.round(1 - output))
        elif reward > self.stats["mean"] + self.stats["std"]:
            self.model.train(input, output)


class StrategyPicker:

    def __init__(self):
        self.strategies = [LearnOnMistakes, LearnOnSuccess, LearnOnBoth]

    def pick_class(self) -> type[TrainingStrategy]:
        return random.choice(self.strategies)

    def pick(self, model: MlModel) -> TrainingStrategy:
        return self.pick_class()(model)
