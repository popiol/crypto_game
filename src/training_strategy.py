import random

import numpy as np

from src.ml_model import MlModel


class TrainingStrategy:

    def __init__(self, model: MlModel):
        self.model = model
        self.stats: dict = None

    def reset(self):
        pass

    def predict(self, input: np.array) -> np.array:
        raise NotImplementedError()

    def train(self, input: np.array, output: np.array, reward: float):
        raise NotImplementedError()

    def add_to_stats(self, reward: float):
        if self.stats is None:
            self.stats = {"mean": 0, "mean_squared": 0, "std": 0, "count": 0}
        self.stats["mean"] = (self.stats["mean"] * self.stats["count"] + reward) / (self.stats["count"] + 1)
        self.stats["mean_squared"] = (self.stats["mean_squared"] * self.stats["count"] + reward**2) / (self.stats["count"] + 1)
        self.stats["std"] = (self.stats["mean_squared"] - self.stats["mean"] ** 2) ** 0.5
        self.stats["count"] += 1


class LearnOnMistakes(TrainingStrategy):

    def predict(self, input: np.array) -> np.array:
        return self.model.predict(np.array([input]))[0]

    def train(self, input: np.array, output: np.array, reward: float):
        self.add_to_stats(reward)
        if reward < 0:
            output = 1 - output
            n_epochs = 1 if reward > self.stats["mean"] - self.stats["std"] else 2
            self.model.train(input, output, n_epochs=n_epochs)


class LearnOnSuccess(TrainingStrategy):

    def __init__(self, model: MlModel):
        super().__init__(model)
        self.reset()
        
    def reset(self):
        self.clone = self.model.copy()
        self.clone.add_noise(0.7)

    def predict(self, input: np.array) -> np.array:
        return self.clone.predict(np.array([input]))[0]

    def train(self, input: np.array, output: np.array, reward: float):
        self.add_to_stats(reward)
        if reward > self.stats["mean"] + self.stats["std"]:
            self.model.train(input, output)


class StrategyPicker:

    def __init__(self):
        self.strategies = [LearnOnMistakes, LearnOnSuccess]

    def pick(self, model: MlModel) -> TrainingStrategy:
        index = random.randrange(len(self.strategies))
        return self.strategies[index](model)
