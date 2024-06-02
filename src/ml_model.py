from __future__ import annotations

import numpy as np
from tensorflow import keras


class MlModel:

    def __init__(self, model: keras.Model):
        self.model = model

    def train(self, x: np.array, y: np.array, n_epochs: int = 1):
        self.model.fit(x, y, epochs=n_epochs, verbose=0)

    def test(self, x: np.array, y: np.array):
        return self.model.evaluate(x, y)

    def predict(self, x: np.array):
        return self.model.predict(x, verbose=0)

    def copy(self) -> MlModel:
        return MlModel(keras.models.clone_model(self.model))

    def add_noise(self, std: float):
        for layer in self.model.get_weights():
            layer += np.random.normal(loc=0.0, scale=std, size=layer.shape)
