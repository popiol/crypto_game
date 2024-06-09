from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from src.keras import keras


@dataclass
class MlModelLayer:
    name: str
    shape: tuple
    input_shape: tuple


class MlModel:

    def __init__(self, model: keras.Model):
        self.model = model

    def train(self, x: np.array, y: np.array, n_epochs: int = 1):
        self.model.fit(x, y, epochs=n_epochs, verbose=0)

    def test(self, x: np.array, y: np.array):
        return self.model.evaluate(x, y)

    def predict(self, x: np.array) -> np.array:
        return self.model.predict(x, verbose=0)

    def copy(self) -> MlModel:
        clone = keras.models.clone_model(self.model)
        clone.set_weights(self.model.get_weights())
        return MlModel(clone)

    def add_noise(self, std: float):
        weights = self.model.get_weights()
        for layer in weights:
            layer += np.random.normal(loc=0.0, scale=std, size=layer.shape)
        self.model.set_weights(weights)

    def get_layers(self) -> list[MlModelLayer]:
        layers = []
        input_shape = self.model.layers[0].batch_shape
        for l in self.model.layers[1:]:
            layers.append(
                MlModelLayer(
                    name=l.name.split("_")[0], shape=tuple(l.weights[0].shape) if l.weights else None, input_shape=input_shape[1:]
                )
            )
            input_shape = l.compute_output_shape(input_shape)
        return layers

    def __str__(self):
        s = io.StringIO()
        self.model.summary(print_fn=lambda x: s.write(x + "\n"))
        return s.getvalue()
