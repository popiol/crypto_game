import numpy as np
from tensorflow import keras


class MlModel:

    def __init__(self, model: keras.Model):
        self.model = model

    def train(self, x: np.array, y: np.array):
        self.model.fit(x, y)

    def test(self, x: np.array, y: np.array):
        return self.model.evaluate(x, y)

    def predict(self, x: np.array):
        return self.model.predict(x, verbose=0)
