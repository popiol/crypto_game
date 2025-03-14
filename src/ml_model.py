from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.keras import keras


@dataclass
class MlModelLayer:
    name: str
    layer_type: str
    shape: tuple
    input_shape: tuple
    output_shape: tuple
    activation: str


class MlModel:

    def __init__(self, model: keras.Model):
        self.model = model

    def train(self, x: np.ndarray, y: np.ndarray, n_epochs: int = 1):
        self.model.fit(x, y, epochs=n_epochs, verbose=0)

    def test(self, x: np.ndarray, y: np.ndarray):
        return self.model.evaluate(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(np.array([x]), verbose=0)[0]

    def copy(self) -> MlModel:
        clone = keras.models.clone_model(self.model)
        clone.set_weights(self.model.get_weights())
        return MlModel(clone)

    def add_noise(self, std: float):
        weights = self.model.get_weights()
        for layer in weights:
            layer += np.random.normal(loc=0.0, scale=std, size=layer.shape)
        self.model.set_weights(weights)

    def get_parent_layer_names(self, index) -> list[str]:
        model_config = self.model.get_config()
        args = model_config["layers"][index + 1]["inbound_nodes"][0]["args"][0]
        if type(args) == dict:
            args = [args]
        return [x["config"]["keras_history"][0] for x in args]

    def get_layers(self) -> list[MlModelLayer]:
        layers = []
        input_shapes = {self.model.layers[0].name: self.model.layers[0].batch_shape}
        format_activation = lambda x: None if x == "linear" else x
        for index, l in enumerate(self.model.layers[1:]):
            parent_layers = self.get_parent_layer_names(index)
            input_shape = input_shapes[parent_layers[0]] if len(parent_layers) == 1 else [input_shapes[x] for x in parent_layers]
            input_shape_without_batch = input_shape[1:] if type(input_shape) == tuple else [x[1:] for x in input_shape]
            output_shape = l.compute_output_shape(input_shape)
            layers.append(
                MlModelLayer(
                    name=l.name,
                    layer_type=l.name.split("_")[0],
                    shape=tuple(l.weights[0].shape) if l.weights else None,
                    input_shape=input_shape_without_batch,
                    output_shape=output_shape[1:],
                    activation=format_activation(l.activation.__name__) if l.weights else None,
                )
            )
            input_shapes[l.name] = output_shape
        return layers

    def get_layer_short_desc(self, layer: keras.layers.Layer) -> str:
        node = layer.name.split("_")[0]
        if node == "input":
            node = "INPUT"
        if node == "dropout":
            node = f"DR {layer.rate}"
        if node == "dense":
            node = f"D {layer.units}"
        if node == "permute":
            node = f"P {','.join([str(x) for x in layer.dims])}"
        if node == "reshape":
            node = f"R {','.join([str(x) for x in layer.target_shape])}"
        if node == "unit":
            node = "NORM"
        if node == "outer":
            node = "OP"
        if node == "concatenate":
            node = "C"
        if self._layer_ids is not None:
            node = f"{self._layer_ids[layer.name]}. {node}"
        return node

    def get_edges(self) -> list[tuple[str, str]]:
        edges = []
        layers = {self.model.layers[0].name: self.model.layers[0]}
        self._layer_ids = {self.model.layers[0].name: 0}
        for index, l in enumerate(self.model.layers[1:]):
            layers[l.name] = l
            self._layer_ids[l.name] = index + 1
            parent_layers = self.get_parent_layer_names(index)
            for parent in parent_layers:
                edges.append((self.get_layer_short_desc(layers[parent]), self.get_layer_short_desc(l)))
        return edges

    def get_model_length(self):
        lengths: dict[str, int] = {}
        for index, l in enumerate(self.model.layers[1:]):
            parent_layers = self.get_parent_layer_names(index)
            lengths[l.name] = max(lengths.get(parent, 0) + 1 for parent in parent_layers)
        return max(lengths.values())

    def get_model_width(self):
        width = 0
        for index, l in enumerate(self.model.layers[1:]):
            parent_layers = self.get_parent_layer_names(index)
            if parent_layers[0].startswith("input"):
                width += 1
        return width

    def get_n_params(self):
        return np.sum([np.prod(v.shape) for v in self.model.trainable_weights])

    def get_weight_stats(self):
        all_weights = []
        for weights in self.model.trainable_weights:
            all_weights.extend(np.reshape(weights, -1))
        return {
            "mean": np.mean(all_weights).tolist(),
            "std": np.std(all_weights).tolist(),
            "min": np.min(all_weights).tolist(),
            "max": np.max(all_weights).tolist(),
        }

    def __str__(self):
        s = io.StringIO()
        df = pd.DataFrame(columns=["name", "parent", "out_shape", "act"])
        df.loc[len(df)] = ["input_layer", "", self.model.layers[0].batch_shape[1:], ""]
        for index, layer in enumerate(self.get_layers()):
            parents = self.get_parent_layer_names(index)
            df.loc[len(df)] = [layer.name, ",".join(parents), layer.output_shape, layer.activation or ""]
        print()
        print(tabulate(df, headers="keys", tablefmt="psql"))
        print("Params:", self.get_n_params())
        return s.getvalue()

    @property
    def name(self):
        return self.model.name

    @name.setter
    def name(self, x):
        self.model.name = x
