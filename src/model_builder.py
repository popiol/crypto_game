from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.keras import keras
from src.ml_model import MlModel


@dataclass
class ModificationResult:
    skip: bool = False
    tensor: keras.KerasTensor = None
    end_index: int = None


@dataclass
class ModelBuilder:

    n_steps: int
    n_assets: int
    n_features: int
    n_outputs: int

    def build_model(self, asset_dependant=False) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        l = inputs
        if asset_dependant:
            l = keras.layers.Permute((1, 3, 2))(l)
            l = keras.layers.Dense(100)(l)
            l = keras.layers.Dense(self.n_assets)(l)
            l = keras.layers.Permute((3, 1, 2))(l)
        else:
            l = keras.layers.Permute((2, 1, 3))(l)
        l = keras.layers.Reshape((self.n_assets, self.n_steps * self.n_features))(l)
        l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(100)(l)
        l = keras.layers.Dense(self.n_outputs)(l)
        model = keras.Model(inputs=inputs, outputs=l)
        self.compile_model(model)
        return MlModel(model)

    def compile_model(self, model):
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")

    @staticmethod
    def adjust_array_shape(array: np.array, dim: int, size: int) -> np.array:
        assert size > 0
        old_shape = np.shape(array)
        assert 0 <= dim < len(old_shape)
        old_size = old_shape[dim]
        if size == old_size:
            return array
        if size < old_size:
            index = [slice(x) for x in old_shape]
            index[dim] = slice(size)
            array = array[*index]
        elif size > old_size:
            add_shape = list(old_shape)
            add_shape[dim] = size - old_size
            array = np.concatenate((array, np.random.normal(0, array.std(), add_shape)), axis=dim)
        return array

    def adjust_weights_shape(self, weights: list[np.array], target_shape: tuple[int]) -> list[np.array]:
        new_weights = []
        if not weights:
            return new_weights
        assert len(weights) <= 2
        for x in target_shape:
            assert x > 0
        w0 = weights[0]
        for dim, size in enumerate(target_shape):
            w0 = self.adjust_array_shape(w0, dim, size)
        new_weights.append(w0)
        if len(weights) > 1:
            w1 = self.adjust_array_shape(weights[1], 0, target_shape[-1])
            new_weights.append(w1)
        return new_weights

    def copy_weights(self, from_model: keras.Model, to_model: keras.Model, skip_start: int = None, skip_end: int = None):
        if skip_start is not None:
            skip_end = skip_start if skip_end is None else skip_end
            if skip_end < skip_start:
                skip_start = None
                skip_end = None
        else:
            assert skip_end is None
        for index, l in enumerate(to_model.layers[1:]):
            if not l.get_weights():
                continue
            if skip_start is not None:
                if len(from_model.layers) < len(to_model.layers):
                    if skip_start <= index <= skip_end:
                        continue
                    elif index > skip_end:
                        index -= skip_end - skip_start + 1
                elif len(from_model.layers) > len(to_model.layers) and index >= skip_start:
                    index += skip_end - skip_start + 1
            weights = from_model.layers[index + 1].get_weights()
            new_weights = self.adjust_weights_shape(weights, np.shape(l.get_weights()[0]))
            l.set_weights(new_weights)

    def fix_reshape(self, config: dict, input_shape: tuple[int]):
        if config["name"].split("_")[0] == "reshape":
            target_n_dim = len(config["target_shape"])
            assert 0 < target_n_dim <= len(input_shape)
            output_shape = input_shape[: target_n_dim - 1] + tuple([np.prod(input_shape[target_n_dim - 1 :])])
            config["target_shape"] = output_shape

    def modify_model(self, model: MlModel, modification: Callable, start_index: int = None, end_index: int = None) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        tensor = inputs
        for index, l in enumerate(model.model.layers[1:]):
            config = l.get_config()
            resp: ModificationResult = modification(index, config, tensor)
            if resp and resp.skip:
                continue
            if resp and resp.tensor is not None:
                tensor = resp.tensor
            if resp and resp.end_index is not None:
                end_index = resp.end_index
            self.fix_reshape(config, tensor.shape[1:])
            new_layer = l.from_config(config)
            tensor = new_layer(tensor)
        if tensor.shape != (None, self.n_assets, self.n_outputs):
            return model
        new_model = keras.Model(inputs=inputs, outputs=tensor)
        self.compile_model(new_model)
        self.copy_weights(model.model, new_model, start_index, end_index)
        return MlModel(new_model)

    def adjust_n_assets(self, model: MlModel) -> MlModel:
        n_assets = model.model.layers[0].batch_shape[2]
        assert self.n_assets >= n_assets
        if self.n_assets == n_assets:
            return model
        layer_names = []

        def modification(index: int, config: dict, tensor: keras.KerasTensor):
            layer_names.append(config["name"].split("_")[0])
            if layer_names[-3:] == ["permute", "dense", "dense"] and config["units"] == n_assets:
                config["units"] = self.n_assets

        return self.modify_model(model, modification)

    def remove_layer(self, model: MlModel, start_index: int, end_index: int) -> MlModel:
        print("remove layers", start_index, end_index)
        assert 0 <= start_index <= end_index < len(model.model.layers) - 2

        def modification(index: int, config: dict, tensor: keras.KerasTensor):
            if start_index <= index <= end_index:
                return ModificationResult(skip=True)

        return self.modify_model(model, modification, start_index, end_index)

    def add_dense_layer(self, model: MlModel, before_index: int, size: int):
        print("add dense layer", before_index, size)
        assert 0 <= before_index < len(model.model.layers) - 1

        def modification(index: int, config: dict, tensor: keras.KerasTensor):
            if index == before_index:
                return ModificationResult(tensor=keras.layers.Dense(size)(tensor))

        return self.modify_model(model, modification, before_index)

    def add_conv_layer(self, model: MlModel, before_index: int):
        print("add conv layer", before_index)
        assert 0 <= before_index < len(model.model.layers) - 1

        def modification(index: int, config: dict, tensor: keras.KerasTensor):
            if index == before_index:
                if len(tensor.shape) == 3:
                    tensor = keras.layers.Permute((2, 1))(tensor)
                    tensor = keras.layers.Conv1D(tensor.shape[-1], 3)(tensor)
                    tensor = keras.layers.Permute((2, 1))(tensor)
                    return ModificationResult(tensor=tensor, end_index=before_index + 2)
                if len(tensor.shape) == 4:
                    tensor = keras.layers.Conv2D(tensor.shape[-1], 3)(tensor)
                    return ModificationResult(tensor=tensor, end_index=before_index)

        return self.modify_model(model, modification, before_index, before_index - 1)

    def resize_layer(self, model: MlModel, layer_index: int, new_size: int):
        print("resize layer", layer_index, new_size)
        assert 0 <= layer_index < len(model.model.layers) - 2

        def modification(index: int, config: dict, tensor: keras.KerasTensor):
            if index == layer_index and config["name"].split("_")[0] == "dense":
                config["units"] = new_size

        return self.modify_model(model, modification)

    def merge_models(self, model_1: MlModel, model_2: MlModel) -> MlModel:
        return MlModel(model_1)
