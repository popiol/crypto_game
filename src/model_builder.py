from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.keras import keras
from src.ml_model import MlModel


@dataclass
class ModificationResult:
    skip: bool = False
    tensor: keras.KerasTensor = None


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

    def copy_weights(self, from_model: keras.Model, to_model: keras.Model):
        for l in to_model.layers[1:]:
            if not l.get_weights():
                continue
            for from_l in from_model.layers[1:]:
                if from_l.name == l.name:
                    weights = from_l.get_weights()
                    new_weights = self.adjust_weights_shape(weights, np.shape(l.get_weights()[0]))
                    l.set_weights(new_weights)
                    break

    def fix_reshape(self, config: dict, tensor: keras.KerasTensor):
        if config["name"].split("_")[0] == "reshape":
            target_n_dim = len(config["target_shape"])
            input_shape = tensor.shape[1:]
            assert 0 < target_n_dim <= len(input_shape)
            output_shape = input_shape[: target_n_dim - 1] + tuple([np.prod(input_shape[target_n_dim - 1 :])])
            config["target_shape"] = output_shape

    def modify_model(self, model: MlModel, modification: Callable) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features), name=model.model.layers[0].name)
        tensors = {inputs.name: inputs}
        for index, l in enumerate(model.model.layers[1:]):
            parent_layers = model.get_parent_layer_names(index)
            tensor = tensors[parent_layers[0]] if len(parent_layers) == 1 else [tensors[x] for x in parent_layers]
            config = l.get_config()
            try:
                resp: ModificationResult = modification(index, config, tensor)
                if resp and resp.skip:
                    tensors[l.name] = tensor
                    continue
                if resp and resp.tensor is not None:
                    tensor = resp.tensor
                self.fix_reshape(config, tensor)
                new_layer = l.from_config(config)
                tensor = new_layer(tensor)
            except ValueError:
                return model
            tensors[l.name] = tensor
        if tensor.shape != (None, self.n_assets, self.n_outputs):
            return model
        new_model = keras.Model(inputs=inputs, outputs=tensor)
        self.compile_model(new_model)
        self.copy_weights(model.model, new_model)
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

        return self.modify_model(model, modification)

    def add_dense_layer(self, model: MlModel, before_index: int, size: int):
        print("add dense layer", before_index, size)
        assert 0 <= before_index < len(model.model.layers) - 1

        def modification(index: int, config: dict, tensor: keras.KerasTensor):
            if index == before_index:
                return ModificationResult(tensor=keras.layers.Dense(size)(tensor))

        return self.modify_model(model, modification)

    def add_conv_layer(self, model: MlModel, before_index: int):
        print("add conv layer", before_index)
        assert 0 <= before_index < len(model.model.layers) - 1

        def modification(index: int, config: dict, tensor: keras.KerasTensor | list):
            if index == before_index and isinstance(tensor, keras.KerasTensor):
                if len(tensor.shape) == 3:
                    tensor = keras.layers.Permute((2, 1))(tensor)
                    tensor = keras.layers.Conv1D(tensor.shape[-1], 3)(tensor)
                    tensor = keras.layers.Permute((2, 1))(tensor)
                    return ModificationResult(tensor=tensor)
                if len(tensor.shape) == 4:
                    tensor = keras.layers.Conv2D(tensor.shape[-1], 3)(tensor)
                    return ModificationResult(tensor=tensor)

        return self.modify_model(model, modification)

    def resize_layer(self, model: MlModel, layer_index: int, new_size: int):
        print("resize layer", layer_index, new_size)
        assert 0 <= layer_index < len(model.model.layers) - 2

        def modification(index: int, config: dict, tensor: keras.KerasTensor):
            if index == layer_index and config["name"].split("_")[0] == "dense":
                config["units"] = new_size

        return self.modify_model(model, modification)

    def merge_models(self, model_1: MlModel, model_2: MlModel) -> MlModel:
        model_1 = self.adjust_n_assets(model_1)
        model_2 = self.adjust_n_assets(model_2)
        model_id_1 = model_1.model.name.split("_")[-1]
        model_id_2 = model_2.model.name.split("_")[-1]
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        tensors = {model_1.model.layers[0].name: inputs}
        for index, l in enumerate(model_1.model.layers[1:-1]):
            parent_layers = model_1.get_parent_layer_names(index)
            tensor_1 = tensors[parent_layers[0]] if len(parent_layers) == 1 else [tensors[x] for x in parent_layers]
            config = l.get_config()
            config["name"] += "_" + model_id_1
            new_layer = l.from_config(config)
            tensor_1 = new_layer(tensor_1)
            tensors[l.name] = tensor_1
        tensors = {model_2.model.layers[0].name: inputs}
        for index, l in enumerate(model_2.model.layers[1:-1]):
            parent_layers = model_2.get_parent_layer_names(index)
            tensor_2 = tensors[parent_layers[0]] if len(parent_layers) == 1 else [tensors[x] for x in parent_layers]
            config = l.get_config()
            config["name"] += "_" + model_id_2
            new_layer = l.from_config(config)
            tensor_2 = new_layer(tensor_2)
            tensors[l.name] = tensor_2
        tensor = keras.layers.Concatenate()([tensor_1, tensor_2])
        tensor = keras.layers.Dense(self.n_outputs)(tensor)
        new_model = keras.Model(inputs=inputs, outputs=tensor)
        self.compile_model(new_model)
        for l in new_model.layers[1:-2]:
            if not l.get_weights():
                continue
            from_layer = None
            for l_1 in model_1.model.layers[1:]:
                if l.name == l_1.name + "_" + model_id_1:
                    from_layer = l_1
                    break
            if from_layer is None:
                for l_2 in model_2.model.layers[1:]:
                    if l.name == l_2.name + "_" + model_id_2:
                        from_layer = l_2
                        break
            if from_layer is not None:
                weights = from_layer.get_weights()
                new_weights = self.adjust_weights_shape(weights, np.shape(l.get_weights()[0]))
                l.set_weights(new_weights)
        return MlModel(new_model)
