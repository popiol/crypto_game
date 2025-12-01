import random
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import numpy as np

from src.keras import keras, tf
from src.ml_model import MlModel


@dataclass
class ModificationInput:
    index: int
    config: dict
    tensor: keras.KerasTensor
    layer_names: set[str]
    parents: list[str]
    input_tensor: keras.layers.Input


@dataclass
class ModificationOutput:
    skip: bool = False
    skip_branch: bool = False
    tensor: keras.KerasTensor = None


class ModificationError(Exception):
    pass


@dataclass
class ModelBuilder:
    n_steps: int
    n_assets: int
    n_features: int
    n_outputs: int

    class ModelVersion(Enum):
        V2DEP = auto()
        V2IND = auto()
        V6DEP = auto()
        V6IND = auto()
        V8DEP = auto()
        V10DEP = auto()
        V11DEP = auto()
        V12DEP = auto()

    def build_model(self, version: ModelVersion = ModelVersion.V6IND) -> MlModel:
        if version == self.ModelVersion.V2DEP:
            return self.build_model_v2dep()
        if version == self.ModelVersion.V2IND:
            return self.build_model_v2ind()
        if version == self.ModelVersion.V6DEP:
            return self.build_model_v6dep()
        if version == self.ModelVersion.V6IND:
            return self.build_model_v6ind()
        if version == self.ModelVersion.V8DEP:
            return self.build_model_v8dep()
        if version == self.ModelVersion.V10DEP:
            return self.build_model_v10dep()
        if version == self.ModelVersion.V11DEP:
            return self.build_model_v11dep()
        if version == self.ModelVersion.V12DEP:
            return self.build_model_v12dep()

    def build_model_v2dep(self) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        l = inputs
        l = keras.layers.Permute((1, 3, 2))(l)
        l = keras.layers.Dense(10)(l)
        l = keras.layers.Dense(self.n_assets)(l)
        l = keras.layers.Permute((3, 1, 2))(l)
        l = keras.layers.Reshape((self.n_assets, self.n_steps * self.n_features))(l)
        l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(100)(l)
        l = keras.layers.Dense(100)(l)
        l = keras.layers.Dense(self.n_outputs)(l)
        model = keras.Model(inputs=inputs, outputs=l)
        self.compile_model(model)
        return MlModel(model)

    def build_model_v2ind(self) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        l = inputs
        l = keras.layers.Permute((2, 1, 3))(l)
        l = keras.layers.Reshape((self.n_assets, self.n_steps * self.n_features))(l)
        l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(100)(l)
        l = keras.layers.Dense(100)(l)
        l = keras.layers.Dense(self.n_outputs)(l)
        model = keras.Model(inputs=inputs, outputs=l)
        self.compile_model(model)
        return MlModel(model)

    def build_model_v6dep(self) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        l = inputs
        l = keras.layers.Permute((1, 3, 2))(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(self.n_assets, activation="relu")(l)
        l = keras.layers.Permute((3, 1, 2))(l)
        l = keras.layers.Reshape((self.n_assets, -1))(l)
        l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(self.n_outputs)(l)
        model = keras.Model(inputs=inputs, outputs=l)
        self.compile_model(model)
        return MlModel(model)

    def build_model_v6ind(self) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        l = inputs
        l = keras.layers.Permute((2, 1, 3))(l)
        l = keras.layers.Reshape((self.n_assets, -1))(l)
        l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(self.n_outputs)(l)
        model = keras.Model(inputs=inputs, outputs=l)
        self.compile_model(model)
        return MlModel(model)

    def build_model_v8dep(self) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        l = inputs
        l = keras.layers.Permute((2, 1, 3))(l)
        l = keras.layers.Reshape((self.n_assets, -1))(l)
        l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(100)(l)
        l2 = keras.layers.Activation("softmax", name="softmax")(l)
        l2 = keras.layers.Dot(axes=2)([l2, l2])
        l = keras.layers.Dot(axes=[2, 1])([l2, l])
        l = keras.layers.Dense(self.n_outputs)(l)
        model = keras.Model(inputs=inputs, outputs=l)
        self.compile_model(model)
        return MlModel(model)

    def build_model_v10dep(self) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        l = inputs
        l = keras.layers.Permute((2, 1, 3))(l)
        n_convs2 = 4
        l = keras.layers.ZeroPadding2D((n_convs2, 0))(l)
        for _ in range(n_convs2):
            l = keras.layers.Conv2D(10, 3, activation="relu")(l)
        l = keras.layers.Reshape((self.n_assets, -1))(l)
        l = keras.layers.UnitNormalization()(l)
        n_convs1 = 4
        l = keras.layers.ZeroPadding1D(n_convs1 * 4)(l)
        for _ in range(n_convs1):
            l = keras.layers.Conv1D(10, 9, activation="relu")(l)
        l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(self.n_outputs)(l)
        model = keras.Model(inputs=inputs, outputs=l)
        self.compile_model(model)
        return MlModel(model)

    def build_model_v11dep(self) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        l = inputs
        l = keras.layers.Permute((2, 1, 3))(l)
        l = keras.layers.Reshape((self.n_assets, -1))(l)
        l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        for _ in range(3):
            l1 = l
            l = keras.layers.Dense(10, activation="relu")(l)
            l = keras.layers.Concatenate()([l, l1])
        l1 = l
        l = keras.layers.Permute((2, 1))(l)
        l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(1)(l)
        l = keras.layers.Flatten()(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        l = keras.layers.Reshape((10, 1))(l)
        l = keras.layers.Dense(self.n_assets)(l)
        l = keras.layers.Permute((2, 1))(l)
        l = keras.layers.Concatenate()([l, l1])
        l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(self.n_outputs)(l)
        model = keras.Model(inputs=inputs, outputs=l)
        self.compile_model(model)
        return MlModel(model)

    def build_model_v12dep(self) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        l = inputs
        l = keras.layers.Permute((2, 1, 3))(l)
        l = keras.layers.Reshape((self.n_assets, -1))(l)
        l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(10)(l)
        l = keras.layers.Dot(axes=2)([l, l])
        l = keras.layers.Dense(self.n_outputs)(l)
        model = keras.Model(inputs=inputs, outputs=l)
        self.compile_model(model)
        return MlModel(model)

    def compile_model(self, model):
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")

    def most_important_dims(self, array: np.ndarray, dim: int, size: int):
        return np.argsort([np.abs(x).sum() for x in np.rollaxis(array, dim)])[-size:]

    def adjust_array_shape(self, array: np.ndarray, dim: int, size: int, filter_indices: list[int] | None = None) -> np.ndarray:
        assert size > 0, f"Invalid size {size}"
        old_shape = np.shape(array)
        assert 0 <= dim < len(old_shape), f"Invalid dimension {dim}"
        old_size = old_shape[dim]
        if size == old_size:
            return array
        if size < old_size:
            index = [slice(x) for x in old_shape]
            if filter_indices and len(filter_indices) == size:
                index[dim] = filter_indices
            else:
                index[dim] = self.most_important_dims(array, dim, size)
            array = array[*index]
        elif size > old_size:
            add_shape = list(old_shape)
            add_shape[dim] = size - old_size
            array = np.concatenate((array, np.zeros(add_shape)), axis=dim)
        return array

    def adjust_weights_shape(
        self, weights: list[np.ndarray], target_shape: tuple[int], filter_indices: list[int] = None
    ) -> list[np.ndarray]:
        new_weights = []
        if not weights:
            return new_weights
        assert len(weights) <= 2, f"Invalid weights length {len(weights)}"
        for x in target_shape:
            assert x > 0, f"Invalid dimension {x} in target_shape {target_shape}"
        w0 = weights[0]
        for dim, size in enumerate(target_shape):
            w0 = self.adjust_array_shape(w0, dim, size, filter_indices)
        new_weights.append(w0)
        if len(weights) > 1:
            w1 = self.adjust_array_shape(weights[1], 0, target_shape[-1], filter_indices)
            new_weights.append(w1)
        return new_weights

    def copy_weights(
        self,
        from_model: keras.Model,
        to_model: keras.Model,
        names_map: dict | None = None,
        filter_indices: list[int] | None = None,
    ):
        for l in to_model.layers[1:]:
            if not l.get_weights():
                continue
            for from_l in from_model.layers[1:]:
                if l.name == (names_map.get(from_l.name) if names_map else from_l.name):
                    weights = from_l.get_weights()
                    new_weights = self.adjust_weights_shape(weights, np.shape(l.get_weights()[0]), filter_indices)
                    l.set_weights(new_weights)
                    break

    def fix_layer_name(self, layer_name: str, layer_names: set[str], add: bool = True) -> str:
        parts = layer_name.split("_")
        layer_name = "_".join(parts[:1] + parts[max(1, len(parts) - 30) :])
        while layer_name in layer_names:
            parts = layer_name.split("_")
            if re.match("^[0-9]+$", parts[-1]):
                parts[-1] = str(int(parts[-1]) + 1)
            else:
                parts[-1] = parts[-1] + "_1"
            layer_name = "_".join(parts)
        if add:
            layer_names.add(layer_name)
        return layer_name

    def expects_list_of_tensors(self, config: dict):
        return config["name"].split("_")[0] in ["concatenate", "outer", "dot"]

    def fix_reshape(self, config: dict):
        if not config["name"].startswith("reshape"):
            return
        if self.n_assets not in config["target_shape"]:
            return
        if len(config["target_shape"]) != 2:
            return
        if config["target_shape"][-1] == self.n_assets:
            config["target_shape"] = (-1, self.n_assets)
        elif config["target_shape"][0] == self.n_assets:
            config["target_shape"] = (self.n_assets, -1)

    def get_model_tensor(
        self,
        model: MlModel,
        inputs: keras.layers.Input,
        layer_names: set,
        on_layer_start: Callable[[ModificationInput], ModificationOutput | None] | None = None,
        on_layer_end: Callable[[ModificationInput], ModificationOutput | None] | None = None,
        raise_on_layer_failure: bool = True,
    ) -> keras.KerasTensor:
        tensors = {inputs.name: inputs}
        for index, l in enumerate(model.model.layers[1:]):
            parent_layers = model.get_parent_layer_names(index)
            tensor = [tensors[x] for x in parent_layers if x in tensors]
            if not tensor:
                continue
            config = l.get_config()
            if len(tensor) == 1:
                if self.expects_list_of_tensors(config):
                    tensors[l.name] = tensor[0]
                    continue
                tensor = tensor[0]
            if on_layer_start:
                resp: ModificationOutput | None = on_layer_start(
                    ModificationInput(index, config, tensor, layer_names, parent_layers, inputs)
                )
                if resp and resp.tensor is not None:
                    tensor = resp.tensor
                if resp and resp.skip:
                    tensors[l.name] = tensor
                    continue
                if resp and resp.skip_branch:
                    continue
            config["name"] = self.fix_layer_name(config["name"], layer_names)
            self.fix_reshape(config)
            if index == len(model.model.layers) - 2 and config["name"].startswith("dense"):
                config["units"] = self.n_outputs
            try:
                new_layer = l.from_config(config)
                tensor = new_layer(tensor)
            except Exception as ex:
                if raise_on_layer_failure:
                    raise ex
                print("Creating layer failed:", ex)
            if on_layer_end:
                resp: ModificationOutput | None = on_layer_end(
                    ModificationInput(index, config, tensor, layer_names, parent_layers, inputs)
                )
                if resp and resp.tensor is not None:
                    tensor = resp.tensor
            tensors[l.name] = tensor
        return tensor

    def modify_model(
        self,
        model: MlModel,
        on_layer_start: Callable[[ModificationInput], ModificationOutput | None] | None = None,
        on_layer_end: Callable[[ModificationInput], ModificationOutput | None] | None = None,
        filter_indices: list[int] | None = None,
        raise_on_failure: bool = False,
        raise_on_layer_failure: bool = True,
        check_output_shape: bool = True,
    ) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features), name=model.model.layers[0].name)
        self.last_failed = False
        layer_names = set()
        try:
            tensor = self.get_model_tensor(model, inputs, layer_names, on_layer_start, on_layer_end, raise_on_layer_failure)
            if check_output_shape:
                assert len(tensor.shape) == 3 and tensor.shape[2] == self.n_outputs, f"Invalid output shape {tensor.shape}"
            new_model = keras.Model(inputs=inputs, outputs=tensor)
            self.compile_model(new_model)
            self.copy_weights(model.model, new_model, filter_indices=filter_indices)
            print("Modification succeeded")
            return MlModel(new_model)
        except Exception as e:
            self.last_failed = True
            print("Modification failed")
            print(e)
            if raise_on_failure:
                print(model)
                raise e
            return model

    def adjust_dimensions(self, model: MlModel) -> MlModel:
        model = self.adjust_n_assets(model)
        model = self.adjust_n_features(model)
        model = self.adjust_n_outputs(model)
        return model

    def adjust_n_assets(self, model: MlModel) -> MlModel:
        n_assets = model.model.layers[0].batch_shape[2]
        assert self.n_assets >= n_assets, f"New n_assets has to be higher got {self.n_assets} < {n_assets}"
        print("Adjust n_assets from", n_assets, "to", self.n_assets)

        def on_layer_start(input: ModificationInput):
            nonlocal n_assets
            if "units" in input.config and input.config["units"] == n_assets:
                input.config["units"] = self.n_assets
            elif "filters" in input.config and input.config["filters"] == n_assets:
                input.config["filters"] = self.n_assets
            elif input.config["name"].startswith("reshape") and n_assets in input.config["target_shape"]:
                shape = list(input.config["target_shape"])
                shape[shape.index(n_assets)] = self.n_assets
                input.config["target_shape"] = shape
            elif input.config["name"].startswith("gather"):
                n_assets = len(input.config["indices"])
                return ModificationOutput(skip=True)

        return self.modify_model(model, on_layer_start, raise_on_failure=True)

    def adjust_n_features(self, model: MlModel) -> MlModel:
        n_features = model.model.layers[0].batch_shape[3]
        assert self.n_features >= n_features, f"New n_features has to be higher got {self.n_features} < {n_features}"
        if self.n_features == n_features:
            return model

        print("Adjust n_features")

        def on_layer_start(input: ModificationInput):
            if "units" in input.config and input.config["units"] == n_features:
                input.config["units"] = self.n_features
            elif "filters" in input.config and input.config["filters"] == n_features:
                input.config["filters"] = self.n_features
            elif input.config["name"].startswith("reshape") and n_features in input.config["target_shape"]:
                shape = list(input.config["target_shape"])
                shape[shape.index(n_features)] = self.n_features
                input.config["target_shape"] = shape

        return self.modify_model(model, on_layer_start, raise_on_failure=True)

    def adjust_n_outputs(self, model: MlModel) -> MlModel:
        n_outputs = model.model.output_shape[-1]
        assert self.n_outputs >= n_outputs, f"New n_features has to be higher got {self.n_features} < {n_features}"
        if self.n_outputs == n_outputs:
            return model

        print("Adjust n_outputs")

        def on_layer_start(input: ModificationInput):
            if "units" in input.config and input.config["units"] == n_outputs:
                input.config["units"] = self.n_outputs
            elif "filters" in input.config and input.config["filters"] == n_outputs:
                input.config["filters"] = self.n_outputs
            elif input.config["name"].startswith("reshape") and n_outputs in input.config["target_shape"]:
                shape = list(input.config["target_shape"])
                shape[shape.index(n_outputs)] = self.n_outputs
                input.config["target_shape"] = shape

        return self.modify_model(model, on_layer_start, raise_on_failure=True)

    def filter_assets(self, model: MlModel, asset_list: list[str], current_assets: set[str]):
        indices = [index for index, asset in enumerate(asset_list) if asset in current_assets]
        n_assets = len(indices)
        print("Filter assets from", len(asset_list), "to", n_assets)

        def on_layer_start(input: ModificationInput):
            if "units" in input.config and input.config["units"] == self.n_assets:
                input.config["units"] = n_assets
            elif "filters" in input.config and input.config["filters"] == self.n_assets:
                input.config["filters"] = n_assets
            elif input.config["name"].startswith("reshape") and self.n_assets in input.config["target_shape"]:
                shape = list(input.config["target_shape"])
                shape[shape.index(self.n_assets)] = n_assets
                input.config["target_shape"] = shape
            skip = input.config["name"].startswith("gather")
            tensor = None
            if input.parents[0].split("_")[0] == "input":
                layer_name = self.fix_layer_name("gather", input.layer_names)
                tensor = self.Gather(indices, axis=2, name=layer_name)(input.tensor)
            return ModificationOutput(tensor=tensor, skip=skip)

        return self.modify_model(model, on_layer_start, filter_indices=indices, raise_on_failure=True)

    def pretrain(self, model: MlModel, asset_list: list[str], current_assets: set[str], asset_only: bool = False):
        print("Pretrain as autoencoder")
        indices = [index for index, asset in enumerate(asset_list) if asset in current_assets]
        n_assets = len(indices)
        names_map = {}
        names_map_inv = {}

        def on_layer_start(input: ModificationInput):
            names_map[input.config["name"]] = self.fix_layer_name(input.config["name"], input.layer_names, add=False)
            names_map_inv[self.fix_layer_name(input.config["name"], input.layer_names, add=False)] = input.config["name"]

        def on_layer_end(input: ModificationInput):
            if not input.config["name"].startswith("dense"):
                return
            if asset_only and input.config["units"] != n_assets:
                return
            if input.index == len(model.model.layers) - 2:
                return
            try:
                tensor = self.add_layers_for_pretrain(
                    input.tensor, n_assets, (None, self.n_steps, n_assets, self.n_features), input.layer_names
                )
                model_2 = keras.Model(inputs=input.input_tensor, outputs=tensor)
                self.compile_model(model_2)
                self.copy_weights(model.model, model_2, names_map)
                x = np.random.normal(size=(10, self.n_steps, self.n_assets, self.n_features))
                y = x.take(indices, 2)
                MlModel(model_2).train(x, y, n_epochs=10)
                self.copy_weights(model_2, model.model, names_map_inv)
            except Exception as ex:
                print(ex)
                print(model)
                print("Pretrain failed for a layer", input.tensor)

        self.modify_model(model, on_layer_start, on_layer_end, raise_on_failure=True)

    def add_layers_for_pretrain(self, tensor: keras.KerasTensor, n_assets: int, y_shape: tuple[int], layer_names: set[str]):
        shape = tensor.shape[1:]
        target_shape = y_shape[1:]
        if n_assets not in shape:
            layer_name = self.fix_layer_name("dense", layer_names)
            tensor = keras.layers.Dense(n_assets, name=layer_name)(tensor)
            shape = tensor.shape[1:]
        if np.prod(shape) != np.prod(target_shape):
            index = shape.index(n_assets)
            if index > 0:
                permute = list(range(1, len(shape) + 1))
                permute[0] = index + 1
                permute[index] = 1
                layer_name = self.fix_layer_name("permute", layer_names)
                tensor = keras.layers.Permute(permute, name=layer_name)(tensor)
                shape = tensor.shape[1:]
            if len(shape) > 2:
                layer_name = self.fix_layer_name("reshape", layer_names)
                tensor = keras.layers.Reshape((n_assets, -1), name=layer_name)(tensor)
                shape = tensor.shape[1:]
            dense_size = round(np.prod(target_shape) / n_assets)
            if dense_size != shape[-1]:
                layer_name = self.fix_layer_name("dense", layer_names)
                tensor = keras.layers.Dense(dense_size, name=layer_name)(tensor)
                shape = tensor.shape[1:]
            if len([dim for dim in target_shape if dim != n_assets]) > 1:
                layer_name = self.fix_layer_name("reshape", layer_names)
                tensor = keras.layers.Reshape((n_assets, *[dim for dim in target_shape if dim != n_assets]), name=layer_name)(
                    tensor
                )
                shape = tensor.shape[1:]
        index = shape.index(n_assets)
        target_index = target_shape.index(n_assets)
        if index != target_index:
            permute = list(range(1, len(target_shape) + 1))
            permute[index] = target_index + 1
            permute[target_index] = index + 1
            layer_name = self.fix_layer_name("permute", layer_names)
            tensor = keras.layers.Permute(permute, name=layer_name)(tensor)
            shape = tensor.shape[1:]
        assert shape == target_shape, f"Invalid shape {shape}, should be {target_shape}"
        return tensor

    def pretrain_with(self, model: MlModel, asset_list: list[str], current_assets: set[str], x: np.ndarray, y: np.ndarray = None):
        indices = [index for index, asset in enumerate(asset_list) if asset in current_assets]
        n_assets = len(indices)
        y = x if y is None else y
        try:
            index = x.shape.index(self.n_assets)
            y = y.take(indices, index)
        except ValueError:
            pass
        n_layers = len(model.model.layers) - 1
        names_map_inv = {}

        def on_layer_start(input: ModificationInput):
            if input.index == n_layers - 1:
                tensor = self.add_layers_for_pretrain(input.tensor, n_assets, y.shape, input.layer_names)
                return ModificationOutput(tensor=tensor, skip=True)
            names_map_inv[self.fix_layer_name(input.config["name"], input.layer_names, add=False)] = input.config["name"]

        try:
            model_2 = self.modify_model(model, on_layer_start, raise_on_failure=True, check_output_shape=False)
            model_2.train(x, y, n_epochs=10)
            self.copy_weights(model_2.model, model.model, names_map_inv)
        except Exception as ex:
            print(ex)
            print("Pretrain failed")

    @keras.utils.register_keras_serializable()
    class Gather(keras.layers.Layer):
        def __init__(self, indices: list[int], axis: int, name: str, **kwargs):
            self.indices = indices
            self.axis = axis
            name = name or "gather"
            super().__init__(name=name, **kwargs)

        def build(self, input_shape):
            pass

        def call(self, inputs):
            return tf.gather(inputs, self.indices, axis=self.axis)

        def compute_output_shape(self, input_shape):
            return (*input_shape[:-2], len(self.indices), input_shape[-1])

    def remove_layer(self, model: MlModel, start_index: int, end_index: int) -> MlModel:
        print("Remove layers", start_index, end_index)
        assert 0 <= start_index <= end_index < len(model.model.layers) - 2, "Index out of bounds"

        def on_layer_start(input: ModificationInput):
            if start_index <= input.index <= end_index:
                return ModificationOutput(skip=True)

        return self.modify_model(model, on_layer_start)

    def add_dense_layer(self, model: MlModel, before_index: int, size: int):
        print("Add dense layer", before_index, size)
        assert 0 <= before_index < len(model.model.layers) - 1, "Index out of bounds"

        def on_layer_start(input: ModificationInput):
            if input.index == before_index:
                if not isinstance(input.tensor, keras.KerasTensor):
                    raise ModificationError(f"Invalid tensor type: {type(input.tensor)}")
                layer_name = self.fix_layer_name("dense", input.layer_names)
                return ModificationOutput(tensor=keras.layers.Dense(size, name=layer_name)(input.tensor))

        return self.modify_model(model, on_layer_start)

    def add_conv_layer(self, model: MlModel, before_index: int, size: int):
        print("Add conv layer", before_index, size)
        assert 0 <= before_index < len(model.model.layers) - 1, "Index out of bounds"

        def on_layer_start(input: ModificationInput):
            if input.index == before_index:
                if not isinstance(input.tensor, keras.KerasTensor):
                    raise ModificationError(f"Invalid tensor type: {type(input.tensor)}")
                if self.n_assets not in input.tensor.shape or len(input.tensor.shape) != 4:
                    raise ModificationError(f"Invalid shape: {input.tensor.shape}")
                tensor = input.tensor
                permutation = None
                if tensor.shape[1] != self.n_assets:
                    index = input.tensor.shape.index(self.n_assets)
                    permutation = list(range(len(input.tensor.shape)))
                    permutation[index] = 1
                    permutation[1] = index
                    permutation = permutation[1:]
                    layer_name = self.fix_layer_name("permute", input.layer_names)
                    tensor = keras.layers.Permute(permutation, name=layer_name)(tensor)
                for size in [100, 10]:
                    layer_name = self.fix_layer_name("conv1d", input.layer_names)
                    tensor = keras.layers.TimeDistributed(keras.layers.Conv1D(size, 3, activation="relu"), name=layer_name)(
                        tensor
                    )
                if permutation is not None:
                    layer_name = self.fix_layer_name("permute", input.layer_names)
                    tensor = keras.layers.Permute(permutation, name=layer_name)(tensor)
                return ModificationOutput(tensor=tensor)

        return self.modify_model(model, on_layer_start)

    def add_dropout(self, model: MlModel, before_index: int):
        print("Add dropout", before_index)
        assert 0 <= before_index < len(model.model.layers) - 1, "Index out of bounds"

        def on_layer_start(input: ModificationInput):
            if input.index == before_index:
                if not isinstance(input.tensor, keras.KerasTensor):
                    raise ModificationError(f"Invalid tensor type: {type(input.tensor)}")
                layer_name = self.fix_layer_name("dropout", input.layer_names)
                return ModificationOutput(tensor=keras.layers.Dropout(0.3, name=layer_name)(input.tensor))

        return self.modify_model(model, on_layer_start)

    def resize_layer(self, model: MlModel, layer_index: int, new_size: int):
        print("Resize layer", layer_index, new_size)
        assert 0 <= layer_index < len(model.model.layers) - 2, "Index out of bounds"

        def on_layer_start(input: ModificationInput):
            if input.index == layer_index:
                layer_type = input.config["name"].split("_")[0]
                if layer_type != "dense":
                    raise ModificationError(f"Can only resize Dense layer, {layer_type} given")
                input.config["units"] = new_size

        return self.modify_model(model, on_layer_start)

    def add_relu(self, model: MlModel, layer_index: int):
        print("Add relu", layer_index)
        assert 0 <= layer_index < len(model.model.layers) - 2, "Index out of bounds"

        def on_layer_start(input: ModificationInput):
            if input.index == layer_index:
                if "activation" not in input.config:
                    raise ModificationError("Layer does not have activation attribute")
                if input.config["activation"] == "relu":
                    raise ModificationError("Already has RELU activation")
                input.config["activation"] = "relu"

        return self.modify_model(model, on_layer_start)

    def remove_relu(self, model: MlModel, layer_index: int):
        print("Remove relu", layer_index)
        assert 0 <= layer_index < len(model.model.layers) - 2, "Index out of bounds"

        def on_layer_start(input: ModificationInput):
            if input.index == layer_index:
                if "activation" not in input.config:
                    raise ModificationError("Layer does not have activation attribute")
                if input.config["activation"] == "linear":
                    raise ModificationError("No activation to remove")
                input.config["activation"] = "linear"

        return self.modify_model(model, on_layer_start)

    def reuse_layer(self, model: MlModel, layer_index: int):
        print("Reuse layer", layer_index)
        assert 0 <= layer_index < len(model.model.layers) - 3, "Index out of bounds"
        tensor_to_reuse = None
        n_layers = len(model.model.layers) - 1

        def on_layer_end(input: ModificationInput):
            nonlocal tensor_to_reuse
            if input.index == layer_index:
                if not isinstance(input.tensor, keras.KerasTensor):
                    raise ModificationError(f"Invalid tensor type: {type(input.tensor)}")
                if len(input.tensor.shape) != 3 or input.tensor.shape[1] != self.n_assets:
                    raise ModificationError(f"Invalid shape  {input.tensor.shape}")
                tensor_to_reuse = input.tensor

        def on_layer_start(input: ModificationInput):
            if input.index == n_layers - 1:
                if tensor_to_reuse is None:
                    raise ModificationError("No tensor to reuse")
                layer_name = self.fix_layer_name("concatenate", input.layer_names)
                tensor = keras.layers.Concatenate(name=layer_name)([input.tensor, tensor_to_reuse])
                layer_name = self.fix_layer_name("unit_normalization", input.layer_names)
                tensor = keras.layers.UnitNormalization(name=layer_name)(tensor)
                layer_name = self.fix_layer_name("dense", input.layer_names)
                tensor = keras.layers.Dense(100, name=layer_name)(tensor)
                return ModificationOutput(tensor=tensor)

        return self.modify_model(model, on_layer_start, on_layer_end)

    class MergeVersion(Enum):
        CONCAT = auto()
        TRANSFORM = auto()
        SELECT = auto()
        MULTIPLY = auto()
        DOT = auto()
        SERIAL = auto()

    @keras.utils.register_keras_serializable()
    class OuterProduct(keras.layers.Layer):
        def __init__(self, name: str, **kwargs):
            name = name or "outer_product"
            super().__init__(name=name, **kwargs)

        def build(self, input_shape):
            assert isinstance(input_shape, list), f"Invalid input_shape type {type(input_shape)}"
            assert len(input_shape) == 2, f"Expected 2 shapes got {input_shape}"
            shape_1, shape_2 = input_shape
            assert shape_1[:-1] == shape_2[:-1], f"Shapes need to have same last dimension, got {shape_1}, {shape_2}"

        def call(self, inputs):
            x, y = inputs
            result = tf.expand_dims(x, axis=-1) * tf.expand_dims(y, axis=-2)
            return keras.layers.Reshape((*x.shape[1:-1], x.shape[-1] * y.shape[-1]))(result)

        def compute_output_shape(self, input_shape):
            shape_1, shape_2 = input_shape
            return (*shape_1[:-1], shape_1[-1] * shape_2[-1])

    def prepare_for_merge(
        self,
        model: MlModel,
        inputs: keras.layers.Input,
        merge_version: MergeVersion,
        names_map: dict,
        layer_names: set,
        remove_last: bool = True,
        raise_on_layer_failure: bool = True,
    ) -> keras.KerasTensor:
        if merge_version == self.MergeVersion.SELECT:
            first_layers = []
            for index, l in enumerate(model.model.layers[1:]):
                parent_layers = model.get_parent_layer_names(index)
                if [x for x in parent_layers if x.startswith("input")]:
                    first_layers.append(l.name)
            chosen_branch = random.choice(first_layers)

        def on_layer_start(input: ModificationInput):
            names_map[input.config["name"]] = self.fix_layer_name(input.config["name"], input.layer_names, add=False)
            if remove_last and input.index == n_layers - 1:
                return ModificationOutput(skip=True)
            if merge_version == self.MergeVersion.SELECT:
                if input.parents[0].split("_")[0] == "input":
                    if input.config["name"] != chosen_branch:
                        return ModificationOutput(skip_branch=True)

        n_layers = len(model.model.layers) - 1
        inputs.name = model.model.layers[0].name
        return self.get_model_tensor(model, inputs, layer_names, on_layer_start, raise_on_layer_failure=raise_on_layer_failure)

    def merge_models(self, model_1: MlModel, model_2: MlModel, merge_version: MergeVersion = MergeVersion.CONCAT) -> MlModel:
        model_1 = self.adjust_dimensions(model_1)
        model_2 = self.adjust_dimensions(model_2)
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        names_map = {}
        layer_names = set()
        tensor_1 = self.prepare_for_merge(model_1, inputs, merge_version, names_map, layer_names)
        if merge_version != self.MergeVersion.SERIAL:
            remove_last = merge_version != self.MergeVersion.DOT
            tensor_2 = self.prepare_for_merge(model_2, inputs, merge_version, names_map, layer_names, remove_last=remove_last)
        if merge_version == self.MergeVersion.SERIAL:
            tensor = self.prepare_for_merge(
                model_2, tensor_1, merge_version, names_map, layer_names, raise_on_layer_failure=False
            )
        elif merge_version == self.MergeVersion.DOT:
            tensor_1 = keras.layers.Activation("softmax", name=self.fix_layer_name("softmax", layer_names))(tensor_1)
            tensor_2 = keras.layers.Permute((2, 1), name=self.fix_layer_name("permute", layer_names))(tensor_2)
            tensor_2 = keras.layers.Dense(tensor_1.shape[-1], name=self.fix_layer_name("dense", layer_names))(tensor_2)
            tensor = keras.layers.Dot(axes=2, name=self.fix_layer_name("dot", layer_names))([tensor_1, tensor_2])
            tensor = keras.layers.UnitNormalization(name=self.fix_layer_name("unit_normalization", layer_names))(tensor)
            tensor = keras.layers.Dense(self.n_outputs, name=self.fix_layer_name("dense", layer_names))(tensor)
        elif merge_version == self.MergeVersion.MULTIPLY:
            tensor_1_10 = keras.layers.Dense(10, name=self.fix_layer_name("dense", layer_names))(tensor_1)
            tensor_2_10 = keras.layers.Dense(10, name=self.fix_layer_name("dense", layer_names))(tensor_2)
            tensor = self.OuterProduct(name=self.fix_layer_name("outer_product", layer_names))([tensor_1_10, tensor_2_10])
            tensor = keras.layers.Concatenate(name=self.fix_layer_name("concatenate", layer_names))([tensor_1, tensor_2, tensor])
            tensor = keras.layers.UnitNormalization(name=self.fix_layer_name("unit_normalization", layer_names))(tensor)
        else:
            tensor = keras.layers.Concatenate(name=self.fix_layer_name("concatenate", layer_names))([tensor_1, tensor_2])
            tensor = keras.layers.UnitNormalization(name=self.fix_layer_name("unit_normalization", layer_names))(tensor)
        if merge_version == self.MergeVersion.TRANSFORM:
            tensor = keras.layers.Dense(100, name=self.fix_layer_name("dense", layer_names))(tensor)
        if tensor.shape[-1] != self.n_outputs:
            tensor = keras.layers.Dense(self.n_outputs, name=self.fix_layer_name("dense", layer_names))(tensor)
        new_model = keras.Model(inputs=inputs, outputs=tensor)
        self.compile_model(new_model)
        self.copy_weights(model_1.model, new_model, names_map)
        self.copy_weights(model_2.model, new_model, names_map)
        return MlModel(new_model)
