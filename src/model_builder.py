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
        V1 = auto()
        V5 = auto()
        V6 = auto()

    def build_model(self, asset_dependent=False, version: ModelVersion = ModelVersion.V1) -> MlModel:
        if version == self.ModelVersion.V1:
            return self.build_model_v1(asset_dependent)
        if version == self.ModelVersion.V5:
            return self.build_model_v5(asset_dependent)
        if version == self.ModelVersion.V6:
            return self.build_model_v6(asset_dependent)

    def build_model_v1(self, asset_dependent=False) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        l = inputs
        if asset_dependent:
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

    def build_model_v5(self, asset_dependent=False) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        l = inputs
        if asset_dependent:
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

    def build_model_v6(self, asset_dependent=False) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        l = inputs
        if asset_dependent:
            l = keras.layers.Permute((1, 3, 2))(l)
            l = keras.layers.Dense(100)(l)
            l = keras.layers.Dense(self.n_assets)(l)
            l = keras.layers.Permute((3, 1, 2))(l)
        else:
            l = keras.layers.Permute((2, 1, 3))(l)
        l = keras.layers.Reshape((self.n_assets, self.n_steps * self.n_features))(l)
        l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(100, activation="softsign")(l)
        l = keras.layers.Dense(self.n_outputs)(l)
        model = keras.Model(inputs=inputs, outputs=l)
        self.compile_model(model)
        return MlModel(model)

    def compile_model(self, model):
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")

    def most_important_dims(self, array: np.ndarray, dim: int, size: int):
        return np.argsort([np.abs(x).sum() for x in np.rollaxis(array, dim)])[-size:]

    def adjust_array_shape(self, array: np.ndarray, dim: int, size: int, filter_indices: list[int] = None) -> np.ndarray:
        assert size > 0
        old_shape = np.shape(array)
        assert 0 <= dim < len(old_shape)
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
            array = np.concatenate((array, np.random.normal(array.mean(), array.std(), add_shape)), axis=dim)
        return array

    def adjust_weights_shape(
        self, weights: list[np.ndarray], target_shape: tuple[int], filter_indices: list[int] = None
    ) -> list[np.ndarray]:
        new_weights = []
        if not weights:
            return new_weights
        assert len(weights) <= 2
        for x in target_shape:
            assert x > 0
        w0 = weights[0]
        for dim, size in enumerate(target_shape):
            w0 = self.adjust_array_shape(w0, dim, size, filter_indices)
        new_weights.append(w0)
        if len(weights) > 1:
            w1 = self.adjust_array_shape(weights[1], 0, target_shape[-1], filter_indices)
            new_weights.append(w1)
        return new_weights

    def copy_weights(
        self, from_model: keras.Model, to_model: keras.Model, names_map: dict = None, filter_indices: list[int] = None
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
        return config["name"].split("_")[0] in ["concatenate", "outer"]

    def get_model_tensor(
        self,
        model: MlModel,
        inputs: keras.layers.Input,
        layer_names: set,
        on_layer_start: Callable[[ModificationInput], ModificationOutput] = None,
        on_layer_end: Callable[[ModificationInput], ModificationOutput] = None,
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
                resp: ModificationOutput = on_layer_start(ModificationInput(index, config, tensor, layer_names, parent_layers))
                if resp and resp.tensor is not None:
                    tensor = resp.tensor
                if resp and resp.skip:
                    tensors[l.name] = tensor
                    continue
                if resp and resp.skip_branch:
                    continue
            config["name"] = self.fix_layer_name(config["name"], layer_names)
            if index == len(model.model.layers) - 2:
                config["units"] = self.n_outputs
            new_layer = l.from_config(config)
            tensor = new_layer(tensor)
            if on_layer_end:
                resp: ModificationOutput = on_layer_end(ModificationInput(index, config, tensor, layer_names, parent_layers))
                if resp and resp.tensor is not None:
                    tensor = resp.tensor
            tensors[l.name] = tensor
        return tensor

    def modify_model(
        self,
        model: MlModel,
        on_layer_start: Callable[[ModificationInput], ModificationOutput] = None,
        on_layer_end: Callable[[ModificationInput], ModificationOutput] = None,
        filter_indices: list[int] = None,
        raise_on_failure: bool = False,
    ) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features), name=model.model.layers[0].name)
        self.last_failed = False
        layer_names = set()
        try:
            tensor = self.get_model_tensor(model, inputs, layer_names, on_layer_start, on_layer_end)
            assert len(tensor.shape) == 3 and tensor.shape[2] == self.n_outputs
        except Exception as e:
            self.last_failed = True
            print("Modification failed")
            print(e)
            if raise_on_failure:
                print(model)
                raise e
            return model
        new_model = keras.Model(inputs=inputs, outputs=tensor)
        self.compile_model(new_model)
        self.copy_weights(model.model, new_model, filter_indices=filter_indices)
        print("Modification succeeded")
        return MlModel(new_model)

    def adjust_dimensions(self, model: MlModel) -> MlModel:
        model = self.adjust_n_assets(model)
        model = self.adjust_n_features(model)
        return model

    def adjust_n_assets(self, model: MlModel) -> MlModel:
        n_assets = model.model.layers[0].batch_shape[2]
        assert self.n_assets >= n_assets

        print("Adjust n_assets from", n_assets, "to", self.n_assets)

        def on_layer_start(input: ModificationInput):
            nonlocal n_assets
            if input.config["name"].startswith("dense") and input.config["units"] == n_assets:
                input.config["units"] = self.n_assets
            elif input.config["name"].startswith("conv") and input.config["filters"] == n_assets:
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
        assert self.n_features >= n_features
        if self.n_features == n_features:
            return model

        print("Adjust n_features")

        def on_layer_start(input: ModificationInput):
            if input.config["name"].startswith("dense") and input.config["units"] == n_features:
                input.config["units"] = self.n_features
            elif input.config["name"].startswith("conv1d") and input.config["filters"] == n_features:
                input.config["filters"] = self.n_features
            elif input.config["name"].startswith("reshape") and n_features in input.config["target_shape"]:
                shape = list(input.config["target_shape"])
                shape[shape.index(n_features)] = self.n_features
                input.config["target_shape"] = shape

        return self.modify_model(model, on_layer_start, raise_on_failure=True)

    def filter_assets(self, model: MlModel, asset_list: list[str], current_assets: set[str]):
        indices = [index for index, asset in enumerate(asset_list) if asset in current_assets]
        n_assets = len(indices)
        print("Filter assets from", len(asset_list), "to", n_assets)

        def on_layer_start(input: ModificationInput):
            if input.config["name"].startswith("dense") and input.config["units"] == self.n_assets:
                input.config["units"] = n_assets
            elif input.config["name"].startswith("conv") and input.config["filters"] == self.n_assets:
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
        assert 0 <= start_index <= end_index < len(model.model.layers) - 2

        def on_layer_start(input: ModificationInput):
            if start_index <= input.index <= end_index:
                return ModificationOutput(skip=True)

        return self.modify_model(model, on_layer_start)

    def add_dense_layer(self, model: MlModel, before_index: int, size: int):
        print("Add dense layer", before_index, size)
        assert 0 <= before_index < len(model.model.layers) - 1

        def on_layer_start(input: ModificationInput):
            if input.index == before_index:
                if not isinstance(input.tensor, keras.KerasTensor):
                    raise ModificationError(f"Invalid tensor type: {type(input.tensor)}")
                layer_name = self.fix_layer_name("dense", input.layer_names)
                return ModificationOutput(tensor=keras.layers.Dense(size, name=layer_name)(input.tensor))

        return self.modify_model(model, on_layer_start)

    def add_dropout(self, model: MlModel, before_index: int):
        print("Add dropout", before_index)
        assert 0 <= before_index < len(model.model.layers) - 1

        def on_layer_start(input: ModificationInput):
            if input.index == before_index:
                if not isinstance(input.tensor, keras.KerasTensor):
                    raise ModificationError(f"Invalid tensor type: {type(input.tensor)}")
                layer_name = self.fix_layer_name("dropout", input.layer_names)
                return ModificationOutput(tensor=keras.layers.Dropout(0.3, name=layer_name)(input.tensor))

        return self.modify_model(model, on_layer_start)

    def resize_layer(self, model: MlModel, layer_index: int, new_size: int):
        print("Resize layer", layer_index, new_size)
        assert 0 <= layer_index < len(model.model.layers) - 2

        def on_layer_start(input: ModificationInput):
            if input.index == layer_index:
                layer_type = input.config["name"].split("_")[0]
                if layer_type != "dense":
                    raise ModificationError(f"Can only resize Dense layer, {layer_type} given")
                input.config["units"] = new_size

        return self.modify_model(model, on_layer_start)

    def add_relu(self, model: MlModel, layer_index: int):
        print("Add relu", layer_index)
        assert 0 <= layer_index < len(model.model.layers) - 2

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
        assert 0 <= layer_index < len(model.model.layers) - 2

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
        assert 0 <= layer_index < len(model.model.layers) - 3
        tensor_to_reuse = None
        n_layers = len(model.model.layers) - 1

        def on_layer_end(input: ModificationInput):
            nonlocal tensor_to_reuse
            if input.index == layer_index:
                if not isinstance(input.tensor, keras.KerasTensor):
                    raise ModificationError(f"Invalid tensor type: {type(input.tensor)}")
                if len(input.tensor.shape) != 3 or input.tensor.shape[1] != self.n_assets:
                    raise ModificationError("Cannot reuse the layer")
                tensor_to_reuse = input.tensor

        def on_layer_start(input: ModificationInput):
            if input.index == n_layers - 1:
                if tensor_to_reuse is None:
                    raise ModificationError("No tensor to reuse")
                layer_name = self.fix_layer_name("concatenate", input.layer_names)
                return ModificationOutput(tensor=keras.layers.Concatenate(name=layer_name)([input.tensor, tensor_to_reuse]))

        return self.modify_model(model, on_layer_start, on_layer_end)

    class MergeVersion(Enum):
        CONCAT = auto()
        TRANSFORM = auto()
        SELECT = auto()
        MULTIPLY = auto()
        NORM = auto()

    @keras.utils.register_keras_serializable()
    class OuterProduct(keras.layers.Layer):
        def __init__(self, name: str, **kwargs):
            name = name or "outer_product"
            super().__init__(name=name, **kwargs)

        def build(self, input_shape):
            assert type(input_shape) == list
            assert len(input_shape) == 2
            shape_1, shape_2 = input_shape
            assert shape_1[:-1] == shape_2[:-1]

        def call(self, inputs):
            x, y = inputs
            result = tf.expand_dims(x, axis=-1) * tf.expand_dims(y, axis=-2)
            return keras.layers.Reshape((*x.shape[1:-1], x.shape[-1] * y.shape[-1]))(result)

        def compute_output_shape(self, input_shape):
            shape_1, shape_2 = input_shape
            return (*shape_1[:-1], shape_1[-1] * shape_2[-1])

    def prepare_for_merge(
        self, model: MlModel, inputs: keras.layers.Input, merge_version: MergeVersion, names_map: dict, layer_names: dict
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
            if input.index == n_layers - 1:
                return ModificationOutput(skip=True)
            if merge_version == self.MergeVersion.SELECT:
                if input.parents[0].split("_")[0] == "input":
                    if input.config["name"] != chosen_branch:
                        return ModificationOutput(skip_branch=True)

        n_layers = len(model.model.layers) - 1
        inputs.name = model.model.layers[0].name
        return self.get_model_tensor(model, inputs, layer_names, on_layer_start)

    def merge_models(self, model_1: MlModel, model_2: MlModel, merge_version: MergeVersion = MergeVersion.CONCAT) -> MlModel:
        model_1 = self.adjust_dimensions(model_1)
        model_2 = self.adjust_dimensions(model_2)
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        names_map = {}
        layer_names = set()
        tensor_1 = self.prepare_for_merge(model_1, inputs, merge_version, names_map, layer_names)
        tensor_2 = self.prepare_for_merge(model_2, inputs, merge_version, names_map, layer_names)
        if merge_version == self.MergeVersion.MULTIPLY:
            tensor_1_10 = keras.layers.Dense(10, name=self.fix_layer_name("dense", layer_names))(tensor_1)
            tensor_2_10 = keras.layers.Dense(10, name=self.fix_layer_name("dense", layer_names))(tensor_2)
            tensor = self.OuterProduct(name=self.fix_layer_name("outer_product", layer_names))([tensor_1_10, tensor_2_10])
            tensor = keras.layers.Concatenate(name=self.fix_layer_name("concatenate", layer_names))([tensor_1, tensor_2, tensor])
            tensor = keras.layers.Dense(100, name=self.fix_layer_name("dense", layer_names))(tensor)
        else:
            tensor = keras.layers.Concatenate(name=self.fix_layer_name("concatenate", layer_names))([tensor_1, tensor_2])
        if merge_version == self.MergeVersion.TRANSFORM:
            tensor = keras.layers.Dense(100, name=self.fix_layer_name("dense", layer_names))(tensor)
        elif merge_version == self.MergeVersion.NORM:
            tensor = keras.layers.UnitNormalization(name=self.fix_layer_name("unit_normalization", layer_names))(tensor)
        tensor = keras.layers.Dense(self.n_outputs, name=self.fix_layer_name("dense", layer_names))(tensor)
        new_model = keras.Model(inputs=inputs, outputs=tensor)
        self.compile_model(new_model)
        self.copy_weights(model_1.model, new_model, names_map)
        self.copy_weights(model_2.model, new_model, names_map)
        return MlModel(new_model)
