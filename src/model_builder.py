import random
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import numpy as np

from src.keras import keras
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
        V2 = auto()

    def build_model(self, asset_dependent=False, version: ModelVersion = ModelVersion.V1) -> MlModel:
        if version == self.ModelVersion.V1:
            return self.build_model_v1(asset_dependent)
        if version == self.ModelVersion.V2:
            return self.build_model_v2(asset_dependent)

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

    def build_model_v2(self, asset_dependent=False) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        l = inputs
        l = keras.layers.Dropout(0.3)(l)
        if asset_dependent:
            l = keras.layers.Permute((1, 3, 2))(l)
            l = keras.layers.Dense(100)(l)
            l = keras.layers.Dense(self.n_assets)(l)
            l = keras.layers.Permute((3, 2, 1))(l)
        else:
            l = keras.layers.Permute((2, 3, 1))(l)
        l = keras.layers.Dense(100)(l)
        l = keras.layers.Permute((1, 3, 2))(l)
        l = keras.layers.Dense(100)(l)
        l = keras.layers.Reshape((self.n_assets, 10000))(l)
        l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(self.n_outputs)(l)
        model = keras.Model(inputs=inputs, outputs=l)
        self.compile_model(model)
        return MlModel(model)

    def compile_model(self, model):
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")

    def most_important_dims(self, array: np.ndarray, dim: int, size: int):
        return np.argsort([np.abs(x).sum() for x in np.rollaxis(array, dim)])[-size:]

    def adjust_array_shape(self, array: np.ndarray, dim: int, size: int) -> np.ndarray:
        assert size > 0
        old_shape = np.shape(array)
        assert 0 <= dim < len(old_shape)
        old_size = old_shape[dim]
        if size == old_size:
            return array
        if size < old_size:
            index = [slice(x) for x in old_shape]
            index[dim] = self.most_important_dims(array, dim, size)
            array = array[*index]
        elif size > old_size:
            add_shape = list(old_shape)
            add_shape[dim] = size - old_size
            array = np.concatenate((array, np.random.normal(0, array.std(), add_shape)), axis=dim)
        return array

    def adjust_weights_shape(self, weights: list[np.ndarray], target_shape: tuple[int]) -> list[np.ndarray]:
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

    def get_model_tensor(
        self,
        model: MlModel,
        inputs: keras.layers.Input,
        layer_names: set,
        modification: Callable[[ModificationInput], ModificationOutput],
    ):
        tensors = {inputs.name: inputs}
        for index, l in enumerate(model.model.layers[1:]):
            parent_layers = model.get_parent_layer_names(index)
            tensor = [tensors[x] for x in parent_layers if x in tensors]
            if not tensor:
                continue
            config = l.get_config()
            if len(tensor) == 1:
                if config["name"].split("_")[0] == "concatenate":
                    continue
                tensor = tensor[0]
            resp: ModificationOutput = modification(ModificationInput(index, config, tensor, layer_names, parent_layers))
            if resp and resp.skip:
                tensors[l.name] = tensor
                continue
            if resp and resp.skip_branch:
                continue
            if resp and resp.tensor is not None:
                tensor = resp.tensor
            self.fix_reshape(config, tensor)
            config["name"] = self.fix_layer_name(config["name"], layer_names)
            new_layer = l.from_config(config)
            tensor = new_layer(tensor)
            tensors[l.name] = tensor
        return tensor

    def modify_model(self, model: MlModel, modification: Callable[[ModificationInput], ModificationOutput]) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features), name=model.model.layers[0].name)
        self.last_failed = False
        layer_names = set()
        try:
            tensor = self.get_model_tensor(model, inputs, layer_names, modification)
        except (ValueError, TypeError, AttributeError, ModificationError):
            self.last_failed = True
            print("Modification failed")
            return model
        if tensor.shape != (None, self.n_assets, self.n_outputs):
            self.last_failed = True
            print("Modification failed")
            return model
        new_model = keras.Model(inputs=inputs, outputs=tensor)
        self.compile_model(new_model)
        self.copy_weights(model.model, new_model)
        print("Modification succeeded")
        return MlModel(new_model)

    def adjust_dimensions(self, model: MlModel) -> MlModel:
        model = self.adjust_n_assets(model)
        model = self.adjust_n_features(model)
        return model

    def adjust_n_assets(self, model: MlModel) -> MlModel:
        n_assets = model.model.layers[0].batch_shape[2]
        assert self.n_assets >= n_assets
        if self.n_assets == n_assets:
            return model

        print("Adjust n_assets")

        def modification(input: ModificationInput):
            if input.config["name"].startswith("dense") and input.config["units"] == n_assets:
                input.config["units"] = self.n_assets
            elif input.config["name"].startswith("conv") and input.config["filters"] == n_assets:
                input.config["filters"] = self.n_assets

        return self.modify_model(model, modification)

    def adjust_n_features(self, model: MlModel) -> MlModel:
        n_features = model.model.layers[0].batch_shape[3]
        assert self.n_features >= n_features
        if self.n_features == n_features:
            return model

        print("Adjust n_features")

        def modification(input: ModificationInput):
            if input.config["name"].startswith("dense") and input.config["units"] == n_features:
                input.config["units"] = self.n_features
            elif input.config["name"].startswith("conv1d") and input.config["filters"] == n_features:
                input.config["filters"] = self.n_features

        return self.modify_model(model, modification)

    def remove_layer(self, model: MlModel, start_index: int, end_index: int) -> MlModel:
        print("Remove layers", start_index, end_index)
        assert 0 <= start_index <= end_index < len(model.model.layers) - 2

        def modification(input: ModificationInput):
            if start_index <= input.index <= end_index:
                return ModificationOutput(skip=True)

        return self.modify_model(model, modification)

    def add_dense_layer(self, model: MlModel, before_index: int, size: int):
        print("Add dense layer", before_index, size)
        assert 0 <= before_index < len(model.model.layers) - 1

        def modification(input: ModificationInput):
            if input.index == before_index:
                if not isinstance(input.tensor, keras.KerasTensor):
                    raise ModificationError(f"Invalid tensor type: {type(input.tensor)}")
                layer_name = self.fix_layer_name("dense", input.layer_names)
                return ModificationOutput(tensor=keras.layers.Dense(size, name=layer_name)(input.tensor))

        return self.modify_model(model, modification)

    def add_conv_layer(self, model: MlModel, before_index: int):
        print("Add conv layer", before_index)
        assert 0 <= before_index < len(model.model.layers) - 1

        def modification(input: ModificationInput):
            if input.index == before_index:
                if not isinstance(input.tensor, keras.KerasTensor):
                    raise ModificationError(f"Invalid tensor type: {type(input.tensor)}")
                if len(input.tensor.shape) == 3:
                    layer_name = self.fix_layer_name("permute", input.layer_names)
                    tensor = keras.layers.Permute((2, 1), name=layer_name)(input.tensor)
                    layer_name = self.fix_layer_name("conv1d", input.layer_names)
                    tensor = keras.layers.Conv1D(tensor.shape[-1], 3, name=layer_name)(tensor)
                    layer_name = self.fix_layer_name("permute", input.layer_names)
                    tensor = keras.layers.Permute((2, 1), name=layer_name)(tensor)
                    return ModificationOutput(tensor=tensor)
                if len(input.tensor.shape) == 4:
                    layer_name = self.fix_layer_name("conv2d", input.layer_names)
                    tensor = keras.layers.Conv2D(input.tensor.shape[-1], 3, name=layer_name)(input.tensor)
                    return ModificationOutput(tensor=tensor)
                raise ModificationError(f"Invalid tensor shape: {input.tensor.shape}")

        return self.modify_model(model, modification)

    def resize_layer(self, model: MlModel, layer_index: int, new_size: int):
        print("Resize layer", layer_index, new_size)
        assert 0 <= layer_index < len(model.model.layers) - 2

        def modification(input: ModificationInput):
            if input.index == layer_index:
                layer_type = input.config["name"].split("_")[0]
                if layer_type != "dense":
                    raise ModificationError(f"Can only resize Dense layer, {layer_type} given")
                input.config["units"] = new_size

        return self.modify_model(model, modification)

    def add_relu(self, model: MlModel, layer_index: int):
        print("Add relu", layer_index)
        assert 0 <= layer_index < len(model.model.layers) - 2

        def modification(input: ModificationInput):
            if input.index == layer_index:
                if input.config["activation"] == "relu":
                    raise ModificationError("Already has RELU activation")
                input.config["activation"] = "relu"

        return self.modify_model(model, modification)

    def remove_relu(self, model: MlModel, layer_index: int):
        print("Remove relu", layer_index)
        assert 0 <= layer_index < len(model.model.layers) - 2

        def modification(input: ModificationInput):
            if input.index == layer_index:
                if input.config["activation"] == "linear":
                    raise ModificationError("No activation to remove")
                input.config["activation"] = "linear"

        return self.modify_model(model, modification)

    class MergeVersion(Enum):
        CONCAT = auto()
        TRANSFORM = auto()
        SELECT = auto()

    def merge_models(self, model_1: MlModel, model_2: MlModel, merge_version: MergeVersion = MergeVersion.CONCAT) -> MlModel:
        model_1 = self.adjust_dimensions(model_1)
        model_2 = self.adjust_dimensions(model_2)
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        names_map = {}
        layer_names = set()

        def modification(input: ModificationInput):
            names_map[input.config["name"]] = self.fix_layer_name(input.config["name"], input.layer_names, add=False)
            if input.index == n_layers - 2:
                return ModificationOutput(skip=True)
            if merge_version == self.MergeVersion.SELECT:
                if input.parents[0].split("_")[0] == "input":
                    if random.random() < 0.5:
                        return ModificationOutput(skip_branch=True)

        n_layers = len(model_1.model.layers)
        inputs.name = model_1.model.layers[0].name
        tensor_1 = self.get_model_tensor(model_1, inputs, layer_names, modification)
        n_layers = len(model_2.model.layers)
        inputs.name = model_2.model.layers[0].name
        tensor_2 = self.get_model_tensor(model_2, inputs, layer_names, modification)
        tensor = keras.layers.Concatenate(name=self.fix_layer_name("concatenate", layer_names))([tensor_1, tensor_2])
        if merge_version == self.MergeVersion.TRANSFORM:
            tensor = keras.layers.Dense(100, name=self.fix_layer_name("dense", layer_names))(tensor)
        tensor = keras.layers.Dense(self.n_outputs, name=self.fix_layer_name("dense", layer_names))(tensor)
        new_model = keras.Model(inputs=inputs, outputs=tensor)
        self.compile_model(new_model)
        for l in new_model.layers[1:-2]:
            if not l.get_weights():
                continue
            from_layer = None
            for l_1 in model_1.model.layers[1:]:
                if l.name == names_map[l_1.name]:
                    from_layer = l_1
                    break
            if from_layer is None:
                for l_2 in model_2.model.layers[1:]:
                    if l.name == names_map[l_2.name]:
                        from_layer = l_2
                        break
            if from_layer is not None:
                weights = from_layer.get_weights()
                new_weights = self.adjust_weights_shape(weights, np.shape(l.get_weights()[0]))
                l.set_weights(new_weights)
        return MlModel(new_model)
