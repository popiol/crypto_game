import uuid
from dataclasses import dataclass

import numpy as np

from src.keras import keras
from src.ml_model import MlModel


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

    def adjust_weights_shape(self, weights: list[np.array], input_size: int, output_size: int) -> list[np.array]:
        new_weights = []
        if not weights:
            return new_weights
        assert len(weights) <= 2
        assert input_size > 0
        assert output_size > 0
        w0 = self.adjust_array_shape(weights[0], 0, input_size)
        w0 = self.adjust_array_shape(w0, 1, output_size)
        new_weights.append(w0)
        if len(weights) > 1:
            w1 = self.adjust_array_shape(weights[1], 0, output_size)
            new_weights.append(w1)
        return new_weights

    def copy_weights(self, from_model: keras.Model, to_model: keras.Model, skip_start: int = None, skip_end: int = None):
        if skip_start is not None:
            skip_end = skip_start if skip_end is None else skip_end
            assert skip_start <= skip_end
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
            new_weights = self.adjust_weights_shape(weights, *np.shape(l.get_weights()[0]))
            l.set_weights(new_weights)

    def fix_reshape(self, config: dict, input_shape: tuple[int]):
        if config["name"].split("_")[0] == "reshape":
            target_n_dim = len(config["target_shape"])
            assert 0 < target_n_dim <= len(input_shape)
            output_shape = input_shape[: target_n_dim - 1] + tuple([np.prod(input_shape[target_n_dim - 1 :])])
            config["target_shape"] = output_shape

    def adjust_n_assets(self, model: MlModel) -> MlModel:
        n_assets = model.model.layers[0].batch_shape[2]
        assert self.n_assets >= n_assets
        if self.n_assets == n_assets:
            return model
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        tensor = inputs
        layer_names = []
        for l in model.model.layers[1:]:
            config = l.get_config()
            layer_names.append(config["name"].split("_")[0])
            self.fix_reshape(config, tensor.shape[1:])
            if layer_names[-3:] == ["permute", "dense", "dense"] and config["units"] == n_assets:
                config["units"] = self.n_assets
            new_layer = l.from_config(config)
            tensor = new_layer(tensor)
        new_model = keras.Model(inputs=inputs, outputs=tensor)
        self.compile_model(new_model)
        self.copy_weights(model.model, new_model)
        return MlModel(new_model)

    def remove_layer(self, model: MlModel, start_index: int, end_index: int) -> MlModel:
        assert 0 <= start_index <= end_index < len(model.model.layers) - 2
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        tensor = inputs
        for index, l in enumerate(model.model.layers[1:]):
            if start_index <= index <= end_index:
                continue
            config = l.get_config()
            self.fix_reshape(config, tensor.shape[1:])
            new_layer = l.from_config(config)
            tensor = new_layer(tensor)
        if tensor.shape != model.model.output_shape:
            return model
        new_model = keras.Model(inputs=inputs, outputs=tensor)
        self.compile_model(new_model)
        self.copy_weights(model.model, new_model, start_index, end_index)
        return MlModel(new_model)

    def add_dense_layer(self, model: MlModel, before_index: int, size: int):
        assert 0 <= before_index <= len(model.model.layers) - 1
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features))
        tensor = inputs
        for index, l in enumerate(model.model.layers[1:]):
            if index == before_index:
                tensor = keras.layers.Dense(size)(tensor)
            config = l.get_config()
            self.fix_reshape(config, tensor.shape[1:])
            new_layer = l.from_config(config)
            tensor = new_layer(tensor)
        if tensor.shape != model.model.output_shape:
            return model
        new_model = keras.Model(inputs=inputs, outputs=tensor)
        self.compile_model(new_model)
        self.copy_weights(model.model, new_model, before_index)
        return MlModel(new_model)

    def add_conv_layer(self, model: MlModel, layer_index: int):
        return model

    def resize_layer(self, model: MlModel, layer_index: int, new_size: int):
        return model

    def merge_models(self, model_1: MlModel, model_2: MlModel) -> MlModel:
        inputs, hidden1, outputs1 = self.copy_structure(model_1.model)
        model_id_1 = hidden1.name.split("/")[0].split("_")[-1]
        if len(hidden1.shape) > 2:
            hidden1 = keras.layers.Flatten(name=f"flatten_{model_id_1}")(hidden1)
        hidden1 = keras.layers.Dense(50, name=f"dense_{model_id_1}")(hidden1)
        inputs, hidden2, outputs2 = self.copy_structure(model_2.model, inputs)
        model_id_2 = hidden2.name.split("/")[0].split("_")[-1]
        if len(hidden2.shape) > 2:
            hidden2 = keras.layers.Flatten(name=f"flatten_{model_id_2}")(hidden2)
        hidden2 = keras.layers.Dense(50, name=f"dense_{model_id_2}")(hidden2)
        model_id_3 = self.new_model_id(model_id_1, model_id_2)
        l = keras.layers.Concatenate(name=f"concatenate_{model_id_1}_{model_id_2}_{model_id_3}")([hidden1, hidden2])
        l = keras.layers.Dense(self.OUTPUT_SIZE, name="output")(l)
        outputs = l
        model = keras.Model(inputs=inputs, outputs=outputs, name=f"{model_id_1}_{model_id_2}")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error",
        )
        return MlModel(model)

    def copy_structure(self, model, inputs=None, new_output=False):
        self.stacks = {}
        self.model_id = uuid.uuid4().hex[:5]
        for layer in model.layers:
            if layer.name.startswith("input"):
                if inputs is None:
                    inputs = keras.layers.Input(shape=layer.output_shape[0][1:], name="input")
            elif layer.name.startswith("reshape"):
                self.add_layer(inputs, layer, keras.layers.Reshape, layer.output_shape[1:])
            elif layer.name.startswith("permute"):
                if not new_output:
                    l = self.add_layer(inputs, layer, keras.layers.Permute, layer.dims)
                    outputs = l
            elif layer.name.startswith("conv1d"):
                l = self.add_layer(
                    inputs,
                    layer,
                    keras.layers.Conv1D,
                    layer.output_shape[2],
                    layer.kernel_size,
                )
                hidden = l
            elif layer.name.startswith("flatten"):
                l = self.add_layer(inputs, layer, keras.layers.Flatten)
                hidden = l
            elif layer.name.startswith("concatenate"):
                l = self.add_layer(inputs, layer, keras.layers.Concatenate)
                hidden = l
            elif layer.name.startswith("dense"):
                self.add_layer(inputs, layer, keras.layers.Dense, layer.output_shape[-1])
            elif layer.name == "output":
                if new_output:
                    l = keras.layers.Flatten()(hidden)
                    hidden = l
                    l = keras.layers.Dense(self.OUTPUT_SIZE, name="output")(l)
                    outputs = l
                else:
                    l = self.add_layer(
                        inputs,
                        layer,
                        keras.layers.Dense,
                        layer.output_shape[-1],
                        name="output",
                    )
                    outputs = l
        return inputs, hidden, outputs

    def add_layer(self, inputs, old_layer, new_layer, *args, **kwargs):
        model_id = old_layer.name.split("_")[-1]
        l = self.stacks.get(model_id, inputs)
        if model_id == "output":
            l = list(self.stacks.values())[0]
            self.stacks = {}
        elif old_layer.name.startswith("concatenate"):
            model_ids = old_layer.name.split("_")[1:3]
            l = [self.stacks[x] for x in model_ids]
            for model_id2 in model_ids:
                del self.stacks[model_id2]
            kwargs["name"] = (
                "concatenate_" + "_".join([self.new_model_id(x) for x in model_ids]) + "_" + self.new_model_id(model_id)
            )
        if "name" not in kwargs:
            kwargs["name"] = old_layer.name.split("_")[0] + "_" + self.new_model_id(model_id)
        l = new_layer(*args, **kwargs, **self.get_initializers(old_layer))(l)
        self.stacks[model_id] = l
        return l
