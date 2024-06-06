import uuid
from dataclasses import dataclass

from src.keras import keras
from src.ml_model import MlModel


@dataclass
class ModelBuilder:

    n_steps: int
    n_assets: int
    n_features: int
    n_outputs: int

    def build_model(self) -> MlModel:
        inputs = keras.layers.Input(shape=(self.n_steps, self.n_assets, self.n_features), name="input")
        l = inputs
        l = keras.layers.Permute((1, 3, 2))(l)
        l = keras.layers.Dense(100)(l)
        l = keras.layers.Dense(self.n_assets)(l)
        l = keras.layers.Permute((3, 1, 2))(l)
        l = keras.layers.Reshape((self.n_assets, self.n_steps * self.n_features))(l)
        l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(100)(l)
        l = keras.layers.Dense(self.n_outputs)(l)
        outputs = l
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")
        return MlModel(model)
    
    def remove_layer(self, model: MlModel, layer_index: int) -> MlModel:
        pass

    def add_dense_layer(self, model: MlModel, layer_index: int, size: int):
        pass

    def add_conv_layer(self, model: MlModel, layer_index: int):
        pass
    
    def resize_layer(self, model: MlModel, layer_index: int, new_size: int):
        pass

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
