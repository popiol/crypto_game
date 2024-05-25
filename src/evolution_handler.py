import random

from tensorflow import keras


class EvolutionHandler:

    def __init__(self, config: dict) -> None:
        self.model_registry = config["model_registry"]
        self.model_serializer = config["model_serializer"]
        self.memory_length = config["memory_length"]

    def create_model(self, input_dim: int, output_dim: int) -> keras.Model:
        method = random.randint(0, 2)
        if method == 0:
            return self.create_new_model(input_dim, output_dim)
        elif method == 1:
            return self.load_existing_model(input_dim, output_dim)
        elif method == 2:
            return self.merge_existing_models(input_dim, output_dim)

    def create_new_model(self, input_dim: int, output_dim: int) -> keras.Model:
        inputs = keras.layers.Input(shape=(self.memory_length, input_dim), name="input")
        l = inputs
        l = keras.layers.Flatten()(l)
        l = keras.layers.UnitNormalization()(l)
        l = keras.layers.Dense(100)(l)
        l = keras.layers.Dense(output_dim)(l)
        outputs = l
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error",
        )
        return model

    def get_random_model(self) -> keras.Model:
        model_name, serialized_model = self.model_registry.get_random_model()
        if model_name is None:
            return None, None
        model = self.model_serializer.deserialize(serialized_model)
        return model_name, model

    def load_existing_model(self, input_dim: int, output_dim: int) -> keras.Model:
        model_name, self.model = self.get_random_model()
        if self.model is None:
            return self.create_new_model(input_dim, output_dim)
        print("Existing model loaded:", model_name)
        return self.model

    def merge_existing_models(self, input_dim: int, output_dim: int) -> keras.Model:
        model_name_1, model_1 = self.get_random_model()
        model_name_2, model_2 = self.get_random_model()
        if model_name_1 is None:
            return self.create_new_model(input_dim, output_dim)
        if model_name_1 == model_name_2:
            print("Existing model loaded:", model_name_1)
            return model_1
        self.merge_models(model_1, model_2)

    def merge_models(self, model_1, model_2):
        inputs, hidden1, outputs1 = self.copy_structure(model_1)
        model_id_1 = hidden1.name.split("/")[0].split("_")[-1]
        if len(hidden1.shape) > 2:
            hidden1 = keras.layers.Flatten(name=f"flatten_{model_id_1}")(hidden1)
        hidden1 = keras.layers.Dense(50, name=f"dense_{model_id_1}")(hidden1)
        inputs, hidden2, outputs2 = self.copy_structure(model_2, inputs)
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
        return model

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
