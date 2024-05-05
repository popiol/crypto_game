import random
from src.config import Config
from tensorflow import keras


class EvolutionHandler:

    def __init__(self, config: Config) -> None:
        self.model_registry = config.model_registry
        self.model_serializer = config.model_serializer

    def create_model(self) -> keras.Model:
        method = random.randint(0, 2)
        if method == 0:
            return self.create_new_model()
        elif method == 1:
            return self.load_existing_model()
        elif method == 2:
            return self.merge_existing_models()

    def create_new_model(self) -> keras.Model:
        inputs = keras.layers.Input(shape=(seq_len, n_features), name="input")
        l = inputs
        n_filters = round(self.HIDDEN_SIZE / n_features) * n_features
        for index in range(seq_len // 10):
            l = keras.layers.Conv1D(n_filters, 3, activation="relu", name=f"conv1d{index}_{model_id}")(l)
        l = keras.layers.Reshape((round(l.shape[1] * n_filters / n_features), n_features), name=f"reshape_{model_id}")(
            l
        )
        l = keras.layers.Permute((2, 1), name=f"permute1_{model_id}")(l)
        l = keras.layers.Dense(seq_len, name="output")(l)
        l = keras.layers.Permute((2, 1), name=f"permute2_output")(l)
        outputs = l
        model = keras.Model(inputs=inputs, outputs=outputs, name=model_id)
        model.compile(optimizer=optimizers.Nadam(learning_rate=0.001), loss="mean_squared_error")
        return model

    def get_random_model(self) -> keras.Model:
        model_name, serialized_model = self.model_registry.get_random_model()
        if model_name is None:
            return None, None
        model = self.model_serializer.deserialize(serialized_model)
        return model_name, model

    def load_existing_model(self) -> keras.Model:
        model_name, self.model = self.get_random_model()
        if self.model is None:
            return self.create_new_model()
        print("Existing model loaded:", model_name)
        return self.model

    def merge_existing_models(self) -> keras.Model:
        model_name_1, model_1 = self.get_random_model()
        model_name_2, model_2 = self.get_random_model()
        if model_name_1 == model_name_2:
            print("Existing model loaded:", model_name_1)
            return model_1
