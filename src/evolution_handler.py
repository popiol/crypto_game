import random
from dataclasses import dataclass

from src.ml_model import MlModel
from src.model_builder import ModelBuilder
from src.model_registry import ModelRegistry
from src.model_serializer import ModelSerializer


@dataclass
class EvolutionHandler:

    model_registry: ModelRegistry
    model_serializer: ModelSerializer
    model_builder: ModelBuilder
    remove_layer_prob: float
    add_layer_prob: float
    shrink_prob: float
    extend_prob: float
    resize_by: int
    max_n_params: int

    def create_model(self) -> MlModel:
        method = random.randint(0, 2)
        if method == 0:
            model = self.create_new_model()
        elif method == 1:
            model = self.load_existing_model()
        elif method == 2:
            model = self.merge_existing_models()
        model = self.model_builder.adjust_dimensions(model)
        model = self.mutate(model)
        return model

    def create_new_model(self) -> MlModel:
        asset_dependant = bool(random.randint(0, 1))
        print("create model, asset_dependant:", asset_dependant)
        return self.model_builder.build_model(asset_dependant)

    def load_existing_model(self) -> MlModel:
        model_name, serialized_model = self.model_registry.get_random_model()
        if model_name is None:
            return self.create_new_model()
        model = self.model_serializer.deserialize(serialized_model)
        print("Existing model loaded:", model_name)
        return model

    def merge_existing_models(self) -> MlModel:
        model_name_1, serialized_model_1 = self.model_registry.get_random_model()
        if model_name_1 is None:
            return self.create_new_model()
        model_1 = self.model_serializer.deserialize(serialized_model_1)
        model_name_2, serialized_model_2 = self.model_registry.get_random_model()
        model_2 = self.model_serializer.deserialize(serialized_model_2)
        if model_name_1 == model_name_2 or model_1.get_n_params() + model_2.get_n_params() > self.max_n_params:
            print("Existing model loaded:", model_name_1)
            return model_1
        print("Merging models:", model_name_1, "and", model_name_2)
        return self.model_builder.merge_models(model_1, model_2)

    def mutate(self, model: MlModel) -> MlModel:
        skip = 0
        n_layers = len(model.get_layers())
        n_layers_diff = 0
        for index, layer in enumerate(model.get_layers()[:-1]):
            if skip > 0:
                skip -= 1
                continue
            if random.random() < self.remove_layer_prob:
                offset = abs(round(random.gauss(0, 2)))
                offset = min(offset, n_layers - index - 2)
                prev_n_layers = len(model.get_layers())
                model = self.model_builder.remove_layer(model, index + n_layers_diff, index + offset + n_layers_diff)
                n_layers_diff += len(model.get_layers()) - prev_n_layers
                skip = offset
                continue
            resize_by = abs(round(random.gauss(self.resize_by - 1, self.resize_by))) + 1
            if random.random() < self.shrink_prob and layer.shape and layer.shape[1] >= 2 * resize_by:
                model = self.model_builder.resize_layer(model, index, layer.shape[1] - resize_by)
            elif random.random() < self.extend_prob and layer.shape:
                model = self.model_builder.resize_layer(model, index, layer.shape[1] + resize_by)
            if random.random() < self.add_layer_prob:
                choice = random.randint(0, 1)
                prev_n_layers = len(model.get_layers())
                if choice == 0:
                    size = max(10, round(random.gauss(100, 30)))
                    model = self.model_builder.add_dense_layer(model, index, size)
                elif choice == 1:
                    model = self.model_builder.add_conv_layer(model, index)
                n_layers_diff += len(model.get_layers()) - prev_n_layers
        return model
