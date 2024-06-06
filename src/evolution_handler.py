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

    def create_model(self) -> MlModel:
        method = random.randint(0, 1)  # TODO: random.randint(0, 2)
        if method == 0:
            model = self.create_new_model()
        elif method == 1:
            model = self.load_existing_model()
        elif method == 2:
            model = self.merge_existing_models()
        return self.mutate(model)

    def create_new_model(self) -> MlModel:
        return self.model_builder.build_model()

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
        if model_name_1 == model_name_2:
            print("Existing model loaded:", model_name_1)
            return model_1
        model_2 = self.model_serializer.deserialize(serialized_model_2)
        print("Merging models:", model_name_1, "and", model_name_2)
        return self.model_builder.merge_models(model_1, model_2)
    
    def adjust_n_assets(self, model: MlModel):
        pass

    def mutate(self, model: MlModel) -> MlModel:
        return model
        for index, layer in enumerate(model.get_layers()):
            if random.random() < self.remove_layer_prob:
                model = self.model_builder.remove_layer(model, index)
                continue
            if random.random() < self.shrink_prob and layer.shape[1] >= 2 * self.resize_by:
                model = self.model_builder.resize_layer(model, index, layer.shape[1] - self.resize_by)
            elif random.random() < self.extend_prob:
                model = self.model_builder.resize_layer(model, index, layer.shape[1] + self.resize_by)
            if random.random() < self.add_layer_prob:
                choice = random.randint(0, 1)
                if choice == 0:
                    model = self.model_builder.add_dense_layer(model, index, 100)
                elif choice == 1:
                    model = self.model_builder.add_conv_layer(model, index)
        return model
