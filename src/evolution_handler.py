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

    def create_model(self) -> MlModel:
        method = random.randint(0, 1)  # TODO: random.randint(0, 2)
        if method == 0:
            return self.create_new_model()
        elif method == 1:
            return self.load_existing_model()
        elif method == 2:
            return self.merge_existing_models()

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
