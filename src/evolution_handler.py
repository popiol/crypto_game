import random
from src.config import Config


class EvolutionHandler:

    def __init__(self, config: Config) -> None:
        self.model_registry = config.model_registry
        self.model_serializer = config.model_serializer

    def create_model(self):
        method = random.randint(0, 2)
        if method == 0:
            return self.create_new_model()
        elif method == 1:
            return self.load_existing_model()
        elif method == 2:
            return self.merge_existing_models()

    def create_new_model(self):
        pass

    def get_random_model(self):
        model_name, serialized_model = self.model_registry.get_random_model()
        if model_name is None:
            return None, None
        model = self.model_serializer.deserialize(serialized_model)
        return model_name, model

    def load_existing_model(self):
        model_name, self.model = self.get_random_model()
        if self.model is None:
            self.model = self.create_new_model()
        print("Load existing model", model_name)
        return self.model

    def merge_existing_models(self):
        model_name_1, model_1 = self.get_random_model()
        model_name_2, model_2 = self.get_random_model()
        if model_name_1 == model_name_2:
            print("Load existing model", model_name_1)
            return model_1
