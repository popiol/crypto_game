from src.model_registry import ModelRegistry
import random


class EvolutionHandler:

    def __init__(self, model_registry: ModelRegistry) -> None:
        self.model_registry = model_registry

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
        model_name, model_body = self.model_registry.get_random_model()
        if model_body is None:
            return None, None
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
