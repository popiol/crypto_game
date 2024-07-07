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
    relu_prob: float
    resize_by: int
    max_n_params: int

    def create_model(self) -> tuple[MlModel, dict]:
        method = random.randint(0, 2)
        if method == 0:
            model, metrics = self.create_new_model()
        elif method == 1:
            model, metrics = self.load_existing_model()
        elif method == 2:
            model, metrics = self.merge_existing_models()
        model = self.model_builder.adjust_dimensions(model)
        model, metrics = self.mutate(model, metrics)
        return model, metrics

    def create_new_model(self) -> tuple[MlModel, dict]:
        asset_dependent = bool(random.randint(0, 1))
        print("create model, asset_dependent:", asset_dependent)
        metrics = {"n_asset_dependent": int(asset_dependent), "n_asset_independent": int(not asset_dependent)}
        return self.model_builder.build_model(asset_dependent), metrics

    def load_spicific_model(self, model_name: str, model: MlModel = None, serialized_model: bytes = None) -> tuple[MlModel, dict]:
        if model is None:
            if serialized_model is None:
                serialized_model = self.model_registry.get_model(model_name)
            model = self.model_serializer.deserialize(serialized_model)
        metrics = self.model_registry.get_metrics(model_name)
        metrics["parents"] = {model_name: metrics.get("parents")}
        print("Existing model loaded:", model_name)
        return model, metrics

    def load_existing_model(self) -> tuple[MlModel, dict]:
        model_name, serialized_model = self.model_registry.get_random_model()
        if model_name is None:
            return self.create_new_model()
        return self.load_spicific_model(model_name, serialized_model=serialized_model)

    def merge_existing_models(self) -> tuple[MlModel, dict]:
        model_name_1, serialized_model_1 = self.model_registry.get_random_model()
        if model_name_1 is None:
            return self.create_new_model()
        model_1 = self.model_serializer.deserialize(serialized_model_1)
        model_name_2, serialized_model_2 = self.model_registry.get_random_model()
        model_2 = self.model_serializer.deserialize(serialized_model_2)
        if model_name_1 == model_name_2 or model_1.get_n_params() + model_2.get_n_params() > self.max_n_params:
            return self.load_spicific_model(model_name_1, model_1)
        metrics_1 = self.model_registry.get_metrics(model_name_1)
        metrics_2 = self.model_registry.get_metrics(model_name_2)
        metrics = {"parents": {model_name_1: metrics_1.get("parents"), model_name_2: metrics_2.get("parents")}}
        for metric_name in metrics_1:
            if type(metrics_1[metric_name]) == int and type(metrics_2.get(metric_name)) == int:
                metrics[metric_name] = metrics_1[metric_name] + metrics_2[metric_name]
        print("Merging models:", model_name_1, "and", model_name_2)
        return self.model_builder.merge_models(model_1, model_2), metrics

    def mutate(self, model: MlModel, metrics: dict) -> tuple[MlModel, dict]:
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
                metrics["remove_layer"] = metrics.get("remove_layer", 0) + 1
                continue
            resize_by = abs(round(random.gauss(self.resize_by - 1, self.resize_by))) + 1
            if random.random() < self.shrink_prob and layer.shape and layer.shape[1] >= 2 * resize_by:
                model = self.model_builder.resize_layer(model, index, layer.shape[1] - resize_by)
                metrics["shrink_layer"] = metrics.get("shrink_layer", 0) + 1
            elif random.random() < self.extend_prob and layer.shape:
                model = self.model_builder.resize_layer(model, index, layer.shape[1] + resize_by)
                metrics["extend_layer"] = metrics.get("extend_layer", 0) + 1
            if random.random() < self.relu_prob and layer.shape:
                model = self.model_builder.add_relu(model, index)
            if random.random() < self.add_layer_prob:
                choice = random.randint(0, 1)
                prev_n_layers = len(model.get_layers())
                if choice == 0:
                    size = max(10, round(random.gauss(100, 30)))
                    model = self.model_builder.add_dense_layer(model, index, size)
                    metrics["add_dense_layer"] = metrics.get("add_dense_layer", 0) + 1
                elif choice == 1:
                    model = self.model_builder.add_conv_layer(model, index)
                    metrics["add_conv_layer"] = metrics.get("add_conv_layer", 0) + 1
                n_layers_diff += len(model.get_layers()) - prev_n_layers
        return model, metrics
