import random
from dataclasses import dataclass

from src.evolution_randomizer import EvolutionRandomizer
from src.ml_model import MlModel
from src.model_builder import ModelBuilder
from src.model_registry import ModelRegistry
from src.model_serializer import ModelSerializer


@dataclass
class EvolutionHandler:

    model_registry: ModelRegistry
    model_serializer: ModelSerializer
    model_builder: ModelBuilder
    evolution_randomizer: EvolutionRandomizer
    asset_list: list[str]
    current_assets: set[str]
    resize_by: int
    max_n_params: int

    def create_model(self) -> tuple[MlModel, dict]:
        method = self.evolution_randomizer.model_creation_method()
        if method == self.evolution_randomizer.ModelCreationMethod.NEW_MODEL:
            model, metrics = self.create_new_model()
        elif method == self.evolution_randomizer.ModelCreationMethod.EXISTING_MODEL:
            model, metrics = self.load_existing_model()
        elif method == self.evolution_randomizer.ModelCreationMethod.MERGE_MODELS:
            model, metrics = self.merge_existing_models()
        model = self.model_builder.adjust_dimensions(model)
        model = self.model_builder.filter_assets(model, self.asset_list, self.current_assets)
        model, metrics = self.mutate(model, metrics)
        return model, metrics

    def create_new_model(self) -> tuple[MlModel, dict]:
        version = self.evolution_randomizer.model_version(self.model_builder.ModelVersion)
        asset_dependent = self.evolution_randomizer.asset_dependent()
        print("create model, asset_dependent:", asset_dependent, ", version:", version.name)
        metrics = {
            "n_asset_dependent": int(asset_dependent),
            "n_asset_independent": int(not asset_dependent),
            "model_version": {version.name: 1},
        }
        return self.model_builder.build_model(asset_dependent, version), metrics

    def load_spicific_model(self, model_name: str, model: MlModel = None, serialized_model: bytes = None) -> tuple[MlModel, dict]:
        if model is None:
            if serialized_model is None:
                serialized_model = self.model_registry.get_model(model_name)
            model = self.model_serializer.deserialize(serialized_model)
        metrics = self.model_registry.get_metrics(model_name)
        metrics["parents"] = {model_name: metrics.get("parents")}
        metrics["evaluation_score"] = None
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
        metrics = {}
        metrics["parents"] = {model_name_1: metrics_1.get("parents"), model_name_2: metrics_2.get("parents")}
        for key in ["mutations", "merge_version", "model_version"]:
            values_1 = metrics_1.get(key, {})
            values_2 = metrics_2.get(key, {})
            metrics[key] = {
                key2: values_1.get(key2, 0) + values_2.get(key2, 0) for key2 in set(values_1.keys()).union(set(values_2.keys()))
            }
        for metric_name in metrics_1:
            if type(metrics_1[metric_name]) == int and type(metrics_2.get(metric_name)) == int:
                metrics[metric_name] = metrics_1[metric_name] + metrics_2[metric_name]
        merge_version = random.choice(list(self.model_builder.MergeVersion))
        merge_version_metrics: dict = metrics.get("merge_version", {})
        merge_version_metrics[merge_version.name] = merge_version_metrics.get(merge_version.name, 0) + 1
        metrics["merge_version"] = merge_version_metrics
        print("Merging models:", model_name_1, "and", model_name_2)
        print("Merge type:", merge_version.name)
        return self.model_builder.merge_models(model_1, model_2, merge_version), metrics

    def mutate(self, model: MlModel, metrics: dict) -> tuple[MlModel, dict]:
        skip = 0
        n_layers_diff = 0
        mutations = metrics.get("mutations", {})
        metrics["mutations"] = mutations
        for index, layer in enumerate(model.get_layers()[:-1]):
            if layer.layer_type == "gather":
                continue
            if skip > 0:
                skip -= 1
                continue
            index += n_layers_diff
            if self.evolution_randomizer.remove_layer():
                prev_n_layers = len(model.get_layers())
                offset = abs(round(random.gauss(0, 2)))
                offset = min(offset, prev_n_layers - index - 2)
                model = self.model_builder.remove_layer(model, index, index + offset)
                if not self.model_builder.last_failed:
                    n_removed = prev_n_layers - len(model.get_layers())
                    n_layers_diff -= n_removed
                    skip = offset
                    mutations["remove_layer"] = mutations.get("remove_layer", 0) + n_removed
                    continue
            resize_by = abs(round(random.gauss(self.resize_by - 1, self.resize_by))) + 1
            resize_action = self.evolution_randomizer.resize_layer()
            if resize_action == self.evolution_randomizer.ResizeAction.SHRINK and layer.shape and layer.shape[1] >= 2 * resize_by:
                model = self.model_builder.resize_layer(model, index, layer.shape[1] - resize_by)
                if not self.model_builder.last_failed:
                    mutations["shrink_layer"] = mutations.get("shrink_layer", 0) + 1
            elif resize_action == self.evolution_randomizer.ResizeAction.EXTEND and layer.shape:
                model = self.model_builder.resize_layer(model, index, layer.shape[1] + resize_by)
                if not self.model_builder.last_failed:
                    mutations["extend_layer"] = mutations.get("extend_layer", 0) + 1
            if self.evolution_randomizer.add_relu() and layer.shape:
                model = self.model_builder.add_relu(model, index)
                if not self.model_builder.last_failed:
                    mutations["add_relu"] = mutations.get("add_relu", 0) + 1
            elif self.evolution_randomizer.remove_relu() and layer.shape:
                model = self.model_builder.remove_relu(model, index)
                if not self.model_builder.last_failed:
                    mutations["remove_relu"] = mutations.get("remove_relu", 0) + 1
            add_layer_action = self.evolution_randomizer.add_layer()
            if add_layer_action != self.evolution_randomizer.AddLayerAction.NO_ACTION:
                prev_n_layers = len(model.get_layers())
                if add_layer_action == self.evolution_randomizer.AddLayerAction.DENSE:
                    size = max(10, round(random.gauss(100, 30)))
                    model = self.model_builder.add_dense_layer(model, index, size)
                    if not self.model_builder.last_failed:
                        mutations["add_dense_layer"] = mutations.get("add_dense_layer", 0) + 1
                n_layers_diff += len(model.get_layers()) - prev_n_layers
            if self.evolution_randomizer.reuse_layer() and index < len(model.get_layers()) - 3:
                prev_n_layers = len(model.get_layers())
                model = self.model_builder.reuse_layer(model, index)
                if not self.model_builder.last_failed:
                    mutations["reuse_layer"] = mutations.get("reuse_layer", 0) + 1
                    n_layers_diff += len(model.get_layers()) - prev_n_layers
        return model, metrics
