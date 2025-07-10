import random
from dataclasses import dataclass
from enum import Enum, auto
from src.model_builder import ModelBuilder

@dataclass
class EvolutionRandomizer:
    new_basic_model_prob: float
    merge_prob: float
    remove_layer_prob: float
    add_layer_prob: float
    resize_prob: float
    relu_prob: float
    reuse_prob: float
    select_weight: float
    multiply_weight: float
    dot_weight: float
    transform_weight: float
    concat_weight: float


    class ModelCreationMethod(Enum):
        NEW_MODEL = auto()
        EXISTING_MODEL = auto()
        MERGE_MODELS = auto()

    def model_creation_method(self):
        if random.random() < self.new_basic_model_prob:
            return self.ModelCreationMethod.NEW_MODEL
        if random.random() < self.merge_prob:
            return self.ModelCreationMethod.MERGE_MODELS
        return self.ModelCreationMethod.EXISTING_MODEL

    def asset_dependent(self):
        return bool(random.randint(0, 1))

    def remove_layer(self):
        return random.random() < self.remove_layer_prob

    class ResizeAction(Enum):
        NO_ACTION = auto()
        SHRINK = auto()
        EXTEND = auto()

    def resize_layer(self):
        if random.random() > self.resize_prob:
            return self.ResizeAction.NO_ACTION
        return random.choice(list(self.ResizeAction)[1:])

    def add_relu(self):
        return random.random() < self.relu_prob

    def remove_relu(self):
        return random.random() < self.relu_prob * 0.5

    class AddLayerAction(Enum):
        NO_ACTION = auto()
        DENSE = auto()
        CONV = auto()
        DROPOUT = auto()

    def add_layer(self):
        if random.random() > self.add_layer_prob:
            return self.AddLayerAction.NO_ACTION
        return random.choice(list(self.AddLayerAction)[1:])

    def reuse_layer(self):
        return random.random() < self.reuse_prob

    def model_version(self, versions: type[ModelBuilder.ModelVersion]):
        return random.choice(list(versions))

    def merge_version(self, versions: type[ModelBuilder.MergeVersion]):
        random.choices(
            [versions.SELECT, versions.MULTIPLY, versions.DOT, versions.TRANSFORM, versions.CONCAT], 
            [self.select_weight, self.multiply_weight, self.dot_weight, self.transform_weight, self.concat_weight],
        )[0]
