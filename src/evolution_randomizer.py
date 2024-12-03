import random
from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class EvolutionRandomizer:

    new_basic_model_prob: float
    remove_layer_prob: float
    add_layer_prob: float
    resize_prob: float
    relu_prob: float

    class ModelCreationMethod(Enum):
        EXISTING_MODEL = auto()
        MERGE_MODELS = auto()
        NEW_MODEL = auto()

    def model_creation_method(self):
        if random.random() < self.new_basic_model_prob:
            return self.ModelCreationMethod.NEW_MODEL
        return list(self.ModelCreationMethod)[random.randrange(len(self.ModelCreationMethod) - 1)]

    def remove_layer(self):
        return random.random() < self.remove_layer_prob

    class ResizeAction(Enum):
        SHRINK = auto()
        EXTEND = auto()
        NO_ACTION = auto()

    def resize_layer(self):
        if random.random() > self.resize_prob:
            return self.ResizeAction.NO_ACTION
        return list(self.ResizeAction)[random.randrange(len(self.ResizeAction) - 1)]

    def add_relu(self):
        return random.random() < self.relu_prob

    def remove_relu(self):
        return random.random() < self.relu_prob

    class AddLayerAction(Enum):
        DENSE = auto()
        CONV = auto()
        NO_ACTION = auto()

    def add_layer(self):
        if random.random() > self.add_layer_prob:
            return self.AddLayerAction.NO_ACTION
        return list(self.AddLayerAction)[random.randrange(len(self.AddLayerAction) - 1)]
