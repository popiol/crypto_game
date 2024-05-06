from dataclasses import dataclass
from src.model_registry import ModelRegistry
from src.evolution_handler import EvolutionHandler
from src.model_serializer import ModelSerializer
import yaml


@dataclass
class Config:
    model_registry: ModelRegistry = None
    model_serializer: ModelSerializer = None
    evolution_handler: EvolutionHandler = None
    names: str
    memory_length: int
    
    @staticmethod
    def from_yaml_file(file_path: str):
        with open(file_path) as f:
            raw_config = yaml.load(f, Loader=yaml.FullLoader)
        config = Config(**raw_config)
        config.model_registry = ModelRegistry(**raw_config["model_registry"])
        config.model_serializer = ModelSerializer()
        config.evolution_handler = EvolutionHandler(config)
        return config
