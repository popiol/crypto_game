from dataclasses import dataclass

import yaml

from src.agent_builder import AgentBuilder
from src.data_registry import DataRegistry
from src.data_transformer import DataTransformer
from src.evolution_handler import EvolutionHandler
from src.model_registry import ModelRegistry
from src.model_serializer import ModelSerializer


@dataclass
class Config:
    model_registry: ModelRegistry = None
    data_registry: DataRegistry = None
    agent_builder: AgentBuilder = None
    data_transformer: DataTransformer = None
    model_serializer: ModelSerializer = None
    evolution_handler: EvolutionHandler = None

    @staticmethod
    def from_yaml_file(file_path: str):
        with open(file_path) as f:
            raw_config = yaml.load(f, Loader=yaml.FullLoader)
        config = Config()
        config.model_registry = ModelRegistry(**raw_config["model_registry"])
        config.data_registry = DataRegistry(**raw_config["data_registry"])
        config.agent_builder = AgentBuilder(**raw_config["agent_builder"])
        config.data_transformer = DataTransformer(**raw_config["data_transformer"])
        config.model_serializer = ModelSerializer()
        config.evolution_handler = EvolutionHandler()
        return config
