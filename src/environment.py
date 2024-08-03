from functools import cached_property

import yaml

from src.agent_builder import AgentBuilder
from src.data_registry import DataRegistry
from src.data_transformer import DataTransformer
from src.evolution_handler import EvolutionHandler
from src.model_builder import ModelBuilder
from src.model_registry import ModelRegistry
from src.model_serializer import ModelSerializer
from src.portfolio_manager import PortfolioManager
from src.trainset import Trainset


class Environment:

    def __init__(self, config_path: str = None):
        self.config = {}
        if config_path:
            self.load_config(config_path)
        self.eval_mode = False
        self.n_assets: int = None

    def load_config(self, file_path: str):
        with open(file_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    @cached_property
    def model_registry(self):
        return ModelRegistry(**self.config["model_registry"])

    @cached_property
    def data_registry(self):
        return DataRegistry(**self.config["data_registry"])

    @cached_property
    def data_transformer(self):
        return DataTransformer(**self.config["data_transformer"])

    @cached_property
    def model_serializer(self):
        return ModelSerializer()

    @cached_property
    def trainset(self):
        if self.eval_mode:
            return None
        return Trainset(**self.config["trainset"])

    @cached_property
    def training_time_hours(self) -> int:
        return self.config["rl_runner"]["training_time_hours"]

    @cached_property
    def model_builder(self):
        return ModelBuilder(
            self.data_transformer.memory_length, self.n_assets, self.data_transformer.n_features, self.data_transformer.n_outputs
        )

    @cached_property
    def agent_builder(self):
        evolution_handler = EvolutionHandler(
            self.model_registry, self.model_serializer, self.model_builder, **self.config["evolution_handler"]
        )
        return AgentBuilder(evolution_handler, self.data_transformer, self.trainset, **self.config["agent_builder"])

    def get_portfolio_managers(self, n_managers: int):
        return [PortfolioManager(**self.config["portfolio_manager"]) for _ in range(n_managers)]
