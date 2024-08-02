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

    def load_config(self, file_path: str):
        with open(file_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def get_model_registry(self):
        return ModelRegistry(**self.config["model_registry"])

    def get_data_registry(self):
        return DataRegistry(**self.config["data_registry"])

    def get_data_transformer(self):
        return DataTransformer(**self.config["data_transformer"])

    def get_model_serializer(self):
        return ModelSerializer()

    def get_trainset(self):
        if self.eval_mode:
            return None
        return Trainset(**self.config["trainset"])

    def get_training_time_hours(self) -> int:
        return self.config["rl_runner"]["training_time_hours"]

    def get_model_builder(self, data_transformer: DataTransformer, n_assets: int):
        return ModelBuilder(data_transformer.memory_length, n_assets, data_transformer.n_features, data_transformer.n_outputs)

    def get_agent_builder(
        self,
        model_registry: ModelRegistry,
        model_serializer: ModelSerializer,
        model_builder: ModelBuilder,
        data_transformer: DataTransformer,
        trainset: Trainset,
    ):
        evolution_handler = EvolutionHandler(model_registry, model_serializer, model_builder, **self.config["evolution_handler"])
        return AgentBuilder(evolution_handler, data_transformer, trainset, **self.config["agent_builder"])

    def get_portfolio_managers(self, n_managers: int):
        return [PortfolioManager(**self.config["portfolio_manager"]) for _ in range(n_managers)]
