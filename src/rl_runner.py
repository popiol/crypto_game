from src.config import Config
from src.model_registry import ModelRegistry
from src.evolution_handler import EvolutionHandler
from src.model_serializer import ModelSerializer
from src.agent import Agent
import random


class RlRunner:

    names_file_path = "data/names.csv"

    def run(self):
        model_registry = ModelRegistry(root_path="s3://popiol-crypto-models/models")
        model_serializer = ModelSerializer()
        config = Config(model_registry, model_serializer)
        config.evolution_handler = EvolutionHandler(config)
        with open(self.names_file_path) as f:
            names = f.read().splitlines()
        random.shuffle(names)
        agent = Agent(names[0], config)
        agent.process_quotes()
