from dataclasses import dataclass
from src.model_registry import ModelRegistry
from src.evolution_handler import EvolutionHandler
from src.model_serializer import ModelSerializer


@dataclass
class Config:
    model_registry: ModelRegistry = None
    model_serializer: ModelSerializer = None
    evolution_handler: EvolutionHandler = None
