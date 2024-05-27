import uuid
from datetime import datetime
from socket import gethostname

from src.evolution_handler import EvolutionHandler
from src.model_registry import ModelRegistry
from src.model_serializer import ModelSerializer
from src.portfolio import Portfolio, PortfolioAction


class Agent:

    def __init__(
        self,
        agent_name: str,
        model_registry: ModelRegistry,
        model_serializer: ModelSerializer,
        evolution_handler: EvolutionHandler,
    ):
        self.agent_name = agent_name
        self.model_id = uuid.uuid4().hex[:5]
        model_dt = datetime.now().strftime("%Y%m%d")
        host_name = gethostname()
        self.model_name = f"{agent_name}_{host_name}_{model_dt}_{self.model_id}"
        self.model_registry = model_registry
        self.model_serializer = model_serializer
        self.model = evolution_handler.create_model(len(self.ModelInputs), len(self.ModelOutputs))
        self.metrics = {}

    def process_quotes(self, quotes: dict, portfolio: Portfolio) -> list[PortfolioAction]:
        pass

    def train(self):
        pass

    def save_model(self):
        serialized_model = self.model_serializer.serialize(self.model)
        self.model_registry.save_model(self.model_name, serialized_model, self.metrics)
