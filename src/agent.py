import uuid
from datetime import datetime
from socket import gethostname

from portfolio import Portfolio, PortfolioAction
from src.config import Config


class Agent:

    def __init__(self, agent_name: str, config: Config):
        self.agent_name = agent_name
        self.memory_length = config.memory_length
        self.model_id = uuid.uuid4().hex[:5]
        model_dt = datetime.now().strftime("%Y%m%d")
        host_name = gethostname()
        self.model_name = f"{agent_name}_{host_name}_{model_dt}_{self.model_id}"
        self.model_registry = config.model_registry
        self.model_serializer = config.model_serializer
        self.model = config.evolution_handler.create_model(
            len(self.ModelInputs), len(self.ModelOutputs)
        )
        self.metrics = {}

    def process_quotes(
        self, quotes: dict, portfolio: Portfolio
    ) -> list[PortfolioAction]:
        pass

    def train(self):
        pass

    def save_model(self):
        serialized_model = self.model_serializer.serialize(self.model)
        self.model_registry.save_model(self.model_name, serialized_model, self.metrics)
