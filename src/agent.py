from portfolio import PortfolioAction, Portfolio
from src.config import Config
import uuid


class Agent:

    def __init__(self, agent_name: str, config: Config):
        self.agent_name = agent_name
        self.model_id = uuid.uuid4().hex[:5]
        self.model_name = f"{agent_name}_{self.model_id}"
        self.model_registry = config.model_registry
        self.model = config.evolution_handler.create_model()

    def process_quotes(self, quotes: dict, portfolio: Portfolio) -> list[PortfolioAction]:
        pass

    def train(self):
        pass

    def save_model(self):
        self.model_registry.save_model(self.model_name, body)
