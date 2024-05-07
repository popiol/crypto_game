from enum import Enum, auto
from portfolio import PortfolioAction, Portfolio
from src.config import Config
import uuid


class Agent:

    class ModelInputs(Enum):
        ask_price = auto()
        ask_whole_lot_volume = auto()
        ask_lot_volume = auto()
        bid_price = auto()
        bid_whole_lot_volume = auto()
        bid_lot_volume = auto()
        closing_price = auto()
        closing_lot_volume = auto()
        volume_today = auto()
        volume_24h = auto()
        volume_weighted_price_today = auto()
        volume_weighted_price_24h = auto()
        n_trades_today = auto()
        n_trades_24h = auto()
        low_today = auto()
        low_24h = auto()
        high_today = auto()
        high_24h = auto()
        opening_price = auto()

    class ModelOutputs(Enum):
        score = auto()
        buy_price = auto()
        sell_price = auto()
        
    def __init__(self, agent_name: str, config: Config):
        self.agent_name = agent_name
        self.memory_length = config.memory_length
        self.model_id = uuid.uuid4().hex[:5]
        self.model_name = f"{agent_name}_{self.model_id}"
        self.model_registry = config.model_registry
        self.model_serializer = config.model_serializer
        self.model = config.evolution_handler.create_model(len(self.ModelInputs), len(self.ModelOutputs))
        self.metrics = {}

    def process_quotes(self, quotes: dict, portfolio: Portfolio) -> list[PortfolioAction]:
        pass

    def train(self):
        pass

    def save_model(self):
        serialized_model = self.model_serializer.serialize(self.model)
        self.model_registry.save_model(self.model_name, serialized_model, self.metrics)
