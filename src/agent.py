import uuid
from datetime import datetime
from socket import gethostname

import numpy as np

from ml_model import MlModel
from src.portfolio import Portfolio, PortfolioAction


class Agent:

    def __init__(self, agent_name: str, model: MlModel):
        self.agent_name = agent_name
        self.model_id = uuid.uuid4().hex[:5]
        model_dt = datetime.now().strftime("%Y%m%d")
        host_name = gethostname()
        self.model_name = f"{agent_name}_{host_name}_{model_dt}_{self.model_id}"
        self.model = model
        self.metrics = {}

    def process_quotes(self, features: np.array, portfolio: Portfolio) -> list[PortfolioAction]:
        pass
