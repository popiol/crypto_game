from unittest.mock import patch

from src.environment import Environment
from src.predictor import Predictor


class TestPrediction:

    @patch("src.model_registry.ModelRegistry.set_leader_portfolio")
    @patch("src.model_registry.ModelRegistry.set_leader_memory")
    @patch("src.model_registry.ModelRegistry.add_transactions")
    def test_predict(self, add_transactions, set_leader_memory, set_leader_portfolio):
        environment = Environment("config/config.yml")
        predictor = Predictor(environment)
        predictor.predict()
        portfolio = set_leader_portfolio.call_args.args[0]
        transactions = add_transactions.call_args.args[0]
        print("portfolio", portfolio)
        print("transactions", transactions)
