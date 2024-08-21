from unittest.mock import patch

from src.environment import Environment
from src.predictor import Predictor


class TestPrediction:

    @patch("src.model_registry.ModelRegistry.set_portfolio")
    @patch("src.model_registry.ModelRegistry.add_transactions")
    def test_predict(self, add_transactions, set_portfolio):
        environment = Environment("config/config.yml")
        predictor = Predictor(environment)
        predictor.predict()
        portfolio = set_portfolio.call_args.args[0]
        transactions = add_transactions.call_args.args[0]
        print("portfolio", portfolio)
        print("transactions", transactions)
