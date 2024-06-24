from src.agent import Agent


class Metrics:

    def __init__(self, agent: Agent, metrics: dict = None):
        self.agent = agent
        self.model = agent.training_strategy.model
        self.metrics = metrics or {}

    def set_evaluation_score(self, score: float):
        self.metrics["evaluation_score"] = score

    def get_n_ancestors(self) -> int:
        return len([1 for x in self.model.get_layers()[0].name.split("_")[1:] if len(x) >= 4])

    def get_metrics(self):
        return {
            "model_id": self.agent.model_id,
            "reward_stats": self.agent.training_strategy.stats,
            **self.metrics,
            "n_ancestors": self.get_n_ancestors(),
        }
