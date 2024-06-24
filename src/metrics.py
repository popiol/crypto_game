from src.agent import Agent


class Metrics:

    def __init__(self, agent: Agent, metrics: dict = None):
        self.agent = agent
        self.metrics = metrics or {}

    def set_evaluation_score(self, score: float):
        self.metrics["evaluation_score"] = score

    def get_metrics(self):
        return {"model_id": self.agent.model_id, "reward_stats": self.agent.training_strategy.stats, **self.metrics}
