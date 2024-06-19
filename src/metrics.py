from src.agent import Agent


class Metrics:

    def __init__(self, agent: Agent, metrics: dict = None):
        self.agent = agent
        self.metrics = metrics or {}

    def set_evaluation_score(self, score: float):
        self.metrics["evaluation_score"] = score

    def get_metrics(self):
        if "model_id" not in self.metrics:
            self.metrics["model_id"] = self.agent.model_id
        if "reward_stats" not in self.metrics:
            self.metrics["reward_stats"] = self.agent.training_strategy.stats
        return self.metrics
