from src.agent import Agent
from src.data_transformer import QuotesSnapshot


class Metrics:

    def __init__(self, agent: Agent, quotes: QuotesSnapshot = None):
        self.agent = agent
        self.quotes = quotes
        self.model = agent.training_strategy.model
        self.metrics = agent.metrics

    def set_evaluation_score(self, score: float):
        self.metrics["evaluation_score"] = score

    def get_n_merge_ancestors(self) -> int:
        return len(
            set.union(
                *[set([x for x in l.name.split("_")[1:] if len(x) == self.agent.model_id_len]) for l in self.model.get_layers()]
            )
        )

    def get_bitcoin_quote(self) -> float:
        if self.quotes is None:
            return None
        return (self.quotes.closing_price("TBTCUSD") + self.quotes.closing_price("WBTCUSD")) / 2

    def get_bitcoin_change(self) -> float:
        if "BTCUSD" not in self.metrics:
            return None
        return self.get_bitcoin_quote() / self.metrics["BTCUSD"] - 1

    def get_n_params(self) -> int:
        return int(self.model.get_n_params())

    def get_n_layers(self) -> int:
        return len(self.model.get_layers())

    def get_n_layers_per_type(self) -> dict[str, int]:
        counts = {}
        for l in self.model.get_layers():
            counts[l.layer_type] = counts.get(l.layer_type, 0) + 1
        return counts

    def get_n_ancestors(self) -> int:
        parents = self.metrics
        n_ancestors = -1
        while parents is not None:
            n_ancestors += 1
            parents = parents.get("parents")
        return n_ancestors

    def get_n_trainings(self) -> int:
        reward_stats = self.metrics.get("reward_stats", self.agent.training_strategy.stats)
        if not reward_stats:
            return 0
        return reward_stats["count"]

    def get_trained_ratio(self) -> float:
        return self.get_n_trainings() / self.get_n_params()

    def get_metrics(self):
        return {
            "model_id": self.agent.model_id,
            "reward_stats": self.agent.training_strategy.stats,
            **self.metrics,
            "n_merge_ancestors": self.get_n_merge_ancestors(),
            "BTCUSD": self.get_bitcoin_quote(),
            "BTCUSD_change": self.get_bitcoin_change(),
            "n_params": self.get_n_params(),
            "n_layers": self.get_n_layers(),
            "n_layers_per_type": self.get_n_layers_per_type(),
            "n_ancestors": self.get_n_ancestors(),
            "n_trainings": self.get_n_trainings(),
            "trained_ratio": self.get_trained_ratio(),
        }
