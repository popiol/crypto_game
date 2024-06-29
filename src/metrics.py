from src.agent import Agent
from src.data_transformer import QuotesSnapshot


class Metrics:

    def __init__(self, agent: Agent, metrics: dict = None, quotes: QuotesSnapshot = None):
        self.agent = agent
        self.model = agent.training_strategy.model
        self.metrics = metrics or {}
        self.quotes = quotes

    def set_evaluation_score(self, score: float):
        self.metrics["evaluation_score"] = score

    def get_n_merge_ancestors(self) -> int:
        return len(
            set.union(
                *[set([x for x in l.name.split("_")[1:] if len(x) == self.agent.model_id_len]) for l in self.model.get_layers()]
            )
        )

    def get_bitcoin_quote(self):
        if self.quotes is None:
            return None
        return (self.quotes.closing_price("TBTCUSD") + self.quotes.closing_price("WBTCUSD")) / 2

    def get_bitcoin_change(self):
        if "BTCUSD" not in self.metrics:
            return None
        return self.get_bitcoin_quote() / self.metrics["BTCUSD"] - 1

    def get_n_params(self):
        return self.model.get_n_params()

    def get_n_layers(self):
        return len(self.model.get_layers())

    def get_n_layers_per_type(self):
        counts = {}
        for l in self.model.get_layers():
            counts[l.layer_type] = counts.get(l.layer_type, 0) + 1
        return counts

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
        }
