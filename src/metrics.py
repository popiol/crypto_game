import psutil

from src.agent import Agent
from src.data_transformer import QuotesSnapshot
from src.portfolio import ClosedTransaction


class Metrics:

    def __init__(self, agent: Agent, quotes: QuotesSnapshot = None, transactions: list[ClosedTransaction] = None):
        self.agent = agent
        self.quotes = quotes
        self.transactions = transactions
        self.model = agent.training_strategy.model
        self.metrics = agent.metrics

    def set_evaluation_score(self, score: float):
        self.metrics["evaluation_score"] = score

    def set_bitcoin_quote(self, BTCUSD: float):
        self.metrics["BTCUSD"] = BTCUSD

    def set_bitcoin_change(self, BTCUSD_change: float):
        self.metrics["BTCUSD_change"] = BTCUSD_change

    def set_n_transactions(self, n_transactions: int):
        self.metrics["n_transactions"] = n_transactions

    def get_bitcoin_change(self) -> float:
        price_1 = self.metrics.get("BTCUSD")
        price_2 = self.get_bitcoin_quote()
        if price_1 is None or price_2 is None:
            return None
        return price_2 / price_1 - 1

    def get_n_merge_ancestors(self) -> int:
        return len(
            set.union(
                *[set([x for x in l.name.split("_")[1:] if len(x) == self.agent.model_id_len]) for l in self.model.get_layers()]
            )
        )

    def get_n_params(self) -> int:
        return int(self.model.get_n_params())

    def get_n_layers(self) -> int:
        return len(self.model.get_layers())

    def get_n_layers_per_type(self) -> dict[str, int]:
        counts = {}
        for l in self.model.get_layers():
            counts[l.layer_type] = counts.get(l.layer_type, 0) + 1
        return counts

    @staticmethod
    def parents_as_list(parents: dict) -> list[str]:
        parents_list = []
        stack = [parents]
        while stack:
            parents = stack.pop()
            if parents is None:
                continue
            parents_list.extend(parents.keys())
            stack.extend(parents.values())
        return parents_list

    def get_n_ancestors(self) -> int:
        return len(self.parents_as_list(self.metrics.get("parents")))

    def get_n_trainings(self) -> int:
        n_trainings = self.metrics.get("n_trainings", 0)
        stats = self.metrics.get("reward_stats", self.agent.training_strategy.stats)
        reward_stats_count = stats["count"] if stats else 0
        return n_trainings + reward_stats_count

    def get_n_mutations(self) -> int:
        return sum(self.metrics.get("mutations", {}).values())

    def get_trained_ratio(self) -> float:
        return self.get_n_trainings() / (
            self.get_n_params() + self.get_n_mutations() * 50000 + self.get_n_merge_ancestors() * 50000
        )

    def get_training_strategy(self):
        strategy = self.agent.training_strategy.__class__.__name__
        return self.metrics.get("training_strategy") if strategy == "TrainingStrategy" else strategy

    def get_shared_input_stats(self):
        return self.agent.data_transformer.get_shared_input_stats() or self.metrics.get("shared_input_stats")

    def get_agent_input_stats(self):
        return self.agent.data_transformer.get_agent_input_stats() or self.metrics.get("agent_input_stats")

    def get_output_stats(self):
        return self.agent.data_transformer.get_output_stats() or self.metrics.get("output_stats")

    def get_weight_stats(self):
        return self.agent.training_strategy.model.get_weight_stats()

    def get_available_memory(self):
        return psutil.virtual_memory().available

    def get_metrics(self):
        return {
            "reward_stats": self.agent.training_strategy.stats,
            **self.metrics,
            "n_ancestors": self.get_n_ancestors(),
            "n_merge_ancestors": self.get_n_merge_ancestors(),
            "n_params": self.get_n_params(),
            "n_layers": self.get_n_layers(),
            "n_layers_per_type": self.get_n_layers_per_type(),
            "n_mutations": self.get_n_mutations(),
            "n_trainings": self.get_n_trainings(),
            "trained_ratio": self.get_trained_ratio(),
            "training_strategy": self.get_training_strategy(),
            "shared_input_stats": self.get_shared_input_stats(),
            "agent_input_stats": self.get_agent_input_stats(),
            "output_stats": self.get_output_stats(),
            "weight_stats": self.get_weight_stats(),
            "available_memory": self.get_available_memory(),
        }
