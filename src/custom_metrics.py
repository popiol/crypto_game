import numpy as np

from src.aggregated_metrics import AggregatedMetrics


class CustomMetrics:

    def __init__(self, aggregated: AggregatedMetrics = None, all_metrics: list[dict] = None):
        self.df = aggregated.df if aggregated else AggregatedMetrics.get_metrics_as_dataframe(all_metrics)

    def parents_as_list(self, parents: dict) -> list[str]:
        parents_list = []
        stack = [parents]
        while stack:
            parents = stack.pop()
            if parents is None:
                continue
            parents_list.extend(parents.keys())
            stack.extend(parents.values())
        return parents_list

    def get_metrics(self):
        df = self.df
        metrics = {}
        asset_dependent_score = df[df.n_asset_dependent == 1].evaluation_score.mean()
        asset_independent_score = df[df.n_asset_independent == 1].evaluation_score.mean()
        metrics["asset_dependent_score"] = {0: asset_independent_score, 1: asset_dependent_score}
        for col in ["n_merge_ancestors", "n_transactions"]:
            metrics[col + "_score"] = df[[col, "evaluation_score"]].groupby(col)["evaluation_score"].mean().to_dict()
        parents_score = {}
        for index, (parents, score) in df[["parents", "evaluation_score"]].iterrows():
            if type(parents) != dict:
                continue
            for parent in self.parents_as_list(parents):
                parents_score[parent] = parents_score.get(parent, [])
                parents_score[parent].append(score)
        for key, val in parents_score.items():
            parents_score[key] = np.nanmean(val)
        metrics["parents_score"] = parents_score
        return metrics
