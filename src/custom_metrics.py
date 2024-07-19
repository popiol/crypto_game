import numpy as np
import pandas as pd


class CustomMetrics:

    def __init__(self, df: pd.DataFrame, aggregated: dict):
        self.df = df
        self.aggregated = aggregated

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
        asset_dependent_score = df[df.n_asset_dependent > 0].evaluation_score.mean()
        asset_dependent_score = 0 if np.isnan(asset_dependent_score) else asset_dependent_score.tolist()
        asset_independent_score = df[df.n_asset_independent > 0].evaluation_score.mean()
        asset_independent_score = 0 if np.isnan(asset_independent_score) else asset_independent_score.tolist()
        metrics["asset_dependent_score"] = {0: asset_independent_score, 1: asset_dependent_score}
        for col in [
            "n_merge_ancestors",
            "n_layers",
            "n_ancestors",
            "add_conv_layer",
            "add_dense_layer",
            "remove_layer",
            "extend_layer",
            "shrink_layer",
            "n_conv1d",
            "n_conv2d",
            "n_permute",
            "n_dense",
            "n_reshape",
            "n_unit",
            "n_concatenate",
            "trained_ratio",
            "BTCUSD_change",
            "n_trainings",
            "n_transactions",
            "n_params",
        ]:
            if col not in self.aggregated:
                continue
            min_val = self.aggregated[col]["min"]
            max_val = self.aggregated[col]["max"]
            n_buckets = 10
            if max_val - min_val < 10:
                n_buckets = round(max_val - min_val)
            if col not in df:
                df[col] = df["n_layers_per_type"].apply(lambda x: x.get(col[2:], 0))
            if n_buckets == 0:
                continue
            df["grouping"] = df[col].apply(
                lambda x: (
                    round((x - min_val) / (max_val - min_val) * n_buckets) / n_buckets * (max_val - min_val) + min_val
                    if not np.isnan(x)
                    else np.nan
                )
            )
            metrics[col + "_score"] = (
                df[["grouping", "evaluation_score"]].groupby("grouping")["evaluation_score"].mean().to_dict()
            )
        parents_score = {}
        if "parents" in df:
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
