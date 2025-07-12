import numpy as np
import pandas as pd


class CustomMetrics:

    def __init__(self, df: pd.DataFrame, aggregated: dict):
        self.df = df
        self.aggregated = aggregated

    def get_metrics(self):
        df = self.df
        metrics = {}
        asset_dependent_score = df.n_asset_dependent.sum().tolist()
        asset_independent_score = df.n_asset_independent.sum().tolist()
        metrics["asset_dependency"] = {"independent": asset_independent_score, "dependent": asset_dependent_score}
        merge_stats = {col[len("merge_") :]: self.aggregated[col]["sum"] for col in self.aggregated if col.startswith("merge_")}
        if merge_stats:
            metrics["merge_version"] = merge_stats
        version_stats = {
            col[len("version_") :]: self.aggregated[col]["sum"] for col in self.aggregated if col.startswith("version_")
        }
        if version_stats:
            metrics["model_version"] = version_stats
        metrics["training_strategy_score"] = (
            df[["training_strategy", "evaluation_score"]].groupby("training_strategy")["evaluation_score"].mean().to_dict()
        )
        for col in [
            "model_age",
            "n_merge_ancestors",
            "n_layers",
            "n_ancestors",
            "merge_concat",
            "merge_transform",
            "merge_select",
            "merge_multiply",
            "version_v2ind",
            "version_v2dep",
            "version_v6ind",
            "version_v6dep",
            "version_v5dep",
            "version_v8dep",
            "n_mutations_add_dense_layer",
            "n_mutations_add_conv_layer",
            "n_mutations_remove_layer",
            "n_mutations_extend_layer",
            "n_mutations_shrink_layer",
            "n_mutations_add_relu",
            "n_layers_conv1d",
            "n_layers_conv2d",
            "n_layers_permute",
            "n_layers_dense",
            "n_layers_reshape",
            "n_layers_unit",
            "n_layers_concatenate",
            "n_layers_dropout",
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
            if min_val == max_val:
                continue
            n_buckets = 10
            if type(max_val) == int and max_val - min_val < 10:
                n_buckets = round(max_val - min_val + 1)
            for key, prefix in [
                ("n_layers_per_type", "n_layers_"),
                ("mutations", "n_mutations_"),
                ("merge_version", "merge_"),
                ("model_version", "version_"),
            ]:
                if col not in df and col.startswith(prefix):
                    df[col] = df[key].apply(
                        lambda x: x.get(col[len(prefix) :], x.get(col[len(prefix) :].upper(), 0)) / (sum(x.values()) or 1) if type(x) == dict else 0
                    )
            df["grouping"] = df[col].apply(
                lambda x: (
                    round(round((x - min_val) / (max_val - min_val) * n_buckets) / n_buckets * (max_val - min_val) + min_val, 2)
                    if not np.isnan(x)
                    else np.nan
                )
            )
            metrics[col + "_score"] = (
                df[["grouping", "evaluation_score"]].groupby("grouping")["evaluation_score"].mean().to_dict()
            )
        return metrics
