from datetime import datetime

import numpy as np
import pandas as pd


class AggregatedMetrics:

    def __init__(self, all_metrics: list[dict]):
        self.df = self.get_metrics_as_dataframe(all_metrics)

    @staticmethod
    def get_metrics_as_dataframe(all_metrics: list[dict]):
        df = pd.concat([pd.DataFrame([m.values()], columns=m.keys()) for m in all_metrics])
        df = df.replace({None: np.nan})
        return df

    @staticmethod
    def stats(x: list[float]) -> dict:
        return {
            "min": np.nanmin(x).tolist(),
            "max": np.nanmax(x).tolist(),
            "mean": np.nanmean(x).tolist(),
            "sum": np.nansum(x).tolist(),
        }

    def get_metrics(self):
        df = self.df
        aggregated = {"n_models": len(df)}
        for col in df.columns:
            col_type = df[col][~df[col].isna()].iloc[0].__class__.__name__
            if col_type.startswith("int") or col_type.startswith("float"):
                aggregated[col] = self.stats(df[col].tolist())
        for n_layers_per_type in df["n_layers_per_type"]:
            for key, val in n_layers_per_type.items():
                aggregated_key = "n_layers_" + key
                aggregated[aggregated_key] = aggregated.get(aggregated_key, [])
                aggregated[aggregated_key].append(val)
        if "mutations" in df:
            for mutations in df["mutations"]:
                if type(mutations) == dict:
                    for key, val in mutations.items():
                        aggregated_key = "n_mutations_" + key
                        aggregated[aggregated_key] = aggregated.get(aggregated_key, [])
                        aggregated[aggregated_key].append(val)
        if "merge_version" in df:
            for merge_version in df["merge_version"]:
                if type(merge_version) == dict:
                    for key, val in merge_version.items():
                        aggregated_key = "merge_" + key.lower()
                        aggregated[aggregated_key] = aggregated.get(aggregated_key, [])
                        aggregated[aggregated_key].append(val)
        for key, val in aggregated.items():
            if type(val) == list:
                aggregated[key] = self.stats(aggregated[key])
        aggregated["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return aggregated
