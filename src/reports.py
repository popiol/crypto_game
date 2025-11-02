import glob
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from src.aggregated_metrics import AggregatedMetrics
from src.custom_metrics import CustomMetrics
from src.model_registry import ModelRegistry


@dataclass
class Reports:
    model_registry: ModelRegistry
    aggregated_path: str
    quick_stats_path: str
    change_in_time_path: str
    custom_metrics_path: str
    portfolio_path: str
    transactions_path: str
    leader_history_path: str
    leader_stats_path: str
    baseline_portfolio_path: str
    baseline_transactions_path: str
    real_portfolio_path: str
    real_transactions_path: str

    def get_leader_portfolio_value(self) -> float | None:
        portfolio = self.model_registry.get_leader_portfolio()
        return portfolio.get("value")

    def get_real_portfolio_value(self) -> float | None:
        portfolio = self.model_registry.get_real_portfolio()
        return portfolio.get("value")

    def get_baseline_portfolio_value(self) -> float | None:
        portfolio = self.model_registry.get_baseline_portfolio()
        return portfolio.get("value")

    def aggregate_metrics(self, all_metrics: list[dict]):
        if not all_metrics:
            return {}
        aggregated = AggregatedMetrics(all_metrics)
        aggregated_dict = aggregated.get_metrics()
        aggregated_dict["leader_value"] = self.get_leader_portfolio_value()
        aggregated_dict["real_portfolio_value"] = self.get_real_portfolio_value()
        aggregated_dict["baseline_value"] = self.get_baseline_portfolio_value()
        custom = CustomMetrics(aggregated.df, aggregated_dict)
        return {**aggregated_dict, "custom": custom.get_metrics()}

    def get_quick_stats(self, model_names: list[str], all_metrics: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(
            columns=[
                "model",
                "score",
                "market",
                "n_params",
                "n_layers",
                "n_ancestors",
                "training_strategy",
                "n_transactions",
            ]
        )
        for model_name, metrics in zip(model_names, all_metrics):
            df.loc[len(df)] = [
                model_name,
                metrics["evaluation_score"],
                metrics["BTCUSD_change"],
                metrics["n_params"],
                metrics["n_layers"],
                metrics["n_ancestors"],
                metrics["training_strategy"],
                metrics["n_transactions"],
            ]
        return df

    def copy_custom_metrics(self, files: list[str]):
        if not files:
            return
        with open(max(files)) as f:
            aggregated = json.load(f)
        with open(self.custom_metrics_path, "w") as f:
            json.dump(aggregated["custom"], f)

    def calc_change_in_time(self, files: list[str]):
        df = pd.DataFrame()
        if os.path.exists(self.change_in_time_path):
            df = pd.read_csv(self.change_in_time_path)
            df = df[df.datetime > (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")]
            last_dt = df.datetime.max()
            last_dt = re.sub("[^0-9]", "", last_dt)[:10]
            files = [file for file in files if file.split("/")[-1][:10] > last_dt]
        dfs = [df]
        for file in sorted(files):
            with open(file) as f:
                aggregated = json.load(f)
            values = {}
            for key, val in aggregated.items():
                if key in ["custom"]:
                    continue
                if type(val) == dict:
                    for key2, val2 in val.items():
                        if key2 == "sum":
                            continue
                        values[f"{key}_{key2}"] = [val2]
                else:
                    values[key] = [val]
            dfs.append(pd.DataFrame.from_dict(values))
        return pd.concat(dfs)

    def calc_leader_stats(self):
        # prepare portfolio history
        self.model_registry.download_real_history(self.leader_history_path)
        files = glob.glob(self.leader_history_path + "/*.json")
        df = pd.DataFrame()
        if os.path.exists(self.leader_stats_path):
            df = pd.read_csv(self.leader_stats_path)
            start_dt = int((datetime.now() - timedelta(days=90)).strftime("%Y%m%d%H"))
            df = df[df.datetime > start_dt]
            last_dt = str(df.datetime.max())
            files = [file for file in files if file.split("/")[-1].split("_")[1][:10] > last_dt]
        dfs = [df]
        all_data = {}
        for file in sorted(files):
            file_name = file.split("/")[-1]
            data_type = file_name.split("_")[0]
            timestamp = file_name.split("_")[1][:10]
            with open(file) as f:
                all_data[data_type] = all_data.get(data_type, {})
                all_data[data_type][timestamp] = json.load(f)
        if len(all_data) < 2:
            return df
        min_len = min(len(all_data["portfolio"]), len(all_data["metrics"]))
        portfolio_timestamps = list(all_data["portfolio"].keys())[-min_len:]
        metrics_timestamps = list(all_data["metrics"].keys())[-min_len:]

        # prepare transactions
        self.model_registry.download_real_portfolio(self.real_portfolio_path, self.real_transactions_path)
        files = sorted(glob.glob(self.real_transactions_path + "/*.json"))
        transactions = {}
        for file in files:
            timestamp = file.split("/")[-1].split(".")[0]
            with open(file) as f:
                transactions[timestamp] = json.load(f)

        for portfolio_timestamp, metrics_timestamp in zip(portfolio_timestamps, metrics_timestamps):
            timestamp = max(portfolio_timestamp, metrics_timestamp)
            stats = {"datetime": timestamp}
            metrics = all_data["metrics"][metrics_timestamp]
            for key, value in metrics.items():
                if type(value) in [str, int, float] and key not in [
                    "BTCUSD",
                    "BTCUSD_change",
                    "evaluation_score",
                    "available_memory",
                ]:
                    stats[key] = [value]
            portfolio = all_data["portfolio"][portfolio_timestamp]
            stats["n_open_positions"] = [len(portfolio["positions"])]
            stats["n_orders"] = [len(portfolio["orders"])]
            from_timestamp = datetime.strftime(datetime.strptime(timestamp, "%Y%m%d%H") - timedelta(days=1), "%Y%m%d%H")
            stats["n_closed_transactions"] = [len([key for key in transactions if from_timestamp < key[:10] <= timestamp])]
            dfs.append(pd.DataFrame.from_dict(stats))
        return pd.concat(dfs)

    def run(self, model_names: list[str], all_metrics: list[dict]):
        aggregated = self.aggregate_metrics(all_metrics)
        self.model_registry.set_aggregated_metrics(aggregated)
        stats = self.get_quick_stats(model_names, all_metrics)
        stats.to_csv(self.quick_stats_path, index=False)
        self.model_registry.download_aggregated_metrics(self.aggregated_path)
        files = glob.glob(self.aggregated_path + "/*.json")
        self.copy_custom_metrics(files)
        df = self.calc_change_in_time(files)
        df.to_csv(self.change_in_time_path, index=False)
        df = self.calc_leader_stats()
        if len(df) > 0:
            df.to_csv(self.leader_stats_path, index=False)
        self.upload_reports()

    def upload_reports(self):
        self.model_registry.upload_report(self.quick_stats_path)
        self.model_registry.upload_report(self.change_in_time_path)
        if os.path.exists(self.leader_history_path):
            self.model_registry.upload_report(self.leader_history_path)
        self.model_registry.upload_report(self.leader_stats_path)
        self.model_registry.upload_report(self.custom_metrics_path)
