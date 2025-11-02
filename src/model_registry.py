import glob
import os
import random
import re
from datetime import datetime, timedelta
from functools import cached_property

import numpy as np
import pandas as pd

from src.s3_utils import S3Utils


class ModelRegistry:
    def __init__(
        self,
        remote_path: str,
        maturity_levels_day: list[int],
        max_models_per_level: int,
        archive_retention_days: int,
        local_path: str,
        local_retention_days: int,
    ):
        self.s3_utils = S3Utils(remote_path)
        self.maturity_levels_day = maturity_levels_day
        self.max_models_per_level = max_models_per_level
        self.archive_retention_days = archive_retention_days
        self.local_path = local_path
        self.local_retention_days = local_retention_days
        self.current_prefix = os.path.join(self.s3_utils.path, "models")
        self.archived_prefix = os.path.join(self.s3_utils.path, "archived")
        self.metrics_prefix = os.path.join(self.s3_utils.path, "metrics")
        self.aggregated_prefix = os.path.join(self.s3_utils.path, "aggregated_metrics")
        self.leader_prefix = os.path.join(self.s3_utils.path, "leader")
        self.baseline_prefix = os.path.join(self.s3_utils.path, "baseline")
        self.reports_prefix = os.path.join(self.s3_utils.path, "reports")
        self.real_prefix = os.path.join(self.s3_utils.path, "real")

    def save_model(self, model_name: str, serialized_model: bytes, metrics: dict):
        self.s3_utils.upload_bytes(f"{self.current_prefix}/{model_name}", serialized_model)
        self.s3_utils.upload_json(f"{self.metrics_prefix}/{model_name}", metrics)

    def get_random_model(self) -> tuple[str, bytes] | tuple[None, None]:
        keys = list(self.s3_utils.list_files(self.current_prefix + "/"))
        if not keys:
            return None, None
        key = random.choice(keys)
        model_name = key.split("/")[-1]
        return model_name, self.get_model(model_name)

    def get_model(self, model_name: str) -> bytes:
        os.makedirs(self.local_path, exist_ok=True)
        local_path = f"{self.local_path}/{model_name}"
        if not os.path.exists(local_path):
            self.s3_utils.download_file(f"{self.current_prefix}/{model_name}", local_path)
        with open(local_path, "rb") as f:
            return f.read()

    def iterate_models(self):
        for files in self.models_by_level:
            for file in files:
                model_name = file.split("/")[-1]
                yield model_name, self.get_model(model_name)

    def get_metrics(self, model_name: str) -> dict:
        return self.s3_utils.download_json(f"{self.metrics_prefix}/{model_name}")

    def set_metrics(self, model_name: str, metrics: dict):
        self.s3_utils.upload_json(f"{self.metrics_prefix}/{model_name}", metrics)

    @cached_property
    def models_by_level(self) -> list[list[str]]:
        models = self.s3_utils.list_files(self.current_prefix + "/")
        maturity_days = []
        for model in models:
            create_dt = model.split("/")[-1].split("_")[1]
            create_dt = datetime.strptime(create_dt, "%Y%m%d%H%M%S")
            maturity_days.append((datetime.now() - create_dt).days)
        result = []
        for start, end in zip([None] + self.maturity_levels_day, self.maturity_levels_day + [None]):
            level_models = []
            for model, days in zip(models, maturity_days):
                if (start is None or days >= start) and (end is None or days < end):
                    level_models.append(model)
            result.append(level_models)
        return result

    def archive_models(self, scores: dict):
        self.show_new_mature_models()
        self.archive_weak_models(scores)
        self.archive_weak_models(scores)
        self.clean_archive()
        self.clean_local_cache()

    def show_new_mature_models(self):
        younger = None
        for level, models in enumerate(self.models_by_level):
            older = set(models)
            if younger is not None:
                files = older & younger
                for file in files:
                    model_name = file.split("/")[-1]
                    print(f"new mature level {level + 1}", model_name)
            younger = older

    def archive_model(self, model_name: str):
        self.s3_utils.move_file(f"{self.current_prefix}/{model_name}", f"{self.archived_prefix}/{model_name}")
        self.s3_utils.move_file(f"{self.metrics_prefix}/{model_name}", f"{self.archived_prefix}/{model_name}.json")

    def get_weak_models(self, scores: dict, maturity_level: int, files: list[str]) -> list[tuple[str, str]]:
        df = pd.DataFrame(columns=["model", "score"])
        max_models = self.max_models_per_level
        models = []
        to_archive = []
        for file in files:
            model_name = file.split("/")[-1]
            try:
                score = scores[model_name]
                df.loc[len(df)] = [model_name, score]
                if np.isnan(score) or score == 0:
                    to_archive.append((model_name, "inactive"))
                else:
                    models.append((model_name, score))
            except Exception as ex:
                print("error", model_name, ex)
                if maturity_level > 0:
                    to_archive.append((model_name, "invalid"))
        if len(models) > max_models:
            weak_models = sorted(models, key=lambda x: x[1])[:-max_models]
            to_archive.extend([(model, "weak") for model, _ in weak_models])
        print(maturity_level, "level models")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", None):
            print(df.sort_values("score", ascending=False).reset_index())
        return to_archive

    def archive_weak_models(self, scores: dict):
        for maturity_level, files in enumerate(self.models_by_level):
            models = self.get_weak_models(scores, maturity_level, files)
            for model, reason in models:
                print("archive", reason, "level", maturity_level, model)
                self.archive_model(model)

    def clean_archive(self):
        files = self.s3_utils.list_files(self.archived_prefix + "/", self.archive_retention_days * 24)
        for file in files:
            self.s3_utils.delete_file(file)

    def restore_archived_model(self, model_name: str):
        self.s3_utils.move_file(f"{self.archived_prefix}/{model_name}", f"{self.current_prefix}/{model_name}")
        self.s3_utils.move_file(f"{self.archived_prefix}/{model_name}.json", f"{self.metrics_prefix}/{model_name}")

    def restore_all_archived_models(self):
        for file in self.s3_utils.list_files(self.archived_prefix + "/"):
            if file.endswith(".json"):
                continue
            model_name = file.split("/")[-1]
            self.restore_archived_model(model_name)
            print("restored", model_name)

    def clean_local_cache(self):
        for file in glob.iglob(self.local_path + "/*"):
            local_mtime = datetime.fromtimestamp(os.path.getmtime(file))
            if datetime.now() - local_mtime > timedelta(days=self.local_retention_days):
                os.remove(file)

    def set_aggregated_metrics(self, metrics: dict):
        file_name = re.sub("[^0-9]", "", metrics["datetime"])[:10] + ".json"
        self.s3_utils.upload_json(f"{self.aggregated_prefix}/{file_name}", metrics)

    def download_aggregated_metrics(self, local_path: str):
        self.s3_utils.sync(f"s3://{self.s3_utils.bucket_name}/{self.aggregated_prefix}/", local_path)

    # leader

    def set_leader(self, model_name: str):
        self.s3_utils.copy_file(f"{self.current_prefix}/{model_name}", f"{self.leader_prefix}/model")
        self.s3_utils.copy_file(f"{self.metrics_prefix}/{model_name}", f"{self.leader_prefix}/metrics.json")

    def get_leader(self):
        model = self.s3_utils.download_bytes(f"{self.leader_prefix}/model")
        metrics = self.s3_utils.download_json(f"{self.leader_prefix}/metrics.json")
        create_dt = self.s3_utils.get_last_modification_time(f"{self.leader_prefix}/model")
        return model, metrics, create_dt

    def get_leader_metrics(self):
        return self.s3_utils.download_json(f"{self.leader_prefix}/metrics.json")

    def set_leader_metrics(self, metrics: dict):
        self.s3_utils.upload_json(f"{self.leader_prefix}/metrics.json", metrics)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.s3_utils.upload_json(f"{self.leader_prefix}/history/metrics_{timestamp}.json", metrics)
        self.s3_utils.upload_json(f"{self.real_prefix}/history/metrics_{timestamp}.json", metrics)

    def get_leader_portfolio(self):
        return self.s3_utils.download_json(f"{self.leader_prefix}/portfolio.json")

    def download_leader_portfolio(self, file_path: str, transactions_path: str):
        self.s3_utils.download_file(f"{self.leader_prefix}/portfolio.json", file_path, only_updated=True)
        self.s3_utils.sync(f"s3://{self.s3_utils.bucket_name}/{self.leader_prefix}/transactions/", transactions_path)

    def set_leader_portfolio(self, portfolio: dict):
        self.s3_utils.upload_json(f"{self.leader_prefix}/portfolio.json", portfolio)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.s3_utils.upload_json(f"{self.leader_prefix}/history/portfolio_{timestamp}.json", portfolio)

    def get_leader_memory(self):
        return self.s3_utils.download_bytes(f"{self.leader_prefix}/memory.pickle")

    def set_leader_memory(self, memory: bytes):
        self.s3_utils.upload_bytes(f"{self.leader_prefix}/memory.pickle", memory)

    def add_transactions(self, transactions: list[dict], copy_to: str):
        if transactions:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.s3_utils.upload_json(f"{self.leader_prefix}/transactions/{timestamp}.json", transactions)
        self.s3_utils.sync(f"s3://{self.s3_utils.bucket_name}/{self.leader_prefix}/transactions/", copy_to)

    def download_leader_history(self, local_path: str):
        self.s3_utils.sync(f"s3://{self.s3_utils.bucket_name}/{self.leader_prefix}/history/", local_path)

    # baseline

    def get_baseline_metrics(self):
        return self.s3_utils.download_json(f"{self.baseline_prefix}/metrics.json")

    def set_baseline_metrics(self, metrics: dict):
        self.s3_utils.upload_json(f"{self.baseline_prefix}/metrics.json", metrics)

    def get_baseline_portfolio(self):
        return self.s3_utils.download_json(f"{self.baseline_prefix}/portfolio.json")

    def set_baseline_portfolio(self, portfolio: dict):
        self.s3_utils.upload_json(f"{self.baseline_prefix}/portfolio.json", portfolio)

    def add_baseline_transactions(self, transactions: list[dict], copy_to: str):
        if transactions:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.s3_utils.upload_json(f"{self.baseline_prefix}/transactions/{timestamp}.json", transactions)
        self.s3_utils.sync(f"s3://{self.s3_utils.bucket_name}/{self.baseline_prefix}/transactions/", copy_to)

    def upload_report(self, file_path: str):
        basename = os.path.basename(file_path)
        self.s3_utils.upload_file(file_path, f"{self.reports_prefix}/{basename}")

    def download_report(self, file_path: str):
        basename = os.path.basename(file_path)
        self.s3_utils.download_file(f"{self.reports_prefix}/{basename}", file_path, only_updated=True)

    # real portfolio

    def get_real_portfolio(self):
        return self.s3_utils.download_json(f"{self.real_prefix}/portfolio.json")

    def get_real_portfolio_last_update(self):
        return self.s3_utils.get_last_modification_time(f"{self.real_prefix}/portfolio.json")

    def download_real_portfolio(self, file_path: str, transactions_path: str):
        self.s3_utils.download_file(f"{self.real_prefix}/portfolio.json", file_path, only_updated=True)
        self.s3_utils.sync(f"s3://{self.s3_utils.bucket_name}/{self.real_prefix}/transactions/", transactions_path)

    def set_real_portfolio(self, portfolio: dict):
        self.s3_utils.upload_json(f"{self.real_prefix}/portfolio.json", portfolio)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.s3_utils.upload_json(f"{self.real_prefix}/history/portfolio_{timestamp}.json", portfolio)

    def get_real_memory(self):
        return self.s3_utils.download_bytes(f"{self.real_prefix}/memory.pickle")

    def set_real_memory(self, memory: bytes):
        self.s3_utils.upload_bytes(f"{self.real_prefix}/memory.pickle", memory)

    def add_real_transactions(self, transactions: list[dict], copy_to: str):
        if transactions:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            self.s3_utils.upload_json(f"{self.real_prefix}/transactions/{timestamp}.json", transactions)
        self.s3_utils.sync(f"s3://{self.s3_utils.bucket_name}/{self.real_prefix}/transactions/", copy_to)

    def download_real_history(self, local_path: str):
        self.s3_utils.sync(f"s3://{self.s3_utils.bucket_name}/{self.real_prefix}/history/", local_path)
