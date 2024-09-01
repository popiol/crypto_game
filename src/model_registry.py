import os
import random
import re
from datetime import datetime

import numpy as np

from src.s3_utils import S3Utils


class ModelRegistry:

    def __init__(
        self,
        remote_path: str,
        maturity_min_hours: int,
        max_mature_models: int,
        retirement_min_hours: int,
        archive_retention_days: int,
    ):
        self.s3_utils = S3Utils(remote_path)
        self.maturity_min_hours = maturity_min_hours
        self.max_mature_models = max_mature_models
        self.retirement_min_hours = retirement_min_hours
        self.archive_retention_days = archive_retention_days
        self.current_prefix = os.path.join(self.s3_utils.path, "models")
        self.archived_prefix = os.path.join(self.s3_utils.path, "archived")
        self.metrics_prefix = os.path.join(self.s3_utils.path, "metrics")
        self.aggregated_prefix = os.path.join(self.s3_utils.path, "aggregated_metrics")
        self.leader_prefix = os.path.join(self.s3_utils.path, "leader")
        self.baseline_prefix = os.path.join(self.s3_utils.path, "baseline")

    def save_model(self, model_name: str, serialized_model: bytes, metrics: dict):
        self.s3_utils.upload_bytes(f"{self.current_prefix}/{model_name}", serialized_model)
        self.s3_utils.upload_json(f"{self.metrics_prefix}/{model_name}", metrics)

    def get_random_model(self) -> tuple[str, bytes]:
        keys = list(self.s3_utils.list_files(self.current_prefix + "/"))
        if not keys:
            return None, None
        key = random.choice(keys)
        model_name = key.split("/")[-1]
        return model_name, self.s3_utils.download_bytes(key)

    def get_model(self, model_name: str) -> bytes:
        return self.s3_utils.download_bytes(f"{self.current_prefix}/{model_name}")

    def iterate_models(self):
        for key in self.s3_utils.list_files(self.current_prefix + "/"):
            model_name = key.split("/")[-1]
            yield model_name, self.s3_utils.download_bytes(key)

    def get_metrics(self, model_name: str) -> dict:
        return self.s3_utils.download_json(f"{self.metrics_prefix}/{model_name}")

    def set_metrics(self, model_name: str, metrics: dict) -> dict:
        self.s3_utils.upload_json(f"{self.metrics_prefix}/{model_name}", metrics)

    def archive_models(self):
        self.archive_old_models()
        self.archive_weak_models()
        self.clean_archive()

    def archive_old_models(self):
        files = self.s3_utils.list_files(self.current_prefix + "/", self.retirement_min_hours)
        for file in files:
            model = file.split("/")[-1]
            print("archive", model)
            self.s3_utils.move_file(f"{self.current_prefix}/{model}", f"{self.archived_prefix}/{model}")
            self.s3_utils.move_file(f"{self.metrics_prefix}/{model}", f"{self.archived_prefix}/{model}.json")

    def get_weak_models(self) -> list[str]:
        files = self.s3_utils.list_files(self.current_prefix + "/", self.maturity_min_hours)
        models = []
        to_archive = []
        for file in files:
            model_name = file.split("/")[-1]
            metrics = self.s3_utils.download_json(f"{self.metrics_prefix}/{model_name}")
            try:
                score = float(metrics["evaluation_score"])
                model_and_score = (model_name, score)
                if np.isnan(score) or score == 0:
                    to_archive.append(model_and_score)
                else:
                    models.append(model_and_score)
            except:
                to_archive.append((model_name, 0))
        if len(models) > self.max_mature_models:
            weak_models = sorted(models, key=lambda x: x[1])[: -self.max_mature_models]
            to_archive.extend(weak_models)
        return [model for model, _ in to_archive]

    def archive_weak_models(self):
        models = self.get_weak_models()
        for model in models:
            print("archive", model)
            self.s3_utils.move_file(f"{self.current_prefix}/{model}", f"{self.archived_prefix}/{model}")
            self.s3_utils.move_file(f"{self.metrics_prefix}/{model}", f"{self.archived_prefix}/{model}.json")

    def clean_archive(self):
        files = self.s3_utils.list_files(self.archived_prefix + "/", self.archive_retention_days * 24)
        for file in files:
            self.s3_utils.delete_file(file)

    def set_aggregated_metrics(self, metrics: dict):
        file_name = re.sub("[^0-9]", "", metrics["datetime"])[:10] + ".json"
        self.s3_utils.upload_json(f"{self.aggregated_prefix}/{file_name}", metrics)

    def download_aggregated_metrics(self, local_path: str):
        self.s3_utils.sync(f"s3://{self.s3_utils.bucket_name}/{self.aggregated_prefix}/", local_path)

    def set_leader(self, model_name: str):
        self.s3_utils.copy_file(f"{self.current_prefix}/{model_name}", f"{self.leader_prefix}/model")
        self.s3_utils.copy_file(f"{self.metrics_prefix}/{model_name}", f"{self.leader_prefix}/metrics.json")

    def get_leader(self):
        model = self.s3_utils.download_bytes(f"{self.leader_prefix}/model")
        metrics = self.s3_utils.download_json(f"{self.leader_prefix}/metrics.json")
        return model, metrics

    def get_leader_metrics(self):
        return self.s3_utils.download_json(f"{self.leader_prefix}/metrics.json")

    def set_leader_metrics(self, metrics: dict):
        self.s3_utils.upload_json(f"{self.leader_prefix}/metrics.json", metrics)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.s3_utils.upload_json(f"{self.leader_prefix}/history/metrics_{timestamp}.json", metrics)

    def get_leader_portfolio(self):
        return self.s3_utils.download_json(f"{self.leader_prefix}/portfolio.json")

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
