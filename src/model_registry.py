import os
import random

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
                stats = metrics["reward_stats"]
                score = float(metrics["evaluation_score"])
                model_and_score = (model_name, score)
                if np.isnan(score):
                    to_archive.append(model_and_score)
                else:
                    models.append(model_and_score)
            except:
                to_archive.append((model_name, 0))
        if len(models) > self.max_mature_models:
            to_archive.extend(sorted(models, key=lambda x: x[1])[: -self.max_mature_models])
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
