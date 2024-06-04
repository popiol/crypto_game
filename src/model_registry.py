import os
import random
import re

from src.s3_utils import S3Utils


class ModelRegistry:

    def __init__(self, remote_path: str, maturity_min_days: int, maturity_min_stats_count: int, max_mature_models: int):
        self.s3_utils = S3Utils(remote_path)
        self.maturity_min_days = maturity_min_days
        self.maturity_min_stats_count = maturity_min_stats_count
        self.max_mature_models = max_mature_models
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
        model_name = re.sub("^" + self.current_prefix + "/", "", key)
        return model_name, self.s3_utils.download_bytes(key)

    def archive_old_models(self):
        files = self.s3_utils.list_files(self.metrics_prefix + "/", self.maturity_min_days)
        models = []
        for file in files:
            metrics = self.s3_utils.download_json(file)
            stats = metrics["reward_stats"]
            print(file.split("/")[-1], stats["count"], stats["mean"], stats["std"])
            if stats["count"] < self.   :
                continue
            models.append((file.split("/")[-1], stats["mean"] - stats["std"]))
        if len(models) > self.max_mature_models:
            to_archive = sorted(models, key=lambda x: x[1])[: len(models) - self.max_mature_models]
            for model, _ in to_archive:
                print("archive", model)
                self.s3_utils.move_file(f"{self.current_prefix}/{model}", f"{self.archived_prefix}/{model}")
                self.s3_utils.move_file(f"{self.metrics_prefix}/{model}", f"{self.archived_prefix}/{model}.json")
