import json
import os
import random
import re

from src.s3_utils import S3Utils


class ModelRegistry:

    def __init__(self, remote_path: str):
        self.remote_path = remote_path
        assert remote_path.startswith("s3://")
        self.bucket_name = remote_path.split("/")[2]
        self.prefix = os.path.normpath("/".join(remote_path.split("/")[3:]))
        self.current_prefix = self.prefix + "/models"
        self.archived_prefix = self.prefix + "/archived"
        self.metrics_prefix = self.prefix + "/metrics"
        self.s3_utils = S3Utils(remote_path)

    def save_model(self, model_name: str, serialized_model: bytes, metrics: dict):
        self.s3_utils.upload_bytes(f"{self.current_prefix}/{model_name}", serialized_model)
        self.s3_utils.upload_bytes(f"{self.metrics_prefix}/{model_name}", json.dumps(metrics))

    def get_random_model(self) -> tuple[str, bytes]:
        keys = self.s3_utils.list_files(self.current_prefix + "/")
        if not keys:
            return None, None
        key = random.choice(keys)
        model_name = re.sub("^" + self.current_prefix + "/", "", key)
        return model_name, self.s3_utils.download_bytes(key)

    def archive_old_models(self):
        pass
