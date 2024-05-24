import json
import os
import random
import re

import boto3
import boto3.resources
import boto3.resources.model
import boto3.s3


class ModelRegistry:

    def __init__(self, remote_path: str):
        self.remote_path = remote_path
        assert remote_path.startswith("s3://")
        self.bucket_name = remote_path.split("/")[2]
        self.prefix = os.path.normpath("/".join(remote_path.split("/")[3:]))
        self.current_prefix = self.prefix + "/models"
        self.archived_prefix = self.prefix + "/archived"
        self.metrics_prefix = self.prefix + "/metrics"

    def save_model(self, model_name: str, serialized_model: bytes, metrics: dict):
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(self.bucket_name)
        bucket.put_object(
            Key=f"{self.current_prefix}/{model_name}", Body=serialized_model
        )
        bucket.put_object(
            Key=f"{self.metrics_prefix}/{model_name}", Body=json.dumps(metrics)
        )

    def get_random_model(self) -> tuple[str, bytes]:
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(self.bucket_name)
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.bucket_name, Prefix=self.current_prefix + "/"
        )
        keys = pages.search("Contents[].Key")
        if not keys:
            return None, None
        key = random.choice(keys)
        model_name = re.sub("^" + self.current_prefix + "/", "", key)
        return model_name, bucket.Object().get()["Body"]

    def archive_old_models(self):
        pass
