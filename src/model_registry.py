import random
import os
import boto3
import boto3.resources
import boto3.resources.model
import boto3.s3
import re


class ModelRegistry:

    def __init__(self, root_path: str):
        self.root_path = root_path
        assert root_path.startswith("s3://")
        self.bucket_name = root_path.split("/")[2]
        self.prefix = os.path.normpath("/".join(root_path.split("/")[3:]))

    def save_model(self, model_name: str, body: bytes):
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(self.bucket_name)
        bucket.put_object(Key=f"{self.prefix}/{model_name}", Body=body)

    def get_random_model(self) -> tuple[str, bytes]:
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(self.bucket_name)
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)
        keys = pages.search("Contents[].Key")
        if not keys:
            return None, None
        key = random.choice(keys)
        model_name = re.sub("^" + self.prefix + "/", "", key)
        return model_name, bucket.Object().get()["Body"]
