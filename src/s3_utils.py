import os
import subprocess

import boto3
from botocore.exceptions import ClientError


class S3Utils:

    def __init__(self, s3_path: str):
        assert s3_path.startswith("s3://")
        self.bucket_name = s3_path.split("/")[2]
        self.path = os.path.normpath("/".join(s3_path.split("/")[3:]))

    def sync(self, source: str, target: str) -> bool:
        proc = subprocess.Popen(["aws", "s3", "sync", source, target], stdout=subprocess.PIPE)
        out, err = proc.communicate()
        return b"download" in out or b"upload" in out

    def download_file(self, remote_path: str, local_path: str) -> bool:
        print("Download", local_path)
        s3 = boto3.client("s3")
        try:
            s3.download_file(self.bucket_name, remote_path, local_path)
            return True
        except ClientError:
            return False

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        print("Upload", local_path)
        s3 = boto3.client("s3")
        try:
            s3.upload_file(local_path, self.bucket_name, remote_path)
            return True
        except ClientError:
            return False

    def download_bytes(self, remote_path: str) -> bytes:
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(self.bucket_name)
        try:
            return bucket.Object(remote_path).get()["Body"]
        except ClientError:
            return None

    def upload_bytes(self, remote_path: str, contents: bytes) -> bool:
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(self.bucket_name)
        try:
            bucket.put_object(Key=remote_path, Body=contents)
            return True
        except ClientError:
            return False

    def list_files(self, remote_path: str) -> list[str]:
        s3 = boto3.resource("s3")
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=remote_path)
        return pages.search("Contents[].Key")
