import json
import re
import subprocess
from datetime import datetime, timedelta, timezone

import boto3
from botocore.exceptions import ClientError


class S3Utils:

    def __init__(self, s3_path: str):
        assert s3_path.startswith("s3://")
        self.bucket_name = s3_path.split("/")[2]
        self.path = "/".join(s3_path.split("/")[3:])
        self.path = re.sub("/+$", "", self.path)

    def sync(self, source: str, target: str) -> bool:
        proc = subprocess.Popen(["/usr/local/bin/aws", "s3", "sync", source, target], stdout=subprocess.PIPE)
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
            return bucket.Object(remote_path).get()["Body"].read()
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

    def download_json(self, remote_path: str):
        contents = self.download_bytes(remote_path)
        if contents is None:
            return {}
        return json.loads(contents.decode())

    def upload_json(self, remote_path: str, json_obj) -> bool:
        return self.upload_bytes(remote_path, json.dumps(json_obj).encode())

    def list_files(self, remote_path: str, older_than_hours: int = None) -> list[str]:
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=remote_path)
        filter = ""
        if older_than_hours is not None:
            timestamp = (datetime.now(timezone.utc) - timedelta(hours=older_than_hours)).strftime("%Y-%m-%d %H:%M:%S+0000")
            filter = f"?to_string(LastModified)<'\"{timestamp}\"'"
        return (x for x in pages.search(f"Contents[{filter}].Key") if x is not None)

    def move_file(self, source: str, target: str):
        s3 = boto3.resource("s3")
        s3.Object(self.bucket_name, target).copy_from(CopySource=f"{self.bucket_name}/{source}")
        s3.Object(self.bucket_name, source).delete()
