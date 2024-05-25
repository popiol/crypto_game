import os
import subprocess

import boto3


class S3Utils:

    def __init__(self, s3_path: str):
        assert s3_path.startswith("s3://")
        self.bucket_name = s3_path.split("/")[2]
        self.path = os.path.normpath("/".join(s3_path.split("/")[3:]))

    def sync(self, source: str, target: str) -> bool:
        proc = subprocess.Popen(["aws", "s3", "sync", source, target], stdout=subprocess.PIPE)
        out, err = proc.communicate()
        return b"download" in out or b"upload" in out

    def download_file(self, local_path: str):
        s3 = boto3.client("s3")
        s3.download_file(self.bucket_name, self.path, local_path)
