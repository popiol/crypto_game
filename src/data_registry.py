import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta

import boto3


@dataclass
class DataRegistry:

    def __init__(self, remote_path: str, local_path: str):
        self.remote_path = remote_path
        self.local_path = local_path
        assert remote_path.startswith("s3://")
        self.bucket_name = remote_path.split("/")[2]
        self.prefix = os.path.normpath("/".join(remote_path.split("/")[3:]))

    def sync(self, source: str, target: str) -> bool:
        proc = subprocess.Popen(
            ["aws", "s3", "sync", source, target], stdout=subprocess.PIPE
        )
        out, err = proc.communicate()
        return b"download" in out or b"upload" in out

    def get_start_dt(self):
        current_dt = datetime.now()
        return current_dt - timedelta(days=6)

    def download_last_week(self):
        current_dt = datetime.now()
        dt = self.get_start_dt()
        while dt <= current_dt:
            year = dt.strftime("%Y")
            month = dt.strftime("%m")
            day = dt.strftime("%d")
            relative_path = f"{self.prefix}/year={year}/month={month}/day={day}"
            source = f"s3://{self.bucket_name}/{relative_path}"
            target = f"{self.local_path}/{relative_path}"
            self.sync(source, target)
            dt = dt + timedelta(days=1)

    def remove_old_data(self):
        start_dt = self.get_start_dt()
        for root, dirs, files in os.walk(self.local_path):
            for file in files:
                if file < start_dt:
                    print("Removing", os.join(root, file))
                    # os.remove(os.join(root, file))
