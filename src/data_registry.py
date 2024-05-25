import os
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.s3_utils import S3Utils


@dataclass
class DataRegistry:

    def __init__(
        self,
        quotes_remote_path: str,
        quotes_local_path: str,
        asset_list_remote_path: str,
        asset_list_local_path: str,
        retention_days: int,
    ):
        self.remote_quotes = S3Utils(quotes_remote_path)
        self.quotes_local_path = quotes_local_path
        self.remote_asset_list = S3Utils(asset_list_remote_path)
        self.asset_list_local_path = asset_list_local_path
        self.retention_days = retention_days

    def get_start_dt(self):
        current_dt = datetime.now()
        return current_dt - timedelta(days=self.retention_days - 1)

    def sync(self):
        current_dt = datetime.now()
        start_dt = self.get_start_dt()
        self.remove_old_data(start_dt)
        self.download_since_until(start_dt, current_dt)
        self.download_asset_list()

    def download_since_until(self, start_dt: datetime, end_dt: datetime):
        dt = start_dt
        while dt <= end_dt:
            datetime_path = dt.strftime("year=%Y/month=%m/day=%d")
            relative_path = f"{self.remote_quotes.path}/{datetime_path}"
            source = f"s3://{self.remote_quotes.bucket_name}/{relative_path}"
            target = f"{self.quotes_local_path}/{relative_path}"
            print("Download", source)
            self.remote_quotes.sync(source, target)
            dt = dt + timedelta(days=1)

    def remove_old_data(self, older_than: datetime):
        for root, dirs, files in os.walk(self.quotes_local_path):
            for file in files:
                if file < older_than:
                    print("Removing", os.join(root, file))
                    # os.remove(os.join(root, file))
        for root, dirs, files in os.walk(self.quotes_local_path):
            if not files and not dirs:
                print("Removing", root)
                # os.rmdir(root)

    def download_asset_list(self):
        print("Download", self.asset_list_local_path)
        self.remote_asset_list.download_file(self.asset_list_local_path)
