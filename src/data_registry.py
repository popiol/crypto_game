import json
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
        config_remote_path: str,
        config_local_path: str,
        retention_days: int,
    ):
        self.remote_quotes = S3Utils(quotes_remote_path)
        self.quotes_local_path = quotes_local_path
        self.remote_config = S3Utils(config_remote_path)
        self.config_local_path = config_local_path
        self.retention_days = retention_days
        self.asset_list_file = "asset_list.csv"
        self.stats_file = "stats.json"

    def get_start_dt(self, end_dt: datetime):
        return

    def sync(self):
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=self.retention_days - 1)
        self.remove_old_data(start_dt)
        self.download_since_until(start_dt, end_dt)
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
        older_than_str = older_than.strftime("%Y%m%d")
        for root, dirs, files in os.walk(self.quotes_local_path):
            for file in files:
                if file[:8] < older_than_str or not file.endswith(".json"):
                    print("Removing", os.join(root, file))
                    # os.remove(os.join(root, file))
        for root, dirs, files in os.walk(self.quotes_local_path, topdown=False):
            if not files and not dirs:
                print("Removing", root)
                # os.rmdir(root)

    def get_asset_list(self) -> list[str]:
        assets = []
        if self.remote_config.download_file(f"{self.remote_config.path}/{self.asset_list_file}", self.config_local_path):
            with open(f"{self.config_local_path}/{self.asset_list_file}") as f:
                assets = f.read().splitlines()
        return assets

    def set_asset_list(self, assets: list[str]):
        with open(self.asset_list_local_path, "w"):
        self.remote_asset_list.upload_file(self.asset_list_local_path)

    def download_stats(self) -> dict:
        stats = {}
        if self.remote_config.download_file(f"{self.remote_config.path}/{self.stats_file}", self.config_local_path):
            with open(f"{self.config_local_path}/{self.stats_file}") as f:
                stats = json.load(f)
        return stats
