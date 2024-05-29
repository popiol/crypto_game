import glob
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
        retention_days: int,
    ):
        self.remote_quotes = S3Utils(quotes_remote_path)
        self.quotes_local_path = quotes_local_path
        self.remote_config = S3Utils(config_remote_path)
        self.retention_days = retention_days
        self.asset_list_file = "asset_list.csv"
        self.stats_file = "stats.json"

    def sync(self):
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=self.retention_days)
        self.remove_old_data(start_dt)
        self.download_since_until(start_dt, end_dt)

    def download_since_until(self, start_dt: datetime, end_dt: datetime):
        dt = start_dt
        while dt <= end_dt:
            datetime_path = dt.strftime("year=%Y/month=%m/day=%d")
            relative_path = f"{self.remote_quotes.path}/{datetime_path}"
            source = f"s3://{self.remote_quotes.bucket_name}/{relative_path}"
            target = f"{self.quotes_local_path}/{datetime_path}"
            if not os.path.exists(target) or dt.strftime("%Y%m%d") == datetime.now().strftime("%Y%m%d"):
                print("Download", source)
                self.remote_quotes.sync(source, target)
            dt = dt + timedelta(days=1)

    def remove_old_data(self, older_than: datetime):
        older_than_str = older_than.strftime("%Y%m%d")
        for root, dirs, files in os.walk(self.quotes_local_path):
            for file in files:
                if file[:8] < older_than_str or not file.endswith(".json"):
                    print("Removing", os.path.join(root, file))
                    os.remove(os.path.join(root, file))
        for root, dirs, files in os.walk(self.quotes_local_path, topdown=False):
            if not files and not dirs:
                print("Removing", root)
                os.rmdir(root)

    def quotes_iterator(self):
        prefix = self.quotes_local_path
        for year in sorted(glob.glob(prefix + "/*")):
            for month in sorted(glob.glob(year + "/*")):
                for day in sorted(glob.glob(month + "/*")):
                    for file in sorted(glob.glob(day + "/*.json")):
                        timestamp = datetime.strptime(file.split("/")[-1].split(".")[0], "%Y%m%d%H%M%S")
                        with open(file) as f:
                            yield timestamp, json.load(f)

    def get_asset_list(self) -> list[str]:
        assets = []
        remote_path = f"{self.remote_config.path}/{self.asset_list_file}"
        contents = self.remote_config.download_bytes(remote_path)
        if contents:
            assets = contents.decode().splitlines()
        return assets

    def set_asset_list(self, assets: list[str]):
        remote_path = f"{self.remote_config.path}/{self.asset_list_file}"
        contents = "\n".join(assets).encode()
        self.remote_config.upload_bytes(remote_path, contents)

    def get_stats(self) -> dict:
        stats = {}
        remote_path = f"{self.remote_config.path}/{self.stats_file}"
        contents = self.remote_config.download_bytes(remote_path)
        if contents:
            stats = json.loads(contents.decode())
        return stats

    def set_stats(self, stats: dict):
        remote_path = f"{self.remote_config.path}/{self.stats_file}"
        contents = json.dumps(stats).encode()
        self.remote_config.upload_bytes(remote_path, contents)
