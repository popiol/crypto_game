import glob
import json
import lzma
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.constants import RlTrainset
from src.s3_utils import S3Utils


@dataclass
class DataRegistry:

    def __init__(
        self,
        quotes_remote_path: str,
        quotes_local_path: str,
        config_remote_path: str,
        retention_days: int,
        trainset_remote_path: str,
    ):
        self.remote_quotes = S3Utils(quotes_remote_path)
        self.quotes_local_path = quotes_local_path
        self.remote_config = S3Utils(config_remote_path)
        self.retention_days = retention_days
        self.remote_trainset = S3Utils(trainset_remote_path)
        self.asset_list_file = "asset_list.csv"
        self.stats_file = "stats.json"
        self._files_and_timestamps = None

    def sync(self):
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=self.retention_days)
        self.remove_old_data(start_dt)
        self.download(start_dt, end_dt)

    def download(self, start_dt: datetime, end_dt: datetime):
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

    def files_and_timestamps(self, eval_mode: bool = False) -> list[tuple[str, datetime, bool]]:
        if self._files_and_timestamps is not None:
            return self._files_and_timestamps
        if eval_mode:
            end_dt = datetime.now()
            middle_dt = end_dt - timedelta(days=self.retention_days / 3)
            start_dt = end_dt - timedelta(days=self.retention_days * 2 / 3)
        else:
            start_dt = datetime.now() - timedelta(days=self.retention_days)
            middle_dt = start_dt + timedelta(days=self.retention_days / 3)
            end_dt = start_dt + timedelta(days=self.retention_days * 2 / 3)
        prefix = self.quotes_local_path
        self._files_and_timestamps = []
        for year in sorted(glob.glob(prefix + "/*")):
            for month in sorted(glob.glob(year + "/*")):
                for day in sorted(glob.glob(month + "/*")):
                    for file in sorted(glob.glob(day + "/??????????????.json")):
                        timestamp = datetime.strptime(file.split("/")[-1].split(".")[0], "%Y%m%d%H%M%S")
                        if timestamp < start_dt or timestamp >= end_dt:
                            continue
                        self._files_and_timestamps.append((file, timestamp, timestamp < middle_dt))
        return self._files_and_timestamps

    def get_quotes_and_bidask(self, file: str):
        with open(file) as f:
            quotes = json.load(f)
        bidask_file = file.replace(".json", "_bidask.json")
        bidask = None
        if os.path.exists(bidask_file):
            with open(bidask_file) as f:
                bidask = json.load(f)
        return quotes, bidask

    def get_asset_list(self) -> list[str]:
        remote_path = f"{self.remote_config.path}/{self.asset_list_file}"
        contents = self.remote_config.download_bytes(remote_path)
        if contents:
            return contents.decode().splitlines()
        return []

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

    def add_to_trainset(self, trainset: RlTrainset):
        if not trainset:
            return
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        trainset_bytes = lzma.compress(pickle.dumps(trainset))
        self.remote_trainset.upload_bytes(f"{self.remote_trainset.path}/{timestamp}.pickle.lzma", trainset_bytes)

    def get_random_trainset(self) -> RlTrainset:
        keys = list(self.remote_trainset.list_files(self.remote_trainset.path + "/"))
        if not keys:
            return []
        key = random.choice(keys)
        trainset_bytes = self.remote_trainset.download_bytes(key)
        return pickle.loads(lzma.decompress(trainset_bytes))
