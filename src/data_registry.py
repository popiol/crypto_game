import glob
import json
import lzma
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

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
        trainset_keys_remote_path: str,
        trainset_keys_local_path: str,
    ):
        self.remote_quotes = S3Utils(quotes_remote_path)
        self.quotes_local_path = quotes_local_path
        self.remote_config = S3Utils(config_remote_path)
        self.retention_days = retention_days
        self.remote_trainset = S3Utils(trainset_remote_path)
        self.remote_trainset_keys = S3Utils(trainset_keys_remote_path)
        self.trainset_keys_local_path = trainset_keys_local_path
        self.asset_list_file = "asset_list.csv"
        self.current_assets_file = "current_assets.csv"
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

    def get_current_assets(self) -> set[str]:
        remote_path = f"{self.remote_config.path}/{self.current_assets_file}"
        contents = self.remote_config.download_bytes(remote_path)
        if contents:
            return set(contents.decode().splitlines())
        return set()

    def set_current_assets(self, assets: set[str]):
        remote_path = f"{self.remote_config.path}/{self.current_assets_file}"
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

    def update_local_trainset_keys(self, keys_timestamp: str, local_keys_path: str, last_trainset: str):
        file_names = [last_trainset]
        download_since = keys_timestamp + "01000000"
        if os.path.exists(local_keys_path):
            with open(local_keys_path) as f:
                local_keys = f.read().splitlines()
            if local_keys:
                download_since = local_keys[-1].split(".")[0]
        download_last_hours = (datetime.now() - datetime.strptime(download_since, "%Y%m%d%H%M%S")).total_seconds() / 3600
        if download_last_hours > 24:
            print("Update trainset keys with data from the last", int(download_last_hours), "hours")
            remote_keys = self.remote_trainset.list_files(
                self.remote_trainset.path + "/", younger_than_hours=int(download_last_hours)
            )
            file_names = [key.split("/")[-1] for key in remote_keys]
            if last_trainset not in file_names:
                file_names.append(last_trainset)
        with open(local_keys_path, "a") as f:
            for file_name in file_names:
                f.write(file_name + "\n")

    def add_trainset_key(self, trainset_file: str):
        keys_timestamp = datetime.now().strftime("%Y%m")
        local_keys_path = f"{self.trainset_keys_local_path}/{keys_timestamp}"
        remote_keys_path = f"{self.remote_trainset_keys.path}/{keys_timestamp}"
        self.remote_trainset_keys.download_file(remote_keys_path, local_keys_path, only_updated=True)
        self.update_local_trainset_keys(keys_timestamp, local_keys_path, trainset_file)
        self.remote_trainset_keys.upload_file(local_keys_path, remote_keys_path)

    def add_to_trainset(self, trainset: RlTrainset):
        if not trainset:
            return
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        trainset_file = f"{timestamp}.pickle.lzma"
        self.add_trainset_key(trainset_file)
        trainset_bytes = lzma.compress(pickle.dumps(trainset))
        self.remote_trainset.upload_bytes(f"{self.remote_trainset.path}/{trainset_file}", trainset_bytes)

    def get_random_trainset(self, asset_list: list[str], current_assets: set[str]) -> RlTrainset:
        trainset_keys_url = f"s3://{self.remote_trainset_keys.bucket_name}/{self.remote_trainset_keys.path}/"
        self.remote_trainset_keys.sync(trainset_keys_url, self.trainset_keys_local_path)
        files = glob.glob(self.trainset_keys_local_path + "/*")
        if not files:
            return []
        file = random.choice(files)
        with open(file) as f:
            local_keys = f.read().splitlines()
        trainset_file = random.choice(local_keys)
        print(trainset_file)
        trainset_bytes = self.remote_trainset.download_bytes(f"{self.remote_trainset.path}/{trainset_file}")
        if not trainset_bytes:
            return []
        trainset = pickle.loads(lzma.decompress(trainset_bytes))
        n_assets = len(asset_list)
        trainset = self.fix_n_assets(trainset, n_assets)
        indices = [index for index, asset in enumerate(asset_list) if asset in current_assets]
        trainset = self.filter_assets(trainset, indices)
        return trainset

    def fix_n_assets(self, trainset: RlTrainset, n_assets: int):
        orig_n_assets = len(trainset[0][0][0][0])
        orig_n_assets_output = len(trainset[0][1][0])
        if n_assets == orig_n_assets:
            return trainset
        print("Fix n_assets", orig_n_assets, "->", n_assets)
        print("input:", np.shape(trainset[0][0]))
        print("output:", np.shape(trainset[0][1]))
        assert n_assets > orig_n_assets
        fixed = []
        for input, output, reward in trainset:
            input_shape = list(np.shape(input))
            input_shape[2] = n_assets - orig_n_assets
            output_shape = list(np.shape(output))
            output_shape[1] = n_assets - orig_n_assets_output
            fixed.append(
                (
                    np.concatenate([input, np.zeros(input_shape, dtype=input.dtype)], axis=2),
                    np.concatenate([output, np.zeros(output_shape, dtype=output.dtype)], axis=1),
                    reward,
                )
            )
        return fixed

    def filter_assets(self, trainset: RlTrainset, indices: list[int]):
        orig_n_assets = len(trainset[0][1][0])
        print("Filter assets from", orig_n_assets, "to", len(indices))
        print("input:", np.shape(trainset[0][0]))
        print("output:", np.shape(trainset[0][1]))
        return [(input, output[:, indices], reward) for input, output, reward in trainset]
