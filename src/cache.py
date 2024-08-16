import copy
from datetime import datetime
from typing import Callable

import psutil


class Cache:

    def __init__(self, min_memory_mb: float):
        self.min_memory_mb = min_memory_mb
        self.memory_cache = {}

    def get(self, func: Callable, timestamp: datetime, *args, **kwargs):
        name = func.__name__
        timestamp_str = timestamp.strftime("%Y%m%d%H%M%S")
        if self.memory_cache.get(name, {}).get(timestamp_str) is not None:
            return self.memory_cache[name][timestamp_str]
        result = func(*args, **kwargs)
        memory = psutil.virtual_memory().available
        if memory > self.min_memory_mb * 1000000:
            self.memory_cache[name] = self.memory_cache.get(name, {})
            self.memory_cache[name][timestamp_str] = copy.deepcopy(result)
        return result
