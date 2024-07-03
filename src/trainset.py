import glob
import os
import pickle
from datetime import datetime

import numpy as np


class Trainset:

    def __init__(self, local_path: str):
        self.local_path = local_path
        self.clean()

    def clean(self):
        if os.path.exists(self.local_path):
            for file in glob.iglob(self.local_path + "/*.pickle"):
                os.remove(file)
        else:
            os.makedirs(self.local_path)

    def get_shared_input_file_path(self, timestamp: datetime):
        return f"{self.local_path}/input_{timestamp.strftime('%Y%m%d%H%M%S')}.pickle"

    def get_agent_input_file_path(self, timestamp: datetime, agent: str):
        return f"{self.local_path}/input_{agent}_{timestamp.strftime('%Y%m%d%H%M%S')}.pickle"

    def get_output_file_path(self, timestamp: datetime, agent: str):
        return f"{self.local_path}/output_{agent}_{timestamp.strftime('%Y%m%d%H%M%S')}.pickle"

    def store_shared_input(self, timestamp: datetime, input: np.array):
        with open(self.get_shared_input_file_path(timestamp), "wb") as f:
            pickle.dump(input, f)

    def store_agent_input(self, timestamp: datetime, input: np.array, agent: str):
        with open(self.get_agent_input_file_path(timestamp, agent), "wb") as f:
            pickle.dump(input, f)

    def store_output(self, timestamp: datetime, output: np.array, agent: str):
        with open(self.get_output_file_path(timestamp, agent), "wb") as f:
            pickle.dump(output, f)

    def get_by_timestamp(self, timestamp: datetime, agent: str) -> tuple[np.array, np.array, np.array]:
        with open(self.get_shared_input_file_path(timestamp), "rb") as f:
            shared_input = pickle.load(f)
        with open(self.get_agent_input_file_path(timestamp, agent), "rb") as f:
            agent_input = pickle.load(f)
        with open(self.get_output_file_path(timestamp, agent), "rb") as f:
            output = pickle.load(f)
        return shared_input, agent_input, output
