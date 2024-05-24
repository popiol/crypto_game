import random
from dataclasses import dataclass


@dataclass
class AgentBuilder:

    n_agents: int
    name_list_file: str

    def get_names(self):
        with open(self.name_list_file) as f:
            names = f.read().splitlines()
        random.shuffle(names)
        return names[: self.n_agents]
