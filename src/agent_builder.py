import random
from dataclasses import dataclass

from src.agent import Agent
from src.data_transformer import DataTransformer
from src.evolution_handler import EvolutionHandler


@dataclass
class AgentBuilder:

    evolution_handler: EvolutionHandler
    data_transformer: DataTransformer
    n_agents: int
    name_list_file: str

    def get_names(self):
        with open(self.name_list_file) as f:
            names = f.read().splitlines()
        random.shuffle(names)
        return names[: self.n_agents]

    def create_agents(self) -> list[Agent]:
        agents = []
        for name in self.get_names():
            print(name)
            model = self.evolution_handler.create_model()
            agent = Agent(name, model, self.data_transformer)
            agents.append(agent)
        return agents
