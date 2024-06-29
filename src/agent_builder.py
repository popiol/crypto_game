import random
from dataclasses import dataclass

from src.agent import Agent
from src.data_transformer import DataTransformer
from src.evolution_handler import EvolutionHandler
from src.training_strategy import StrategyPicker
from src.trainset import Trainset


@dataclass
class AgentBuilder:

    evolution_handler: EvolutionHandler
    data_transformer: DataTransformer
    trainset: Trainset
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
            model, metrics = self.evolution_handler.create_model()
            training_strategy = StrategyPicker().pick(model)
            agent = Agent(name, self.data_transformer, self.trainset, training_strategy, metrics)
            model.name = agent.model_name
            agents.append(agent)
        return agents
