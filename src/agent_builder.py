import random
from dataclasses import dataclass

from src.agent import Agent
from src.data_transformer import DataTransformer
from src.evolution_handler import EvolutionHandler
from src.training_strategy import LearnOnSuccess, LearnOnMistakes, LearnOnBoth
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
    
    def get_strategy(self):
        strategy = self.evolution_handler.evolution_randomizer.training_strategy()
        if strategy == self.evolution_handler.evolution_randomizer.TrainingStrategy.LEARN_ON_SUCCESS:
            return LearnOnSuccess
        elif strategy == self.evolution_handler.evolution_randomizer.TrainingStrategy.LEARN_ON_MISTAKE:
            return LearnOnMistakes
        elif strategy == self.evolution_handler.evolution_randomizer.TrainingStrategy.LEARN_ON_BOTH:
            return LearnOnBoth
        

    def create_agents(self) -> list[Agent]:
        agents = []
        for name in self.get_names():
            model, metrics = self.evolution_handler.create_model()
            training_strategy = self.get_strategy()(model)
            agent = Agent(name, self.data_transformer, self.trainset, training_strategy, metrics)
            model.name = agent.model_name
            agents.append(agent)
        return agents
