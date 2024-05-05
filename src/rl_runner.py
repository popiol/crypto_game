from src.config import Config
from src.agent import Agent
import random
import argparse
import sys
import time


class RlRunner:

    def __init__(self, config: Config):
        self.config = config

    def run(self):
        with open(self.config.names) as f:
            names = f.read().splitlines()
        random.shuffle(names)
        agent = Agent(names[0], self.config)
        agent.process_quotes()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args, other = parser.parse_known_args(argv)
    config = Config.from_yaml_file(args.config)
    RlRunner(config).run()


if __name__ == "__main__":
    time1 = time.time()
    main(sys.argv)
    time2 = time.time()
    print("Overall execution time:", time2 - time1)
    