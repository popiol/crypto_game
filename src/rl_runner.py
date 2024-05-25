import argparse
import random
import sys
import time

import psutil

from src.agent import Agent
from src.config import Config


class RlRunner:

    def __init__(self, config: Config):
        self.config = config

    def run(self):
        names = self.config.agent_builder.get_names()
        agent = Agent(names[0], self.config)
        agent.process_quotes()


def check_still_running(process_name: str):
    processes = [
        p.cmdline()
        for p in psutil.process_iter()
        if len(p.cmdline()) > 2 and p.cmdline()[2] == process_name and p.cmdline()[3:] == sys.argv[1:]
    ]
    if len(processes) > 1:
        print("Master process up and running")
        exit()
    print("No master process found - starting now")


def main(argv):
    import time

    print("Starting")
    time.sleep(10)
    print("Ending")
    exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args, other = parser.parse_known_args(argv)
    config = Config.from_yaml_file(args.config)
    RlRunner(config).run()


if __name__ == "__main__":
    check_still_running("src.rl_runner")
    time1 = time.time()
    main(sys.argv)
    time2 = time.time()
    print("Overall execution time:", time2 - time1)
