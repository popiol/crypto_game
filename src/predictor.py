import argparse
import sys
import time

import pandas as pd

from src.environment import Environment


def choose_leader(environment: Environment):
    df = pd.read_csv(environment.reports.quick_stats_path)
    row = df[df.score == df.score.max()].iloc[0]
    model_name = row.model
    score = row.score
    print("Deploying", model_name)
    print("Score", score)
    environment.model_registry.deploy_model(model_name)


def predict(environment: Environment):
    pass


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yml")
    parser.add_argument("--choose_leader", action="store_true")
    args, other = parser.parse_known_args(argv)

    environment = Environment(args.config)

    if args.choose_leader:
        choose_leader(environment)
    else:
        predict(environment)


if __name__ == "__main__":
    time1 = time.time()
    main(sys.argv)
    time2 = time.time()
    print("Overall execution time:", time2 - time1)
