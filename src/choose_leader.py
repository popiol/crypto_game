import pandas as pd

from src.environment import Environment


def choose_leader():
    environment = Environment("config/config.yml")
    df = pd.read_csv(environment.reports.quick_stats_path)
    row = df[df.score == df.score.max()].iloc[0]
    model_name = row.model
    score = row.score
    print("Deploying", model_name)
    print("Score", score)
    environment.model_registry.deploy_model(model_name)


if __name__ == "__main__":
    choose_leader()
