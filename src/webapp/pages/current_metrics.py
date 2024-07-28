import glob
import json
import os

import pandas as pd
import streamlit as st

from src.rl_runner import RlRunner

st.set_page_config(page_title="Current metrics")

model_name = st.query_params.get("model_name")


def list_current_models():
    st.title("Current metrics")
    eval_file_path = "logs/evaluate.log"
    if not os.path.exists(eval_file_path):
        eval_file_path = max(glob.glob("logs/archive/evaluate_*.log"))
    with open(eval_file_path) as f:
        log = f.read()
    pos1 = log.index("Evaluate models")
    pos1 = log.index("\n", pos1) + 1
    pos2 = log.index("archive") - 1
    lines = log[pos1:pos2].splitlines()
    scores = [line.split()[2:] for line in lines]
    scores = sorted(scores, key=lambda x: -float(x[1]))
    for row in scores:
        row[0] = f"?model_name={row[0]}"
    df = pd.DataFrame(scores, columns=["model", "score"])
    st.dataframe(
        df, hide_index=True, column_config={"model": st.column_config.LinkColumn(), "score": st.column_config.NumberColumn()}
    )


def show_model(model_name: str):
    st.title(model_name)


if model_name:
    show_model(model_name)
else:
    list_current_models()


exit()

rl_runner = RlRunner()
rl_runner.load_config("config/config.yml")
model_registry = rl_runner.get_model_registry()


def print_dict(obj, level: int = 0):
    for key, val in obj.items():
        if type(val) == dict:
            continue
        col = st.columns([0.05 * level, 1 - 0.05 * level])[1] if level else st
        col.write(key + ": " + json.dumps(val))
    for key, val in obj.items():
        if type(val) != dict:
            continue
        col = st.columns([0.05 * level, 1 - 0.05 * level])[1] if level else st
        col.write(key)
        print_dict(val, level + 1)


metrics_list = []

for model_name, serialized_model in model_registry.iterate_models():
    metrics = model_registry.get_metrics(model_name)
    metrics_list.append((model_name, metrics))
    if len(metrics_list) >= 10:
        break

metrics_list = sorted(metrics_list, key=lambda x: -x[1].get("evaluation_score", -100))

for model_name, metrics in metrics_list:
    with st.expander(model_name):
        print_dict(metrics)
