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
    scores = sorted(scores, key=lambda x: -float(x[1]))[:10]
    for row in scores:
        row[0] = f"?model_name={row[0]}"
    df = pd.DataFrame(scores, columns=["model", "score"])
    st.dataframe(
        df, hide_index=True, column_config={"model": st.column_config.LinkColumn(), "score": st.column_config.NumberColumn()}
    )


def show_model(model_name: str):
    st.title(model_name)
    rl_runner = RlRunner()
    rl_runner.load_config("config/config.yml")
    model_registry = rl_runner.get_model_registry()

    def print_dict(obj, level: int = 0):
        get_col = lambda level: st.columns([0.05 * level, 1 - 0.05 * level])[1] if level else st
        params = ""
        for key, val in obj.items():
            if type(val) == dict:
                continue
            params += "- " + key + ": " + json.dumps(val) + "\n"
        col = get_col(level)
        col.write(params)
        for key, val in obj.items():
            if type(val) != dict:
                continue
            if level == 0:
                with st.expander(key):
                    print_dict(val, level + 1)
            else:
                col = get_col(level)
                col.write(key)
                print_dict(val, level + 1)

    metrics = model_registry.get_metrics(model_name)
    print_dict(metrics)


if model_name:
    show_model(model_name)
else:
    list_current_models()
