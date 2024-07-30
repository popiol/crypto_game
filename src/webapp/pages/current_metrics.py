import json

import pandas as pd
import streamlit as st

from src.rl_runner import RlRunner

model_name = st.query_params.get("model_name")
title = model_name or "Current metrics"
st.set_page_config(page_title=title, layout="wide")


def list_current_models():
    st.title("Current metrics")
    df = pd.read_csv("data/quick_stats.csv")
    df = df[df.score != 0][:15]
    df = df.sort_values("score", ascending=False)
    df["model"] = df["model"].apply(lambda x: f"?model_name={x}")
    st.dataframe(
        df,
        hide_index=True,
        column_config={"model": st.column_config.LinkColumn(), "score": st.column_config.NumberColumn()},
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
