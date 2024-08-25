import json

import pandas as pd
import streamlit as st

from src.environment import Environment

model_name = st.query_params.get("model")
title = model_name or "Current metrics"
st.set_page_config(page_title=title, layout="wide")
environment = Environment("config/config.yml")


def list_current_models():
    st.title("Current metrics")
    df = pd.read_csv(environment.reports.quick_stats_path)
    df = df[df.score != 0]
    df = df.sort_values("score", ascending=False)[:15]
    df["model"] = df["model"].apply(lambda x: f"?model={x}")
    st.dataframe(df, hide_index=True, column_config={"model": st.column_config.LinkColumn()})


def show_model(model_name: str):
    st.title(model_name)
    model_registry = environment.model_registry

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

    if model_name.startswith("Leader_"):
        metrics = model_registry.get_leader_metrics()
    else:
        metrics = model_registry.get_metrics(model_name)
    print_dict(metrics)


if model_name:
    show_model(model_name)
else:
    list_current_models()
