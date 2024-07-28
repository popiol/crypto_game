import json

import streamlit as st

from src.rl_runner import RlRunner

st.set_page_config(page_title="Current metrics")

st.title("Current metrics")

rl_runner = RlRunner()
rl_runner.load_config("config/config.yml")
model_registry = rl_runner.get_model_registry()


def print_dict(obj, level: int = 0):
    for key, val in obj.items():
        col = st.columns([0.05 * level, 1 - 0.05 * level])[1] if level else st
        if type(val) == dict:
            col.write(key)
            print_dict(val, level + 1)
            continue
        col.write(key + ": " + json.dumps(val))


for model_name, serialized_model in model_registry.iterate_models():
    metrics = model_registry.get_metrics(model_name)
    with st.expander(model_name):
        print_dict(metrics)
