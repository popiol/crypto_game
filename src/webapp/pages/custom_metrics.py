import json

import altair as alt
import pandas as pd
import streamlit as st

from src.environment import Environment

st.set_page_config(page_title="Custom metrics", layout="wide")

st.title("Custom metrics")

environment = Environment("config/config.yml")
with open(environment.reports.custom_metrics_path) as f:
    custom = json.load(f)

ignore = ["parents_score"]

for metric, values in custom.items():
    if metric in ignore:
        continue
    st.write("## " + metric)
    chart = (
        alt.Chart(pd.DataFrame.from_dict(values, orient="index", columns=["y"]).reset_index(names="x"))
        .mark_bar()
        .encode(
            x=alt.X("x", axis=alt.Axis(title=None, labelAngle=0), sort=None),
            y=alt.Y("y", axis=alt.Axis(title=None), scale=alt.Scale(zero=False)),
        )
    )
    st.altair_chart(chart, use_container_width=True, theme=None)
