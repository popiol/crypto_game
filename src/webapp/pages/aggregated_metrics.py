import re

import altair as alt
import pandas as pd
import streamlit as st

from src.environment import Environment

st.set_page_config(page_title="Aggregated metrics", layout="wide")

st.title("Aggregated metrics")

environment = Environment("config/config.yml")
environment.model_registry.download_report(environment.reports.change_in_time_path)
df = pd.read_csv(environment.reports.change_in_time_path)
df["datetime"] = pd.to_datetime(df["datetime"])

groups = set()
suffixes = ["_min", "_max", "_mean"]
head = ["evaluation_score"]
ignore = ["leader_value", "baseline_value", "real_portfolio_value"]
ignore_prefix = ["n_layers_", "version_", "merge_"]

# print mean score vs BTCUSD change
st.write("## Mean score vs BTCUSD change")
chart = alt.Chart(df).encode(x=alt.X("datetime", axis=alt.Axis(title=None)))
chart = alt.layer(
    chart.mark_line(color="#7f7", strokeWidth=2).encode(
        y=alt.Y("evaluation_score_mean", axis=alt.Axis(title=None), scale=alt.Scale(zero=False))
    ),
    chart.mark_line(color="#f77", strokeWidth=2).encode(
        y=alt.Y("BTCUSD_change_mean", axis=alt.Axis(title=None), scale=alt.Scale(zero=False))
    )
)
st.altair_chart(chart, use_container_width=True, theme=None)

for col in df.columns:
    group = re.sub("|".join([s + "$" for s in suffixes]), "", col)
    if col == "datetime" or group in ignore or group in head:
        continue
    if any(group.startswith(prefix) for prefix in ignore_prefix):
        continue
    groups.add(group)

groups = head + sorted(groups)
colors = ["#7f7", "#f77", "#77f"]

for group in groups:
    st.write("## " + group)
    cols = [group + s for s in suffixes] if group + suffixes[0] in df else [group]
    chart = alt.Chart(df).encode(x=alt.X("datetime", axis=alt.Axis(title=None)))
    chart = alt.layer(
        *[
            chart.mark_line(color=color, strokeWidth=2).encode(
                y=alt.Y(col, axis=alt.Axis(title=None), scale=alt.Scale(zero=False))
            )
            for col, color in zip(cols, colors)
        ]
    )
    st.altair_chart(chart, use_container_width=True, theme=None)
