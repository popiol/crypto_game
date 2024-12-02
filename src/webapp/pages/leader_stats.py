import altair as alt
import pandas as pd
import streamlit as st

from src.environment import Environment

st.set_page_config(page_title="Leader stats", layout="wide")

st.title("Leader stats")

environment = Environment("config/config.yml")
environment.model_registry.download_report(environment.reports.leader_stats_path)
df = pd.read_csv(environment.reports.leader_stats_path)
df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d%H")

head = [
    "n_open_positions",
    "n_orders",
    "n_closed_transactions",
    "n_params",
    "n_mutations",
    "n_layers",
    "n_trainings",
    "trained_ratio",
]
ignore = ["datetime"]

cols = head + [c for c in df.columns if c not in ignore and c not in head]

for col in cols:
    if col in ignore:
        continue
    st.write("## " + col)
    chart = alt.Chart(df).encode(x=alt.X("datetime", axis=alt.Axis(title=None)), y=col)
    chart = chart.mark_line(strokeWidth=5).encode(y=alt.Y(col, axis=alt.Axis(title=None), scale=alt.Scale(zero=False)))
    st.altair_chart(chart, use_container_width=True, theme=None)
