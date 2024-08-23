import altair as alt
import pandas as pd
import streamlit as st

from src.environment import Environment

st.set_page_config(page_title="Aggregated metrics", layout="wide")

st.title("Aggregated metrics")

environment = Environment("config/config.yml")
df = pd.read_csv(environment.reports.change_in_time_path)
df["datetime"] = pd.to_datetime(df["datetime"])

df = df[~df.leader_value.isna()]

chart = (
    alt.Chart(df)
    .mark_line(strokeWidth=5)
    .encode(
        x=alt.X("datetime", axis=alt.Axis(title=None)),
        y=alt.Y("leader_value", axis=alt.Axis(title=None), scale=alt.Scale(zero=False)),
    )
)
st.altair_chart(chart, use_container_width=True, theme=None)
