import glob
import json

import altair as alt
import pandas as pd
import streamlit as st

from src.environment import Environment

st.set_page_config(page_title="Portfolio", layout="wide")

st.title("Portfolio")

environment = Environment("config/config.yml")

with open(environment.reports.portfolio_path) as f:
    portfolio = json.load(f)

st.write(
    " ".join(
        [
            f"Value: <font size='6'>\${round(portfolio['value'],2)}</font>",
            f"Cash: <font size='6'>\${round(portfolio['cash'],2)}</font>",
        ]
    ),
    unsafe_allow_html=True,
)

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

st.write("## Open positions")

data = {index: row for index, row in enumerate(portfolio["positions"])}

if data:
    df = pd.DataFrame.from_dict(data, orient="index")
    st.dataframe(df, hide_index=True)
else:
    st.write("No open positions")

st.write("## Closed transactions")

files = sorted(glob.glob(environment.reports.transactions_path + "/*.json"))[-10:]
transactions = []
for file in files:
    with open(file) as f:
        transactions.extend(json.load(f))

data = {index: row for index, row in enumerate(transactions)}

if data:
    df = pd.DataFrame.from_dict(data, orient="index")
    st.dataframe(df, hide_index=True)
else:
    st.write("No closed transactions")
