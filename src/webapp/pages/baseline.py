import glob
import json

import altair as alt
import pandas as pd
import streamlit as st

from src.environment import Environment

st.set_page_config(page_title="Baseline", layout="wide")

st.title("Baseline")

environment = Environment("config/config.yml")

with open(environment.reports.baseline_portfolio_path) as f:
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
df = df[~df.baseline_value.isna()]

df.baseline_value = df.baseline_value / df.iloc[0].baseline_value - 1
df["BTCUSD"] = df.BTCUSD_mean / df.iloc[0].BTCUSD_mean - 1

chart = (
    alt.Chart(df)
    .mark_line(strokeWidth=5)
    .transform_fold(fold=["baseline_value", "BTCUSD"], as_=["variable", "value"])
    .encode(
        x=alt.X("datetime", axis=alt.Axis(title=None)),
        y=alt.Y("max(value):Q", axis=alt.Axis(title=None), scale=alt.Scale(zero=False)),
        color=alt.Color("variable:N", sort="descending"),
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

st.write("## Orders")

data = {index: row for index, row in enumerate(portfolio["orders"])}

if data:
    df = pd.DataFrame.from_dict(data, orient="index")
    st.dataframe(df, hide_index=True)
else:
    st.write("No orders")

st.write("## Closed transactions")

files = sorted(glob.glob(environment.reports.baseline_transactions_path + "/*.json"))[-10:]
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
