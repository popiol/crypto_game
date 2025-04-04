import glob
import json
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from src.environment import Environment

st.set_page_config(page_title="Real portfolio", layout="wide")

st.title("Real portfolio")

environment = Environment("config/config.yml")

environment.model_registry.download_real_portfolio(
    environment.reports.real_portfolio_path, environment.reports.real_transactions_path
)
with open(environment.reports.real_portfolio_path) as f:
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

environment.model_registry.download_report(environment.reports.change_in_time_path)
df = pd.read_csv(environment.reports.change_in_time_path)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df[df.datetime > datetime(2025, 3, 10)]
df = df[~df.real_portfolio_value.isna()]

df.real_portfolio_value = df.real_portfolio_value / df.iloc[0].real_portfolio_value - 1
df["BTCUSD"] = df.BTCUSD_mean / df.iloc[0].BTCUSD_mean - 1

chart = (
    alt.Chart(df)
    .mark_line(strokeWidth=5)
    .transform_fold(fold=["real_portfolio_value", "BTCUSD"], as_=["variable", "value"])
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

files = sorted(glob.glob(environment.reports.real_transactions_path + "/*.json"))[-10:]
transactions = []
for file in files:
    with open(file) as f:
        transactions.extend(json.load(f))

data = {index: row for index, row in enumerate(transactions)}

if data:
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.drop_duplicates()
    df = df.sort_values("place_sell_dt", ascending=False)
    st.dataframe(df, hide_index=True)
else:
    st.write("No closed transactions")
