import re

import pandas as pd
import streamlit as st

from src.environment import Environment

st.set_page_config(page_title="Aggregated metrics", layout="wide")

st.title("Aggregated metrics")

environment = Environment("config/config.yml")
df = pd.read_csv(environment.reports.change_in_time_path)
df["datetime"] = pd.to_datetime(df["datetime"])

groups = set()
suffixes = ["_min", "_max", "_mean"]
head = ["evaluation_score"]
for col in df.columns:
    group = re.sub("|".join([s + "$" for s in suffixes]), "", col)
    if group not in [col, *head]:
        groups.add(group)
groups = head + sorted(groups)

for group in groups:
    st.write("## " + group)
    cols = [group + s for s in suffixes]
    st.line_chart(df[["datetime", *cols]], x="datetime", x_label="", y=cols, y_label="", color=["#f55", "#55f", "#5f5"])
