"""
Parses a log file and returns a list of dictionaries representing each log entry.

There are three types of log entries:
1. Sandbox logs: (dictionary with keys: "sandboxLog", "lambdaLog", "timestamp")
    {
    "sandboxLog": "",
    "lambdaLog": "traderData: \nObservations: (plainValueObservations: {}, conversionObservations: {})\nAcceptable price : 10\nBuy Order depth : 3, Sell order depth : 1\nSELL 1x 2028\nAcceptable price : 10\nBuy Order depth : 3, Sell order depth : 2\nSELL 1x 10002",
    "timestamp": 0
    }
2. Activities log: (csv-like format)
    day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss
    -1;0;RAINFOREST_RESIN;10002;1;9996;2;9995;29;10004;2;10005;29;;;10003.0;0.0
    -1;0;KELP;2028;1;2026;2;2025;29;2029;31;;;;;2028.5;0.0
    -1;100;KELP;2025;24;;;;;2028;2;2029;22;;;2026.5;1.157470703125
    -1;100;RAINFOREST_RESIN;9996;2;9995;22;;;10004;2;10005;22;;;10000.0;2.0
3. Trade History: (list of dictionaries with keys: "timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity")
    [
    {
        "timestamp": 0,
        "buyer": "",
        "seller": "SUBMISSION",
        "symbol": "KELP",
        "currency": "SEASHELLS",
        "price": 2028,
        "quantity": 1
    },
"""

import json
import re
from io import StringIO
from typing import Any, Dict, List

import pandas as pd


def parse_json_every_n_lines(lines, n=5):
    parsed = []
    for i in range(0, len(lines), n):
        chunk = lines[i : i + n]
        chunk_str = "\n".join(chunk)
        try:
            obj = json.loads(chunk_str)
            parsed.append(obj)
        except json.JSONDecodeError as e:
            print(f"fail at lines {i}-{i+n}: {e}")
    return parsed


with open("example.log", "r", encoding="utf-8") as f:
    content = f.read()

# 抽出 Sandbox Logs 部分
sandbox_section = content.split("Sandbox logs:")[1].split("Activities log:")[0].strip()
lines = [line.strip() for line in sandbox_section.strip().splitlines() if line.strip()]
grouped = []
current = []
for line in lines:
    if line.startswith('"lambdaLog'):
        print("Skipping lambdaLog line")
    current.append(line)
    if line == "}":
        grouped.append("\n".join(current))
        current = []
# 在每个对象后加逗号，最后一个不加
json_objects_with_commas = ",\n".join(grouped)
# 包裹成 JSON 数组
wrapped_json = "[\n" + json_objects_with_commas + "\n]"
# sandbox_lines = [line.strip() for line in sandbox_section.splitlines()]
# sandbox_logs = parse_json_every_n_lines(sandbox_lines, n=5)
# # TODO: 解析内部
# sandbox_list = "[\n" + sandbox_section + "\n]"
# print(sandbox_list)
# sandbox_loaded = json.loads(sandbox_list)
sandbox_logs = json.loads(wrapped_json)
print(f"提取了 {len(sandbox_logs)} 条 sandbox logs", sandbox_logs[0].items())

# 抽出 Activities Logs 部分
activities_section = (
    content.split("Activities log:")[1].split("Trade History:")[0].strip()
)
activities_df = pd.read_csv(StringIO(activities_section), sep=";")
print(activities_df.head(), activities_df.shape, activities_df.columns)

# 抽出 Trade History 部分
trade_history_section = content.split("Trade History:")[1].strip()
trade_history = json.loads(trade_history_section)
print(f"提取了 {len(trade_history)} 条 trade history", trade_history[0].items())
