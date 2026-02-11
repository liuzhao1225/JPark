"""科大国创 60日日K：关键价位、量价、均线"""
import baostock as bs
import pandas as pd
import numpy as np

CODE, NAME = "sz.300520", "科大国创"
START, END = "2025-10-01", "2026-02-11"

lg = bs.login()
rs = bs.query_history_k_data_plus(
    CODE, "date,open,high,low,close,volume,amount,turn,pctChg",
    start_date=START, end_date=END, frequency="d", adjustflag="2"
)
rows = []
while rs.error_code == '0' and rs.next():
    rows.append(rs.get_row_data())
bs.logout()

df = pd.DataFrame(rows, columns=rs.fields)
for c in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["close"]).reset_index(drop=True)

# 均线
df["ma5"] = df["close"].rolling(5).mean()
df["ma10"] = df["close"].rolling(10).mean()
df["ma20"] = df["close"].rolling(20).mean()
df["ma60"] = df["close"].rolling(60).mean()

# 近期高低点
high_60, low_60 = df["high"].max(), df["low"].min()
idx_high = df["high"].idxmax()
idx_low = df["low"].idxmin()

print(f"{NAME} 日K {len(df)} 根 ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})\n")
print(f"区间最高: {high_60:.2f} ({df['date'].iloc[idx_high]})  区间最低: {low_60:.2f} ({df['date'].iloc[idx_low]})")
print(f"最新: 收{df['close'].iloc[-1]:.2f} 量{df['volume'].iloc[-1]/1e4:.0f}万手 额{df['amount'].iloc[-1]/1e8:.2f}亿 换手{df['turn'].iloc[-1]:.2f}%")
print(f"MA5/10/20/60: {df['ma5'].iloc[-1]:.2f} / {df['ma10'].iloc[-1]:.2f} / {df['ma20'].iloc[-1]:.2f} / {df['ma60'].iloc[-1]:.2f}")

# 近10日量价
print("\n近10日 日期 | 收 | 涨跌% | 量(万手) | 额(亿) | 换手%")
for _, r in df.tail(10).iterrows():
    print(f"  {r['date']} | {r['close']:.2f} | {r['pctChg']:+.2f}% | {r['volume']/1e4:.0f} | {r['amount']/1e8:.2f} | {r['turn']:.2f}")

# 前高前低（支撑压力）
recent = df.tail(30)
peaks = recent[recent["high"] == recent["high"].rolling(5, center=True).max()]["high"].tolist()
lows = recent[recent["low"] == recent["low"].rolling(5, center=True).min()]["low"].tolist()
print("\n近期波动区间(约): 压力", sorted(set(peaks), reverse=True)[:3], "支撑", sorted(set(lows))[:3])
