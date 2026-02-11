"""粤万年青 SZ301111 近5个交易日 30分钟K线"""
import baostock as bs
import pandas as pd

CODE, NAME = "sz.301111", "粤万年青"
end_date = "2026-02-11"
start_date = "2026-01-30"

lg = bs.login()
rs = bs.query_history_k_data_plus(
    CODE,
    "date,time,open,high,low,close,volume,amount",
    start_date=start_date, end_date=end_date,
    frequency="30", adjustflag="2"
)
rows = []
while rs.error_code == '0' and rs.next():
    rows.append(rs.get_row_data())
bs.logout()

df = pd.DataFrame(rows, columns=rs.fields)
if df.empty:
    print("无数据")
    raise SystemExit(1)

for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
    df[col] = df[col].astype(float)

print(f"{NAME} ({CODE}) 近5个交易日 30分钟K线\n")
for date, gdf in df.groupby('date'):
    o, c = gdf['open'].iloc[0], gdf['close'].iloc[-1]
    h, l = gdf['high'].max(), gdf['low'].min()
    vol, amt = gdf['volume'].sum(), gdf['amount'].sum()
    chg = (c - o) / o * 100
    m_vol = gdf[gdf['time'].str[8:12] <= '1130']['volume'].sum()
    a_vol = gdf['volume'].sum() - m_vol
    print(f"{date}: 开{o:.2f} 高{h:.2f} 低{l:.2f} 收{c:.2f} 涨跌{chg:+.2f}% 量{vol/1e4:.0f}万手 额{amt/1e8:.2f}亿 上午量占{m_vol/(m_vol+a_vol)*100:.0f}%")
    print("  30min:", " | ".join([f"{r['time'][8:12]} C:{r['close']:.2f}" for _, r in gdf.iterrows()]))
