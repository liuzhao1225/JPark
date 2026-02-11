"""获取中文在线/博纳影业/光线传媒/捷成股份/每日互动 近5个交易日分钟级K线数据"""
import baostock as bs
import pandas as pd
from datetime import datetime, timedelta

STOCKS = [
    ("sz.300364", "中文在线"),
    ("sz.001330", "博纳影业"),
    ("sz.300251", "光线传媒"),
    ("sz.300182", "捷成股份"),
    ("sz.300766", "每日互动"),
]

# 近5个交易日范围（留buffer取10天自然日）
end_date = "2026-02-11"
start_date = "2026-01-30"

lg = bs.login()
print(f"login: {lg.error_code} {lg.error_msg}")

for code, name in STOCKS:
    print(f"\n{'='*60}")
    print(f"  {name} ({code}) - 30分钟K线 近5个交易日")
    print(f"{'='*60}")
    
    rs = bs.query_history_k_data_plus(
        code,
        "date,time,open,high,low,close,volume,amount",
        start_date=start_date, end_date=end_date,
        frequency="30", adjustflag="2"
    )
    
    rows = []
    while (rs.error_code == '0') and rs.next():
        rows.append(rs.get_row_data())
    
    df = pd.DataFrame(rows, columns=rs.fields)
    if df.empty:
        print("  无数据")
        continue
    
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['amount'] = df['amount'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    
    # 按日分组输出摘要
    for date, gdf in df.groupby('date'):
        o = gdf['open'].iloc[0]
        c = gdf['close'].iloc[-1]
        h = gdf['high'].max()
        l = gdf['low'].min()
        vol = gdf['volume'].sum()
        amt = gdf['amount'].sum()
        chg = (c - o) / o * 100
        
        # 分时量能分布
        morning = gdf[gdf['time'].apply(lambda x: x[8:12]) <= '1130']
        afternoon = gdf[gdf['time'].apply(lambda x: x[8:12]) > '1130']
        m_vol = morning['volume'].sum()
        a_vol = afternoon['volume'].sum()
        
        # 尾盘30分钟
        tail = gdf.iloc[-1:] if len(gdf) > 0 else gdf
        tail_vol = tail['volume'].sum()
        
        print(f"\n  {date}: 开{o:.2f} 高{h:.2f} 低{l:.2f} 收{c:.2f} 涨幅{chg:+.2f}%")
        print(f"    成交量:{vol/10000:.0f}万手 成交额:{amt/100000000:.2f}亿")
        print(f"    上午量:{m_vol/10000:.0f}万手 下午量:{a_vol/10000:.0f}万手 上午占比:{m_vol/(m_vol+a_vol)*100:.1f}%")
        
        # 30分钟K线明细
        print(f"    30min明细:")
        for _, row in gdf.iterrows():
            t = row['time'][8:12]
            bar_chg = (row['close'] - row['open']) / row['open'] * 100 if row['open'] > 0 else 0
            print(f"      {t} O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f} {bar_chg:+.2f}% Vol:{row['volume']/10000:.0f}万")

bs.logout()
print("\n完成!")
