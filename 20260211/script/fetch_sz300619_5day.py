"""金银河(SZ300619) 近5日分钟级K线数据获取"""
import baostock as bs
import pandas as pd
from datetime import datetime, timedelta

bs.login()

end_date = datetime(2026, 2, 11)
start_date = end_date - timedelta(days=10)  # 多取几天确保覆盖5个交易日

for freq in ['5', '15', '30', '60']:
    rs = bs.query_history_k_data_plus(
        "sz.300619",
        "date,time,open,high,low,close,volume,amount",
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        frequency=freq, adjustflag="2"
    )
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    if not df.empty:
        df['volume'] = pd.to_numeric(df['volume'])
        df['amount'] = pd.to_numeric(df['amount'])
        df['close'] = pd.to_numeric(df['close'])
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])

        print(f"\n{'='*60}")
        print(f"金银河(300619) {freq}分钟K线 最近5个交易日")
        print(f"{'='*60}")

        # 按日统计
        df['trade_date'] = df['date']
        for date, gdf in df.groupby('trade_date'):
            daily_vol = gdf['volume'].sum()
            daily_amt = gdf['amount'].sum()
            print(f"\n--- {date} | 成交量:{daily_vol:,.0f}股 | 成交额:{daily_amt/1e8:.2f}亿 ---")

            if freq == '30':
                # 30分钟级别打印详细数据
                for _, r in gdf.iterrows():
                    t = r['time'][8:12]  # HHmm
                    t_fmt = f"{t[:2]}:{t[2:]}"
                    print(f"  {t_fmt} O:{r['open']:.2f} H:{r['high']:.2f} L:{r['low']:.2f} C:{r['close']:.2f} V:{r['volume']:>10,.0f} A:{r['amount']/1e4:>8,.0f}万")

            if freq == '5':
                # 5分钟级别只打印开盘前4根(20分钟)和尾盘2根
                morning = gdf.head(4)
                closing = gdf.tail(2)
                print("  [开盘20分钟]")
                for _, r in morning.iterrows():
                    t = r['time'][8:12]
                    t_fmt = f"{t[:2]}:{t[2:]}"
                    print(f"    {t_fmt} O:{r['open']:.2f} C:{r['close']:.2f} V:{r['volume']:>10,.0f}")
                print("  [尾盘10分钟]")
                for _, r in closing.iterrows():
                    t = r['time'][8:12]
                    t_fmt = f"{t[:2]}:{t[2:]}"
                    print(f"    {t_fmt} O:{r['open']:.2f} C:{r['close']:.2f} V:{r['volume']:>10,.0f}")

bs.logout()
