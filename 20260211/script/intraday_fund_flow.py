"""五股近5交易日 分钟级资金承接分析
拉取5min K线, 分析:
  1. 分时段量能分布(早盘/盘中/尾盘)
  2. 主动买卖力度(上涨bar成交 vs 下跌bar成交)
  3. 大单承接(高成交量bar的方向)
  4. 日内VWAP偏离
  5. 跳空缺口与回补
  6. 每日资金行为总结
"""
import baostock as bs
import pandas as pd
import numpy as np

STOCKS = [
    ("sz.300364", "中文在线"),
    ("sz.001330", "博纳影业"),
    ("sz.300251", "光线传媒"),
    ("sz.300182", "捷成股份"),
    ("sz.300766", "每日互动"),
]

START = "2026-02-04"
END = "2026-02-11"


def fetch_5min(code):
    rs = bs.query_history_k_data_plus(
        code, "date,time,open,high,low,close,volume,amount",
        start_date=START, end_date=END, frequency="5", adjustflag="2",
    )
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    df["time_hm"] = df["time"].apply(lambda x: x[8:12])  # HHMM
    return df


def time_slot(hm):
    """分时段: 竞价冲击/早盘/盘中/午后/尾盘"""
    if hm <= "0935":
        return "竞价冲击"
    if hm <= "1030":
        return "早盘"
    if hm <= "1130":
        return "盘中上午"
    if hm <= "1400":
        return "午后"
    if hm <= "1445":
        return "盘中下午"
    return "尾盘"


def analyze_day(gdf, date, prev_close=None):
    """分析单日分钟线"""
    o = gdf["open"].iloc[0]
    c = gdf["close"].iloc[-1]
    h = gdf["high"].max()
    l = gdf["low"].min()
    total_vol = gdf["volume"].sum()
    total_amt = gdf["amount"].sum()

    # 开盘缺口
    gap = 0
    gap_str = ""
    if prev_close and prev_close > 0:
        gap = (o / prev_close - 1) * 100
        if abs(gap) > 1:
            gap_str = f"{'跳空高开' if gap > 0 else '跳空低开'}{gap:+.1f}%"

    # VWAP
    vwap = total_amt / total_vol if total_vol > 0 else c

    # 分时段量能
    gdf = gdf.copy()
    gdf["slot"] = gdf["time_hm"].apply(time_slot)
    slot_stats = gdf.groupby("slot").agg(
        vol=("volume", "sum"), amt=("amount", "sum"), bars=("volume", "count")
    )

    # 上涨bar vs 下跌bar 的成交量(主动买卖力度)
    gdf["bar_dir"] = np.sign(gdf["close"] - gdf["open"])
    up_vol = gdf[gdf["bar_dir"] > 0]["volume"].sum()
    dn_vol = gdf[gdf["bar_dir"] < 0]["volume"].sum()
    flat_vol = gdf[gdf["bar_dir"] == 0]["volume"].sum()
    buy_ratio = up_vol / total_vol * 100 if total_vol > 0 else 50

    # 大单承接: 成交量 > 日均bar量*2 的bar分析
    avg_bar_vol = total_vol / len(gdf) if len(gdf) > 0 else 0
    big_bars = gdf[gdf["volume"] > avg_bar_vol * 2].copy()
    big_up = big_bars[big_bars["bar_dir"] > 0]["volume"].sum()
    big_dn = big_bars[big_bars["bar_dir"] < 0]["volume"].sum()

    # 收盘 vs VWAP
    vwap_dev = (c / vwap - 1) * 100 if vwap > 0 else 0

    # 尾盘(14:45-15:00)行为
    tail = gdf[gdf["time_hm"] > "1445"]
    tail_vol = tail["volume"].sum()
    tail_chg = 0
    if len(tail) > 0:
        tail_chg = (tail["close"].iloc[-1] / tail["open"].iloc[0] - 1) * 100

    # 盘中最大回撤(从日内高点)
    gdf["cum_high"] = gdf["high"].cummax()
    gdf["drawdown"] = (gdf["low"] / gdf["cum_high"] - 1) * 100
    max_dd = gdf["drawdown"].min()

    # 盘中最大反弹(从日内低点)
    gdf["cum_low"] = gdf["low"].cummin()
    gdf["rebound"] = (gdf["high"] / gdf["cum_low"] - 1) * 100
    max_rb = gdf["rebound"].max()

    return {
        "date": date, "open": o, "close": c, "high": h, "low": l,
        "total_vol": total_vol, "total_amt": total_amt,
        "gap": gap, "gap_str": gap_str, "vwap": vwap, "vwap_dev": vwap_dev,
        "buy_ratio": buy_ratio, "up_vol": up_vol, "dn_vol": dn_vol,
        "big_up": big_up, "big_dn": big_dn,
        "tail_vol": tail_vol, "tail_chg": tail_chg, "tail_vol_pct": tail_vol / total_vol * 100 if total_vol > 0 else 0,
        "max_dd": max_dd, "max_rb": max_rb,
        "slot_stats": slot_stats, "avg_bar_vol": avg_bar_vol,
    }


def print_day_analysis(r):
    chg = (r["close"] / r["open"] - 1) * 100
    print(f"\n  {r['date']}: 开{r['open']:.2f} → 收{r['close']:.2f} ({chg:+.1f}%)  高{r['high']:.2f} 低{r['low']:.2f}")
    if r["gap_str"]:
        print(f"    ▶ {r['gap_str']}")
    print(f"    成交额:{r['total_amt']/1e8:.1f}亿  VWAP={r['vwap']:.2f}  收盘偏离VWAP:{r['vwap_dev']:+.1f}%")
    print(f"    盘中最大回撤:{r['max_dd']:.1f}%  最大反弹:{r['max_rb']:.1f}%")

    # 主动买卖
    buy_bar = "█" * int(r["buy_ratio"] / 5)
    sell_bar = "░" * (20 - int(r["buy_ratio"] / 5))
    print(f"    主动买入量占比: [{buy_bar}{sell_bar}] {r['buy_ratio']:.0f}%", end="")
    if r["buy_ratio"] > 60:
        print("  ✅ 买盘主导")
    elif r["buy_ratio"] < 40:
        print("  ⚠ 卖盘主导")
    else:
        print("  多空均衡")

    # 大单方向
    big_total = r["big_up"] + r["big_dn"]
    if big_total > 0:
        big_buy_pct = r["big_up"] / big_total * 100
        print(f"    大单(>2倍均量): 买{r['big_up']/1e4:.0f}万 卖{r['big_dn']/1e4:.0f}万 → 买方占{big_buy_pct:.0f}%", end="")
        if big_buy_pct > 60:
            print(" ✅")
        elif big_buy_pct < 40:
            print(" ⚠")
        else:
            print("")

    # 尾盘
    print(f"    尾盘(14:45后): 量占{r['tail_vol_pct']:.1f}% 涨跌{r['tail_chg']:+.2f}%", end="")
    if r["tail_chg"] > 0.5 and r["tail_vol_pct"] > 10:
        print(" 尾盘拉升(承接)")
    elif r["tail_chg"] < -0.5 and r["tail_vol_pct"] > 10:
        print(" 尾盘杀跌(出货)")
    else:
        print("")

    # 分时段量能
    print(f"    分时段量能分布:")
    slot_order = ["竞价冲击", "早盘", "盘中上午", "午后", "盘中下午", "尾盘"]
    for slot in slot_order:
        if slot not in r["slot_stats"].index:
            continue
        sv = r["slot_stats"].loc[slot]
        pct = sv["vol"] / r["total_vol"] * 100
        bar = "▓" * max(1, int(pct / 3))
        print(f"      {slot:<6} {bar} {pct:.0f}% ({sv['vol']/1e4:.0f}万手)")


def judge_fund_behavior(days):
    """综合多日资金行为判断"""
    if len(days) < 2:
        return "数据不足"
    
    signals = []
    latest = days[-1]
    
    # 1. 买入量占比趋势
    buy_ratios = [d["buy_ratio"] for d in days]
    if buy_ratios[-1] > 55 and buy_ratios[-1] > np.mean(buy_ratios[:-1]):
        signals.append("✅ 主动买入力度增强")
    elif buy_ratios[-1] < 45 and buy_ratios[-1] < np.mean(buy_ratios[:-1]):
        signals.append("⚠ 主动卖出力度增强")
    
    # 2. VWAP偏离趋势
    if latest["vwap_dev"] > 1:
        signals.append("✅ 收盘在VWAP上方(资金承接)")
    elif latest["vwap_dev"] < -1:
        signals.append("⚠ 收盘在VWAP下方(资金流出)")
    
    # 3. 大单方向
    big_total = latest["big_up"] + latest["big_dn"]
    if big_total > 0:
        big_ratio = latest["big_up"] / big_total
        if big_ratio > 0.6:
            signals.append("✅ 大单以买入为主")
        elif big_ratio < 0.4:
            signals.append("⚠ 大单以卖出为主")
    
    # 4. 尾盘行为
    tail_chgs = [d["tail_chg"] for d in days[-3:]]
    if np.mean(tail_chgs) > 0.3:
        signals.append("✅ 近期尾盘持续拉升")
    elif np.mean(tail_chgs) < -0.3:
        signals.append("⚠ 近期尾盘持续杀跌")
    
    # 5. 量能变化
    vols = [d["total_vol"] for d in days]
    if len(vols) >= 3 and vols[-1] > vols[-2] > vols[-3]:
        signals.append("✅ 成交量持续放大")
    elif len(vols) >= 3 and vols[-1] < vols[-2] < vols[-3]:
        signals.append("量能持续萎缩")
    
    # 6. 回撤控制
    if latest["max_dd"] > -5:
        signals.append("✅ 盘中回撤可控(<5%)")
    elif latest["max_dd"] < -10:
        signals.append("⚠ 盘中回撤剧烈(>10%)")
    
    # 7. 缺口
    gaps = [d for d in days if abs(d["gap"]) > 2]
    unfilled = [d for d in gaps if d["gap"] > 0 and d["low"] > d["open"] * 0.98]
    if unfilled:
        signals.append(f"有{len(unfilled)}个未回补跳空缺口(支撑)")
    
    return signals


def overall_verdict(signals):
    bulls = sum(1 for s in signals if "✅" in s)
    bears = sum(1 for s in signals if "⚠" in s)
    if bulls >= 4:
        return "资金承接良好, 短线偏多"
    if bulls > bears:
        return "资金略偏积极, 可关注"
    if bears >= 4:
        return "资金明显流出, 短线回避"
    if bears > bulls:
        return "资金偏谨慎, 观望为主"
    return "多空胶着, 等待方向"


# ── 主流程 ──────────────────────────────────────────
lg = bs.login()
print(f"login: {lg.error_code} {lg.error_msg}")

all_results = {}

for code, name in STOCKS:
    print(f"\n\n{'='*70}")
    print(f"  {name} ({code}) - 5min K线资金承接分析")
    print(f"{'='*70}")

    df = fetch_5min(code)
    if df.empty:
        print("  无数据")
        continue

    dates = sorted(df["date"].unique())
    days = []
    prev_close = None

    for date in dates:
        gdf = df[df["date"] == date].copy()
        if gdf.empty:
            continue
        r = analyze_day(gdf, date, prev_close)
        days.append(r)
        print_day_analysis(r)
        prev_close = r["close"]

    # 综合判断
    signals = judge_fund_behavior(days)
    verdict = overall_verdict(signals)
    print(f"\n  ┌─ 资金行为综合判断 ─────────────────────────────┐")
    for s in signals:
        print(f"  │  {s}")
    print(f"  ├──────────────────────────────────────────────────┤")
    print(f"  │  结论: {verdict}")
    print(f"  └──────────────────────────────────────────────────┘")

    all_results[name] = {"signals": signals, "verdict": verdict, "days": days}

# ── 五股横向对比 ──────────────────────────────────────
print(f"\n\n{'='*70}")
print(f"  五股资金承接横向对比")
print(f"{'='*70}")
for name, r in all_results.items():
    bulls = sum(1 for s in r["signals"] if "✅" in s)
    bears = sum(1 for s in r["signals"] if "⚠" in s)
    last = r["days"][-1] if r["days"] else {}
    buy_r = last.get("buy_ratio", 0)
    vwap_d = last.get("vwap_dev", 0)
    print(f"\n  {name}:")
    print(f"    多头信号:{bulls}个  空头信号:{bears}个  最新日买入占比:{buy_r:.0f}%  VWAP偏离:{vwap_d:+.1f}%")
    print(f"    → {r['verdict']}")

bs.logout()
print("\n\n完成!")
