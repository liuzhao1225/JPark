"""五股中短线技术分析: 中文在线/博纳影业/光线传媒/捷成股份/每日互动
拉取60+交易日日K线, 计算:
  MACD / KDJ / RSI / 布林带 / 均线系统 / 量价背离 / 支撑压力位 / 筹码集中度
输出中短线综合评分与操作建议

注: baostock日线数据通常延迟1日, 2/11当日数据需手动补充
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

# 2/11实时收盘数据(来自MCP), 补充baostock延迟
TODAY_DATA = {
    "sz.300364": {"date": "2026-02-11", "open": 42.50, "high": 43.80, "low": 38.95, "close": 39.80, "volume": 260150000, "amount": 10504000000, "turn": 35.71, "pctChg": -6.00},
    "sz.001330": {"date": "2026-02-11", "open": 12.94, "high": 13.72, "low": 12.88, "close": 13.40, "volume": 438560000, "amount": 5929000000, "turn": 31.91, "pctChg": 7.46},
    "sz.300251": {"date": "2026-02-11", "open": 24.56, "high": 29.88, "low": 24.56, "close": 26.86, "volume": 520880000, "amount": 13883000000, "turn": 17.76, "pctChg": 5.09},
    "sz.300182": {"date": "2026-02-11", "open": 8.30, "high": 9.46, "low": 8.30, "close": 8.85, "volume": 1208390000, "amount": 10672000000, "turn": 45.58, "pctChg": 4.73},
    "sz.300766": {"date": "2026-02-11", "open": 47.00, "high": 49.23, "low": 46.48, "close": 48.60, "volume": 65800000, "amount": 3129000000, "turn": 16.67, "pctChg": 2.75},
}

START = "2025-10-01"
END = "2026-02-11"


# ── 数据获取 ──────────────────────────────────────────────
def fetch_daily(code):
    rs = bs.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume,amount,turn,pctChg",
        start_date=START, end_date=END,
        frequency="d", adjustflag="2",
    )
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    for c in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


# ── 指标计算 ──────────────────────────────────────────────
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def calc_macd(df, fast=12, slow=26, signal=9):
    dif = ema(df["close"], fast) - ema(df["close"], slow)
    dea = ema(dif, signal)
    macd = 2 * (dif - dea)
    return dif, dea, macd


def calc_kdj(df, n=9, m1=3, m2=3):
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calc_rsi(df, periods=(6, 12, 24)):
    result = {}
    for p in periods:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        rs = gain / loss
        result[p] = 100 - 100 / (1 + rs)
    return result


def calc_boll(df, n=20, k=2):
    mid = df["close"].rolling(n).mean()
    std = df["close"].rolling(n).std()
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower


def calc_ma(df):
    mas = {}
    for n in [5, 10, 20, 60]:
        mas[n] = df["close"].rolling(n).mean()
    return mas


def calc_obv(df):
    signs = np.sign(df["close"].diff()).fillna(0)
    return (signs * df["volume"]).cumsum()


def calc_atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def calc_vwap_5d(df):
    """近5日量价重心 (baostock: volume=股, amount=元)"""
    tail = df.tail(5)
    if tail["volume"].sum() == 0:
        return tail["close"].iloc[-1]
    return tail["amount"].sum() / tail["volume"].sum()  # 元/股


def calc_vol_ratio(df):
    """量比: 今日成交量 / 近5日均量"""
    if len(df) < 6:
        return 1.0
    avg5 = df["volume"].iloc[-6:-1].mean()
    return df["volume"].iloc[-1] / avg5 if avg5 > 0 else 1.0


def calc_chip_concentration(df, n=20):
    """近n日筹码集中度: 用收盘价标准差/均值衡量"""
    tail = df.tail(n)
    mean = tail["close"].mean()
    std = tail["close"].std()
    return std / mean * 100 if mean > 0 else 0


def detect_vol_price_divergence(df):
    """量价背离检测: 近5日价涨量缩 or 价跌量增"""
    if len(df) < 6:
        return "数据不足"
    recent = df.tail(5)
    price_chg = recent["close"].iloc[-1] / recent["close"].iloc[0] - 1
    vol_chg = recent["volume"].iloc[-1] / recent["volume"].iloc[0] - 1
    if price_chg > 0.05 and vol_chg < -0.2:
        return "⚠ 顶背离(价涨量缩)"
    if price_chg < -0.05 and vol_chg > 0.2:
        return "✅ 底背离(价跌量增)"
    return "无明显背离"


def find_support_resistance(df):
    """基于近20日高低点找支撑/压力位"""
    tail = df.tail(20)
    cur = df["close"].iloc[-1]
    highs = tail["high"].nlargest(3).values
    lows = tail["low"].nsmallest(3).values
    resistance = [h for h in highs if h > cur]
    support = [l for l in lows if l < cur]
    return (
        round(min(resistance), 2) if resistance else round(highs[0], 2),
        round(max(support), 2) if support else round(lows[0], 2),
    )


# ── 综合评分 ──────────────────────────────────────────────
def score_stock(df):
    """中短线综合评分 (0~100), 越高越看多"""
    score = 50  # 基准分
    reasons = []
    cur = df["close"].iloc[-1]

    # 1. 均线多头排列 (+/-15)
    mas = calc_ma(df)
    ma_vals = {n: mas[n].iloc[-1] for n in [5, 10, 20, 60] if not np.isnan(mas[n].iloc[-1])}
    if len(ma_vals) == 4:
        vals = [ma_vals[5], ma_vals[10], ma_vals[20], ma_vals[60]]
        if vals == sorted(vals, reverse=True):
            score += 15
            reasons.append("均线完美多头排列 +15")
        elif cur > ma_vals[5] > ma_vals[10]:
            score += 8
            reasons.append("短期均线多头 +8")
        elif cur < ma_vals[5] < ma_vals[10]:
            score -= 10
            reasons.append("短期均线空头 -10")

    # 2. MACD (+/-10)
    dif, dea, macd = calc_macd(df)
    if dif.iloc[-1] > dea.iloc[-1] and macd.iloc[-1] > macd.iloc[-2]:
        score += 10
        reasons.append("MACD金叉且柱放大 +10")
    elif dif.iloc[-1] > dea.iloc[-1]:
        score += 5
        reasons.append("MACD金叉 +5")
    elif dif.iloc[-1] < dea.iloc[-1] and macd.iloc[-1] < macd.iloc[-2]:
        score -= 10
        reasons.append("MACD死叉且柱放大 -10")

    # 3. KDJ (+/-10)
    k, d, j = calc_kdj(df)
    if j.iloc[-1] > 80:
        score -= 8
        reasons.append(f"KDJ超买(J={j.iloc[-1]:.0f}) -8")
    elif j.iloc[-1] < 20:
        score += 8
        reasons.append(f"KDJ超卖(J={j.iloc[-1]:.0f}) +8")
    elif k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2]:
        score += 6
        reasons.append("KDJ金叉 +6")

    # 4. RSI (+/-8)
    rsi = calc_rsi(df)
    rsi6 = rsi[6].iloc[-1]
    if rsi6 > 80:
        score -= 8
        reasons.append(f"RSI6超买({rsi6:.0f}) -8")
    elif rsi6 < 30:
        score += 8
        reasons.append(f"RSI6超卖({rsi6:.0f}) +8")
    elif 40 < rsi6 < 60:
        score += 2
        reasons.append(f"RSI6中性({rsi6:.0f}) +2")

    # 5. 布林带位置 (+/-8)
    upper, mid, lower = calc_boll(df)
    if cur > upper.iloc[-1]:
        score -= 8
        reasons.append("突破布林上轨(超买) -8")
    elif cur < lower.iloc[-1]:
        score += 8
        reasons.append("跌破布林下轨(超卖) +8")
    elif cur < mid.iloc[-1]:
        score -= 3
        reasons.append("低于布林中轨 -3")

    # 6. 量比 (+/-5)
    vr = calc_vol_ratio(df)
    if vr > 3:
        score -= 3
        reasons.append(f"量比过大({vr:.1f}x) 警惕 -3")
    elif vr > 1.5:
        score += 3
        reasons.append(f"量比放大({vr:.1f}x) +3")
    elif vr < 0.5:
        score -= 3
        reasons.append(f"量比萎缩({vr:.1f}x) -3")

    # 7. 量价背离 (+/-5)
    div = detect_vol_price_divergence(df)
    if "顶背离" in div:
        score -= 5
        reasons.append("量价顶背离 -5")
    elif "底背离" in div:
        score += 5
        reasons.append("量价底背离 +5")

    # 8. 近5日涨幅过大的回调风险 (-5~-10)
    chg5 = (cur / df["close"].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
    if chg5 > 40:
        score -= 10
        reasons.append(f"5日涨幅{chg5:.0f}%过大 -10")
    elif chg5 > 25:
        score -= 5
        reasons.append(f"5日涨幅{chg5:.0f}%偏大 -5")
    elif chg5 < -15:
        score += 5
        reasons.append(f"5日跌幅{chg5:.0f}%超卖 +5")

    # 9. 换手率过高 (-5)
    if df["turn"].iloc[-1] > 30:
        score -= 5
        reasons.append(f"换手率{df['turn'].iloc[-1]:.0f}%过高 -5")

    score = max(0, min(100, score))
    return score, reasons


# ── 输出分析 ──────────────────────────────────────────────
def analyze(code, name, df):
    cur = df["close"].iloc[-1]
    prev = df["close"].iloc[-2]
    chg_today = (cur / prev - 1) * 100

    dif, dea, macd_bar = calc_macd(df)
    k, d, j = calc_kdj(df)
    rsi = calc_rsi(df)
    upper, mid, lower = calc_boll(df)
    mas = calc_ma(df)
    atr = calc_atr(df)
    obv = calc_obv(df)
    vwap5 = calc_vwap_5d(df)
    vr = calc_vol_ratio(df)
    chip = calc_chip_concentration(df)
    div = detect_vol_price_divergence(df)
    resistance, support = find_support_resistance(df)
    score, reasons = score_stock(df)

    # 近5/10/20日涨幅
    chg5 = (cur / df["close"].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
    chg10 = (cur / df["close"].iloc[-11] - 1) * 100 if len(df) >= 11 else 0
    chg20 = (cur / df["close"].iloc[-21] - 1) * 100 if len(df) >= 21 else 0

    # 判断趋势
    ma5v = mas[5].iloc[-1]
    ma10v = mas[10].iloc[-1]
    ma20v = mas[20].iloc[-1]
    ma60v = mas[60].iloc[-1]

    # OBV趋势
    obv_5 = obv.iloc[-1] - obv.iloc[-6] if len(df) >= 6 else 0
    obv_dir = "↑" if obv_5 > 0 else "↓"

    # ATR止损位
    atr_val = atr.iloc[-1]
    stop_loss = round(cur - 2 * atr_val, 2)

    # 操作建议
    if score >= 70:
        advice = "偏多 — 可逢低参与"
    elif score >= 55:
        advice = "中性偏多 — 持有观望"
    elif score >= 40:
        advice = "中性偏空 — 谨慎持有"
    else:
        advice = "偏空 — 回避或减仓"

    print(f"\n{'='*70}")
    print(f"  {name} ({code})  收盘:{cur:.2f}  今日:{chg_today:+.2f}%  评分:{score}/100")
    print(f"{'='*70}")

    print(f"\n  ┌─ 涨跌幅 ────────────────────────────────────────┐")
    print(f"  │ 5日:{chg5:+.1f}%  10日:{chg10:+.1f}%  20日:{chg20:+.1f}%          │")
    print(f"  └─────────────────────────────────────────────────┘")

    print(f"\n  ┌─ 均线系统 ──────────────────────────────────────┐")
    print(f"  │ MA5={ma5v:.2f}  MA10={ma10v:.2f}  MA20={ma20v:.2f}  MA60={ma60v:.2f}")
    pos = "上方" if cur > ma5v else "下方"
    print(f"  │ 当前价在MA5{pos}, ", end="")
    if ma5v > ma10v > ma20v > ma60v:
        print("均线完美多头排列")
    elif ma5v > ma10v > ma20v:
        print("短中期多头")
    elif ma5v < ma10v < ma20v:
        print("短中期空头")
    else:
        print("均线交织")
    print(f"  └─────────────────────────────────────────────────┘")

    print(f"\n  ┌─ MACD ─────────────────────────────────────────┐")
    print(f"  │ DIF={dif.iloc[-1]:.3f}  DEA={dea.iloc[-1]:.3f}  柱={macd_bar.iloc[-1]:.3f}")
    macd_status = "金叉" if dif.iloc[-1] > dea.iloc[-1] else "死叉"
    bar_dir = "放大" if abs(macd_bar.iloc[-1]) > abs(macd_bar.iloc[-2]) else "缩小"
    print(f"  │ 状态: {macd_status}, 柱体{bar_dir}")
    print(f"  └─────────────────────────────────────────────────┘")

    print(f"\n  ┌─ KDJ ──────────────────────────────────────────┐")
    print(f"  │ K={k.iloc[-1]:.1f}  D={d.iloc[-1]:.1f}  J={j.iloc[-1]:.1f}")
    if j.iloc[-1] > 80:
        print(f"  │ ⚠ J值超买区域")
    elif j.iloc[-1] < 20:
        print(f"  │ ✅ J值超卖区域")
    else:
        print(f"  │ J值正常区间")
    print(f"  └─────────────────────────────────────────────────┘")

    print(f"\n  ┌─ RSI ──────────────────────────────────────────┐")
    print(f"  │ RSI6={rsi[6].iloc[-1]:.1f}  RSI12={rsi[12].iloc[-1]:.1f}  RSI24={rsi[24].iloc[-1]:.1f}")
    if rsi[6].iloc[-1] > 70:
        print(f"  │ ⚠ 短期超买")
    elif rsi[6].iloc[-1] < 30:
        print(f"  │ ✅ 短期超卖")
    else:
        print(f"  │ 正常区间")
    print(f"  └─────────────────────────────────────────────────┘")

    print(f"\n  ┌─ 布林带 ───────────────────────────────────────┐")
    print(f"  │ 上轨={upper.iloc[-1]:.2f}  中轨={mid.iloc[-1]:.2f}  下轨={lower.iloc[-1]:.2f}")
    boll_pct = (cur - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) * 100
    print(f"  │ 当前位置: 带内{boll_pct:.0f}%处")
    print(f"  └─────────────────────────────────────────────────┘")

    print(f"\n  ┌─ 量能分析 ──────────────────────────────────────┐")
    print(f"  │ 量比={vr:.2f}x  换手={df['turn'].iloc[-1]:.1f}%")
    print(f"  │ 5日VWAP={vwap5:.2f}  OBV趋势={obv_dir}")
    print(f"  │ 量价关系: {div}")
    print(f"  └─────────────────────────────────────────────────┘")

    print(f"\n  ┌─ 波动与筹码 ────────────────────────────────────┐")
    print(f"  │ ATR(14)={atr_val:.2f}  日均波动约{atr_val/cur*100:.1f}%")
    print(f"  │ 20日筹码离散度={chip:.1f}%  {'高度分散' if chip > 15 else '较集中' if chip < 8 else '中等'}")
    print(f"  └─────────────────────────────────────────────────┘")

    print(f"\n  ┌─ 关键价位 ──────────────────────────────────────┐")
    print(f"  │ 压力位: {resistance:.2f}  支撑位: {support:.2f}")
    print(f"  │ ATR止损参考: {stop_loss:.2f} (当前价-2倍ATR)")
    print(f"  └─────────────────────────────────────────────────┘")

    print(f"\n  ┌─ 评分明细 ──────────────────────────────────────┐")
    for r in reasons:
        print(f"  │  {r}")
    print(f"  ├─────────────────────────────────────────────────┤")
    bar = "█" * (score // 5) + "░" * (20 - score // 5)
    print(f"  │  综合评分: [{bar}] {score}/100")
    print(f"  │  操作建议: {advice}")
    print(f"  └─────────────────────────────────────────────────┘")

    return {
        "name": name, "code": code, "price": cur, "score": score,
        "chg5": chg5, "advice": advice,
        "resistance": resistance, "support": support, "stop_loss": stop_loss,
    }


# ── 主流程 ────────────────────────────────────────────────
lg = bs.login()
print(f"login: {lg.error_code} {lg.error_msg}")

results = []
for code, name in STOCKS:
    df = fetch_daily(code)
    if df.empty:
        print(f"\n{name}: 无数据, 跳过")
        continue
    # 补充2/11实时数据
    if code in TODAY_DATA and (df["date"].iloc[-1] != "2026-02-11"):
        td = TODAY_DATA[code]
        new_row = pd.DataFrame([td])
        df = pd.concat([df, new_row], ignore_index=True)
        print(f"  [补充2/11实时数据] {name}: 收盘{td['close']}")
    results.append(analyze(code, name, df))

# ── 横向对比排名 ──────────────────────────────────────────
print(f"\n\n{'='*70}")
print(f"  五股横向对比 (按评分排名)")
print(f"{'='*70}")
print(f"  {'排名':<4} {'股票':<10} {'收盘':>8} {'5日涨幅':>8} {'评分':>6} {'建议':<20}")
print(f"  {'-'*62}")

for i, r in enumerate(sorted(results, key=lambda x: x["score"], reverse=True), 1):
    print(f"  {i:<4} {r['name']:<10} {r['price']:>8.2f} {r['chg5']:>+7.1f}% {r['score']:>5}/100 {r['advice']}")

print(f"\n  关键价位参考:")
for r in sorted(results, key=lambda x: x["score"], reverse=True):
    print(f"    {r['name']}: 压力{r['resistance']:.2f} / 支撑{r['support']:.2f} / 止损{r['stop_loss']:.2f}")

bs.logout()
print("\n完成!")
