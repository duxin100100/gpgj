# web_traader.py
# 量化技术信号面板（回测 + 信号说明）

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============ 基础配置 ============

st.set_page_config(
    page_title="量化技术信号面板",
    page_icon="📈",
    layout="wide",
)

# 一点简单的暗色系样式
st.markdown(
    """
    <style>
    body, .main {
        background-color: #05060a;
        color: #f5f5f5;
        font-family: -apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Icons","PingFang SC","Helvetica Neue",Arial,sans-serif;
    }
    .stock-card {
        background: #101119;
        border-radius: 18px;
        padding: 18px 20px 14px 20px;
        margin-bottom: 18px;
        box-shadow: 0 0 20px rgba(0,0,0,0.35);
    }
    .stock-title {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .stock-price {
        font-size: 24px;
        font-weight: 600;
        margin-left: 8px;
    }
    .stock-chg-pos {
        font-size: 20px;
        margin-left: 8px;
        color: #21c25e;
    }
    .stock-chg-neg {
        font-size: 20px;
        margin-left: 8px;
        color: #ff4b4b;
    }
    .metric-line {
        font-size: 15px;
        line-height: 1.7;
    }
    .metric-label {
        font-weight: 500;
    }
    .divider {
        margin-top: 8px;
        margin-bottom: 10px;
        border-bottom: 1px dashed #333847;
    }
    .prob-line {
        font-size: 15px;
        line-height: 1.7;
    }
    .prob-highlight {
        color: #ffcf5a;
        font-weight: 600;
    }
    .prob-highlight-30 {
        color: #7ee787;
        font-weight: 600;
    }
    .signal-label {
        font-size: 15px;
        font-weight: 600;
    }
    .signal-adv-buy {
        color: #21c25e;
        font-weight: 700;
        font-size: 15px;
    }
    .signal-adv-sell {
        color: #ff4b4b;
        font-weight: 700;
        font-size: 15px;
    }
    .signal-adv-hold {
        color: #ffcf5a;
        font-weight: 700;
        font-size: 15px;
    }
    .arrow-btn > button {
        border-radius: 999px !important;
        padding: 2px 8px !important;
        margin-top: 2px;
    }
    .explain-box {
        margin-top: 8px;
        padding: 10px 12px;
        background: #141623;
        border-radius: 12px;
        font-size: 14px;
        line-height: 1.6;
        color: #e5e5e5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============ 工具函数 ============


@st.cache_data(show_spinner=False)
def load_price_data(symbol: str, years: int) -> pd.DataFrame:
    """从 yfinance 拉历史数据"""
    period = f"{years}y"
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df.dropna()
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算 MACD / RSI / ATR / OBV 等技术指标"""

    df = df.copy()

    # MACD
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # 成交量均量比
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20"]

    # RSI 14
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_gain = pd.Series(gain, index=df.index).rolling(14).mean()
    roll_loss = pd.Series(loss, index=df.index).rolling(14).mean()
    rs = roll_gain / roll_loss
    df["RSI"] = 100 - 100 / (1 + rs)

    # ATR 14
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()
    df["ATR_MA20"] = df["ATR"].rolling(20).mean()
    df["ATR_RATIO"] = df["ATR"] / df["ATR_MA20"]

    # OBV + 均值比
    obv = []
    last_obv = 0
    closes = df["Close"].values
    vols = df["Volume"].values
    for i in range(len(df)):
        if i == 0:
            obv.append(0)
        else:
            if closes[i] > closes[i - 1]:
                last_obv += vols[i]
            elif closes[i] < closes[i - 1]:
                last_obv -= vols[i]
        obv.append(last_obv)
    df["OBV"] = obv
    df["OBV_MA20"] = df["OBV"].rolling(20).mean()
    df["OBV_RATIO"] = df["OBV"] / df["OBV_MA20"]

    df = df.dropna()
    return df


def backtest_stats(returns: pd.Series):
    """根据一组收益率计算胜率、均盈、均亏、盈亏比等"""
    returns = returns.dropna()
    n = len(returns)
    if n == 0:
        return dict(
            count=0,
            win_count=0,
            prob=np.nan,
            avg_win=np.nan,
            avg_loss=np.nan,
            pf=np.nan,
        )

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_count = len(wins)
    prob = win_count / n * 100 if n > 0 else np.nan

    avg_win = wins.mean() * 100 if len(wins) > 0 else np.nan
    avg_loss = losses.mean() * 100 if len(losses) > 0 else np.nan

    if len(losses) > 0:
        pf = wins.sum() / abs(losses.sum())
    else:
        pf = np.nan

    return dict(
        count=n,
        win_count=win_count,
        prob=prob,
        avg_win=avg_win,
        avg_loss=avg_loss,
        pf=pf,
    )


def decide_advice(prob: float, pf: float):
    """
    根据胜率 + 盈亏比给出建议 & 强度 (1~5)
    buy / sell / hold 三档 + 强弱
    """

    if np.isnan(prob) or np.isnan(pf):
        return "观望", 1, "hold"

    # 基本分数
    score = 0

    # 胜率分
    if prob >= 55:
        score += 1
    if prob >= 60:
        score += 1
    if prob >= 70:
        score += 1

    # 盈亏比分
    if pf >= 1.2:
        score += 1
    if pf >= 1.6:
        score += 1

    # 判断方向
    if prob >= 55 and pf >= 1.1:
        kind = "buy"
        label = "建议买入"
    elif prob <= 45 and pf <= 0.9:
        kind = "sell"
        label = "建议卖出"
        # 方向反转评分（越低越想卖）
        score = max(1, 6 - score)
    else:
        kind = "hold"
        label = "观望"
        score = max(1, min(score, 3))

    intensity = int(np.clip(score, 1, 5))
    return label, intensity, kind


def dots(intensity: int, kind: str) -> str:
    """根据强度 + 类型画 5 个点"""
    if kind == "buy":
        on = "🟢"
    elif kind == "sell":
        on = "🔴"
    else:
        on = "🟡"
    off = "⚫"
    intensity = int(np.clip(intensity, 1, 5))
    return on * intensity + off * (5 - intensity)


def color_dot_by_ratio(current: float, target: float):
    """根据当前值 / 阈值给绿黄红"""
    if np.isnan(current):
        return "⚫"
    if current >= target:
        return "🟢"
    elif current >= target * 0.7:
        return "🟡"
    else:
        return "🔴"


def build_signal_explanation(row: dict, horizon: int, lookback_label: str) -> str:
    """生成 7日 / 30日 信号说明文字"""

    # 固定的“数据”描述，直接写死，和阈值保持一致
    macd_desc = "MACD 柱线＞0 的多头结构"
    vol_desc = "成交量 ≥ 20 日均量的 1.10 倍"
    rsi_desc = "RSI ≥ 60"
    atr_desc = "ATR ≥ 近 20 日均值的 1.10 倍"
    obv_desc = "OBV ≥ 近 20 日均值的 1.05 倍"

    if horizon == 7:
        N = row["count7"]
        W = row["win7"]
        prob = row["prob7"]
        avg_win = row["avg_win7"]
        avg_loss = row["avg_loss7"]
        pf = row["pf7"]
    else:
        N = row["count30"]
        W = row["win30"]
        prob = row["prob30"]
        avg_win = row["avg_win30"]
        avg_loss = row["avg_loss30"]
        pf = row["pf30"]

    if N == 0 or np.isnan(prob):
        return (
            f"在过去 **{lookback_label}** 中，没有找到足够多的历史样本满足当前这类技术组合，"
            f"暂时无法给出可靠的 {horizon} 日统计结果，请仅作参考。"
        )

    text = (
        f"在过去 **{lookback_label}**，当这只股票出现「MACD 偏多（{macd_desc}）、"
        f"量能放大（{vol_desc}）、RSI 偏强（{rsi_desc}）、波动放大（{atr_desc}）、"
        f"OBV 偏多（{obv_desc}）」这一类技术组合（以上 5 项指标中至少有 **3 项** "
        f"同时达到当前这次的强度区间）时，历史上共出现 **{N} 次**，其中有 **{W} 次** "
        f"在随后 **{horizon} 个交易日内上涨**。\n\n"
        f"{horizon} 日上涨概率约 **{prob:.1f}%**，上涨时平均涨 **{avg_win:.1f}%**，"
        f"下跌时平均跌 **{avg_loss:.1f}%**，整体盈亏比约 **{pf:.2f} 倍**。"
    )

    return text


def compute_stock_metrics(symbol: str, years: int):
    """对单只股票进行指标计算 + 回测"""
    df = load_price_data(symbol, years)
    if df.empty or len(df) < 80:
        return None

    df = compute_indicators(df)
    if df.empty or len(df) < 60:
        return None

    # 最新价 & 涨跌
    last = df.iloc[-1]
    prev_close = df["Close"].iloc[-2]
    price = float(last["Close"])
    pct_chg = (price - prev_close) / prev_close * 100

    # 阈值
    VOL_TARGET = 1.10
    RSI_TARGET = 60.0
    ATR_TARGET = 1.10
    OBV_TARGET = 1.05

    # 信号定义
    df["SIG_MACD"] = (df["MACD_HIST"] > 0).astype(int)
    df["SIG_VOL"] = (df["VOL_RATIO"] >= VOL_TARGET).astype(int)
    df["SIG_RSI"] = (df["RSI"] >= RSI_TARGET).astype(int)
    df["SIG_ATR"] = (df["ATR_RATIO"] >= ATR_TARGET).astype(int)
    df["SIG_OBV"] = (df["OBV_RATIO"] >= OBV_TARGET).astype(int)
    df["SCORE"] = (
        df["SIG_MACD"]
        + df["SIG_VOL"]
        + df["SIG_RSI"]
        + df["SIG_ATR"]
        + df["SIG_OBV"]
    )

    # 未来收益（7 / 30 日）
    df["RET_7"] = df["Close"].shift(-7) / df["Close"] - 1
    df["RET_30"] = df["Close"].shift(-30) / df["Close"] - 1

    mask_sig = df["SCORE"] >= 3

    stats7 = backtest_stats(df.loc[mask_sig, "RET_7"])
    stats30 = backtest_stats(df.loc[mask_sig, "RET_30"])

    # 建议
    adv7_label, adv7_intensity, adv7_kind = decide_advice(
        stats7["prob"], stats7["pf"]
    )
    adv30_label, adv30_intensity, adv30_kind = decide_advice(
        stats30["prob"], stats30["pf"]
    )

    row = dict(
        symbol=symbol.upper(),
        price=price,
        pct_chg=pct_chg,
        macd_hist=float(last["MACD_HIST"]),
        vol_ratio=float(last["VOL_RATIO"]),
        rsi=float(last["RSI"]),
        atr_ratio=float(last["ATR_RATIO"]),
        obv_ratio=float(last["OBV_RATIO"]),
        # 阈值
        vol_target=VOL_TARGET,
        rsi_target=RSI_TARGET,
        atr_target=ATR_TARGET,
        obv_target=OBV_TARGET,
        # 7 日回测
        prob7=stats7["prob"],
        avg_win7=stats7["avg_win"],
        avg_loss7=stats7["avg_loss"],
        pf7=stats7["pf"],
        count7=stats7["count"],
        win7=stats7["win_count"],
        # 30 日回测
        prob30=stats30["prob"],
        avg_win30=stats30["avg_win"],
        avg_loss30=stats30["avg_loss"],
        pf30=stats30["pf"],
        count30=stats30["count"],
        win30=stats30["win_count"],
        # 建议
        adv7_label=adv7_label,
        adv7_intensity=adv7_intensity,
        adv7_kind=adv7_kind,
        adv30_label=adv30_label,
        adv30_intensity=adv30_intensity,
        adv30_kind=adv30_kind,
    )

    return row


# ============ 页面结构 ============

st.markdown(
    '<div class="stock-title">📊 量化技术信号面板</div>',
    unsafe_allow_html=True,
)

# 默认展示：QQQ + 美股七姐妹
default_watchlist = ["QQQ", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA"]

if "watchlist" not in st.session_state:
    st.session_state.watchlist = default_watchlist.copy()

# 顶部：输入 + 添加/置顶
top_c1, top_c2 = st.columns([4, 1.1])
with top_c1:
    new_symbol = st.text_input(
        "",
        value="",
        max_chars=10,
        placeholder="输入股票代码添加到自选（例：TSLA）",
        label_visibility="collapsed",
    )
with top_c2:
    if st.button("➕ 添加/置顶", use_container_width=True):
        code = new_symbol.strip().upper()
        if code:
            if code in st.session_state.watchlist:
                st.session_state.watchlist.remove(code)
            st.session_state.watchlist.insert(0, code)

# 排序方式 & 回测区间
bar_c1, bar_c2 = st.columns([1.2, 1])
with bar_c1:
    sort_by = st.selectbox(
        "",
        ["默认顺序", "7日盈利概率", "30日盈利概率", "信号强度"],
        index=0,
        label_visibility="collapsed",
    )
with bar_c2:
    lookback_label = st.selectbox(
        "",
        ["1年", "2年", "3年", "5年", "10年"],
        index=2,
        label_visibility="collapsed",
    )

lookback_map = {"1年": 1, "2年": 2, "3年": 3, "5年": 5, "10年": 10}
years = lookback_map[lookback_label]

st.write("")  # 间隔

# ============ 计算所有股票数据 ============

rows = []
for sym in st.session_state.watchlist:
    try:
        metrics = compute_stock_metrics(sym, years)
        if metrics:
            rows.append(metrics)
    except Exception as e:
        st.warning(f"{sym} 数据获取失败：{e}")

# 排序
if sort_by == "7日盈利概率":
    rows = sorted(
        rows,
        key=lambda r: (0 if np.isnan(r["prob7"]) else r["prob7"]),
        reverse=True,
    )
elif sort_by == "30日盈利概率":
    rows = sorted(
        rows,
        key=lambda r: (0 if np.isnan(r["prob30"]) else r["prob30"]),
        reverse=True,
    )
elif sort_by == "信号强度":
    rows = sorted(
        rows,
        key=lambda r: r["adv7_intensity"] + r["adv30_intensity"],
        reverse=True,
    )

# ============ 渲染卡片 ============

if "explain_flags" not in st.session_state:
    st.session_state.explain_flags = {}  # key: f"{symbol}_{horizon}"

n_cols = 3
cols = st.columns(n_cols)

for idx, row in enumerate(rows):
    col = cols[idx % n_cols]
    with col:
        st.markdown('<div class="stock-card">', unsafe_allow_html=True)

        # 顶部：代码 + 价格 + 涨跌
        chg_cls = "stock-chg-pos" if row["pct_chg"] >= 0 else "stock-chg-neg"
        st.markdown(
            f"""
            <div>
              <span class="stock-title">{row['symbol']}</span>
              <span class="stock-price">${row['price']:.2f}</span>
              <span class="{chg_cls}">{row['pct_chg']:+.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 五个指标
        macd_color = "🟢" if row["macd_hist"] > 0 else "🔴"
        vol_dot = color_dot_by_ratio(row["vol_ratio"], row["vol_target"])
        rsi_dot = color_dot_by_ratio(row["rsi"], row["rsi_target"])
        atr_dot = color_dot_by_ratio(row["atr_ratio"], row["atr_target"])
        obv_dot = color_dot_by_ratio(row["obv_ratio"], row["obv_target"])

        st.markdown(
            f"""
            <div class="metric-line"><span class="metric-label">MACD 多头/空头</span>　{macd_color}</div>
            <div class="metric-line"><span class="metric-label">成交量相对20日均量</span> （{row['vol_target']:.2f} / {row['vol_ratio']:.2f}）　{vol_dot}</div>
            <div class="metric-line"><span class="metric-label">RSI 区间</span> （{row['rsi_target']:.1f} / {row['rsi']:.1f}）　{rsi_dot}</div>
            <div class="metric-line"><span class="metric-label">ATR 波动率</span> （{row['atr_target']:.2f} / {row['atr_ratio']:.2f}）　{atr_dot}</div>
            <div class="metric-line"><span class="metric-label">OBV 资金趋势</span> （{row['obv_target']:.2f} / {row['obv_ratio']:.2f}）　{obv_dot}</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # 7 / 30 日盈利概率
        if np.isnan(row["prob7"]):
            prob7_str = "暂无有效样本"
        else:
            prob7_str = (
                f"7日盈利概率 <span class='prob-highlight'>{row['prob7']:.1f}%</span>"
                f"（均盈 +{row['avg_win7']:.1f}% 均亏 {row['avg_loss7']:.1f}% 盈亏 {row['pf7']:.2f}）"
            )

        if np.isnan(row["prob30"]):
            prob30_str = "暂无有效样本"
        else:
            prob30_str = (
                f"30日盈利概率 <span class='prob-highlight-30'>{row['prob30']:.1f}%</span>"
                f"（均盈 +{row['avg_win30']:.1f}% 均亏 {row['avg_loss30']:.1f}% 盈亏 {row['pf30']:.2f}）"
            )

        st.markdown(
            f"<div class='prob-line'>{prob7_str}</div>"
            f"<div class='prob-line'>{prob30_str}</div>",
            unsafe_allow_html=True,
        )

        st.write("")

        # 7 日信号 + 右侧 >
        s7_c1, s7_c2, s7_c3, s7_c4 = st.columns([1.4, 1.8, 2.8, 0.7])
        with s7_c1:
            st.markdown('<span class="signal-label">7日信号</span>', unsafe_allow_html=True)
        with s7_c2:
            cls = (
                "signal-adv-buy"
                if row["adv7_kind"] == "buy"
                else "signal-adv-sell"
                if row["adv7_kind"] == "sell"
                else "signal-adv-hold"
            )
            st.markdown(
                f'<span class="{cls}">{row["adv7_label"]}</span>',
                unsafe_allow_html=True,
            )
        with s7_c3:
            st.markdown(dots(row["adv7_intensity"], row["adv7_kind"]))
        with s7_c4:
            key_btn7 = f"{row['symbol']}_7_btn"
            key_flag7 = f"{row['symbol']}_7_flag"
            if st.button("›", key=key_btn7):
                st.session_state.explain_flags[key_flag7] = not st.session_state.explain_flags.get(
                    key_flag7, False
                )

        # 30 日信号 + 右侧 >
        s30_c1, s30_c2, s30_c3, s30_c4 = st.columns([1.4, 1.8, 2.8, 0.7])
        with s30_c1:
            st.markdown('<span class="signal-label">30日信号</span>', unsafe_allow_html=True)
        with s30_c2:
            cls2 = (
                "signal-adv-buy"
                if row["adv30_kind"] == "buy"
                else "signal-adv-sell"
                if row["adv30_kind"] == "sell"
                else "signal-adv-hold"
            )
            st.markdown(
                f'<span class="{cls2}">{row["adv30_label"]}</span>',
                unsafe_allow_html=True,
            )
        with s30_c3:
            st.markdown(dots(row["adv30_intensity"], row["adv30_kind"]))
        with s30_c4:
            key_btn30 = f"{row['symbol']}_30_btn"
            key_flag30 = f"{row['symbol']}_30_flag"
            if st.button("›", key=key_btn30):
                st.session_state.explain_flags[key_flag30] = not st.session_state.explain_flags.get(
                    key_flag30, False
                )

        # 展开说明：7 日
        if st.session_state.explain_flags.get(key_flag7, False):
            txt7 = build_signal_explanation(row, 7, lookback_label)
            st.markdown(
                f'<div class="explain-box">{txt7}</div>',
                unsafe_allow_html=True,
            )

        # 展开说明：30 日
        if st.session_state.explain_flags.get(key_flag30, False):
            txt30 = build_signal_explanation(row, 30, lookback_label)
            st.markdown(
                f'<div class="explain-box">{txt30}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)  # end card
