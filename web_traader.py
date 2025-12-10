import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ============ 页面基础设置 ============
st.set_page_config(page_title="量化技术信号面板", layout="wide")
st.markdown(
    """
    <style>
    body { background: #05060a; }
    .main { background: #05060a; }
    .card {
        background: #14151d;
        border-radius: 16px;
        padding: 14px 16px 10px;
        border: 1px solid #262736;
        box-shadow: 0 18px 40px rgba(0,0,0,0.45);
        margin-bottom: 12px;
        color: #f5f5f7;
        font-size: 13px;
    }
    .symbol-line {
        display: flex;
        align-items: baseline;
        gap: 8px;
        font-size: 16px;
        font-weight: 600;
    }
    .price {
        font-size: 14px;
        font-weight: 600;
        color: #fefefe;
        margin-top: 2px;
    }
    .change-up { color: #4ade80; font-size: 12px; font-weight: 500; }
    .change-down { color: #fb7185; font-size: 12px; font-weight: 500; }
    .dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        display: inline-block;
        margin-left: 6px;
    }
    .dot-bull { background: #4ade80; }
    .dot-neutral { background: #facc15; }
    .dot-bear { background: #fb7185; }
    .label { color: #9ca3af; }
    .prob-good { color: #4ade80; font-weight:600; }
    .prob-mid { color: #facc15; font-weight:600; }
    .prob-bad { color: #fb7185; font-weight:600; }
    .score { font-size: 11px; color: #9ca3af; margin-top: 4px; }
    .score span { color: #4ade80; margin-left: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 量化技术信号面板")


# ============ 指标计算（不用 ta，全部自己算）===========

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def macd_hist(close):
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    return macd_line - signal

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()

def obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["macd_hist"] = macd_hist(df["Close"])
    df["rsi"] = rsi(df["Close"])
    df["atr"] = atr(df["High"], df["Low"], df["Close"])
    df["obv"] = obv(df["Close"], df["Volume"])

    df["vol20"] = df["Volume"].rolling(20).mean()
    df["atr20"] = df["atr"].rolling(20).mean()
    df["obv20"] = df["obv"].rolling(20).mean()

    # 信号列（0/1），用于回测时快速判断
    df["sig_macd"] = (df["macd_hist"] > 0).astype(int)
    df["sig_vol"] = (df["Volume"] > df["vol20"] * 1.1).astype(int)
    df["sig_rsi"] = (df["rsi"] >= 60).astype(int)
    df["sig_atr"] = (df["atr"] > df["atr20"] * 1.1).astype(int)
    df["sig_obv"] = (df["obv"] > df["obv20"] * 1.05).astype(int)
    df["score"] = (
        df["sig_macd"]
        + df["sig_vol"]
        + df["sig_rsi"]
        + df["sig_atr"]
        + df["sig_obv"]
    )

    return df.dropna()


def indicator_status_from_row(row: pd.Series):
    # 使用已经算好的信号列 + 均线，保证都是标量，不会再有 Series 冲突
    indicators = []

    macd_status = "bull" if row["sig_macd"] == 1 else "bear"
    indicators.append({"name": "MACD 多头/空头", "status": macd_status})

    if row["Volume"] > row["vol20"] * 1.1:
        vol_status = "bull"
    elif row["Volume"] < row["vol20"] * 0.9:
        vol_status = "bear"
    else:
        vol_status = "neutral"
    indicators.append({"name": "成交量相对20日均量", "status": vol_status})

    if row["rsi"] >= 60:
        rsi_status = "bull"
    elif row["rsi"] <= 40:
        rsi_status = "bear"
    else:
        rsi_status = "neutral"
    indicators.append({"name": "RSI 区间", "status": rsi_status})

    if row["atr"] > row["atr20"] * 1.1:
        atr_status = "bull"
    elif row["atr"] < row["atr20"] * 0.9:
        atr_status = "bear"
    else:
        atr_status = "neutral"
    indicators.append({"name": "ATR 波动率", "status": atr_status})

    if row["obv"] > row["obv20"] * 1.05:
        obv_status = "bull"
    elif row["obv"] < row["obv20"] * 0.95:
        obv_status = "bear"
    else:
        obv_status = "neutral"
    indicators.append({"name": "OBV 资金趋势", "status": obv_status})

    score = int(row["score"])
    return indicators, score


def backtest(df: pd.DataFrame, days: int = 7, min_score: int = 3):
    close = df["Close"].values
    scores = df["score"].values

    wins = 0
    total = 0
    rets = []

    for i in range(len(df) - days):
        if scores[i] >= min_score:
            total += 1
            r = close[i + days] / close[i] - 1.0
            rets.append(r)
            if r > 0:
                wins += 1

    if total == 0:
        return 0.0, 0.0
    return wins / total, float(np.mean(rets))


def prob_class(p):
    if p >= 0.65:
        return "prob-good"
    if p >= 0.45:
        return "prob-mid"
    return "prob-bad"


@st.cache_data(show_spinner=False)
def get_stock_metrics(symbol: str):
    df = yf.download(symbol, period="3y", interval="1d").dropna()
    if df.empty:
        raise ValueError("无数据")
    df = calc_indicators(df)
    latest = df.iloc[-1]
    prev_close = df["Close"].iloc[-2]
    change_pct = (latest["Close"] / prev_close - 1.0) * 100

    prob7, avg7 = backtest(df, 7)
    prob30, avg30 = backtest(df, 30)
    indicators, score = indicator_status_from_row(latest)

    return {
        "symbol": symbol,
        "price": float(latest["Close"]),
        "change": float(change_pct),
        "prob7": float(prob7),
        "prob30": float(prob30),
        "avg7": float(avg7),
        "avg30": float(avg30),
        "indicators": indicators,
        "score": int(score),
    }


# ============ Streamlit 交互层：平铺 QQQ + 七姐妹 ============

st.write("默认展示：QQQ + 美股七姐妹，可在上方添加/置顶其它股票。")

default_watchlist = ["QQQ", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA"]
if "watchlist" not in st.session_state:
    st.session_state.watchlist = default_watchlist.copy()

top_c1, top_c2, top_c3 = st.columns([2, 1.5, 1])

with top_c1:
    new_symbol = st.text_input("输入股票代码添加到自选（例：TSLA）", value="", max_chars=10)
with top_c2:
    add_btn = st.button("➕ 添加/置顶")
with top_c3:
    sort_by = st.selectbox(
        "排序方式",
        ["默认顺序", "涨跌幅", "7日盈利概率", "30日盈利概率", "信号强度"],
        index=0,
    )

if add_btn and new_symbol.strip():
    sym = new_symbol.strip().upper()
    if sym in st.session_state.watchlist:
        st.session_state.watchlist.remove(sym)
    st.session_state.watchlist.insert(0, sym)

rows = []

for sym in st.session_state.watchlist:
    try:
        with st.spinner(f"载入 {sym} ..."):
            metrics = get_stock_metrics(sym)
        rows.append(metrics)
    except Exception as e:
        st.warning(f"{sym} 加载失败：{e}")
        continue

# 排序
if sort_by == "涨跌幅":
    rows.sort(key=lambda x: x["change"], reverse=True)
elif sort_by == "7日盈利概率":
    rows.sort(key=lambda x: x["prob7"], reverse=True)
elif sort_by == "30日盈利概率":
    rows.sort(key=lambda x: x["prob30"], reverse=True)
elif sort_by == "信号强度":
    rows.sort(key=lambda x: x["score"], reverse=True)
# 默认顺序就按 watchlist 的顺序（上面 append 时已保证）

# 平铺卡片（4 列网格，更接近你原来的UI）
if not rows:
    st.info("暂无自选股票，请先在上方输入代码添加。")
else:
    cols_per_row = 4
    for i in range(0, len(rows), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, row in zip(cols, rows[i : i + cols_per_row]):
            with col:
                change_class = "change-up" if row["change"] >= 0 else "change-down"
                change_str = f"{row['change']:+.2f}%"
                prob7_text = f"{row['prob7']*100:.1f}%"
                prob30_text = f"{row['prob30']*100:.1f}%"
                prob7_class = prob_class(row["prob7"])
                prob30_class = prob_class(row["prob30"])

                indicators_html = ""
                for ind in row["indicators"]:
                    indicators_html += (
                        f"<div class='label'>{ind['name']}"
                        f"<span class='dot dot-{ind['status']}'></span></div>"
                    )

                html = f"""
                <div class="card">
                  <div class="symbol-line">
                    <span>{row['symbol']}</span>
                    <span class="{change_class}">{change_str}</span>
                  </div>
                  <div class="price">${row['price']:.2f}</div>
                  <div style="margin-top:6px;margin-bottom:6px">
                    {indicators_html}
                  </div>
                  <div style="border-bottom:1px dashed #262736;margin:6px 0 4px;"></div>
                  <div>
                    <div><span class="label">未来7日盈利概率</span>
                      <span class="{prob7_class}">{prob7_text}</span>
                    </div>
                    <div><span class="label">未来30日盈利概率</span>
                      <span class="{prob30_class}">{prob30_text}</span>
                    </div>
                  </div>
                  <div class="score">
                    信号强度：<span>{row['score']}/5</span>
                  </div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)

st.caption("数据来源：yfinance，回测区间约近3年，仅作个人量化研究，不构成投资建议。")
