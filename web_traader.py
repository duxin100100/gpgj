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


# ============ 技术指标计算（纯 pandas 实现）===========

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def macd_diff(close):
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return hist

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
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


# ============ 指标状态 & 回测逻辑 ============

def indicator_status(latest, row_idx, df):
    def status(val, high, low):
        if val > high:
            return "bull"
        elif val < low:
            return "bear"
        else:
            return "neutral"

    macd_status = "bull" if latest["macd_diff"] > 0 else "bear"

    vol = latest["Volume"]
    vol_ma = latest["vol20"]
    vol_status = status(vol, vol_ma * 1.1, vol_ma * 0.9)

    r = latest["rsi"]
    if r >= 60:
        rsi_status = "bull"
    elif r <= 40:
        rsi_status = "bear"
    else:
        rsi_status = "neutral"

    atr_val = latest["atr"]
    atr_ma = latest["atr20"]
    atr_status = status(atr_val, atr_ma * 1.1, atr_ma * 0.9)

    obv_val = latest["obv"]
    obv_ma = latest["obv20"]
    obv_status = status(obv_val, obv_ma * 1.05, obv_ma * 0.95)

    indicators = [
        {"name": "MACD 多头/空头", "status": macd_status},
        {"name": "成交量相对20日均量", "status": vol_status},
        {"name": "RSI 区间", "status": rsi_status},
        {"name": "ATR 波动率", "status": atr_status},
        {"name": "OBV 资金趋势", "status": obv_status},
    ]
    score = sum(1 for x in indicators if x["status"] == "bull")
    return indicators, score


def backtest(df, days=7, min_signals=3):
    wins, total, rets = 0, 0, []
    for i in range(len(df) - days):
        latest = df.iloc[i]
        inds, score = indicator_status(latest, i, df)
        if score >= min_signals:
            total += 1
            future_ret = df["Close"].iloc[i + days] / df["Close"].iloc[i] - 1.0
            rets.append(future_ret)
            if future_ret > 0:
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


# ============ Streamlit 交互 ============

st.write("在下方输入美股代码（如：TSLA、AAPL、QQQ），点击查询。")

symbol = st.text_input("股票代码", value="TSLA", max_chars=10)
run_btn = st.button("开始计算信号 🚀")

if run_btn:
    sym = symbol.strip().upper()
    if not sym:
        st.warning("请输入有效的股票代码。")
    else:
        try:
            with st.spinner("正在加载数据并回测，请稍候……"):
                df = yf.download(sym, period="3y", interval="1d").dropna()
                if df.empty:
                    st.error("下载不到这个代码的数据，可能代码无效或被退市。")
                else:
                    # 计算指标
                    df["macd_diff"] = macd_diff(df["Close"])
                    df["rsi"] = rsi(df["Close"])
                    df["atr"] = atr(df["High"], df["Low"], df["Close"])
                    df["obv"] = obv(df["Close"], df["Volume"])
                    df["vol20"] = df["Volume"].rolling(20).mean()
                    df["atr20"] = df["atr"].rolling(20).mean()
                    df["obv20"] = df["obv"].rolling(20).mean()
                    df = df.dropna()

                    latest = df.iloc[-1]
                    prev_close = df["Close"].iloc[-2]
                    change_pct = (latest["Close"] / prev_close - 1.0) * 100

                    prob7, avg7 = backtest(df, 7)
                    prob30, avg30 = backtest(df, 30)

                    indicators, score = indicator_status(latest, len(df)-1, df)

            # ----- 渲染卡片 -----
            change_class = "change-up" if change_pct >= 0 else "change-down"
            change_str = f"{change_pct:+.2f}%"
            prob7_text = f"{prob7*100:.1f}%"
            prob30_text = f"{prob30*100:.1f}%"
            prob7_class = prob_class(prob7)
            prob30_class = prob_class(prob30)

            indicators_html = ""
            for ind in indicators:
                indicators_html += (
                    f"<div class='label'>{ind['name']}"
                    f"<span class='dot dot-{ind['status']}'></span></div>"
                )

            html = f"""
            <div class="card">
              <div class="symbol-line">
                <span>{sym}</span>
                <span class="{change_class}">{change_str}</span>
              </div>
              <div class="price">${latest['Close']:.2f}</div>
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
                信号强度：<span>{score}/5</span>
              </div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

        except Exception as e:
            # 如果 yfinance 在云端被封/限流，会在这里显示错误
            st.error(f"运行出错：{e}")

st.caption("数据来自 Yahoo Finance（yfinance），仅用于个人量化研究，不构成任何投资建议。")
