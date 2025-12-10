import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange

# ----------------- 页面基础设置 -----------------
st.set_page_config(page_title="量化信号面板", layout="wide")
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

# ----------------- 量化核心逻辑 -----------------

def get_data(symbol: str):
    df = yf.download(symbol, period="3y", interval="1d")
    return df.dropna()


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # MACD
    macd = MACD(df["Close"])
    df["macd"] = macd.macd_diff()
    # RSI
    df["rsi"] = RSIIndicator(df["Close"]).rsi()
    # EMA 趋势
    df["ema8"] = EMAIndicator(df["Close"], window=8).ema_indicator()
    df["ema21"] = EMAIndicator(df["Close"], window=21).ema_indicator()
    # ATR 波动率
    atr = AverageTrueRange(df["High"], df["Low"], df["Close"])
    df["atr"] = atr.average_true_range()
    # OBV 资金
    obv = OnBalanceVolumeIndicator(df["Close"], df["Volume"])
    df["obv"] = obv.on_balance_volume()
    # 均值
    df["atr20"] = df["atr"].rolling(20).mean()
    df["obv20"] = df["obv"].rolling(20).mean()
    df["vol20"] = df["Volume"].rolling(20).mean()
    return df.dropna()


def indicator_status(latest: pd.Series):
    def status(val, high, low):
        if val > high:
            return "bull"
        elif val < low:
            return "bear"
        else:
            return "neutral"

    macd_status = "bull" if latest["macd"] > 0 else "bear"
    vol_status = status(latest["Volume"], latest["vol20"] * 1.1, latest["vol20"] * 0.9)
    rsi_status = "bull" if latest["rsi"] >= 60 else "bear" if latest["rsi"] <= 40 else "neutral"
    atr_status = status(latest["atr"], latest["atr20"] * 1.1, latest["atr20"] * 0.9)
    obv_status = status(latest["obv"], latest["obv20"] * 1.05, latest["obv20"] * 0.95)

    indicators = [
        {"name": "MACD 多头/空头", "status": macd_status},
        {"name": "成交量相对20日均量", "status": vol_status},
        {"name": "RSI 区间", "status": rsi_status},
        {"name": "ATR 波动率", "status": atr_status},
        {"name": "OBV 资金趋势", "status": obv_status},
    ]
    score = sum(1 for i in indicators if i["status"] == "bull")
    return indicators, score


def backtest(df: pd.DataFrame, days: int = 7, min_signals: int = 3):
    wins, total, rets = 0, 0, []
    for i in range(len(df) - days):
        latest = df.iloc[i]
        inds, score = indicator_status(latest)
        if score >= min_signals:
            total += 1
            future_ret = df["Close"].iloc[i + days] / df["Close"].iloc[i] - 1.0
            rets.append(future_ret)
            if future_ret > 0:
                wins += 1
    if total == 0:
        return 0.0, 0.0
    return wins / total, float(np.mean(rets))


def prob_class(p: float):
    if p >= 0.65:
        return "prob-good"
    if p >= 0.45:
        return "prob-mid"
    return "prob-bad"


# ----------------- Streamlit UI -----------------

st.title("📈 量化技术信号面板")

# 初始化自选列表：QQQ + 七姐妹
