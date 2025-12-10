import streamlit as st
import requests
import numpy as np

# ============ 页面基础设置 ============
st.set_page_config(page_title="量化技术信号面板", layout="wide")
st.markdown(
    """
    <style>
    body { background:#05060a; }
    .main { background:#05060a; padding-top:10px !important; }

    /* 标题缩小一倍 */
    h1 { font-size:26px !important; font-weight:700 !important; margin-bottom:6px !important; }

    .card {
        background:#14151d;
        border-radius:14px;
        padding:14px 16px;
        border:1px solid #262736;
        box-shadow:0 18px 36px rgba(0,0,0,0.45);
        color:#f5f5f7;
        font-size:13px;
        transition:0.15s;
        margin-bottom:18px;  /* 卡片之间留缝隙 */
    }
    .card:hover {
        transform:translateY(-3px);
        box-shadow:0 26px 48px rgba(0,0,0,0.6);
    }

    .symbol-line { display:flex; gap:8px; font-size:15px; font-weight:600; }
    .price { font-size:14px; margin-top:2px; }

    .change-up { color:#4ade80; font-size:12px; }
    .change-down { color:#fb7185; font-size:12px; }

    .dot { width:9px;height:9px;border-radius:50%;display:inline-block;margin-left:6px; }
    .dot-bull { background:#4ade80; }
    .dot-neutral { background:#facc15; }
    .dot-bear { background:#fb7185; }

    .label { color:#9ca3af; }
    .prob-good { color:#4ade80; font-weight:600; }
    .prob-mid { color:#facc15; font-weight:600; }
    .prob-bad { color:#fb7185; font-weight:600; }

    .score{
        font-size:12px;
        color:#9ca3af;
        margin-top:6px;
        display:flex;
        align-items:center;
        gap:6px;
    }
    .score-label{
        font-size:13px;
        font-weight:600;
        color:#e5e7eb;
    }
    .dot-score{
        width:10px;
        height:10px;
        border-radius:50%;
        display:inline-block;
        margin-right:2px;
    }
    .dot-score-on{
        background:#4ade80;
    }
    .dot-score-off{
        background:#4b5563;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 量化技术信号面板")

# ============ 时间框架定义 ============
# key -> (range, base_interval, agg)
# base_interval 是请求 Yahoo 的 interval
# agg 是在本地聚合的 bar 数（4 小时 = 4 根 1 小时）
TIMEFRAMES = {
    "1年":  ("1y",  "1d", 1),
    "2年":  ("2y",  "1d", 1),
    "3年":  ("3y",  "1d", 1),
    "5年":  ("5y",  "1d", 1),
    "10年": ("10y", "1d", 1),
    "3月/4小时": ("3mo", "1h", 4),
    "6月/4小时": ("6mo", "1h", 4),
    "3月/1小时": ("3mo", "1h", 1),
    "6月/1小时": ("6mo", "1h", 1),
}

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range}&interval={interval}"


def fetch_yahoo_ohlcv(symbol: str, tf_key: str):
    """根据时间框架获取并（必要时）聚合 OHLCV 数据"""
    if tf_key not in TIMEFRAMES:
        raise ValueError("未知时间框架")

    range_str, base_interval, agg = TIMEFRAMES[tf_key]
    url = YAHOO_URL.format(symbol=symbol, range=range_str, interval=base_interval)
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    data = resp.json()
    if "chart" not in data or not data["chart"].get("result"):
        raise ValueError("Yahoo 无返回数据")

    result = data["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]

    close = np.array(quote["close"], dtype="float64")
    high = np.array(quote["high"], dtype="float64")
    low = np.array(quote["low"], dtype="float64")
    volume = np.array(quote["volume"], dtype="float64")

    # 先按 close 的有效值做掩码
    mask = ~np.isnan(close)
    close = close[mask]
    high = high[mask]
    low = low[mask]
    volume = volume[mask]

    if len(close) < 80:
        raise ValueError("可用历史数据太少")

    # 如果需要 4 小时，就把 1 小时数据聚合
    if agg > 1:
        step = agg
        usable = (len(close) // step) * step
        close = close[-usable:]
        high = high[-usable:]
        low = low[-usable:]
        volume = volume[-usable:]

        close = close.reshape(-1, step)[:, -1]              # 最后一根收盘
        high = high.reshape(-1, step).max(axis=1)           # 区间最高
        low = low.reshape(-1, step).min(axis=1)             # 区间最低
        volume = volume.reshape(-1, step).sum(axis=1)       # 区间成交量

    return close, high, low, volume


# ============ numpy 技术指标 ============

def ema_np(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2 / (span + 1)
    ema = np.zeros_like(x, dtype=float)
    ema[0] = x[0]
    for i in range(1, len(x)):
        ema[i] = alpha * x[i] + (1 - alpha) * ema[i - 1]
    return ema


def macd_hist_np(close: np.ndarray) -> np.ndarray:
    ema12 = ema_np(close, 12)
    ema26 = ema_np(close, 26)
    macd_line = ema12 - ema26
    signal = ema_np(macd_line, 9)
    return macd_line - signal


def rsi_np(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_ema = np.zeros_like(gain)
    loss_ema = np.zeros_like(loss)

    alpha = 1 / period
    gain_ema[0] = gain[0]
    loss_ema[0] = loss[0]
    for i in range(1, len(gain)):
        gain_ema[i] = alpha * gain[i] + (1 - alpha) * gain_ema[i - 1]
        loss_ema[i] = alpha * loss[i] + (1 - alpha) * loss_ema[i - 1]

    rs = gain_ema / (loss_ema + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    alpha = 1 / period
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return atr


def rolling_mean_np(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return np.full_like(x, x.mean(), dtype=float)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    head = np.full(window - 1, ma[0])
    return np.concatenate([head, ma])


def obv_np(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    direction = np.sign(np.diff(close, prepend=close[0]))
    return np.cumsum(direction * volume)


# ============ 回测统计 ============

def backtest_with_stats(close: np.ndarray, score: np.ndarray, days: int, min_score: int = 3):
    """
    返回：
      胜率、平均收益、信号次数、最大回撤、盈亏比、盈利次数
    """
    if len(close) <= days:
        return 0.0, 0.0, 0, 0.0, 0.0, 0

    idx = np.where(score[:-days] >= min_score)[0]
    if len(idx) == 0:
        return 0.0, 0.0, 0, 0.0, 0.0, 0

    rets = close[idx + days] / close[idx] - 1.0
    signals = len(rets)
    win_mask = rets > 0
    wins = int(win_mask.sum())
    win_rate = float(win_mask.mean())
    avg_ret = float(rets.mean())

    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1
    max_dd = float(dd.min())

    profit = rets[rets > 0].sum()
    loss = -rets[rets < 0].sum()
    if loss > 0:
        pf = float(profit / loss)
    else:
        pf = 0.0

    return win_rate, avg_ret, signals, max_dd, pf, wins


# ============ 计算单只股票 ============

def compute_stock_metrics(symbol: str, tf_key: str):
    close, high, low, volume = fetch_yahoo_ohlcv(symbol, tf_key=tf_key)

    macd_hist = macd_hist_np(close)
    rsi = rsi_np(close)
    atr = atr_np(high, low, close)
    obv = obv_np(close, volume)

    vol_ma20 = rolling_mean_np(volume, 20)
    atr_ma20 = rolling_mean_np(atr, 20)
    obv_ma20 = rolling_mean_np(obv, 20)

    sig_macd = (macd_hist > 0).astype(int)
    sig_vol = (volume > vol_ma20 * 1.1).astype(int)
    sig_rsi = (rsi >= 60).astype(int)
    sig_atr = (atr > atr_ma20 * 1.1).astype(int)
    sig_obv = (obv > obv_ma20 * 1.05).astype(int)

    score = sig_macd + sig_vol + sig_rsi + sig_atr + sig_obv

    # “7 天 / 30 天” 在小时级里其实就是 7 个 / 30 个 bar
    prob7, avg7, signals7, max_dd7, _, wins7 = backtest_with_stats(close, score, days=7)
    prob30, avg30, signals30, _, _, wins30 = backtest_with_stats(close, score, days=30)

    last_close = close[-1]
    prev_close = close[-2] if len(close) >= 2 else close[-1]
    change_pct = (last_close / prev_close - 1.0) * 100
    last_idx = -1

    indicators = []

    macd_status = "bull" if macd_hist[last_idx] > 0 else "bear"
    indicators.append({"name": "MACD 多头/空头", "status": macd_status})

    if volume[last_idx] > vol_ma20[last_idx] * 1.1:
        vol_status = "bull"
    elif volume[last_idx] < vol_ma20[last_idx] * 0.9:
        vol_status = "bear"
    else:
        vol_status = "neutral"
    indicators.append({"name": "成交量相对20日均量", "status": vol_status})

    if rsi[last_idx] >= 60:
        rsi_status = "bull"
    elif rsi[last_idx] <= 40:
        rsi_status = "bear"
    else:
        rsi_status = "neutral"
    indicators.append({"name": "RSI 区间", "status": rsi_status})

    if atr[last_idx] > atr_ma20[last_idx] * 1.1:
        atr_status = "bull"
    elif atr[last_idx] < atr_ma20[last_idx] * 0.9:
        atr_status = "bear"
    else:
        atr_status = "neutral"
    indicators.append({"name": "ATR 波动率", "status": atr_status})

    if obv[last_idx] > obv_ma20[last_idx] * 1.05:
        obv_status = "bull"
    elif obv[last_idx] < obv_ma20[last_idx] * 0.95:
        obv_status = "bear"
    else:
        obv_status = "neutral"
    indicators.append({"name": "OBV 资金趋势", "status": obv_status})

    return {
        "symbol": symbol,
        "price": float(last_close),
        "change": float(change_pct),
        "prob7": float(prob7),
        "prob30": float(prob30),
        "avg7": float(avg7),
        "avg30": float(avg30),
        "signals7": int(signals7),
        "wins7": int(wins7),
        "signals30": int(signals30),
        "wins30": int(wins30),
        "max_dd7": float(max_dd7),
        "indicators": indicators,
        "score": int(score[last_idx]),
    }


def prob_class(p):
    if p >= 0.65:
        return "prob-good"
    if p >= 0.45:
        return "prob-mid"
    return "prob-bad"


# version 加上 time frame key，强制新缓存
@st.cache_data(show_spinner=False)
def get_stock_metrics_cached(symbol: str, tf_key: str, version: int = 6):
    return compute_stock_metrics(symbol, tf_key=tf_key)


# ============ Streamlit 交互层 ============

default_watchlist = ["QQQ", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA"]
if "watchlist" not in st.session_state:
    st.session_state.watchlist = default_watchlist.copy()

# 输入框 / 添加按钮 / 排序 / 回测时间框架
top_c1, top_c2, top_c3, top_c4 = st.columns([2.4, 1.1, 1.1, 1.3])

with top_c1:
    new_symbol = st.text_input(
        "",
        value="",
        max_chars=10,
        placeholder="输入股票代码添加到自选（例：TSLA）",
        label_visibility="collapsed",
    )
with top_c2:
    add_btn = st.button("➕ 添加/置顶")
with top_c3:
    sort_by = st.selectbox(
        "",
        ["默认顺序", "涨跌幅", "7日盈利概率", "30日盈利概率", "信号强度"],
        index=0,
        label_visibility="collapsed",
    )
with top_c4:
    tf_label = st.selectbox(
        "",
        list(TIMEFRAMES.keys()),
        index=2,  # 默认 3 年 日线
        label_visibility="collapsed",
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
            metrics = get_stock_metrics_cached(sym, tf_key=tf_label)
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
# 默认顺序就用 watchlist 的顺序

# 平铺卡片
if not rows:
    st.info("暂无自选股票，请先在上方输入代码添加。")
else:
    cols_per_row = 4
    for i in range(0, len(rows), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, row in zip(cols, rows[i: i + cols_per_row]):
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

                signals7 = row.get("signals7", 0)
                wins7 = row.get("wins7", 0)
                signals30 = row.get("signals30", 0)
                wins30 = row.get("wins30", 0)

                # 信号强度五个点
                score_val = int(row.get("score", 0))
                filled = max(0, min(5, score_val))
                empty = 5 - filled
                dots_html = (
                    "<span class='dot-score dot-score-on'></span>" * filled
                    + "<span class='dot-score dot-score-off'></span>" * empty
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
                    <div>
                      <span class="label">未来7日盈利概率（{signals7}/{wins7}）</span>
                      <span class="{prob7_class}">{prob7_text}</span>
                    </div>
                    <div>
                      <span class="label">未来30日盈利概率（{signals30}/{wins30}）</span>
                      <span class="{prob30_class}">{prob30_text}</span>
                    </div>
                  </div>
                  <div class="score">
                    <span class="score-label">信号强度</span>
                    {dots_html}
                  </div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)

st.caption(
    "数据来源：Yahoo Finance HTTP 接口，时间框架基于右侧下拉选项（日线 / 小时线），"
    "回测窗口为该周期内的历史信号，仅作个人量化研究，不构成投资建议。"
)
