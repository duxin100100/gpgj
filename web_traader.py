import streamlit as st
import requests
import numpy as np

# ============ 页面基础设置 ============
st.set_page_config(page_title="回测信号面板", layout="wide")
st.markdown(
    """
    <style>
    body { background:#05060a; }
    .main { background:#05060a; padding-top:10px !important; }

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
        margin-bottom:18px;
    }
    .card:hover {
        transform:translateY(-3px);
        box-shadow:0 26px 48px rgba(0,0,0,0.6);
    }

    .symbol-line {
        display:flex;
        gap:10px;
        align-items:baseline;
        font-size:20px;
        margin-bottom:2px;
    }
    .symbol-code {
        font-weight:800;
    }
    .symbol-price {
        font-size:20px;
    }
    .change-up { color:#4ade80; font-size:16px; }
    .change-down { color:#fb7185; font-size:16px; }

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
        gap:8px;
    }
    .score-label{
        font-size:13px;
        font-weight:700;
        color:#e5e7eb;
        min-width:64px;
    }
    .dot-score{
        width:10px;
        height:10px;
        border-radius:50%;
        display:inline-block;
        margin-right:2px;
    }
    .dot-score-buy{ background:#4ade80; }
    .dot-score-hold{ background:#facc15; }
    .dot-score-sell{ background:#fb7185; }
    .dot-score-off{ background:#4b5563; }
    .advice-text{
        font-size:13px;
        font-weight:600;
    }
    .advice-buy{ color:#4ade80; }
    .advice-hold{ color:#facc15; }
    .advice-sell{ color:#fb7185; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 量化技术信号面板")

# ============ 回测配置（日线+4H+1H） ============
BACKTEST_CONFIG = {
    "1年":  {"range": "1y",  "interval": "1d", "steps_per_day": 1},
    "2年":  {"range": "2y",  "interval": "1d", "steps_per_day": 1},
    "3年":  {"range": "3y",  "interval": "1d", "steps_per_day": 1},
    "5年":  {"range": "5y",  "interval": "1d", "steps_per_day": 1},
    "10年": {"range": "10y", "interval": "1d", "steps_per_day": 1},
    "3月/4小时": {"range": "3mo", "interval": "4h", "steps_per_day": 6},
    "6月/4小时": {"range": "6mo", "interval": "4h", "steps_per_day": 6},
    "3月/1小时": {"range": "3mo", "interval": "1h", "steps_per_day": 24},
    "6月/1小时": {"range": "6mo", "interval": "1h", "steps_per_day": 24},
}

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range}&interval={interval}"


def fetch_yahoo_ohlcv(symbol: str, range_str: str, interval: str):
    url = YAHOO_URL.format(symbol=symbol, range=range_str, interval=interval)
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
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

    mask = ~np.isnan(close)
    close = close[mask]
    high = high[mask]
    low = low[mask]
    volume = volume[mask]

    if len(close) < 80:
        raise ValueError("可用历史数据太少")

    return close, high, low, volume


# ============ 技术指标 ============
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


# ============ 回测统计（带均盈均亏+PF） ============
def backtest_with_stats(close: np.ndarray, score: np.ndarray, steps: int, min_score: int = 3):
    """
    steps: 向前看的 bar 数
    返回：
      胜率、净平均收益、信号次数、最大回撤、盈利因子PF、盈利次数、平均盈利、平均亏损
    """
    if len(close) <= steps:
        return 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0

    idx = np.where(score[:-steps] >= min_score)[0]
    if len(idx) == 0:
        return 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0

    rets = close[idx + steps] / close[idx] - 1.0
    signals = len(rets)
    win_mask = rets > 0
    loss_mask = rets < 0

    wins = int(win_mask.sum())
    losses = int(loss_mask.sum())

    win_rate = float(win_mask.mean())
    avg_ret = float(rets.mean())

    profit_rets = rets[win_mask]
    loss_rets = rets[loss_mask]

    avg_win = float(profit_rets.mean()) if wins > 0 else 0.0
    avg_loss = float(loss_rets.mean()) if losses > 0 else 0.0  # 负数

    profit_sum = float(profit_rets.sum())
    loss_sum = float(-loss_rets.sum())  # 正数

    if loss_sum > 0:
        pf = profit_sum / loss_sum
    else:
        pf = 0.0

    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1
    max_dd = float(dd.min())

    return win_rate, avg_ret, signals, max_dd, pf, wins, avg_win, avg_loss


def prob_class(p):
    if p >= 0.65:
        return "prob-good"
    if p >= 0.45:
        return "prob-mid"
    return "prob-bad"


# ============ 建议逻辑（7日 / 30日 各自一套） ============
def decide_advice(prob: float, pf: float):
    """
    返回: (label, intensity, color_class)
      label: 建议买入 / 观望 / 建议卖出
      intensity: 1~5 点
      color_class: buy / hold / sell
    """
    if pf <= 0:
        pf = 0.0

    if prob >= 0.60 and pf >= 1.20:
        kind = "buy"
    elif prob <= 0.40 and pf <= 0.80:
        kind = "sell"
    else:
        kind = "hold"

    if kind == "buy":
        score = 0
        if prob >= 0.60: score += 1
        if prob >= 0.65: score += 1
        if prob >= 0.70: score += 1
        if pf   >= 1.20: score += 1
        if pf   >= 1.60: score += 1
        intensity = max(1, min(5, score))
        label = "建议买入"
        color = "buy"
    elif kind == "sell":
        score = 0
        if prob <= 0.40: score += 1
        if prob <= 0.35: score += 1
        if prob <= 0.30: score += 1
        if pf   <= 0.80: score += 1
        if pf   <= 0.60: score += 1
        intensity = max(1, min(5, score))
        label = "建议卖出"
        color = "sell"
    else:  # hold
        score = 1
        if 0.45 <= prob <= 0.55: score += 1
        if 0.47 <= prob <= 0.53: score += 1
        if 0.90 <= pf <= 1.10:   score += 1
        if 0.95 <= pf <= 1.05:   score += 1
        intensity = max(1, min(5, score))
        label = "观望"
        color = "hold"

    return label, intensity, color


# ============ 计算单只股票（使用上一根完整K线） ============
def compute_stock_metrics(symbol: str, cfg_key: str):
    cfg = BACKTEST_CONFIG[cfg_key]
    close, high, low, volume = fetch_yahoo_ohlcv(
        symbol, range_str=cfg["range"], interval=cfg["interval"]
    )

    # 丢掉最后一根「可能还没走完」的K线
    # 日线 = 昨日收盘；4H/1H = 最近完整周期
    if len(close) > 81:
        close = close[:-1]
        high = high[:-1]
        low = low[:-1]
        volume = volume[:-1]

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

    score_arr = sig_macd + sig_vol + sig_rsi + sig_atr + sig_obv

    spd = cfg["steps_per_day"]
    steps7 = 7 * spd
    steps30 = 30 * spd

    prob7, avg7, signals7, max_dd7, pf7, wins7, avg_win7, avg_loss7 = backtest_with_stats(
        close, score_arr, steps=steps7
    )
    prob30, avg30, signals30, max_dd30, pf30, wins30, avg_win30, avg_loss30 = backtest_with_stats(
        close, score_arr, steps=steps30
    )

    last_close = close[-1]
    prev_close = close[-2] if len(close) >= 2 else close[-1]
    change_pct = (last_close / prev_close - 1.0) * 100
    last_idx = -1

    indicators = []

    # MACD：只用红/绿点表示多头/空头，不显示数值
    macd_val = float(macd_hist[last_idx])
    macd_status = "bull" if macd_val > 0 else "bear"
    indicators.append({
        "name": "MACD 多头/空头",
        "status": macd_status,
        "desc": ""
    })

    # 成交量相对20日均量：阈值 1.10x
    vol_ratio = float(volume[last_idx] / (vol_ma20[last_idx] + 1e-9))
    vol_target = 1.10
    if vol_ratio > vol_target:
        vol_status = "bull"
    elif vol_ratio < 0.90:
        vol_status = "bear"
    else:
        vol_status = "neutral"
    indicators.append({
        "name": "成交量相对20日均量",
        "status": vol_status,
        "desc": f"{vol_target:.2f} / {vol_ratio:.2f}"
    })

    # RSI：阈值 60
    rsi_val = float(rsi[last_idx])
    if rsi_val >= 60:
        rsi_status = "bull"
    elif rsi_val <= 40:
        rsi_status = "bear"
    else:
        rsi_status = "neutral"
    indicators.append({
        "name": "RSI 区间",
        "status": rsi_status,
        "desc": f"60.0 / {rsi_val:.1f}"
    })

    # ATR 波动率：阈值 1.10x
    atr_ratio = float(atr[last_idx] / (atr_ma20[last_idx] + 1e-9))
    if atr_ratio > 1.10:
        atr_status = "bull"
    elif atr_ratio < 0.90:
        atr_status = "bear"
    else:
        atr_status = "neutral"
    indicators.append({
        "name": "ATR 波动率",
        "status": atr_status,
        "desc": f"1.10 / {atr_ratio:.2f}"
    })

    # OBV 资金趋势：阈值 1.05x
    obv_ratio = float(obv[last_idx] / (obv_ma20[last_idx] + 1e-9))
    if obv_ratio > 1.05:
        obv_status = "bull"
    elif obv_ratio < 0.95:
        obv_status = "bear"
    else:
        obv_status = "neutral"
    indicators.append({
        "name": "OBV 资金趋势",
        "status": obv_status,
        "desc": f"1.05 / {obv_ratio:.2f}"
    })

    return {
        "symbol": symbol,
        "price": float(last_close),
        "change": float(change_pct),
        "prob7": float(prob7),
        "prob30": float(prob30),
        "avg7": float(avg7),
        "avg30": float(avg30),
        "pf7": float(pf7),
        "pf30": float(pf30),
        "avg_win7": float(avg_win7),
        "avg_loss7": float(avg_loss7),
        "avg_win30": float(avg_win30),
        "avg_loss30": float(avg_loss30),
        "indicators": indicators,
    }


# ============ 缓存 ============
@st.cache_data(show_spinner=False)
def get_stock_metrics_cached(symbol: str, cfg_key: str, version: int = 11):
    # version 改成 11，强制刷新缓存
    return compute_stock_metrics(symbol, cfg_key=cfg_key)


# ============ Streamlit 交互层 ============
default_watchlist = ["QQQ", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA"]
if "watchlist" not in st.session_state:
    st.session_state.watchlist = default_watchlist.copy()

top_c1, top_c2, top_c3, top_c4 = st.columns([2.4, 1.1, 1.1, 1.4])

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
        ["默认顺序", "涨跌幅", "7日盈利概率", "30日盈利概率"],
        index=0,
        label_visibility="collapsed",
    )
with top_c4:
    mode_label = st.selectbox(
        "",
        list(BACKTEST_CONFIG.keys()),
        index=2,
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
            metrics = get_stock_metrics_cached(sym, cfg_key=mode_label)
        rows.append(metrics)
    except Exception as e:
        st.warning(f"{sym} 加载失败：{e}")
        continue

if sort_by == "涨跌幅":
    rows.sort(key=lambda x: x["change"], reverse=True)
elif sort_by == "7日盈利概率":
    rows.sort(key=lambda x: x["prob7"], reverse=True)
elif sort_by == "30日盈利概率":
    rows.sort(key=lambda x: x["prob30"], reverse=True)

# ============ 卡片展示 ============
if not rows:
    st.info("暂无自选股票，请先在上方输入代码添加。")
else:
    cols_per_row = 4
    for i in range(0, len(rows), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, row in zip(cols, rows[i:i + cols_per_row]):
            with col:
                change_class = "change-up" if row["change"] >= 0 else "change-down"
                change_str = f"{row['change']:+.2f}%"

                prob7_pct = row["prob7"] * 100
                prob30_pct = row["prob30"] * 100

                avg_win7_pct = row["avg_win7"] * 100
                avg_loss7_pct = row["avg_loss7"] * 100

                avg_win30_pct = row["avg_win30"] * 100
                avg_loss30_pct = row["avg_loss30"] * 100

                pf7 = row["pf7"]
                pf30 = row["pf30"]

                prob7_class = prob_class(row["prob7"])
                prob30_class = prob_class(row["prob30"])

                indicators_html = ""
                for ind in row["indicators"]:
                    if ind["desc"]:
                        line = f"{ind['name']} ({ind['desc']})"
                    else:
                        line = ind["name"]
                    indicators_html += (
                        f"<div class='label'>{line}"
                        f"<span class='dot dot-{ind['status']}'></span></div>"
                    )

                # 7 日 & 30 日 建议分别计算
                adv7_label, adv7_intensity, adv7_kind = decide_advice(row["prob7"], pf7)
                adv30_label, adv30_intensity, adv30_kind = decide_advice(row["prob30"], pf30)

                def build_advice_html(label, intensity, kind):
                    if kind == "buy":
                        dot_on_class = "dot-score dot-score-buy"
                        advice_class = "advice-text advice-buy"
                    elif kind == "sell":
                        dot_on_class = "dot-score dot-score-sell"
                        advice_class = "advice-text advice-sell"
                    else:
                        dot_on_class = "dot-score dot-score-hold"
                        advice_class = "advice-text advice-hold"
                    dots = (
                        f"<span class='{dot_on_class}'></span>" * intensity +
                        "<span class='dot-score dot-score-off'></span>" * (5 - intensity)
                    )
                    return advice_class, label, dots

                adv7_class, adv7_text, adv7_dots = build_advice_html(
                    adv7_label, adv7_intensity, adv7_kind
                )
                adv30_class, adv30_text, adv30_dots = build_advice_html(
                    adv30_label, adv30_intensity, adv30_kind
                )

                html = f"""
                <div class="card">
                  <div class="symbol-line">
                    <span class="symbol-code">{row['symbol']}</span>
                    <span class="symbol-price">${row['price']:.2f}</span>
                    <span class="{change_class}">{change_str}</span>
                  </div>

                  <div style="margin-top:4px;margin-bottom:6px">
                    {indicators_html}
                  </div>

                  <div style="border-bottom:1px dashed #262736;margin:6px 0 6px;"></div>

                  <div>
                    <div>
                      <span class="label">7日盈利概率</span>
                      <span class="{prob7_class}"> {prob7_pct:.1f}%</span>
                      <span class="label"> (均盈 {avg_win7_pct:+.1f}%&nbsp;&nbsp;均亏 {avg_loss7_pct:+.1f}%&nbsp;&nbsp;盈亏 {pf7:.2f})</span>
                    </div>
                    <div>
                      <span class="label">30日盈利概率</span>
                      <span class="{prob30_class}"> {prob30_pct:.1f}%</span>
                      <span class="label"> (均盈 {avg_win30_pct:+.1f}%&nbsp;&nbsp;均亏 {avg_loss30_pct:+.1f}%&nbsp;&nbsp;盈亏 {pf30:.2f})</span>
                    </div>
                  </div>

                  <div class="score">
                    <span class="score-label">7日信号</span>
                    <span class="{adv7_class}">{adv7_text}</span>
                    {adv7_dots}
                  </div>
                  <div class="score">
                    <span class="score-label">30日信号</span>
                    <span class="{adv30_class}">{adv30_text}</span>
                    {adv30_dots}
                  </div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)

st.caption(
    "所有指标和回测均基于“上一根完整K线”（日线=昨日收盘，4小时/1小时=上一完整周期），"
    "不会使用盘中尚未走完的K线数据。"
    "7日/30日盈利概率基于历史同类信号统计，括号为平均盈利、平均亏损和盈亏比（Profit Factor），"
    "仅作个人量化研究，不构成投资建议。"
)
