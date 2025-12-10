from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange
import uvicorn


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ========== 数据与回测 ==========
def load(symbol):
    df = yf.download(symbol, period="3y", interval="1d").dropna()
    return df if not df.empty else None


def calc(df):
    df["macd"] = MACD(df["Close"]).macd_diff()
    df["rsi"] = RSIIndicator(df["Close"]).rsi()
    df["ema8"] = EMAIndicator(df["Close"], window=8).ema_indicator()
    df["ema21"] = EMAIndicator(df["Close"], window=21).ema_indicator()
    df["atr"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    df["obv"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    df["atr20"] = df["atr"].rolling(20).mean()
    df["obv20"] = df["obv"].rolling(20).mean()
    df["vol20"] = df["Volume"].rolling(20).mean()
    return df.dropna()


def signal(latest):
    def status(val, high, low):
        return "bull" if val>high else "bear" if val<low else "neutral"

    return [
        {"name":"MACD趋势", "status":"bull" if latest["macd"]>0 else "bear"},
        {"name":"成交量", "status":status(latest["Volume"], latest["vol20"]*1.1, latest["vol20"]*0.9)},
        {"name":"RSI",   "status":"bull" if latest["rsi"]>60 else "bear" if latest["rsi"]<40 else "neutral"},
        {"name":"ATR波动", "status":status(latest["atr"], latest["atr20"]*1.1, latest["atr20"]*0.9)},
        {"name":"OBV资金流", "status":status(latest["obv"], latest["obv20"]*1.05, latest["obv20"]*0.95)},
    ]


def backtest(df, days=7, base=3):
    wins,total,rets=0,0,[]
    for i in range(len(df)-days):
        s=signal(df.iloc[i])
        score=sum(1 for x in s if x["status"]=="bull")
        if score>=base:
            total+=1
            r=(df["Close"].iloc[i+days]/df["Close"].iloc[i]-1)
            rets.append(r)
            if r>0:wins+=1
    return (wins/total if total>0 else 0),(np.mean(rets) if rets else 0)


# ====== API ======
@app.get("/stock/{symbol}")
def api(symbol:str):
    s=symbol.upper()
    df=load(s); 
    if df is None: return {"error":"代码无效"}
    df=calc(df)
    latest=df.iloc[-1]
    p7,a7=backtest(df,7)
    p30,a30=backtest(df,30)

    return {
        "symbol":s,
        "price":float(latest["Close"]),
        "change":float((latest["Close"]/df["Close"].iloc[-2]-1)*100),
        "prob7":round(p7,4),"avg7":round(a7,4),
        "prob30":round(p30,4),"avg30":round(a30,4),
        "indicators":signal(latest),
        "score":sum(i["status"]=="bull" for i in signal(latest))
    }


# ====== 前端 UI 页面输出 ======
@app.get("/",response_class=HTMLResponse)
def ui():
    return """
<html>
<head>
<title>AI 量化看板</title>
<style>
body{background:#0e1014;color:#fff;font-family:-apple-system;margin:40px}
input{padding:8px 12px;border-radius:6px;border:none;margin-right:8px}
button{padding:8px 14px;border-radius:6px;border:none;background:#4f46e5;color:#fff}
.card{background:#1a1d23;padding:16px;border-radius:10px;margin-top:14px;width:350px}
.dot{width:10px;height:10px;border-radius:50%}
.up{color:#4ade80}.down{color:#f87171}
.bull{background:#4ade80}.neutral{background:#facc15}.bear{background:#fb7185}
</style></head>
<body>

<h2>📈 AI 量化信号系统</h2>
<input id="code" placeholder="输入股票 如 TSLA AAPL NVDA">
<button onclick="load()">查询</button>

<div id="list"></div>

<script>
async function load(){
    let c=document.getElementById("code").value.toUpperCase()
    let r=await fetch('/stock/'+c).then(r=>r.json())
    if(r.error)return alert("股票不存在")
    let ind=r.indicators.map(i=>`<div>
        ${i.name} <span class="dot ${i.status}"></span></div>`).join("")
    document.getElementById("list").innerHTML+=`
    <div class='card'>
        <h3>${r.symbol} <span class="${r.change>0?"up":"down"}">${r.change.toFixed(2)}%</span></h3>
        <p>$${r.price.toFixed(2)}</p>
        ${ind}
        <p>7日盈利概率：${(r.prob7*100).toFixed(1)}%</p>
        <p>30日盈利概率：${(r.prob30*100).toFixed(1)}%</p>
        <p>信号强度：${r.score}/5</p>
    </div>`
}
</script>
</body></html>
"""


if __name__ == "__main__":
    print("🚀 运行成功 → 打开浏览器访问：http://127.0.0.1:8000")
    uvicorn.run(app,host="0.0.0.0",port=8000)
