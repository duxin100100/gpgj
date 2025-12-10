Skip to content
Navigation Menu
duxin100100
gpgj

Type / to search
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Files
Go to file
t
requirements.txt
web_traader.py
gpgj
/
web_traader.py
in
main

Edit

Preview
Indent mode

Spaces
Indent size

4
Line wrap mode

No wrap
Editing web_traader.py file contents
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
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
        <p>信号强度：${r.score}/5</p>
Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.
 
No spaces found. You can create a new space to get started.
