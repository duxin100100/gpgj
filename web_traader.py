import dash
from dash import dcc, html, dash_table, Input, Output, State
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import warnings

# --- 配置区 ---
DEFAULT_STOCKS = ['QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
HISTORY_YEARS = 3
MIN_CONDITIONS_MET = 3

# 忽略 pandas 在特定情况下的警告
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 核心分析函数 (V7 - 加固版) ---
def analyze_stock(ticker_symbol):
    """分析单个股票，返回包含纯净数字的字典用于处理。"""
    try:
        stock = yf.Ticker(ticker_symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=HISTORY_YEARS * 365)
        # 使用 auto_adjust=True 自动调整数据，更稳定
        hist_data = stock.history(start=start_date, end=end_date, interval="1d", auto_adjust=True)

        if hist_data.empty or len(hist_data) < 35:
            return None

        latest_price = hist_data['Close'].iloc[-1]
        previous_close = hist_data['Close'].iloc[-2]
        price_change_percent = ((latest_price - previous_close) / previous_close) * 100

        hist_data.ta.rsi(length=14, append=True)
        hist_data.ta.macd(fast=12, slow=26, signal=9, append=True)
        hist_data.ta.bbands(length=20, std=2, append=True)
        hist_data.ta.ema(length=20, append=True)
        hist_data.ta.obv(append=True)
        
        latest = hist_data.iloc[-1]

        # 定义指标状态 (2=满足, 1=中性, 0=不满足)
        rsi_status = 2 if latest.get('RSI_14', 50) < 40 else 1 if 40 <= latest.get('RSI_14', 50) < 50 else 0
        macd_status = 2 if latest.get('MACD_12_26_9', 0) > latest.get('MACDs_12_26_9', -1) else 0
        bb_status = 2 if latest['Close'] < latest.get('BBL_20_2.0', float('inf')) else 1 if latest.get('BBL_20_2.0', float('inf')) <= latest['Close'] < latest.get('BBM_20_2.0', float('inf')) else 0
        ema_status = 2 if latest['Close'] > latest.get('EMA_20', float('inf')) else 0
        obv_mean = hist_data['OBV'].rolling(window=5).mean().iloc[-1]
        obv_status = 2 if latest.get('OBV', 0) > obv_mean else 0

        # 回测盈利概率
        conditions_df = pd.DataFrame({
            'rsi_ok': hist_data['RSI_14'] < 40,
            'macd_ok': hist_data['MACD_12_26_9'] > hist_data['MACDs_12_26_9'],
            'bb_ok': hist_data['Close'] < hist_data['BBL_20_2.0'],
            'ema_ok': hist_data['Close'] > hist_data['EMA_20'],
            'obv_ok': hist_data['OBV'] > hist_data['OBV'].rolling(window=5).mean()
        })
        conditions_met_count = conditions_df.sum(axis=1)
        buy_signals = (conditions_met_count >= MIN_CONDITIONS_MET)
        
        hist_data['future_7d_close'] = hist_data['Close'].shift(-7)
        hist_data['future_30d_close'] = hist_data['Close'].shift(-30)
        
        signal_df = hist_data[buy_signals].dropna(subset=['future_30d_close'])
        signal_days = len(signal_df)
        
        prob_7d, prob_30d = 0.0, 0.0
        if signal_days > 0:
            win_7d = (signal_df['future_7d_close'] > signal_df['Close']).sum()
            win_30d = (signal_df['future_30d_close'] > signal_df['Close']).sum()
            prob_7d = (win_7d / signal_days) * 100
            prob_30d = (win_30d / signal_days) * 100

        return {
            'id': ticker_symbol, '代码': ticker_symbol, '收盘价': latest_price,
            '涨跌幅': price_change_percent, 'RSI': rsi_status, 'MACD': macd_status,
            '布林带': bb_status, 'EMA': ema_status, 'OBV': obv_status,
            '7日概率': prob_7d, '30日概率': prob_30d,
        }
    except Exception as e:
        print(f"分析 {ticker_symbol} 时出错: {e}")
        return None

# --- Dash 应用定义 ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

# --- 界面布局 ---
app.layout = html.Div(style={'backgroundColor': '#121212', 'color': '#e0e0e0', 'fontFamily': 'sans-serif'}, children=[
    html.H1("美股量化观察列表", style={'textAlign': 'center', 'color': '#4dabf7'}),
    
    html.Div(className='row', style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}, children=[
        html.Div(className='six columns', children=[
            dcc.Input(id='ticker-input', type='text', placeholder='输入股票代码...', style={'backgroundColor': '#333', 'color': '#fff', 'border': '1px solid #555', 'padding': '8px'}),
            html.Button('添加', id='add-btn', n_clicks=0, style={'marginLeft': '10px', 'backgroundColor': '#4dabf7', 'color': 'white', 'border': 'none', 'padding': '8px 12px'}),
        ]),
        html.Div(className='six columns', style={'textAlign': 'right'}, children=[
            dcc.Dropdown(id='sort-select', options=[{'label': '默认', 'value': 'default'},{'label': '涨跌幅 (高到低)', 'value': 'change_desc'},{'label': '7日概率 (高到低)', 'value': 'prob7d_desc'},{'label': '30日概率 (高到低)', 'value': 'prob30d_desc'}], value='default', style={'width': '200px', 'display': 'inline-block'}, clearable=False)
        ])
    ]),
    
    dcc.Loading(id="loading", type="default", children=[
        dash_table.DataTable(
            id='stock-table',
            columns=[
                {'name': '代码', 'id': '代码'}, {'name': '收盘价', 'id': '收盘价'}, {'name': '涨跌幅', 'id': '涨跌幅'},
                {'name': 'RSI', 'id': 'RSI'}, {'name': 'MACD', 'id': 'MACD'}, {'name': '布林带', 'id': '布林带'},
                {'name': 'EMA', 'id': 'EMA'}, {'name': 'OBV', 'id': 'OBV'},
                {'name': '7日概率', 'id': '7日概率'}, {'name': '30日概率', 'id': '30日概率'},
            ],
            style_header={'backgroundColor': '#333', 'fontWeight': 'bold'},
            style_cell={'backgroundColor': '#1e1e1e', 'color': 'white', 'textAlign': 'center', 'border': '1px solid #333'},
            style_data_conditional=[
                {'if': {'column_id': '涨跌幅', 'filter_query': '{涨跌幅_raw} > 0'}, 'color': '#4caf50'},
                {'if': {'column_id': '涨跌幅', 'filter_query': '{涨跌幅_raw} < 0'}, 'color': '#f44336'},
                *[{'if': {'column_id': c, 'filter_query': f'{{{c}_raw} = 2'}, 'color': '#4caf50', 'fontWeight': 'bold'} for c in ['RSI', 'MACD', '布林带', 'EMA', 'OBV']],
                *[{'if': {'column_id': c, 'filter_query': f'{{{c}_raw} = 1'}, 'color': '#ffeb3b'} for c in ['RSI', 'MACD', '布林带', 'EMA', 'OBV']],
                *[{'if': {'column_id': c, 'filter_query': f'{{{c}_raw} = 0'}, 'color': '#f44336'} for c in ['RSI', 'MACD', '布林带', 'EMA', 'OBV']],
            ]
        )
    ]),
    dcc.Store(id='stock-list-store', data=DEFAULT_STOCKS),
])

# --- 交互逻辑 (Callback - V7 防御性编程) ---
@app.callback(Output('stock-list-store', 'data'), Input('add-btn', 'n_clicks'), [State('ticker-input', 'value'), State('stock-list-store', 'data')])
def add_stock(n_clicks, ticker, stock_list):
    if n_clicks > 0 and ticker:
        ticker = ticker.upper().strip()
        if ticker and ticker not in stock_list:
            return [ticker] + stock_list
    return stock_list

@app.callback(Output('stock-table', 'data'), [Input('stock-list-store', 'data'), Input('sort-select', 'value')])
def update_table(stock_list, sort_value):
    raw_data = [analyze_stock(ticker) for ticker in stock_list]
    valid_data = [d for d in raw_data if d is not None]
    if not valid_data:
        return []

    # 先排序：使用 .get() 安全地获取值，如果不存在则使用0
    if sort_value == 'change_desc':
        sorted_data = sorted(valid_data, key=lambda x: x.get('涨跌幅', 0.0), reverse=True)
    elif sort_value == 'prob7d_desc':
        sorted_data = sorted(valid_data, key=lambda x: x.get('7日概率', 0.0), reverse=True)
    elif sort_value == 'prob30d_desc':
        sorted_data = sorted(valid_data, key=lambda x: x.get('30日概率', 0.0), reverse=True)
    else:
        sorted_data = valid_data

    # 后格式化：同样使用 .get()，确保即使数据不完整也不会崩溃
    display_data = []
    for row in sorted_data:
        formatted_row = {
            'id': row.get('id', ''),
            '代码': row.get('代码', 'N/A'),
            '收盘价': f"{row.get('收盘价', 0.0):.2f}",
            '涨跌幅': f"{row.get('涨跌幅', 0.0):.2f}%",
            '涨跌幅_raw': row.get('涨跌幅', 0.0), # 为条件格式化保留原始数字
            '7日概率': f"{row.get('7日概率', 0.0):.2f}%",
            '30日概率': f"{row.get('30日概率', 0.0):.2f}%",
        }
        for col in ['RSI', 'MACD', '布林带', 'EMA', 'OBV']:
            formatted_row[col] = '●'
            formatted_row[f'{col}_raw'] = row.get(col, 0) # 为条件格式化保留原始数字
        
        display_data.append(formatted_row)

    return display_data

# --- 运行应用 ---
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')

