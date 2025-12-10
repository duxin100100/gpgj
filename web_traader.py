import dash
from dash import dcc, html, Input, Output, State
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import warnings

# --- 配置区 ---
DEFAULT_STOCKS = ['QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
HISTORY_YEARS = 3
MIN_CONDITIONS_MET = 3

warnings.filterwarnings("ignore", category=FutureWarning)

# --- 核心分析函数 (V10 - 保持V8的英文内核) ---
def analyze_stock(ticker_symbol):
    """分析单个股票，返回包含纯英文键的字典。"""
    try:
        stock = yf.Ticker(ticker_symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=HISTORY_YEARS * 365)
        hist_data = stock.history(start=start_date, end=end_date, interval="1d", auto_adjust=True)

        if hist_data.empty or len(hist_data) < 40:
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

        rsi_status = 2 if latest.get('RSI_14', 50) < 40 else 1 if 40 <= latest.get('RSI_14', 50) < 50 else 0
        macd_status = 2 if latest.get('MACD_12_26_9', 0) > latest.get('MACDs_12_26_9', -1) else 0
        bb_status = 2 if latest['Close'] < latest.get('BBL_20_2.0', float('inf')) else 1 if latest.get('BBL_20_2.0', float('inf')) <= latest['Close'] < latest.get('BBM_20_2.0', float('inf')) else 0
        ema_status = 2 if latest['Close'] > latest.get('EMA_20', float('inf')) else 0
        obv_mean = hist_data['OBV'].rolling(window=5).mean().iloc[-1]
        obv_status = 2 if latest.get('OBV', 0) > obv_mean else 0

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
            'ticker': ticker_symbol, 'price': latest_price,
            'change_pct': price_change_percent, 'rsi': rsi_status, 'macd': macd_status,
            'bbands': bb_status, 'ema': ema_status, 'obv': obv_status,
            'prob_7d': prob_7d, 'prob_30d': prob_30d,
        }
    except Exception:
        return None

# --- Dash 应用定义 ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

# --- 界面布局 (V10 - 手动构建HTML表格) ---
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
        html.Table(className='u-full-width', children=[
            html.Thead(html.Tr([
                html.Th('代码'), html.Th('收盘价'), html.Th('涨跌幅'),
                html.Th('RSI'), html.Th('MACD'), html.Th('布林带'),
                html.Th('EMA'), html.Th('OBV'),
                html.Th('7日概率'), html.Th('30日概率'),
            ], style={'backgroundColor': '#333', 'color': 'white'})),
            html.Tbody(id='stock-table-body') # 表格内容将由Callback填充
        ])
    ]),
    dcc.Store(id='stock-list-store', data=DEFAULT_STOCKS),
])

# --- 交互逻辑 (Callback - V10 终极版) ---
@app.callback(Output('stock-list-store', 'data'), Input('add-btn', 'n_clicks'), [State('ticker-input', 'value'), State('stock-list-store', 'data')])
def add_stock(n_clicks, ticker, stock_list):
    if n_clicks > 0 and ticker:
        ticker = ticker.upper().strip()
        if ticker and ticker not in stock_list:
            return [ticker] + stock_list
    return stock_list

@app.callback(Output('stock-table-body', 'children'), [Input('stock-list-store', 'data'), Input('sort-select', 'value')])
def update_table(stock_list, sort_value):
    raw_data = [analyze_stock(ticker) for ticker in stock_list]
    valid_data = [d for d in raw_data if d is not None]
    if not valid_data:
        return []

    if sort_value == 'change_desc':
        sorted_data = sorted(valid_data, key=lambda x: x.get('change_pct', 0.0), reverse=True)
    elif sort_value == 'prob7d_desc':
        sorted_data = sorted(valid_data, key=lambda x: x.get('prob_7d', 0.0), reverse=True)
    elif sort_value == 'prob30d_desc':
        sorted_data = sorted(valid_data, key=lambda x: x.get('prob_30d', 0.0), reverse=True)
    else:
        sorted_data = valid_data

    table_rows = []
    for row in sorted_data:
        # 定义单元格的基础样式
        cell_style = {'textAlign': 'center', 'borderBottom': '1px solid #333', 'padding': '12px'}
        
        # 涨跌幅颜色
        change_pct_color = '#4caf50' if row.get('change_pct', 0.0) > 0 else '#f44336' if row.get('change_pct', 0.0) < 0 else 'white'
        
        # 指标单元格
        indicator_cells = []
        for col_id in ['rsi', 'macd', 'bbands', 'ema', 'obv']:
            status = row.get(col_id, 0)
            color = '#4caf50' if status == 2 else '#ffeb3b' if status == 1 else '#f44336'
            indicator_cells.append(html.Td('●', style={**cell_style, 'color': color, 'fontWeight': 'bold' if status == 2 else 'normal'}))

        # 创建HTML表格行
        table_rows.append(html.Tr([
            html.Td(row.get('ticker'), style=cell_style),
            html.Td(f"{row.get('price', 0.0):.2f}", style=cell_style),
            html.Td(f"{row.get('change_pct', 0.0):.2f}%", style={**cell_style, 'color': change_pct_color}),
            *indicator_cells, # 解包指标单元格列表
            html.Td(f"{row.get('prob_7d', 0.0):.2f}%", style=cell_style),
            html.Td(f"{row.get('prob_30d', 0.0):.2f}%", style=cell_style),
        ]))

    return table_rows

# --- 运行应用 ---
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
