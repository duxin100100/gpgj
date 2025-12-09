import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 网页标题和布局
# ==========================================
st.set_page_config(page_title="小白量化助手", page_icon="📈")
st.title("📈 美股趋势探测器")
st.write("这是你的专属量化工具，输入代码即可分析！")

# ==========================================
# 1. 侧边栏：输入框和按钮
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数设置")
    # 创建一个输入框，默认值是 TSLA
    symbol = st.text_input("输入股票代码 (例如: AAPL, NVDA, BABA)", value="TSLA")
    # 创建一个按钮
    run_button = st.button("开始分析 🚀")

# ==========================================
# 2. 核心逻辑 (点击按钮后才运行)
# ==========================================
if run_button:
    st.info(f"正在联网获取 {symbol} 的数据，请稍候...")
    
    # --- 原来的抓取和计算代码 ---
    try:
        data = yf.download(symbol, period="6mo", progress=False)
        
        # 数据清洗
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs('Close', axis=1, level=0, drop_level=False)
            data.columns = ['Close']
            
        if data.empty:
            st.error("❌ 找不到数据！请检查股票代码是否正确 (比如美股代码要是大写)。")
            st.stop() # 停止运行
            
        # 计算 MACD
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # 取最新数据
        last_date = data.index[-1].strftime('%Y-%m-%d')
        last_price = data.iloc[-1]['Close']
        macd = data.iloc[-1]['MACD']
        signal = data.iloc[-1]['Signal_Line']
        
        # --- 3. 在网页上展示结果 ---
        st.success("✅ 分析完成！")
        
        # 显示大指标卡片
        col1, col2, col3 = st.columns(3)
        col1.metric("股票代码", symbol)
        col2.metric("最新日期", last_date)
        col3.metric("当前价格", f"${last_price:.2f}")

        st.divider() # 分割线

        # 判断结论
        if macd > signal:
            st.header("🔥 结论：多头 (买入/持有)")
            st.markdown("MACD线在信号线上方，**上涨动能较强**。")
        else:
            st.header("❄️ 结论：空头 (卖出/观望)")
            st.markdown("MACD线在信号线下方，**下跌风险较大**。")

        # --- 4. 画图 (这是网页版的强项) ---
        st.subheader("📊 趋势图表")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 上图：股价
        ax1.plot(data.index, data['Close'], label='Price', color='black')
        ax1.set_title(f"{symbol} Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下图：MACD
        ax2.plot(data.index, data['MACD'], label='MACD', color='red')
        ax2.plot(data.index, data['Signal_Line'], label='Signal', color='blue')
        # 画红绿柱子
        bars = data['MACD'] - data['Signal_Line']
        ax2.bar(data.index, bars, color=['red' if x > 0 else 'green' for x in bars], alpha=0.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 把图表显示在网页上
        st.pyplot(fig)

    except Exception as e:
        st.error(f"发生错误: {e}")
else:
    st.info("👈 请在左侧输入代码，点击按钮开始运行。")
