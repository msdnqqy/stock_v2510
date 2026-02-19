
def get_data():
    import yfinance as yf

    ticker_symbol = "600031.SS"

    # 使用 Ticker 类
    stock = yf.Ticker(ticker_symbol)

    # 获取历史数据
    data = stock.history(period="1y", interval="1d")

    # 3. 计算均线 (Moving Averages)
    # rolling(window=n) 表示计算过去 n 天的滑动窗口
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA60'] = data['Close'].rolling(window=60).mean()

    data_tem = data.iloc[-140:-100,:]
    # mpf.plot(data_tem, type='candle', style='charles', volume=True)

    return data_tem