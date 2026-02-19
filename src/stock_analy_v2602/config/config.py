"""
配置文件 - 数据库和API配置
"""

# MySQL数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123654aaa',  # 请修改为实际密码
    'database': 'stock_data_v2602',
    'charset': 'utf8mb4'
}

# 股票数据API配置
# 支持多个数据源：tushare, akshare, yfinance等
API_CONFIG = {
    'source': 'akshare',  # 默认使用akshare（免费、无需token）
    'tushare_token': '',  # 如果使用tushare，需要填写token
}

# 数据下载配置
DOWNLOAD_CONFIG = {
    'daily_batch_size': 100,  # 日线数据批量下载数量
    'minute_batch_size': 10,   # 分钟线数据批量下载数量
    'retry_times': 3,          # 失败重试次数
    'retry_delay': 2,          # 重试延迟(秒)
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'stock_analyzer.log'
}

# 目标股票列表配置
# 格式: ['股票代码1', '股票代码2', ...]
TARGET_STOCKS = [
    '000001',  # 平安银行
    '600000',  # 浦发银行
    '600036',  # 招商银行
    '600030',  # 中国平安
    '600031',  # 三一重工
    '600032',  # 中国太保
    '600033',  # 中国交运
    '600034',  # 中国中铁
    # 可以继续添加更多股票代码
]

# 分钟线周期配置
# 支持: '1', '5', '15', '30', '60'
TARGET_PERIODS = ['1', '5', '15', '30', '60']
