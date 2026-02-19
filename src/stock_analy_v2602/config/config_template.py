"""
配置文件模板
使用前请将此文件复制为 config.py 并修改相应配置
"""

# MySQL数据库配置
DB_CONFIG = {
    'host': 'localhost',        # 数据库主机地址
    'port': 3306,               # 数据库端口
    'user': 'root',             # 数据库用户名
    'password': 'YOUR_PASSWORD_HERE',  # ⚠️ 请修改为实际密码
    'database': 'stock_data_v2602',    # 数据库名称
    'charset': 'utf8mb4'        # 字符集
}

# 股票数据API配置
API_CONFIG = {
    # 数据源选择：
    # - 'akshare'：免费，无需token，推荐使用
    # - 'tushare'：需要注册并获取token
    # - 'yfinance'：适合美股和港股
    'source': 'akshare',
    
    # Tushare Token（如果使用tushare）
    # 获取方式：https://tushare.pro/ 注册后在个人中心获取
    'tushare_token': '',
}

# 数据下载配置
DOWNLOAD_CONFIG = {
    'daily_batch_size': 100,   # 日线数据批量下载数量
    'minute_batch_size': 10,   # 分钟线数据批量下载数量
    'retry_times': 3,          # 失败重试次数
    'retry_delay': 2,          # 重试延迟(秒)
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',           # 日志级别: DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'stock_analyzer.log'  # 日志文件名
}

# 使用说明：
# 1. 复制此文件为 config.py
# 2. 修改 DB_CONFIG 中的 password 为实际MySQL密码
# 3. 如果使用 tushare，填写 tushare_token
# 4. 根据需要调整其他配置项
