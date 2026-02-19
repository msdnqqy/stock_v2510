# 股票分析系统

一个功能完整的股票数据下载和分析系统，支持日线、分钟线数据的下载和增量更新。

## 功能特性

### 1. 数据下载
- ✅ 下载指定股票的日线数据（开盘、收盘、最高、最低、成交量、成交额、换手率等）
- ✅ 下载指定股票的分钟线数据（1分钟、5分钟、15分钟、30分钟、60分钟）
- ✅ 支持多个数据源：AKShare（免费）、Tushare、yfinance
- ✅ 自动存储到MySQL数据库（stock_data_v2602）

### 2. 增量更新
- ✅ 智能检测数据库中的最新日期
- ✅ 仅下载增量数据，避免重复下载
- ✅ 支持批量更新多只股票

### 3. 数据字段（日线）
- 基本价格：开盘价、收盘价、最高价、最低价、前收盘价
- 涨跌数据：涨跌额、涨跌幅
- 成交数据：成交量、成交额、换手率、量比
- 估值指标：市盈率、市净率、市销率
- 市值数据：总市值、流通市值

### 4. 数据字段（分钟线）
- 基本价格：开盘价、收盘价、最高价、最低价
- 成交数据：成交量、成交额
- 均价数据

## 项目结构

```
stock_analyzer/
├── config/              # 配置文件
│   └── config.py       # 数据库和API配置
├── database/           # 数据库模块
│   ├── db_manager.py   # 数据库管理
│   └── storage_service.py  # 数据存储服务
├── models/             # 数据模型
│   └── models.py       # SQLAlchemy模型定义
├── downloaders/        # 数据下载器
│   └── data_downloader.py  # 股票数据下载
├── utils/              # 工具模块
├── tests/              # 测试模块
├── main.py             # 主程序入口
├── requirements.txt    # 依赖包
└── README.md          # 说明文档
```

## 安装步骤

### 1. 安装依赖包

```bash
cd /home/claude/src/stock_analyzer
pip install -r requirements.txt --break-system-packages
```

### 2. 配置数据库

编辑 `config/config.py` 文件，修改MySQL配置：

```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'your_password',  # 修改为实际密码
    'database': 'stock_data_v2602',
    'charset': 'utf8mb4'
}
```

### 3. 初始化数据库

首次运行时，系统会自动创建数据库和表。

## 使用方法

### 1. 下载单只股票的日线数据

```bash
# 下载平安银行(000001)的全部历史日线数据
python main.py --action download_daily --stock 000001

# 下载指定日期范围的数据
python main.py --action download_daily --stock 000001 --start 2020-01-01 --end 2023-12-31

# 使用Tushare数据源（需要token）
python main.py --action download_daily --stock 000001 --source tushare
```

### 2. 批量下载多只股票的日线数据

```bash
# 批量下载
python main.py --action download_daily --stocks "000001,000002,600000,600036"
```

### 3. 下载分钟线数据

```bash
# 下载1分钟线数据
python main.py --action download_minute --stock 000001 --period 1

# 下载5分钟线数据
python main.py --action download_minute --stock 000001 --period 5

# 下载指定日期范围的分钟数据
python main.py --action download_minute --stock 000001 --period 1 --start 2024-02-01 --end 2024-02-15
```

### 4. 增量更新日线数据

```bash
# 增量更新单只股票
python main.py --action update_daily --stock 000001

# 批量增量更新
python main.py --action update_daily --stocks "000001,000002,600000,600036"
```

### 5. 增量更新分钟线数据

```bash
# 增量更新分钟线
python main.py --action update_minute --stock 000001 --period 1
```

### 6. 查看数据统计

```bash
# 查看某只股票的数据统计
python main.py --action stats --stock 000001
```

## Python代码调用示例

```python
from main import StockAnalyzer

# 初始化系统
analyzer = StockAnalyzer(data_source='akshare')

# 下载日线数据
analyzer.download_stock_daily('000001', start_date='2020-01-01')

# 下载分钟线数据
analyzer.download_stock_minute('000001', period='5')

# 增量更新
analyzer.update_stock_daily('000001')

# 批量下载
analyzer.batch_download_daily(['000001', '000002', '600000'])

# 批量更新
analyzer.batch_update_daily(['000001', '000002', '600000'])

# 查看统计
analyzer.show_stats('000001')

# 关闭系统
analyzer.close()
```

## 数据源说明

### 1. AKShare（推荐）
- ✅ 免费使用，无需token
- ✅ 数据全面，包含A股、港股等
- ✅ 更新及时
- ⚠️ 网络请求可能较慢

### 2. Tushare
- ✅ 数据质量高
- ✅ 支持更多指标
- ⚠️ 需要注册获取token
- ⚠️ 部分数据需要积分

### 3. yfinance
- ✅ 支持全球市场
- ✅ 适合美股、港股
- ⚠️ A股数据较少

## 数据库表结构

### daily_data（日线数据表）
| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 主键 |
| stock_code | VARCHAR(20) | 股票代码 |
| trade_date | DATE | 交易日期 |
| open | FLOAT | 开盘价 |
| high | FLOAT | 最高价 |
| low | FLOAT | 最低价 |
| close | FLOAT | 收盘价 |
| volume | BIGINT | 成交量 |
| amount | FLOAT | 成交额 |
| turnover_rate | FLOAT | 换手率 |
| ... | ... | 更多字段 |

### minute_data（分钟线数据表）
| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 主键 |
| stock_code | VARCHAR(20) | 股票代码 |
| trade_datetime | DATETIME | 交易时间 |
| open | FLOAT | 开盘价 |
| high | FLOAT | 最高价 |
| low | FLOAT | 最低价 |
| close | FLOAT | 收盘价 |
| volume | BIGINT | 成交量 |
| amount | FLOAT | 成交额 |

## 常见问题

### Q1: 如何获取Tushare token？
访问 https://tushare.pro/ 注册账号，在个人中心获取token。

### Q2: 数据下载失败怎么办？
- 检查网络连接
- 检查股票代码格式是否正确
- 查看日志文件 stock_analyzer.log
- 尝试更换数据源

### Q3: 如何修改数据库配置？
编辑 `config/config.py` 文件中的 DB_CONFIG 字典。

### Q4: 支持哪些股票代码格式？
- A股：000001、600000（6位数字）
- 也可以带前缀：sh600000、sz000001

## 日志文件

系统运行日志保存在 `stock_analyzer.log` 文件中，记录了所有操作和错误信息。

## 注意事项

1. 首次下载建议使用日期范围，避免数据量过大
2. 分钟线数据量较大，建议分批下载
3. 定期运行增量更新脚本，保持数据最新
4. 注意数据源的访问频率限制

## 后续扩展

- [ ] 添加数据清洗功能
- [ ] 添加技术指标计算
- [ ] 添加数据可视化
- [ ] 添加股票筛选功能
- [ ] 添加回测系统
- [ ] 添加定时任务调度

## 开发者

创建日期：2026-02-15

## 许可证

MIT License
