
import logging
import sys

# 添加项目路径
sys.path.append('.')

from main import StockAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    try:
        # 初始化分析器
        analyzer = StockAnalyzer(data_source='akshare')
        
        # 初始化股票列表
        print("开始初始化股票列表...")
        analyzer.init_stock_list()
        print("初始化完成！")
        
    except Exception as e:
        print(f"执行出错: {e}")

if __name__ == "__main__":
    main()
