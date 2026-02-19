
"""
股票数据自动下载脚本
"""
import sys
import logging
import time
from datetime import datetime

# 添加项目路径到系统路径，以便导入模块
sys.path.append('.')

from config.config import TARGET_STOCKS, TARGET_PERIODS, LOG_CONFIG
from main import StockAnalyzer

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['level']),
    format=LOG_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOG_CONFIG['file'], encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def do_download_main():
    """
    主下载函数
    循环下载配置中的股票日线、分时线数据
    """
    logger.info("=" * 60)
    logger.info("开始执行自动下载任务...")
    logger.info(f"目标股票数量: {len(TARGET_STOCKS)}")
    logger.info(f"目标分钟线周期: {TARGET_PERIODS}")
    logger.info("=" * 60)
    
    # 初始化分析器
    try:
        analyzer = StockAnalyzer()
    except Exception as e:
        logger.error(f"初始化 StockAnalyzer 失败: {e}")
        return

    total_start_time = time.time()
    
    # 统计信息
    stats = {
        'daily_success': 0,
        'daily_failed': 0,
        'minute_success': 0,
        'minute_failed': 0
    }

    try:
        # 1. 下载日线数据
        logger.info("\n--- 开始下载日线数据 ---")
        for stock_code in TARGET_STOCKS:
            logger.info(f"处理股票: {stock_code}")
            
            # 检查数据库中是否存在该股票的日线数据
            latest_date = analyzer.storage_service.get_latest_date(stock_code, 'daily')
            
            if latest_date:
                # 存在数据，执行增量更新
                logger.info(f"数据库中已存在 {stock_code} 的日线数据，最新日期: {latest_date}，执行增量更新")
                if analyzer.update_stock_daily(stock_code):
                    stats['daily_success'] += 1
                else:
                    stats['daily_failed'] += 1
                    logger.error(f"股票 {stock_code} 日线数据增量更新失败")
            else:
                # 不存在数据，执行全量下载
                logger.info(f"数据库中不存在 {stock_code} 的日线数据，执行全量下载")
                if analyzer.download_stock_daily(stock_code):
                    stats['daily_success'] += 1
                else:
                    stats['daily_failed'] += 1
                    logger.error(f"股票 {stock_code} 日线数据全量下载失败")
            
            # 避免请求过快
            time.sleep(1)

        # 2. 下载分钟线数据
        logger.info("\n--- 开始下载分钟线数据 ---")
        for stock_code in TARGET_STOCKS:
            for period in TARGET_PERIODS:
                logger.info(f"处理股票 {stock_code} - {period}分钟线")
                
                # 检查数据库中是否存在该股票的分钟线数据
                latest_datetime = analyzer.storage_service.get_latest_date(stock_code, 'minute')
                
                if latest_datetime:
                    # 存在数据，执行增量更新
                    logger.info(f"数据库中已存在 {stock_code} 的{period}分钟线数据，最新时间: {latest_datetime}，执行增量更新")
                    if analyzer.update_stock_minute(stock_code, period):
                        stats['minute_success'] += 1
                    else:
                        stats['minute_failed'] += 1
                        logger.error(f"股票 {stock_code} - {period}分钟线数据增量更新失败")
                else:
                    # 不存在数据，执行全量下载
                    logger.info(f"数据库中不存在 {stock_code} 的{period}分钟线数据，执行全量下载")
                    # 不指定 start_date，让下载器决定（通常是获取可用的全部/近期历史）
                    if analyzer.download_stock_minute(stock_code, period):
                        stats['minute_success'] += 1
                    else:
                        stats['minute_failed'] += 1
                        logger.error(f"股票 {stock_code} - {period}分钟线数据全量下载失败")
                
                # 避免请求过快
                time.sleep(1)

    except KeyboardInterrupt:
        logger.warning("任务被用户中断")
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}")
    finally:
        # 关闭资源
        analyzer.close()
        
        total_end_time = time.time()
        duration = total_end_time - total_start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("下载任务完成统计:")
        logger.info(f"耗时: {duration:.2f} 秒")
        logger.info(f"日线数据: 成功 {stats['daily_success']} / 失败 {stats['daily_failed']}")
        logger.info(f"分钟线数据: 成功 {stats['minute_success']} / 失败 {stats['minute_failed']}")
        logger.info("=" * 60)

if __name__ == "__main__":
    do_download_main()
