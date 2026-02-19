"""
股票分析系统 - 主程序入口
"""
import logging
import sys
import argparse
from datetime import datetime, timedelta
from typing import List

# 添加项目路径
sys.path.append('/home/claude/src/stock_analyzer')

from config.config import LOG_CONFIG
from database.db_manager import DatabaseManager
from database.storage_service import DataStorageService
from downloaders.data_downloader import StockDataDownloader

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


class StockAnalyzer:
    """股票分析系统主类"""
    
    def __init__(self, data_source='akshare'):
        """
        初始化系统
        :param data_source: 数据源 (akshare, tushare, yfinance)
        """
        logger.info("=" * 60)
        logger.info("股票分析系统启动中...")
        logger.info("=" * 60)
        
        # 初始化数据库
        self.db_manager = DatabaseManager()
        self.db_manager.create_tables()
        
        # 初始化存储服务
        self.storage_service = DataStorageService(self.db_manager)
        
        # 初始化下载器
        self.downloader = StockDataDownloader(source=data_source)
        
        logger.info("系统初始化完成")
    
    def init_stock_list(self):
        """
        初始化A股股票列表并保存到数据库
        """
        try:
            logger.info("开始获取A股股票列表...")
            df = self.downloader.get_stock_list()
            
            if df.empty:
                logger.warning("未获取到股票列表数据")
                return

            logger.info(f"获取到 {len(df)} 只股票，开始保存到数据库...")
            
            success_count = 0
            for _, row in df.iterrows():
                # 保存股票信息
                if self.storage_service.save_stock_info(
                    stock_code=row['stock_code'],
                    stock_name=row['stock_name']
                ):
                    success_count += 1
            
            logger.info(f"股票列表初始化完成，成功保存 {success_count} 条记录")
            
        except Exception as e:
            logger.error(f"初始化股票列表失败: {e}")

    def download_stock_daily(self, stock_code: str, start_date: str = None, 
                           end_date: str = None, save_to_db: bool = True) -> bool:
        """
        下载股票日线数据
        :param stock_code: 股票代码 (如: '000001', 'sh600000')
        :param start_date: 开始日期 (格式: '2020-01-01')
        :param end_date: 结束日期
        :param save_to_db: 是否保存到数据库
        :return: 是否成功
        """
        try:
            logger.info(f"开始下载 {stock_code} 的日线数据...")
            
            # 下载数据
            df = self.downloader.download_daily_data(stock_code, start_date, end_date)
            
            if df.empty:
                logger.warning(f"{stock_code} 没有数据")
                return False
            
            # 保存到数据库
            if save_to_db:
                saved_count = self.storage_service.save_daily_data(df, update_mode='replace')
                logger.info(f"{stock_code} 日线数据下载并保存成功，共 {saved_count} 条")
            else:
                logger.info(f"{stock_code} 日线数据下载成功，共 {len(df)} 条")
            
            return True
            
        except Exception as e:
            logger.error(f"下载 {stock_code} 日线数据失败: {e}")
            return False
    
    def download_stock_minute(self, stock_code: str, period: str = '1',
                            start_date: str = None, end_date: str = None,
                            save_to_db: bool = True) -> bool:
        """
        下载股票分钟线数据
        :param stock_code: 股票代码
        :param period: 周期 ('1', '5', '15', '30', '60')
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param save_to_db: 是否保存到数据库
        :return: 是否成功
        """
        try:
            logger.info(f"开始下载 {stock_code} 的{period}分钟线数据...")
            
            # 下载数据
            df = self.downloader.download_minute_data(stock_code, period, start_date, end_date)
            
            if df.empty:
                logger.warning(f"{stock_code} 没有分钟数据")
                return False
            
            # 保存到数据库
            if save_to_db:
                saved_count = self.storage_service.save_minute_data(df, update_mode='replace')
                logger.info(f"{stock_code} {period}分钟线数据下载并保存成功，共 {saved_count} 条")
            else:
                logger.info(f"{stock_code} {period}分钟线数据下载成功，共 {len(df)} 条")
            
            return True
            
        except Exception as e:
            logger.error(f"下载 {stock_code} 分钟线数据失败: {e}")
            return False
    
    def update_stock_daily(self, stock_code: str) -> bool:
        """
        增量更新股票日线数据
        :param stock_code: 股票代码
        :return: 是否成功
        """
        try:
            logger.info(f"开始增量更新 {stock_code} 的日线数据...")
            
            # 获取数据库中最新日期
            latest_date = self.storage_service.get_latest_date(stock_code, 'daily')
            
            if latest_date:
                # 从最新日期的下一天开始更新
                start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
                logger.info(f"数据库最新日期: {latest_date.strftime('%Y-%m-%d')}，从 {start_date} 开始更新")
            else:
                # 如果数据库中没有数据，下载全部历史数据
                start_date = None
                logger.info("数据库中无数据，将下载全部历史数据")
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # 下载并保存
            df = self.downloader.download_daily_data(stock_code, start_date, end_date)
            
            if df.empty:
                logger.info(f"{stock_code} 无新数据需要更新")
                return True
            
            saved_count = self.storage_service.save_daily_data(df, update_mode='update')
            logger.info(f"{stock_code} 增量更新完成，新增/更新 {saved_count} 条数据")
            
            return True
            
        except Exception as e:
            logger.error(f"增量更新 {stock_code} 失败: {e}")
            return False
    
    def update_stock_minute(self, stock_code: str, period: str = '1') -> bool:
        """
        增量更新股票分钟线数据
        :param stock_code: 股票代码
        :param period: 周期
        :return: 是否成功
        """
        try:
            logger.info(f"开始增量更新 {stock_code} 的{period}分钟线数据...")
            
            # 获取数据库中最新时间
            latest_datetime = self.storage_service.get_latest_date(stock_code, 'minute')
            
            if latest_datetime:
                start_date = latest_datetime.strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"数据库最新时间: {start_date}，开始增量更新")
            else:
                # 分钟数据通常只下载最近一段时间
                start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
                logger.info("数据库中无数据，将下载最近5天的分钟数据")
            
            end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 下载并保存
            df = self.downloader.download_minute_data(stock_code, period, start_date, end_date)
            
            if df.empty:
                logger.info(f"{stock_code} 无新数据需要更新")
                return True
            
            saved_count = self.storage_service.save_minute_data(df, update_mode='update')
            logger.info(f"{stock_code} 增量更新完成，新增/更新 {saved_count} 条数据")
            
            return True
            
        except Exception as e:
            logger.error(f"增量更新 {stock_code} 分钟线数据失败: {e}")
            return False
    
    def batch_download_daily(self, stock_codes: List[str], start_date: str = None,
                           end_date: str = None) -> dict:
        """
        批量下载日线数据
        :param stock_codes: 股票代码列表
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: 下载结果统计
        """
        logger.info(f"开始批量下载 {len(stock_codes)} 只股票的日线数据...")
        
        results = {'success': 0, 'failed': 0, 'failed_codes': []}
        
        for stock_code in stock_codes:
            success = self.download_stock_daily(stock_code, start_date, end_date)
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
                results['failed_codes'].append(stock_code)
        
        logger.info(f"批量下载完成: 成功 {results['success']} 只，失败 {results['failed']} 只")
        if results['failed_codes']:
            logger.warning(f"失败的股票: {results['failed_codes']}")
        
        return results
    
    def batch_update_daily(self, stock_codes: List[str]) -> dict:
        """
        批量增量更新日线数据
        :param stock_codes: 股票代码列表
        :return: 更新结果统计
        """
        logger.info(f"开始批量更新 {len(stock_codes)} 只股票的日线数据...")
        
        results = {'success': 0, 'failed': 0, 'failed_codes': []}
        
        for stock_code in stock_codes:
            success = self.update_stock_daily(stock_code)
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
                results['failed_codes'].append(stock_code)
        
        logger.info(f"批量更新完成: 成功 {results['success']} 只，失败 {results['failed']} 只")
        if results['failed_codes']:
            logger.warning(f"失败的股票: {results['failed_codes']}")
        
        return results
    
    def show_stats(self, stock_code: str = None):
        """
        显示数据统计信息
        :param stock_code: 股票代码（为空则显示所有）
        """
        if stock_code:
            daily_count = self.storage_service.get_data_count(stock_code, 'daily')
            minute_count = self.storage_service.get_data_count(stock_code, 'minute')
            latest_daily = self.storage_service.get_latest_date(stock_code, 'daily')
            latest_minute = self.storage_service.get_latest_date(stock_code, 'minute')
            
            logger.info(f"\n{'='*60}")
            logger.info(f"股票 {stock_code} 数据统计:")
            logger.info(f"日线数据: {daily_count} 条，最新日期: {latest_daily}")
            logger.info(f"分钟线数据: {minute_count} 条，最新时间: {latest_minute}")
            logger.info(f"{'='*60}\n")
        else:
            logger.info("请指定股票代码")
    
    def close(self):
        """关闭系统"""
        self.db_manager.close()
        logger.info("系统已关闭")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='股票分析系统')
    parser.add_argument('--action', type=str, required=True, 
                       choices=['download_daily', 'download_minute', 'update_daily', 'update_minute', 'stats'],
                       help='操作类型')
    parser.add_argument('--stock', type=str, help='股票代码')
    parser.add_argument('--stocks', type=str, help='股票代码列表（逗号分隔）')
    parser.add_argument('--start', type=str, help='开始日期 (2020-01-01)')
    parser.add_argument('--end', type=str, help='结束日期')
    parser.add_argument('--period', type=str, default='1', help='分钟线周期 (1, 5, 15, 30, 60)')
    parser.add_argument('--source', type=str, default='akshare', 
                       choices=['akshare', 'tushare', 'yfinance'],
                       help='数据源')
    
    args = parser.parse_args()
    
    # 初始化系统
    analyzer = StockAnalyzer(data_source=args.source)
    
    try:
        if args.action == 'download_daily':
            if args.stocks:
                # 批量下载
                stock_list = [s.strip() for s in args.stocks.split(',')]
                analyzer.batch_download_daily(stock_list, args.start, args.end)
            elif args.stock:
                # 单个下载
                analyzer.download_stock_daily(args.stock, args.start, args.end)
            else:
                logger.error("请指定 --stock 或 --stocks 参数")
        
        elif args.action == 'download_minute':
            if args.stock:
                analyzer.download_stock_minute(args.stock, args.period, args.start, args.end)
            else:
                logger.error("请指定 --stock 参数")
        
        elif args.action == 'update_daily':
            if args.stocks:
                # 批量更新
                stock_list = [s.strip() for s in args.stocks.split(',')]
                analyzer.batch_update_daily(stock_list)
            elif args.stock:
                # 单个更新
                analyzer.update_stock_daily(args.stock)
            else:
                logger.error("请指定 --stock 或 --stocks 参数")
        
        elif args.action == 'update_minute':
            if args.stock:
                analyzer.update_stock_minute(args.stock, args.period)
            else:
                logger.error("请指定 --stock 参数")
        
        elif args.action == 'stats':
            if args.stock:
                analyzer.show_stats(args.stock)
            else:
                logger.error("请指定 --stock 参数")
    
    finally:
        analyzer.close()


if __name__ == '__main__':
    main()
