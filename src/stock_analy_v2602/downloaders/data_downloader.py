"""
股票数据下载器 - 支持多个数据源
"""
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, List
from config.config import API_CONFIG, DOWNLOAD_CONFIG

logger = logging.getLogger(__name__)


class StockDataDownloader:
    """股票数据下载器基类"""
    
    def __init__(self, source='akshare'):
        """
        初始化下载器
        :param source: 数据源 (akshare, tushare, yfinance)
        """
        self.source = source
        self._init_api()
    
    def _init_api(self):
        """初始化API"""
        if self.source == 'akshare':
            try:
                import akshare as ak
                self.ak = ak
                logger.info("AKShare API初始化成功")
            except ImportError:
                logger.error("请先安装akshare: pip install akshare")
                raise
        
        elif self.source == 'tushare':
            try:
                import tushare as ts
                if not API_CONFIG.get('tushare_token'):
                    raise ValueError("请在config.py中配置tushare_token")
                ts.set_token(API_CONFIG['tushare_token'])
                self.ts = ts.pro_api()
                logger.info("Tushare API初始化成功")
            except ImportError:
                logger.error("请先安装tushare: pip install tushare")
                raise
        
        elif self.source == 'yfinance':
            try:
                import yfinance as yf
                self.yf = yf
                logger.info("yfinance API初始化成功")
            except ImportError:
                logger.error("请先安装yfinance: pip install yfinance")
                raise
    
    def download_daily_data(self, stock_code: str, start_date: str = None, 
                          end_date: str = None) -> pd.DataFrame:
        """
        下载日线数据
        :param stock_code: 股票代码 (如: '000001' 或 'sh000001')
        :param start_date: 开始日期 (格式: '20200101' 或 '2020-01-01')
        :param end_date: 结束日期
        :return: DataFrame
        """
        if self.source == 'akshare':
            return self._download_daily_akshare(stock_code, start_date, end_date)
        elif self.source == 'tushare':
            return self._download_daily_tushare(stock_code, start_date, end_date)
        elif self.source == 'yfinance':
            return self._download_daily_yfinance(stock_code, start_date, end_date)
    
    def _download_daily_akshare(self, stock_code: str, start_date: str = None,
                                end_date: str = None) -> pd.DataFrame:
        """使用AKShare下载日线数据"""
        try:
            # 格式化股票代码
            symbol = self._format_stock_code(stock_code)
            
            # 下载历史数据
            df = self.ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace('-', '') if start_date else "19900101",
                end_date=end_date.replace('-', '') if end_date else datetime.now().strftime('%Y%m%d'),
                adjust="qfq"  # 前复权
            )
            
            if df.empty:
                logger.warning(f"股票 {stock_code} 没有数据")
                return pd.DataFrame()
            
            # 标准化列名
            df = self._standardize_daily_columns(df, stock_code)
            
            logger.info(f"成功下载 {stock_code} 的日线数据，共 {len(df)} 条")
            return df
            
        except Exception as e:
            logger.error(f"下载 {stock_code} 日线数据失败: {e}")
            return pd.DataFrame()
    
    def _download_daily_tushare(self, stock_code: str, start_date: str = None,
                               end_date: str = None) -> pd.DataFrame:
        """使用Tushare下载日线数据"""
        try:
            # Tushare股票代码格式: 000001.SZ
            ts_code = self._format_tushare_code(stock_code)
            
            df = self.ts.daily(
                ts_code=ts_code,
                start_date=start_date.replace('-', '') if start_date else None,
                end_date=end_date.replace('-', '') if end_date else None
            )
            
            if df.empty:
                logger.warning(f"股票 {stock_code} 没有数据")
                return pd.DataFrame()
            
            # 获取额外指标
            df_adj = self.ts.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
            df_daily_basic = self.ts.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,ps,total_mv,circ_mv'
            )
            
            # 合并数据
            df = df.merge(df_adj, on=['ts_code', 'trade_date'], how='left')
            df = df.merge(df_daily_basic, on=['ts_code', 'trade_date'], how='left')
            
            df = self._standardize_daily_columns_tushare(df, stock_code)
            
            logger.info(f"成功下载 {stock_code} 的日线数据，共 {len(df)} 条")
            return df
            
        except Exception as e:
            logger.error(f"下载 {stock_code} 日线数据失败: {e}")
            return pd.DataFrame()
    
    def _download_daily_yfinance(self, stock_code: str, start_date: str = None,
                                end_date: str = None) -> pd.DataFrame:
        """使用yfinance下载日线数据（主要用于美股、港股）"""
        try:
            # yfinance股票代码格式示例: 0700.HK, AAPL
            ticker = self.yf.Ticker(stock_code)
            
            df = ticker.history(
                start=start_date if start_date else "1990-01-01",
                end=end_date if end_date else datetime.now().strftime('%Y-%m-%d')
            )
            
            if df.empty:
                logger.warning(f"股票 {stock_code} 没有数据")
                return pd.DataFrame()
            
            df = self._standardize_daily_columns_yfinance(df, stock_code)
            
            logger.info(f"成功下载 {stock_code} 的日线数据，共 {len(df)} 条")
            return df
            
        except Exception as e:
            logger.error(f"下载 {stock_code} 日线数据失败: {e}")
            return pd.DataFrame()
    
    def download_minute_data(self, stock_code: str, period: str = '1',
                           start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        下载分钟线数据
        :param stock_code: 股票代码
        :param period: 周期 ('1', '5', '15', '30', '60')
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: DataFrame
        """
        if self.source == 'akshare':
            return self._download_minute_akshare(stock_code, period, start_date, end_date)
        elif self.source == 'tushare':
            return self._download_minute_tushare(stock_code, period, start_date, end_date)
        elif self.source == 'yfinance':
            return self._download_minute_yfinance(stock_code, period)
    
    def _download_minute_akshare(self, stock_code: str, period: str = '1',
                                start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """使用AKShare下载分钟线数据"""
        try:
            symbol = self._format_stock_code(stock_code)
            
            df = self.ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period=period,
                start_date=start_date.replace('-', ' ') if start_date else None,
                end_date=end_date.replace('-', ' ') if end_date else None,
                adjust="qfq"
            )
            
            if df.empty:
                logger.warning(f"股票 {stock_code} 没有分钟数据")
                return pd.DataFrame()
            
            df = self._standardize_minute_columns(df, stock_code)
            
            logger.info(f"成功下载 {stock_code} 的{period}分钟线数据，共 {len(df)} 条")
            return df
            
        except Exception as e:
            logger.error(f"下载 {stock_code} 分钟线数据失败: {e}")
            return pd.DataFrame()
    
    def _download_minute_tushare(self, stock_code: str, period: str = '1',
                                start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """使用Tushare下载分钟线数据（需要高级权限）"""
        try:
            ts_code = self._format_tushare_code(stock_code)
            
            # Tushare分钟数据需要积分权限
            df = self.ts.stk_mins(
                ts_code=ts_code,
                freq=f'{period}min',
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                logger.warning(f"股票 {stock_code} 没有分钟数据")
                return pd.DataFrame()
            
            df = self._standardize_minute_columns_tushare(df, stock_code)
            
            logger.info(f"成功下载 {stock_code} 的{period}分钟线数据，共 {len(df)} 条")
            return df
            
        except Exception as e:
            logger.error(f"下载 {stock_code} 分钟线数据失败: {e}")
            return pd.DataFrame()
    
    def _download_minute_yfinance(self, stock_code: str, period: str = '1') -> pd.DataFrame:
        """使用yfinance下载分钟线数据（最近几天）"""
        try:
            ticker = self.yf.Ticker(stock_code)
            
            # yfinance分钟数据只能获取最近几天
            interval_map = {'1': '1m', '5': '5m', '15': '15m', '30': '30m', '60': '60m'}
            interval = interval_map.get(period, '1m')
            
            df = ticker.history(period="5d", interval=interval)
            
            if df.empty:
                logger.warning(f"股票 {stock_code} 没有分钟数据")
                return pd.DataFrame()
            
            df = self._standardize_minute_columns_yfinance(df, stock_code)
            
            logger.info(f"成功下载 {stock_code} 的{period}分钟线数据，共 {len(df)} 条")
            return df
            
        except Exception as e:
            logger.error(f"下载 {stock_code} 分钟线数据失败: {e}")
            return pd.DataFrame()
    
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取A股所有股票列表
        :return: DataFrame (code, name)
        """
        if self.source == 'akshare':
            try:
                # 获取实时行情数据，包含所有股票代码和名称
                df = self.ak.stock_zh_a_spot_em()
                
                # 重命名列以匹配模型
                df = df.rename(columns={
                    '代码': 'stock_code',
                    '名称': 'stock_name'
                })
                
                # 只保留需要的列
                return df[['stock_code', 'stock_name']]
            except Exception as e:
                logger.error(f"获取股票列表失败: {e}")
                return pd.DataFrame()
        else:
            logger.warning(f"目前仅支持 akshare 获取股票列表，当前源: {self.source}")
            return pd.DataFrame()

    def _format_stock_code(self, stock_code: str) -> str:
        """格式化股票代码为AKShare格式"""
        # 移除可能的前缀
        code = stock_code.replace('sh', '').replace('sz', '').replace('SH', '').replace('SZ', '')
        return code
    
    def _format_tushare_code(self, stock_code: str) -> str:
        """格式化股票代码为Tushare格式 (如: 000001.SZ)"""
        code = stock_code.replace('sh', '').replace('sz', '').replace('SH', '').replace('SZ', '')
        
        if code.startswith('6'):
            return f"{code}.SH"
        elif code.startswith(('0', '3')):
            return f"{code}.SZ"
        else:
            return code
    
    def _standardize_daily_columns(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """标准化AKShare日线数据列名"""
        column_mapping = {
            '日期': 'trade_date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover_rate',
        }
        
        df = df.rename(columns=column_mapping)
        df['stock_code'] = stock_code
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['created_at'] = datetime.now()
        df['updated_at'] = datetime.now()
        
        return df
    
    def _standardize_daily_columns_tushare(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """标准化Tushare日线数据列名"""
        column_mapping = {
            'trade_date': 'trade_date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'pre_close': 'pre_close',
            'change': 'change',
            'pct_chg': 'pct_change',
            'vol': 'volume',
            'amount': 'amount',
            'turnover_rate': 'turnover_rate',
            'volume_ratio': 'volume_ratio',
            'pe': 'pe_ratio',
            'pb': 'pb_ratio',
            'ps': 'ps_ratio',
            'total_mv': 'total_market_cap',
            'circ_mv': 'circulating_market_cap',
        }
        
        df = df.rename(columns=column_mapping)
        df['stock_code'] = stock_code
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['created_at'] = datetime.now()
        df['updated_at'] = datetime.now()
        
        # 转换单位（Tushare的成交量单位是手，需要转换为股）
        if 'volume' in df.columns:
            df['volume'] = df['volume'] * 100
        
        return df
    
    def _standardize_daily_columns_yfinance(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """标准化yfinance日线数据列名"""
        df = df.reset_index()
        
        column_mapping = {
            'Date': 'trade_date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        }
        
        df = df.rename(columns=column_mapping)
        df['stock_code'] = stock_code
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df['created_at'] = datetime.now()
        df['updated_at'] = datetime.now()
        
        return df
    
    def _standardize_minute_columns(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """标准化AKShare分钟线数据列名"""
        column_mapping = {
            '时间': 'trade_datetime',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
        }
        
        df = df.rename(columns=column_mapping)
        df['stock_code'] = stock_code
        df['trade_datetime'] = pd.to_datetime(df['trade_datetime'])
        df['created_at'] = datetime.now()
        df['updated_at'] = datetime.now()
        
        return df
    
    def _standardize_minute_columns_tushare(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """标准化Tushare分钟线数据列名"""
        column_mapping = {
            'trade_time': 'trade_datetime',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount',
        }
        
        df = df.rename(columns=column_mapping)
        df['stock_code'] = stock_code
        df['trade_datetime'] = pd.to_datetime(df['trade_datetime'])
        df['created_at'] = datetime.now()
        df['updated_at'] = datetime.now()
        
        if 'volume' in df.columns:
            df['volume'] = df['volume'] * 100
        
        return df
    
    def _standardize_minute_columns_yfinance(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """标准化yfinance分钟线数据列名"""
        df = df.reset_index()
        
        column_mapping = {
            'Datetime': 'trade_datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        }
        
        df = df.rename(columns=column_mapping)
        df['stock_code'] = stock_code
        df['trade_datetime'] = pd.to_datetime(df['trade_datetime'])
        df['created_at'] = datetime.now()
        df['updated_at'] = datetime.now()
        
        return df
