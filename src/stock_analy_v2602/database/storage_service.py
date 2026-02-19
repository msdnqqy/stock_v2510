"""
数据存储服务 - 将下载的数据保存到数据库
"""
import logging
import pandas as pd
from datetime import datetime
from sqlalchemy import and_
from typing import List
from database.db_manager import DatabaseManager
from models.models import DailyData, MinuteData, StockInfo

logger = logging.getLogger(__name__)


class DataStorageService:
    """数据存储服务"""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        初始化存储服务
        :param db_manager: 数据库管理器
        """
        self.db_manager = db_manager
    
    def save_daily_data(self, df: pd.DataFrame, update_mode='replace') -> int:
        """
        保存日线数据
        :param df: 数据DataFrame
        :param update_mode: 更新模式 ('replace'-替换, 'append'-追加, 'update'-更新)
        :return: 保存的记录数
        """
        if df.empty:
            logger.warning("数据为空，跳过保存")
            return 0
        
        saved_count = 0
        
        try:
            with self.db_manager.get_session() as session:
                for _, row in df.iterrows():
                    try:
                        # 检查是否已存在
                        existing = session.query(DailyData).filter(
                            and_(
                                DailyData.stock_code == row['stock_code'],
                                DailyData.trade_date == row['trade_date']
                            )
                        ).first()
                        
                        if existing:
                            if update_mode == 'replace' or update_mode == 'update':
                                # 更新现有记录
                                for key, value in row.items():
                                    if key not in ['id', 'created_at'] and pd.notna(value):
                                        setattr(existing, key, value)
                                existing.updated_at = datetime.now()
                                saved_count += 1
                            # append模式下跳过已存在的记录
                        else:
                            # 插入新记录
                            daily_data = DailyData(**{
                                k: (None if pd.isna(v) else v) 
                                for k, v in row.items() 
                                if k in DailyData.__table__.columns.keys()
                            })
                            session.add(daily_data)
                            saved_count += 1
                    
                    except Exception as e:
                        logger.error(f"保存单条日线数据失败: {e}")
                        continue
                
                session.commit()
                logger.info(f"成功保存 {saved_count} 条日线数据")
                
        except Exception as e:
            logger.error(f"保存日线数据失败: {e}")
            raise
        
        return saved_count
    
    def save_minute_data(self, df: pd.DataFrame, update_mode='replace') -> int:
        """
        保存分钟线数据
        :param df: 数据DataFrame
        :param update_mode: 更新模式
        :return: 保存的记录数
        """
        if df.empty:
            logger.warning("数据为空，跳过保存")
            return 0
        
        saved_count = 0
        
        try:
            with self.db_manager.get_session() as session:
                for _, row in df.iterrows():
                    try:
                        # 检查是否已存在
                        existing = session.query(MinuteData).filter(
                            and_(
                                MinuteData.stock_code == row['stock_code'],
                                MinuteData.trade_datetime == row['trade_datetime']
                            )
                        ).first()
                        
                        if existing:
                            if update_mode == 'replace' or update_mode == 'update':
                                # 更新现有记录
                                for key, value in row.items():
                                    if key not in ['id', 'created_at'] and pd.notna(value):
                                        setattr(existing, key, value)
                                existing.updated_at = datetime.now()
                                saved_count += 1
                        else:
                            # 插入新记录
                            minute_data = MinuteData(**{
                                k: (None if pd.isna(v) else v) 
                                for k, v in row.items() 
                                if k in MinuteData.__table__.columns.keys()
                            })
                            session.add(minute_data)
                            saved_count += 1
                    
                    except Exception as e:
                        logger.error(f"保存单条分钟线数据失败: {e}")
                        continue
                
                session.commit()
                logger.info(f"成功保存 {saved_count} 条分钟线数据")
                
        except Exception as e:
            logger.error(f"保存分钟线数据失败: {e}")
            raise
        
        return saved_count
    
    def save_stock_info(self, stock_code: str, stock_name: str = None, 
                       exchange: str = None, list_date: str = None) -> bool:
        """
        保存股票基本信息
        :param stock_code: 股票代码
        :param stock_name: 股票名称
        :param exchange: 交易所
        :param list_date: 上市日期
        :return: 是否成功
        """
        try:
            with self.db_manager.get_session() as session:
                existing = session.query(StockInfo).filter(
                    StockInfo.stock_code == stock_code
                ).first()
                
                if existing:
                    # 更新信息
                    if stock_name:
                        existing.stock_name = stock_name
                    if exchange:
                        existing.exchange = exchange
                    if list_date:
                        existing.list_date = pd.to_datetime(list_date)
                    existing.updated_at = datetime.now()
                else:
                    # 插入新记录
                    stock_info = StockInfo(
                        stock_code=stock_code,
                        stock_name=stock_name,
                        exchange=exchange,
                        list_date=pd.to_datetime(list_date) if list_date else None,
                        status='L',
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    session.add(stock_info)
                
                session.commit()
                logger.info(f"成功保存股票信息: {stock_code}")
                return True
                
        except Exception as e:
            logger.error(f"保存股票信息失败: {e}")
            return False
    
    def get_latest_date(self, stock_code: str, data_type='daily') -> datetime:
        """
        获取数据库中某股票的最新日期
        :param stock_code: 股票代码
        :param data_type: 数据类型 ('daily' 或 'minute')
        :return: 最新日期
        """
        try:
            with self.db_manager.get_session() as session:
                if data_type == 'daily':
                    result = session.query(DailyData.trade_date).filter(
                        DailyData.stock_code == stock_code
                    ).order_by(DailyData.trade_date.desc()).first()
                    
                    return result[0] if result else None
                
                elif data_type == 'minute':
                    result = session.query(MinuteData.trade_datetime).filter(
                        MinuteData.stock_code == stock_code
                    ).order_by(MinuteData.trade_datetime.desc()).first()
                    
                    return result[0] if result else None
                
        except Exception as e:
            logger.error(f"获取最新日期失败: {e}")
            return None
    
    def get_data_count(self, stock_code: str, data_type='daily') -> int:
        """
        获取数据库中某股票的数据条数
        :param stock_code: 股票代码
        :param data_type: 数据类型
        :return: 数据条数
        """
        try:
            with self.db_manager.get_session() as session:
                if data_type == 'daily':
                    count = session.query(DailyData).filter(
                        DailyData.stock_code == stock_code
                    ).count()
                elif data_type == 'minute':
                    count = session.query(MinuteData).filter(
                        MinuteData.stock_code == stock_code
                    ).count()
                else:
                    count = 0
                
                return count
                
        except Exception as e:
            logger.error(f"获取数据条数失败: {e}")
            return 0
    
    def batch_save_daily_data(self, df_list: List[pd.DataFrame]) -> int:
        """
        批量保存日线数据
        :param df_list: DataFrame列表
        :return: 总保存条数
        """
        total_saved = 0
        for df in df_list:
            saved = self.save_daily_data(df)
            total_saved += saved
        
        logger.info(f"批量保存完成，总计 {total_saved} 条")
        return total_saved
