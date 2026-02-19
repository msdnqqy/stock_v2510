"""
数据库模型定义
"""
from sqlalchemy import Column, String, Float, BigInteger, Date, DateTime, Integer, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DailyData(Base):
    """日线数据表"""
    __tablename__ = 'daily_data'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_code = Column(String(20), nullable=False, comment='股票代码')
    trade_date = Column(Date, nullable=False, comment='交易日期')
    open = Column(Float, comment='开盘价')
    high = Column(Float, comment='最高价')
    low = Column(Float, comment='最低价')
    close = Column(Float, comment='收盘价')
    pre_close = Column(Float, comment='前收盘价')
    change = Column(Float, comment='涨跌额')
    pct_change = Column(Float, comment='涨跌幅(%)')
    volume = Column(BigInteger, comment='成交量(股)')
    amount = Column(Float, comment='成交额(元)')
    turnover_rate = Column(Float, comment='换手率(%)')
    volume_ratio = Column(Float, comment='量比')
    pe_ratio = Column(Float, comment='市盈率')
    pb_ratio = Column(Float, comment='市净率')
    ps_ratio = Column(Float, comment='市销率')
    total_market_cap = Column(Float, comment='总市值(元)')
    circulating_market_cap = Column(Float, comment='流通市值(元)')
    created_at = Column(DateTime, comment='创建时间')
    updated_at = Column(DateTime, comment='更新时间')
    
    # 创建复合索引
    __table_args__ = (
        Index('idx_stock_date', 'stock_code', 'trade_date', unique=True),
        Index('idx_trade_date', 'trade_date'),
    )


class MinuteData(Base):
    """分钟线数据表"""
    __tablename__ = 'minute_data'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_code = Column(String(20), nullable=False, comment='股票代码')
    trade_datetime = Column(DateTime, nullable=False, comment='交易时间')
    open = Column(Float, comment='开盘价')
    high = Column(Float, comment='最高价')
    low = Column(Float, comment='最低价')
    close = Column(Float, comment='收盘价')
    volume = Column(BigInteger, comment='成交量(股)')
    amount = Column(Float, comment='成交额(元)')
    avg_price = Column(Float, comment='均价')
    created_at = Column(DateTime, comment='创建时间')
    updated_at = Column(DateTime, comment='更新时间')
    
    # 创建复合索引
    __table_args__ = (
        Index('idx_stock_datetime', 'stock_code', 'trade_datetime', unique=True),
        Index('idx_trade_datetime', 'trade_datetime'),
    )


class StockInfo(Base):
    """股票基本信息表"""
    __tablename__ = 'stock_info'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(20), nullable=False, unique=True, comment='股票代码')
    stock_name = Column(String(100), comment='股票名称')
    exchange = Column(String(20), comment='交易所(SH/SZ)')
    list_date = Column(Date, comment='上市日期')
    delist_date = Column(Date, comment='退市日期')
    status = Column(String(20), comment='状态(L-上市 D-退市 P-暂停)')
    created_at = Column(DateTime, comment='创建时间')
    updated_at = Column(DateTime, comment='更新时间')
    
    __table_args__ = (
        Index('idx_stock_code', 'stock_code'),
    )
