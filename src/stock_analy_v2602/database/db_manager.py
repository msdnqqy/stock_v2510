"""
数据库管理模块
"""
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from config.config import DB_CONFIG
from models.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理类"""
    
    def __init__(self):
        """初始化数据库连接"""
        self.engine = None
        self.Session = None
        self._connect()
    
    def _connect(self):
        """创建数据库连接"""
        try:
            # 先连接到MySQL服务器（不指定数据库）
            connection_string = (
                f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
                f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}"
                f"?charset={DB_CONFIG['charset']}"
            )
            temp_engine = create_engine(connection_string)
            
            # 创建数据库（如果不存在）
            with temp_engine.connect() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} "
                                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
                conn.commit()
            
            temp_engine.dispose()
            
            # 连接到指定数据库
            connection_string = (
                f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
                f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
                f"?charset={DB_CONFIG['charset']}"
            )
            
            self.engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            self.Session = sessionmaker(bind=self.engine)
            logger.info(f"成功连接到数据库: {DB_CONFIG['database']}")
            
        except SQLAlchemyError as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def create_tables(self):
        """创建所有表"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("数据库表创建成功")
        except SQLAlchemyError as e:
            logger.error(f"创建表失败: {e}")
            raise
    
    def drop_tables(self):
        """删除所有表（慎用）"""
        try:
            Base.metadata.drop_all(self.engine)
            logger.info("数据库表已删除")
        except SQLAlchemyError as e:
            logger.error(f"删除表失败: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """获取数据库会话（上下文管理器）"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self):
        """测试数据库连接"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            logger.info("数据库连接测试成功")
            return True
        except SQLAlchemyError as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False
    
    def close(self):
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
            logger.info("数据库连接已关闭")
