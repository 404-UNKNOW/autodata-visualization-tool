import pandas as pd
import json
import os
import requests
import sqlite3
import pymysql
import psycopg2
from sqlalchemy import create_engine
from typing import Union, Optional, Dict, Any, List


class DataLoader:
    """数据加载类，支持多种格式的数据导入"""
    
    @staticmethod
    def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
        """加载CSV文件
        
        Args:
            file_path: CSV文件路径
            **kwargs: 传递给pandas.read_csv的参数
            
        Returns:
            DataFrame对象
        """
        return pd.read_csv(file_path, **kwargs)
    
    @staticmethod
    def load_excel(file_path: str, sheet_name: Optional[Union[str, int]] = 0, **kwargs) -> pd.DataFrame:
        """加载Excel文件
        
        Args:
            file_path: Excel文件路径
            sheet_name: 工作表名称或索引
            **kwargs: 传递给pandas.read_excel的参数
            
        Returns:
            DataFrame对象
        """
        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
    
    @staticmethod
    def load_json(file_path: str, **kwargs) -> pd.DataFrame:
        """加载JSON文件
        
        Args:
            file_path: JSON文件路径
            **kwargs: 传递给pandas.read_json的参数
            
        Returns:
            DataFrame对象
        """
        return pd.read_json(file_path, **kwargs)
    
    @staticmethod
    def load_sql(query: str, conn, **kwargs) -> pd.DataFrame:
        """从SQL数据库加载数据
        
        Args:
            query: SQL查询语句
            conn: 数据库连接对象
            **kwargs: 传递给pandas.read_sql的参数
            
        Returns:
            DataFrame对象
        """
        return pd.read_sql(query, conn, **kwargs)
    
    @staticmethod
    def connect_sqlite(db_path: str) -> sqlite3.Connection:
        """连接SQLite数据库
        
        Args:
            db_path: 数据库文件路径
            
        Returns:
            数据库连接对象
        """
        return sqlite3.connect(db_path)
    
    @staticmethod
    def connect_mysql(host: str, user: str, password: str, database: str, port: int = 3306) -> pymysql.connections.Connection:
        """连接MySQL数据库
        
        Args:
            host: 主机地址
            user: 用户名
            password: 密码
            database: 数据库名
            port: 端口号，默认3306
            
        Returns:
            数据库连接对象
        """
        return pymysql.connect(host=host, user=user, password=password, database=database, port=port)
    
    @staticmethod
    def connect_postgresql(host: str, user: str, password: str, database: str, port: int = 5432) -> psycopg2.extensions.connection:
        """连接PostgreSQL数据库
        
        Args:
            host: 主机地址
            user: 用户名
            password: 密码
            database: 数据库名
            port: 端口号，默认5432
            
        Returns:
            数据库连接对象
        """
        return psycopg2.connect(host=host, user=user, password=password, dbname=database, port=port)
    
    @staticmethod
    def create_sqlalchemy_engine(connection_string: str):
        """创建SQLAlchemy引擎
        
        Args:
            connection_string: 连接字符串，如"sqlite:///database.db"、"mysql+pymysql://user:pass@host/db"
            
        Returns:
            SQLAlchemy引擎对象
        """
        return create_engine(connection_string)
    
    @staticmethod
    def load_from_api(url: str, params: Optional[Dict] = None, 
                     headers: Optional[Dict] = None, format: str = 'json',
                     data_path: Optional[str] = None) -> pd.DataFrame:
        """从API获取数据
        
        Args:
            url: API地址
            params: 请求参数
            headers: 请求头
            format: 响应格式，可选'json'或'csv'
            data_path: JSON响应中的数据路径，如'data.items'
            
        Returns:
            DataFrame对象
        """
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # 如果请求失败则抛出异常
        
        if format.lower() == 'json':
            data = response.json()
            # 如果指定了数据路径，则提取对应路径的数据
            if data_path:
                for key in data_path.split('.'):
                    data = data[key]
            return pd.DataFrame(data)
        elif format.lower() == 'csv':
            return pd.read_csv(pd.io.common.StringIO(response.text))
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    @classmethod
    def auto_load(cls, file_path: str) -> pd.DataFrame:
        """根据文件扩展名自动选择加载方法
        
        Args:
            file_path: 文件路径
            
        Returns:
            DataFrame对象
            
        Raises:
            ValueError: 不支持的文件格式
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.csv':
            return cls.load_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            return cls.load_excel(file_path)
        elif ext == '.json':
            return cls.load_json(file_path)
        elif ext == '.db' or ext == '.sqlite':
            conn = cls.connect_sqlite(file_path)
            # 获取第一个表作为示例
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
            if len(tables) > 0:
                return pd.read_sql_query(f"SELECT * FROM {tables.iloc[0, 0]}", conn)
            else:
                raise ValueError("SQLite数据库中没有表")
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """获取文件基本信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            包含文件信息的字典
        """
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        _, ext = os.path.splitext(file_path)
        
        return {
            "file_name": file_name,
            "file_size": file_size,
            "file_type": ext.lstrip('.'),
            "file_path": file_path,
            "last_modified": os.path.getmtime(file_path)
        }
        
    @staticmethod
    def list_database_tables(conn) -> List[str]:
        """列出数据库中的所有表
        
        Args:
            conn: 数据库连接对象
            
        Returns:
            表名列表
        """
        # 检测连接类型并执行相应的查询
        if isinstance(conn, sqlite3.Connection):
            tables_df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
            return list(tables_df['name'])
        elif isinstance(conn, pymysql.connections.Connection):
            tables_df = pd.read_sql_query("SHOW TABLES;", conn)
            return list(tables_df.iloc[:, 0])
        elif isinstance(conn, psycopg2.extensions.connection):
            tables_df = pd.read_sql_query("SELECT table_name FROM information_schema.tables WHERE table_schema='public';", conn)
            return list(tables_df['table_name'])
        else:
            # 通用方法，尝试使用SQLAlchemy
            try:
                from sqlalchemy import inspect
                inspector = inspect(conn)
                return inspector.get_table_names()
            except:
                raise TypeError("不支持的数据库连接类型") 