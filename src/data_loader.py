import pandas as pd
import json
import os
import requests
import sqlite3
import pymysql
import psycopg2
# from sqlalchemy import create_engine  # 不直接导入SQLAlchemy
from typing import Union, Optional, Dict, Any, List, Tuple
import numpy as np
import io
import streamlit as st
import base64


class DataLoader:
    """数据加载类，支持多种数据源加载"""
    
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
        try:
            from sqlalchemy import create_engine
            return create_engine(connection_string)
        except ImportError:
            raise ImportError("未安装SQLAlchemy库，请使用'pip install sqlalchemy'安装")
        except Exception as e:
            raise Exception(f"SQLAlchemy错误: {str(e)}。这可能是由于SQLAlchemy与Python 3.13不兼容导致，请考虑降级Python版本或使用直接数据库连接。")
    
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

    def __init__(self):
        """初始化数据加载器"""
        self.data = None
        self.data_source = None
        self.meta_info = {}
    
    def load_file(self, file, file_type: Optional[str] = None) -> pd.DataFrame:
        """从上传的文件加载数据
        
        Args:
            file: 上传的文件对象
            file_type: 文件类型，如不提供则自动识别
            
        Returns:
            加载的数据
        """
        if file is None:
            raise ValueError("未提供文件")
        
        # 如果文件类型未提供，则从文件名推断
        if file_type is None:
            file_type = file.name.split('.')[-1].lower()
        
        try:
            if file_type in ['csv', 'txt']:
                # 尝试检测分隔符
                content = file.getvalue().decode('utf-8')
                if ',' in content[:1000]:
                    sep = ','
                elif ';' in content[:1000]:
                    sep = ';'
                elif '\t' in content[:1000]:
                    sep = '\t'
                else:
                    sep = None  # 让pandas自己检测
                
                self.data = pd.read_csv(file, sep=sep)
                
            elif file_type in ['xlsx', 'xls']:
                self.data = pd.read_excel(file)
                
            elif file_type == 'json':
                self.data = pd.read_json(file)
                
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
            
            self.data_source = f"文件: {file.name}"
            self.meta_info = {
                "来源": "文件上传",
                "文件名": file.name,
                "文件类型": file_type,
                "数据维度": f"{self.data.shape[0]}行 × {self.data.shape[1]}列"
            }
            
            return self.data
            
        except Exception as e:
            raise ValueError(f"加载文件时出错: {str(e)}")
    
    def load_database(self, db_path: str, query: str) -> pd.DataFrame:
        """从SQLite数据库加载数据
        
        Args:
            db_path: 数据库文件路径
            query: SQL查询语句
            
        Returns:
            查询结果数据
        """
        try:
            # 检查数据库文件是否存在
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"数据库文件不存在: {db_path}")
            
            # 连接数据库并执行查询
            conn = sqlite3.connect(db_path)
            self.data = pd.read_sql(query, conn)
            conn.close()
            
            self.data_source = f"数据库: {os.path.basename(db_path)}"
            self.meta_info = {
                "来源": "SQLite数据库",
                "数据库": os.path.basename(db_path),
                "SQL查询": query,
                "数据维度": f"{self.data.shape[0]}行 × {self.data.shape[1]}列"
            }
            
            return self.data
            
        except Exception as e:
            raise ValueError(f"从数据库加载数据时出错: {str(e)}")
    
    def load_api(self, url: str, params: Optional[Dict] = None, 
               headers: Optional[Dict] = None, format: str = 'json') -> pd.DataFrame:
        """从API加载数据
        
        Args:
            url: API地址
            params: 请求参数
            headers: 请求头
            format: 响应格式，支持'json'和'csv'
            
        Returns:
            API响应数据
        """
        try:
            # 默认参数和头
            if params is None:
                params = {}
            if headers is None:
                headers = {'Content-Type': 'application/json'}
            
            # 发送请求
            response = requests.get(url, params=params, headers=headers)
            
            # 检查响应状态
            if response.status_code != 200:
                raise ValueError(f"API请求失败，状态码: {response.status_code}, 错误: {response.text}")
            
            # 根据格式处理响应数据
            if format.lower() == 'json':
                # JSON格式响应
                json_data = response.json()
                
                # 处理不同的JSON结构
                if isinstance(json_data, list):
                    # 列表数据直接转换为DataFrame
                    self.data = pd.DataFrame(json_data)
                elif isinstance(json_data, dict):
                    # 尝试找到数据数组
                    for key, value in json_data.items():
                        if isinstance(value, list) and len(value) > 0:
                            self.data = pd.DataFrame(value)
                            break
                    
                    # 如果没有找到数据数组，则使用整个字典
                    if self.data is None:
                        self.data = pd.DataFrame([json_data])
                else:
                    raise ValueError("不支持的JSON响应格式")
                    
            elif format.lower() == 'csv':
                # CSV格式响应
                self.data = pd.read_csv(io.StringIO(response.text))
                
            else:
                raise ValueError(f"不支持的响应格式: {format}")
            
            self.data_source = f"API: {url}"
            self.meta_info = {
                "来源": "API",
                "URL": url,
                "参数": str(params),
                "格式": format,
                "数据维度": f"{self.data.shape[0]}行 × {self.data.shape[1]}列"
            }
            
            return self.data
            
        except Exception as e:
            raise ValueError(f"从API加载数据时出错: {str(e)}")
    
    def generate_sample_data(self, data_type: str = 'sales', rows: int = 100) -> pd.DataFrame:
        """生成示例数据
        
        Args:
            data_type: 数据类型，支持'sales'（销售数据）, 'stock'（股票数据）, 'survey'（问卷数据）
            rows: 生成的行数
            
        Returns:
            生成的示例数据
        """
        np.random.seed(42)  # 固定随机种子以便复现
        
        if data_type == 'sales':
            # 销售数据
            dates = pd.date_range(start='2023-01-01', periods=rows)
            products = ['电子产品', '家居用品', '服装', '食品', '饮料', '办公用品', '健康产品']
            regions = ['华东', '华南', '华北', '西南', '西北', '东北', '华中']
            
            self.data = pd.DataFrame({
                '日期': dates,
                '产品': np.random.choice(products, size=rows),
                '区域': np.random.choice(regions, size=rows),
                '销售额': np.random.randint(1000, 20000, size=rows),
                '成本': np.random.randint(500, 15000, size=rows),
                '利润': np.random.randint(100, 5000, size=rows),
                '客户年龄': np.random.randint(18, 70, size=rows),
                '客户性别': np.random.choice(['男', '女'], size=rows),
                '满意度': np.random.randint(1, 6, size=rows)  # 1-5星评分
            })
            
            self.data_source = "示例数据: 销售数据"
            
        elif data_type == 'stock':
            # 股票数据
            dates = pd.date_range(start='2023-01-01', periods=rows)
            stock_names = ['科技股A', '金融股B', '医疗股C', '消费股D', '能源股E']
            
            base_prices = {name: np.random.randint(50, 500) for name in stock_names}
            
            data_rows = []
            for date in dates:
                for name in stock_names:
                    # 随机波动
                    volatility = np.random.normal(0, 0.02)
                    base_price = base_prices[name]
                    open_price = base_price * (1 + np.random.normal(0, 0.01))
                    close_price = open_price * (1 + volatility)
                    high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
                    low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
                    volume = np.random.randint(10000, 1000000)
                    
                    # 更新基础价格
                    base_prices[name] = close_price
                    
                    data_rows.append({
                        '日期': date,
                        '股票名称': name,
                        '开盘价': round(open_price, 2),
                        '收盘价': round(close_price, 2),
                        '最高价': round(high_price, 2),
                        '最低价': round(low_price, 2),
                        '成交量': volume
                    })
            
            self.data = pd.DataFrame(data_rows)
            self.data_source = "示例数据: 股票数据"
            
        elif data_type == 'survey':
            # 问卷调查数据
            age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
            education = ['高中', '大专', '本科', '硕士', '博士']
            income = ['<3000', '3000-5000', '5000-10000', '10000-20000', '>20000']
            rating_cols = ['产品质量', '价格合理性', '客户服务', '品牌信任度', '整体满意度']
            
            self.data = pd.DataFrame({
                '年龄段': np.random.choice(age_groups, size=rows),
                '性别': np.random.choice(['男', '女'], size=rows),
                '教育程度': np.random.choice(education, size=rows),
                '月收入': np.random.choice(income, size=rows),
                '使用频率': np.random.choice(['每天', '每周几次', '每月几次', '很少'], size=rows),
                '购买渠道': np.random.choice(['线上', '线下', '两者都有'], size=rows)
            })
            
            # 添加评分列(1-5分)
            for col in rating_cols:
                self.data[col] = np.random.randint(1, 6, size=rows)
            
            # 添加推荐意愿(0-10分)
            self.data['推荐意愿'] = np.random.randint(0, 11, size=rows)
            
            # 添加开放性问题回答
            feedback = [
                '很满意，产品质量不错', '价格有点贵，但质量好', '客服响应速度需要提高',
                '包装可以改进', '整体使用体验良好', '功能可以再丰富些', '比竞品好用',
                '会继续购买', '有小问题但可以接受', '超出预期', '一般，符合预期'
            ]
            self.data['反馈意见'] = np.random.choice(feedback, size=rows)
            
            self.data_source = "示例数据: 问卷调查"
        
        else:
            raise ValueError(f"不支持的示例数据类型: {data_type}")
        
        self.meta_info = {
            "来源": "示例数据",
            "数据类型": data_type,
            "样本数量": rows,
            "数据维度": f"{self.data.shape[0]}行 × {self.data.shape[1]}列"
        }
        
        return self.data
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """获取当前加载的数据
        
        Returns:
            当前数据或None(如果未加载)
        """
        return self.data
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据信息
        
        Returns:
            数据来源和元信息
        """
        if self.data is None:
            return {"状态": "未加载数据"}
        
        info = {
            "数据源": self.data_source,
            "行数": self.data.shape[0],
            "列数": self.data.shape[1],
            "内存使用": f"{self.data.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
        }
        
        # 添加元信息
        info.update(self.meta_info)
        
        return info
    
    def get_data_preview(self, rows: int = 5) -> Optional[pd.DataFrame]:
        """获取数据预览
        
        Args:
            rows: 预览的行数
            
        Returns:
            数据预览
        """
        if self.data is None:
            return None
            
        return self.data.head(rows)
    
    def export_data(self, format: str = 'csv') -> Tuple[str, str, str]:
        """导出数据
        
        Args:
            format: 导出格式，支持'csv', 'excel', 'json'
            
        Returns:
            (数据, 文件名, MIME类型)
        """
        if self.data is None:
            raise ValueError("没有数据可导出")
        
        if format == 'csv':
            data = self.data.to_csv(index=False)
            filename = "exported_data.csv"
            mime = "text/csv"
        elif format == 'excel':
            # 使用BytesIO对象导出Excel
            output = io.BytesIO()
            self.data.to_excel(output, index=False)
            data = base64.b64encode(output.getvalue()).decode()
            filename = "exported_data.xlsx"
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif format == 'json':
            data = self.data.to_json(orient='records')
            filename = "exported_data.json"
            mime = "application/json"
        else:
            raise ValueError(f"不支持的导出格式: {format}")
        
        return data, filename, mime
    
    def create_download_link(self, format: str = 'csv') -> str:
        """创建下载链接
        
        Args:
            format: 导出格式
            
        Returns:
            HTML下载链接
        """
        data, filename, mime = self.export_data(format)
        
        if format == 'excel':
            # Excel导出需要特殊处理
            href = f'<a href="data:{mime};base64,{data}" download="{filename}">下载 {filename}</a>'
        else:
            b64 = base64.b64encode(data.encode()).decode()
            href = f'<a href="data:{mime};base64,{b64}" download="{filename}">下载 {filename}</a>'
        
        return href 