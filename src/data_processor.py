import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataProcessor:
    """数据处理类，提供数据清洗、转换和分析功能"""
    
    def __init__(self, df: pd.DataFrame):
        """初始化数据处理器
        
        Args:
            df: 待处理的DataFrame对象
        """
        self.df = df.copy()
        self.original_df = df.copy()
    
    def reset(self) -> None:
        """重置数据到原始状态"""
        self.df = self.original_df.copy()
    
    def get_basic_info(self) -> Dict[str, Any]:
        """获取数据基本信息
        
        Returns:
            包含数据基本信息的字典
        """
        info = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "missing_percentage": (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            "numeric_columns": list(self.df.select_dtypes(include=['int64', 'float64']).columns),
            "categorical_columns": list(self.df.select_dtypes(include=['object', 'category']).columns)
        }
        return info
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息
        
        Returns:
            包含数据统计信息的字典
        """
        numeric_stats = self.df.describe().to_dict()
        
        categorical_stats = {}
        for col in self.df.select_dtypes(include=['object', 'category']).columns:
            categorical_stats[col] = {
                "unique_values": self.df[col].nunique(),
                "top_values": self.df[col].value_counts().head(5).to_dict()
            }
        
        return {
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats
        }
    
    def handle_missing_values(self, strategy: str = 'mean',
                             columns: Optional[List[str]] = None) -> 'DataProcessor':
        """处理缺失值
        
        Args:
            strategy: 填充策略，可选 'mean', 'median', 'most_frequent', 'constant'
            columns: 需要处理的列，如果为None则处理所有数值型列
            
        Returns:
            处理后的DataProcessor对象（链式调用）
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        if strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
            self.df[columns] = imputer.fit_transform(self.df[columns])
        elif strategy == 'drop':
            self.df = self.df.dropna(subset=columns)
        
        return self
    
    def normalize_data(self, method: str = 'standard',
                      columns: Optional[List[str]] = None) -> 'DataProcessor':
        """标准化/归一化数据
        
        Args:
            method: 标准化方法，可选 'standard'(标准化) 或 'minmax'(归一化)
            columns: 需要处理的列，如果为None则处理所有数值型列
            
        Returns:
            处理后的DataProcessor对象（链式调用）
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
        
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self
    
    def encode_categorical(self, columns: Optional[List[str]] = None,
                          method: str = 'onehot') -> 'DataProcessor':
        """对分类变量进行编码
        
        Args:
            columns: 需要编码的列，如果为None则编码所有分类型列
            method: 编码方法，可选 'onehot' 或 'label'
            
        Returns:
            处理后的DataProcessor对象（链式调用）
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns
        
        if method == 'onehot':
            self.df = pd.get_dummies(self.df, columns=columns)
        elif method == 'label':
            for col in columns:
                self.df[col] = self.df[col].astype('category').cat.codes
        
        return self
    
    def filter_data(self, conditions: Dict[str, Any]) -> 'DataProcessor':
        """根据条件筛选数据
        
        Args:
            conditions: 筛选条件字典，格式为 {列名: 条件值}
            
        Returns:
            处理后的DataProcessor对象（链式调用）
        """
        for col, condition in conditions.items():
            if isinstance(condition, (list, tuple)):
                self.df = self.df[self.df[col].isin(condition)]
            else:
                self.df = self.df[self.df[col] == condition]
        
        return self
    
    def add_feature(self, feature_name: str, expression: str) -> 'DataProcessor':
        """添加新特征
        
        Args:
            feature_name: 新特征名称
            expression: 基于现有列的表达式，如 "col1 + col2"
            
        Returns:
            处理后的DataProcessor对象（链式调用）
        """
        self.df[feature_name] = eval(f"self.df.{expression}")
        return self
    
    def get_correlation(self, method: str = 'pearson') -> pd.DataFrame:
        """计算数值型列的相关系数
        
        Args:
            method: 相关系数计算方法，可选 'pearson', 'kendall', 'spearman'
            
        Returns:
            相关系数矩阵
        """
        numeric_df = self.df.select_dtypes(include=['int64', 'float64'])
        return numeric_df.corr(method=method)
    
    def get_result(self) -> pd.DataFrame:
        """获取处理后的DataFrame结果
        
        Returns:
            处理后的DataFrame对象
        """
        return self.df.copy() 