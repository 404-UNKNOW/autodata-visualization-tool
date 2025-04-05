import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Union, Tuple


class Visualizer:
    """数据可视化类，提供各种图表和可视化功能"""
    
    def __init__(self, df: pd.DataFrame):
        """初始化可视化器
        
        Args:
            df: 数据源DataFrame对象
        """
        self.df = df.copy()
        self.numeric_cols = list(df.select_dtypes(include=['int64', 'float64']).columns)
        self.categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
        
        # 设置默认样式
        sns.set(style="whitegrid")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    def distribution_plot(self, column: str, bins: int = 30, 
                         kde: bool = True, fig_size: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """绘制数值分布图
        
        Args:
            column: 列名
            bins: 直方图的箱数
            kde: 是否显示核密度估计曲线
            fig_size: 图形尺寸
            
        Returns:
            matplotlib图形对象
        """
        plt.figure(figsize=fig_size)
        sns.histplot(data=self.df, x=column, bins=bins, kde=kde)
        plt.title(f"{column}的分布")
        plt.xlabel(column)
        plt.ylabel("频数")
        plt.tight_layout()
        return plt.gcf()
    
    def boxplot(self, column: str, by: Optional[str] = None, 
               fig_size: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """绘制箱线图
        
        Args:
            column: 要绘制的数值列名
            by: 可选的分组列名
            fig_size: 图形尺寸
            
        Returns:
            matplotlib图形对象
        """
        plt.figure(figsize=fig_size)
        if by:
            sns.boxplot(data=self.df, x=by, y=column)
            plt.title(f"{column}按{by}分组的箱线图")
        else:
            sns.boxplot(data=self.df, y=column)
            plt.title(f"{column}的箱线图")
        plt.tight_layout()
        return plt.gcf()
    
    def scatter_plot(self, x: str, y: str, hue: Optional[str] = None, 
                    size: Optional[str] = None, fig_size: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """绘制散点图
        
        Args:
            x: X轴列名
            y: Y轴列名
            hue: 可选的颜色分组列名
            size: 可选的点大小列名
            fig_size: 图形尺寸
            
        Returns:
            matplotlib图形对象
        """
        plt.figure(figsize=fig_size)
        sns.scatterplot(data=self.df, x=x, y=y, hue=hue, size=size)
        plt.title(f"{x}与{y}的散点图")
        plt.tight_layout()
        return plt.gcf()
    
    def correlation_heatmap(self, columns: Optional[List[str]] = None, 
                          fig_size: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """绘制相关系数热力图
        
        Args:
            columns: 要包含的列，默认为所有数值列
            fig_size: 图形尺寸
            
        Returns:
            matplotlib图形对象
        """
        if columns is None:
            columns = self.numeric_cols
        
        corr_matrix = self.df[columns].corr()
        plt.figure(figsize=fig_size)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("相关系数热力图")
        plt.tight_layout()
        return plt.gcf()
    
    def bar_chart(self, x: str, y: Optional[str] = None, 
                hue: Optional[str] = None, fig_size: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """绘制条形图
        
        Args:
            x: X轴列名
            y: Y轴列名，默认为计数
            hue: 可选的颜色分组列名
            fig_size: 图形尺寸
            
        Returns:
            matplotlib图形对象
        """
        plt.figure(figsize=fig_size)
        if y:
            sns.barplot(data=self.df, x=x, y=y, hue=hue)
            plt.title(f"{x}与{y}的条形图")
        else:
            # 计数图
            value_counts = self.df[x].value_counts().reset_index()
            value_counts.columns = [x, 'count']
            sns.barplot(data=value_counts, x=x, y='count')
            plt.title(f"{x}的计数条形图")
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def line_plot(self, x: str, y: str, hue: Optional[str] = None, 
                fig_size: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """绘制折线图
        
        Args:
            x: X轴列名
            y: Y轴列名
            hue: 可选的颜色分组列名
            fig_size: 图形尺寸
            
        Returns:
            matplotlib图形对象
        """
        plt.figure(figsize=fig_size)
        sns.lineplot(data=self.df, x=x, y=y, hue=hue)
        plt.title(f"{x}与{y}的折线图")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def pie_chart(self, column: str, fig_size: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """绘制饼图
        
        Args:
            column: 分类列名
            fig_size: 图形尺寸
            
        Returns:
            matplotlib图形对象
        """
        value_counts = self.df[column].value_counts()
        plt.figure(figsize=fig_size)
        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(f"{column}的饼图")
        plt.tight_layout()
        return plt.gcf()
    
    def pair_plot(self, columns: Optional[List[str]] = None, 
                hue: Optional[str] = None, fig_size: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """绘制成对关系图
        
        Args:
            columns: 要包含的列，默认选择前5个数值列
            hue: 可选的颜色分组列名
            fig_size: 图形尺寸
            
        Returns:
            seaborn PairGrid对象
        """
        if columns is None:
            columns = self.numeric_cols[:5]  # 默认选择前5个数值列
        
        plot = sns.pairplot(data=self.df, vars=columns, hue=hue, height=fig_size[0]/len(columns))
        plt.suptitle("成对关系图", y=1.02)
        return plot
    
    # Plotly交互式图表
    def plotly_scatter(self, x: str, y: str, color: Optional[str] = None, 
                     size: Optional[str] = None, hover_data: Optional[List[str]] = None) -> go.Figure:
        """创建Plotly交互式散点图
        
        Args:
            x: X轴列名
            y: Y轴列名
            color: 可选的颜色映射列名
            size: 可选的点大小列名
            hover_data: 悬停显示的额外数据列
            
        Returns:
            plotly图形对象
        """
        fig = px.scatter(self.df, x=x, y=y, color=color, size=size, 
                        hover_data=hover_data or [], title=f"{x}与{y}的交互式散点图")
        fig.update_layout(title_font_size=20, xaxis_title_font_size=16, yaxis_title_font_size=16)
        return fig
    
    def plotly_bar(self, x: str, y: Optional[str] = None, color: Optional[str] = None,
                 hover_data: Optional[List[str]] = None) -> go.Figure:
        """创建Plotly交互式条形图
        
        Args:
            x: X轴列名
            y: Y轴列名，默认为计数
            color: 可选的颜色映射列名
            hover_data: 悬停显示的额外数据列
            
        Returns:
            plotly图形对象
        """
        if y:
            fig = px.bar(self.df, x=x, y=y, color=color, hover_data=hover_data or [],
                        title=f"{x}与{y}的交互式条形图")
        else:
            value_counts = self.df[x].value_counts().reset_index()
            value_counts.columns = [x, 'count']
            fig = px.bar(value_counts, x=x, y='count', title=f"{x}的交互式计数条形图")
        
        fig.update_layout(title_font_size=20, xaxis_title_font_size=16, yaxis_title_font_size=16)
        return fig
    
    def plotly_line(self, x: str, y: str, color: Optional[str] = None,
                  hover_data: Optional[List[str]] = None) -> go.Figure:
        """创建Plotly交互式折线图
        
        Args:
            x: X轴列名
            y: Y轴列名
            color: 可选的颜色映射列名
            hover_data: 悬停显示的额外数据列
            
        Returns:
            plotly图形对象
        """
        fig = px.line(self.df, x=x, y=y, color=color, hover_data=hover_data or [],
                     title=f"{x}与{y}的交互式折线图")
        fig.update_layout(title_font_size=20, xaxis_title_font_size=16, yaxis_title_font_size=16)
        return fig
    
    def plotly_pie(self, column: str) -> go.Figure:
        """创建Plotly交互式饼图
        
        Args:
            column: 分类列名
            
        Returns:
            plotly图形对象
        """
        value_counts = self.df[column].value_counts().reset_index()
        value_counts.columns = [column, 'count']
        
        fig = px.pie(value_counts, names=column, values='count', title=f"{column}的交互式饼图")
        fig.update_layout(title_font_size=20)
        return fig 