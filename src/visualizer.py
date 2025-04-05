import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Union, Tuple
import jieba
import wordcloud
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')  # 忽略警告
from plotly.subplots import make_subplots


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
    
    def set_data(self, df: pd.DataFrame) -> None:
        """设置数据源
        
        Args:
            df: 数据源DataFrame对象
        """
        self.df = df
    
    def get_numerical_columns(self) -> List[str]:
        """获取数值型列名列表
        
        Returns:
            数值型列名列表
        """
        if self.df is None:
            return []
        
        return list(self.df.select_dtypes(include=['int64', 'float64']).columns)
    
    def get_categorical_columns(self) -> List[str]:
        """获取分类型列名列表
        
        Returns:
            分类型列名列表
        """
        if self.df is None:
            return []
        
        # 获取所有非数值型列
        non_numeric = list(self.df.select_dtypes(exclude=['int64', 'float64']).columns)
        
        # 对于数值型列，如果唯一值数量小于总记录数的10%，也视为分类变量
        numeric_cols = self.get_numerical_columns()
        for col in numeric_cols:
            if len(self.df[col].dropna().unique()) < len(self.df) * 0.1:
                non_numeric.append(col)
        
        return non_numeric
    
    def get_datetime_columns(self) -> List[str]:
        """获取日期时间型列名列表
        
        Returns:
            日期时间型列名列表
        """
        if self.df is None:
            return []
        
        datetime_cols = list(self.df.select_dtypes(include=['datetime64']).columns)
        
        # 检查其他可能是日期的字符串列
        str_cols = list(self.df.select_dtypes(include=['object']).columns)
        for col in str_cols:
            # 尝试转换为日期类型
            try:
                pd.to_datetime(self.df[col], errors='raise')
                datetime_cols.append(col)
            except:
                pass
        
        return datetime_cols
    
    def create_bar_chart(self, x: str, y: str, title: str = "条形图", 
                         color: Optional[str] = None, orientation: str = 'v',
                         sort_values: bool = False) -> go.Figure:
        """创建条形图
        
        Args:
            x: X轴列名
            y: Y轴列名
            title: 图表标题
            color: 颜色分组列名
            orientation: 方向，'v'为垂直，'h'为水平
            sort_values: 是否按值排序
            
        Returns:
            条形图对象
        """
        if self.df is None:
            return go.Figure()
        
        # 数据准备
        df = self.df.copy()
        
        # 如果分类变量作为Y轴，则切换到水平方向
        if orientation == 'v' and y in self.get_categorical_columns():
            orientation = 'h'
            x, y = y, x
        
        # 判断是否需要聚合
        if x in self.get_categorical_columns():
            if y in self.get_numerical_columns():
                # 数值按分类聚合
                if color:
                    plot_df = df.groupby([x, color])[y].mean().reset_index()
                else:
                    plot_df = df.groupby(x)[y].mean().reset_index()
                    if sort_values:
                        plot_df = plot_df.sort_values(y)
            else:
                # 分类按数量聚合
                if color:
                    plot_df = df.groupby([x, color]).size().reset_index(name='count')
                    y = 'count'
                else:
                    plot_df = df.groupby(x).size().reset_index(name='count')
                    y = 'count'
                    if sort_values:
                        plot_df = plot_df.sort_values(y)
        else:
            plot_df = df
        
        # 创建图表
        if orientation == 'h':
            fig = px.bar(
                plot_df, x=y, y=x, color=color,
                title=title,
                labels={x: x, y: y},
                orientation='h'
            )
        else:
            fig = px.bar(
                plot_df, x=x, y=y, color=color,
                title=title,
                labels={x: x, y: y}
            )
        
        # 图表样式设置
        fig.update_layout(
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            legend_title_text=color if color else "",
            height=500
        )
        
        return fig
    
    def create_line_chart(self, x: str, y: Union[str, List[str]], title: str = "折线图",
                          color: Optional[str] = None, mode: str = 'lines+markers') -> go.Figure:
        """创建折线图
        
        Args:
            x: X轴列名
            y: Y轴列名或多个Y轴列名列表
            title: 图表标题
            color: 颜色分组列名
            mode: 线条模式，'lines'、'markers'或'lines+markers'
            
        Returns:
            折线图对象
        """
        if self.df is None:
            return go.Figure()
        
        # 数据准备
        df = self.df.copy()
        
        # 处理日期列
        if x in self.get_datetime_columns():
            if not pd.api.types.is_datetime64_any_dtype(df[x]):
                df[x] = pd.to_datetime(df[x])
        
        # 如果是单个Y轴
        if isinstance(y, str):
            y_cols = [y]
            is_multi_y = False
        else:
            y_cols = y
            is_multi_y = True
        
        # 处理多条线
        if color and not is_multi_y:
            # 按颜色分组
            groups = df[color].unique()
            
            fig = go.Figure()
            
            for group in groups:
                group_df = df[df[color] == group]
                
                # 如果需要按X轴排序
                if group_df[x].dtype.kind in 'ifc' or pd.api.types.is_datetime64_any_dtype(group_df[x]):
                    group_df = group_df.sort_values(x)
                
                fig.add_trace(go.Scatter(
                    x=group_df[x],
                    y=group_df[y],
                    mode=mode,
                    name=str(group)
                ))
                
            fig.update_layout(
                title=title,
                title_x=0.5,
                xaxis_title=x,
                yaxis_title=y,
                legend_title=color,
                height=500,
                plot_bgcolor='rgba(0,0,0,0)'
            )
        elif is_multi_y:
            # 多条线(多个Y轴)
            fig = go.Figure()
            
            # 如果需要按X轴排序
            if df[x].dtype.kind in 'ifc' or pd.api.types.is_datetime64_any_dtype(df[x]):
                df = df.sort_values(x)
            
            for y_col in y_cols:
                fig.add_trace(go.Scatter(
                    x=df[x],
                    y=df[y_col],
                    mode=mode,
                    name=y_col
                ))
                
            fig.update_layout(
                title=title,
                title_x=0.5,
                xaxis_title=x,
                yaxis_title="数值",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)'
            )
        else:
            # 单条线
            if df[x].dtype.kind in 'ifc' or pd.api.types.is_datetime64_any_dtype(df[x]):
                df = df.sort_values(x)
            
            fig = px.line(
                df, x=x, y=y_cols[0],
                title=title,
                labels={x: x, y_cols[0]: y_cols[0]},
            )
            
            # 更新为线+点模式
            if mode == 'lines+markers':
                fig.update_traces(mode='lines+markers')
            elif mode == 'markers':
                fig.update_traces(mode='markers')
            
            fig.update_layout(
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                height=500
            )
        
        return fig
    
    def create_scatter_chart(self, x: str, y: str, title: str = "散点图", 
                            color: Optional[str] = None, size: Optional[str] = None,
                            add_trend: bool = False) -> go.Figure:
        """创建散点图
        
        Args:
            x: X轴列名
            y: Y轴列名
            title: 图表标题
            color: 颜色分组列名
            size: 点大小列名
            add_trend: 是否添加趋势线
            
        Returns:
            散点图对象
        """
        if self.df is None:
            return go.Figure()
        
        # 数据准备
        df = self.df.copy()
        
        # 如果X或Y包含缺失值，则删除相应行
        df = df.dropna(subset=[x, y])
        
        # 创建散点图
        fig = px.scatter(
            df, x=x, y=y, color=color, size=size,
            title=title,
            labels={x: x, y: y},
            hover_data=df.columns
        )
        
        # 添加趋势线
        if add_trend and x in self.get_numerical_columns() and y in self.get_numerical_columns():
            # 添加趋势线
            if color:
                # 为每个分组添加趋势线
                for c in df[color].unique():
                    group_df = df[df[color] == c]
                    coeffs = np.polyfit(group_df[x], group_df[y], 1)
                    poly_eq = np.poly1d(coeffs)
                    
                    x_range = np.linspace(df[x].min(), df[x].max(), 100)
                    y_pred = poly_eq(x_range)
                    
                    fig.add_trace(go.Scatter(
                        x=x_range, y=y_pred,
                        mode='lines',
                        line=dict(dash='dash'),
                        name=f'{c} 趋势线',
                        showlegend=True
                    ))
            else:
                # 添加单一趋势线
                coeffs = np.polyfit(df[x], df[y], 1)
                poly_eq = np.poly1d(coeffs)
                
                x_range = np.linspace(df[x].min(), df[x].max(), 100)
                y_pred = poly_eq(x_range)
                
                fig.add_trace(go.Scatter(
                    x=x_range, y=y_pred,
                    mode='lines',
                    line=dict(dash='dash', color='red'),
                    name='趋势线',
                    showlegend=True
                ))
        
        # 图表样式设置
        fig.update_layout(
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            legend_title_text=color if color else "",
            height=600
        )
        
        return fig
    
    def create_pie_chart(self, values: str, names: str, title: str = "饼图",
                        hole: float = 0.0) -> go.Figure:
        """创建饼图或环形图
        
        Args:
            values: 数值列名
            names: 分类列名
            title: 图表标题
            hole: 中心孔径比例，0表示饼图，大于0表示环形图
            
        Returns:
            饼图或环形图对象
        """
        if self.df is None:
            return go.Figure()
        
        # 数据准备：按分类列聚合
        if values in self.get_numerical_columns():
            # 使用数值列的总和
            plot_df = self.df.groupby(names)[values].sum().reset_index()
        else:
            # 使用计数
            plot_df = self.df.groupby(names).size().reset_index(name='count')
            values = 'count'
        
        # 筛选掉特别小的类别，避免饼图太杂
        total = plot_df[values].sum()
        threshold = total * 0.02  # 低于2%的归为"其他"
        small_categories = plot_df[plot_df[values] < threshold]
        
        if len(small_categories) > 1:  # 如果有多个小类别
            # 保留主要类别，其余归为"其他"
            main_categories = plot_df[plot_df[values] >= threshold]
            others_sum = small_categories[values].sum()
            
            if others_sum > 0:  # 确保"其他"类别不为0
                others_row = pd.DataFrame({names: ['其他'], values: [others_sum]})
                plot_df = pd.concat([main_categories, others_row], ignore_index=True)
        
        # 创建饼图
        fig = px.pie(
            plot_df, values=values, names=names,
            title=title,
            hole=hole
        )
        
        # 图表样式设置
        fig.update_layout(
            title_x=0.5,
            height=500
        )
        
        # 显示数值和百分比
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            insidetextorientation='radial'
        )
        
        return fig
    
    def create_histogram(self, column: str, bins: int = 20, title: str = "直方图",
                         color: Optional[str] = None, cumulative: bool = False) -> go.Figure:
        """创建直方图
        
        Args:
            column: 数据列名
            bins: 分组数量
            title: 图表标题
            color: 颜色分组列名
            cumulative: 是否显示累积分布
            
        Returns:
            直方图对象
        """
        if self.df is None:
            return go.Figure()
        
        # 创建直方图
        fig = px.histogram(
            self.df, x=column, color=color,
            title=title,
            nbins=bins,
            labels={column: column},
            cumulative=cumulative,
            histnorm='percent' if cumulative else None
        )
        
        # 图表样式设置
        fig.update_layout(
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            bargap=0.1,
            height=500
        )
        
        if color:
            fig.update_layout(legend_title_text=color)
        
        return fig
    
    def create_box_plot(self, x: Optional[str], y: str, title: str = "箱线图",
                       color: Optional[str] = None) -> go.Figure:
        """创建箱线图
        
        Args:
            x: 分组列名
            y: 数值列名
            title: 图表标题
            color: 颜色分组列名
            
        Returns:
            箱线图对象
        """
        if self.df is None:
            return go.Figure()
        
        # 创建箱线图
        fig = px.box(
            self.df, x=x, y=y, color=color,
            title=title,
            labels={x: x if x else "", y: y},
            points="outliers"  # 只显示异常值点
        )
        
        # 图表样式设置
        fig.update_layout(
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        
        if color:
            fig.update_layout(legend_title_text=color)
        
        return fig
    
    def create_heatmap(self, columns: Optional[List[str]] = None, 
                      title: str = "相关性热力图") -> go.Figure:
        """创建热力图
        
        Args:
            columns: 要包含的列名列表，为None则使用所有数值列
            title: 图表标题
            
        Returns:
            热力图对象
        """
        if self.df is None:
            return go.Figure()
        
        # 如果未指定列，使用所有数值列
        if columns is None:
            columns = self.get_numerical_columns()
        
        # 计算相关系数矩阵
        if len(columns) < 2:
            # 至少需要两列才能计算相关性
            return go.Figure()
        
        df_corr = self.df[columns].corr()
        
        # 创建热力图
        fig = px.imshow(
            df_corr,
            text_auto='.2f',  # 显示相关系数值
            title=title,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            aspect="auto"
        )
        
        # 图表样式设置
        fig.update_layout(
            title_x=0.5,
            height=600
        )
        
        return fig
    
    def create_bubble_chart(self, x: str, y: str, size: str, 
                           title: str = "气泡图", color: Optional[str] = None) -> go.Figure:
        """创建气泡图
        
        Args:
            x: X轴列名
            y: Y轴列名
            size: 气泡大小列名
            title: 图表标题
            color: 颜色分组列名
            
        Returns:
            气泡图对象
        """
        if self.df is None or x not in self.df.columns or y not in self.df.columns:
            return go.Figure()
        
        # 创建气泡图
        fig = px.scatter(
            self.df, x=x, y=y, size=size, color=color,
            title=title,
            labels={x: x, y: y, size: size},
            hover_data=self.df.columns
        )
        
        # 图表样式设置
        fig.update_layout(
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            height=600
        )
        
        if color:
            fig.update_layout(legend_title_text=color)
        
        return fig
    
    def create_radar_chart(self, categories: List[str], values: List[float], 
                          title: str = "雷达图", group_name: str = "组别") -> go.Figure:
        """创建雷达图
        
        Args:
            categories: 分类标签列表
            values: 对应的数值列表
            title: 图表标题
            group_name: 分组名称
            
        Returns:
            雷达图对象
        """
        if not categories or not values or len(categories) != len(values):
            return go.Figure()
        
        # 创建雷达图
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=group_name
        ))
        
        # 图表样式设置
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                )
            ),
            title=title,
            title_x=0.5,
            height=500
        )
        
        return fig
    
    def create_multi_radar_chart(self, categories: List[str], 
                               group_data: Dict[str, List[float]], 
                               title: str = "多组雷达图") -> go.Figure:
        """创建多组雷达图
        
        Args:
            categories: 分类标签列表
            group_data: 分组数据字典，键为分组名，值为对应的数值列表
            title: 图表标题
            
        Returns:
            多组雷达图对象
        """
        if not categories or not group_data:
            return go.Figure()
        
        # 创建雷达图
        fig = go.Figure()
        
        for group_name, values in group_data.items():
            if len(categories) != len(values):
                continue
                
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=group_name
            ))
        
        # 图表样式设置
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                )
            ),
            title=title,
            title_x=0.5,
            height=500
        )
        
        return fig
    
    def create_time_series(self, date_column: str, value_column: str, 
                          title: str = "时间序列图", freq: Optional[str] = None,
                          color: Optional[str] = None) -> go.Figure:
        """创建时间序列图
        
        Args:
            date_column: 日期列名
            value_column: 数值列名
            title: 图表标题
            freq: 重采样频率，如'D', 'W', 'M', 'Q', 'Y'等
            color: 颜色分组列名
            
        Returns:
            时间序列图对象
        """
        if self.df is None:
            return go.Figure()
        
        # 确保日期列为日期类型
        df = self.df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df = df.dropna(subset=[date_column])
        
        # 按日期排序
        df = df.sort_values(date_column)
        
        # 如果指定了重采样频率
        if freq:
            if color:
                # 对每个颜色分组单独重采样
                resampled_dfs = []
                for c in df[color].unique():
                    group_df = df[df[color] == c]
                    group_df = group_df.set_index(date_column)
                    resampled = group_df[[value_column]].resample(freq).mean().reset_index()
                    resampled[color] = c
                    resampled_dfs.append(resampled)
                
                if resampled_dfs:
                    df = pd.concat(resampled_dfs)
            else:
                # 整体重采样
                df = df.set_index(date_column)
                df = df[[value_column]].resample(freq).mean().reset_index()
        
        # 创建时间序列图
        fig = px.line(
            df, x=date_column, y=value_column, color=color,
            title=title,
            labels={date_column: date_column, value_column: value_column},
            markers=True
        )
        
        # 图表样式设置
        fig.update_layout(
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date'
            )
        )
        
        if color:
            fig.update_layout(legend_title_text=color)
        
        return fig
    
    def create_wordcloud(self, text_column: str, title: str = "词云图", 
                        width: int = 800, height: int = 400, 
                        background_color: str = 'white') -> Any:
        """创建词云图
        
        Args:
            text_column: 文本列名
            title: 图表标题
            width: 图像宽度
            height: 图像高度
            background_color: 背景颜色
            
        Returns:
            词云图对象
        """
        if self.df is None or not hasattr(wordcloud, 'WordCloud'):
            return None
        
        # 合并所有文本
        all_text = ' '.join(self.df[text_column].dropna().astype(str))
        
        # 中文分词
        if any('\u4e00' <= char <= '\u9fff' for char in all_text):  # 检测是否包含中文
            words = ' '.join(jieba.cut(all_text))
        else:
            words = all_text
        
        # 创建词云
        wc = wordcloud.WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=200,
            contour_width=0,
            collocations=False
        ).generate(words)
        
        # 创建matplotlib图像
        plt.figure(figsize=(width/100, height/100))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout(pad=0)
        
        return plt
    
    def create_geo_map(self, location_column: str, value_column: Optional[str] = None, 
                       title: str = "地理地图", scope: str = "world", 
                       color_scale: str = "Viridis") -> go.Figure:
        """创建地理地图
        
        Args:
            location_column: 地区/国家列名
            value_column: 数值列名，如不提供则显示计数
            title: 图表标题
            scope: 地图范围，如'world', 'asia', 'europe', 'usa', 'china'等
            color_scale: 颜色比例尺
            
        Returns:
            地理地图对象
        """
        if self.df is None:
            return go.Figure()
        
        # 数据准备
        if value_column is None:
            # 如果未提供值列，使用计数
            plot_df = self.df.groupby(location_column).size().reset_index(name='count')
            value_column = 'count'
        else:
            # 按地区聚合
            plot_df = self.df.groupby(location_column)[value_column].sum().reset_index()
        
        # 创建地理地图
        if scope.lower() == 'china':
            # 中国地图
            fig = px.choropleth(
                plot_df,
                locations=location_column,
                locationmode="country names",
                color=value_column,
                hover_name=location_column,
                color_continuous_scale=color_scale,
                title=title,
                scope="asia",  # 聚焦亚洲区域
            )
            # 缩放到中国区域
            fig.update_geos(
                visible=False,
                lataxis_range=[15, 55],
                lonaxis_range=[70, 140],
                showcountries=True,
                countrycolor="gray"
            )
        else:
            # 其他地图
            fig = px.choropleth(
                plot_df,
                locations=location_column,
                locationmode="country names",
                color=value_column,
                hover_name=location_column,
                color_continuous_scale=color_scale,
                title=title,
                scope=scope,
            )
        
        # 图表样式设置
        fig.update_layout(
            title_x=0.5,
            height=600,
            coloraxis_colorbar=dict(
                title=value_column
            )
        )
        
        return fig
    
    def create_scatter_mapbox(self, lat_column: str, lon_column: str, 
                             title: str = "散点地图", color: Optional[str] = None,
                             size: Optional[str] = None, zoom: int = 1,
                             mapbox_style: str = "carto-positron") -> go.Figure:
        """创建散点地图
        
        Args:
            lat_column: 纬度列名
            lon_column: 经度列名
            title: 图表标题
            color: 颜色分组列名
            size: 点大小列名
            zoom: 缩放级别
            mapbox_style: 地图样式
            
        Returns:
            散点地图对象
        """
        if self.df is None:
            return go.Figure()
        
        # 创建散点地图
        fig = px.scatter_mapbox(
            self.df,
            lat=lat_column,
            lon=lon_column,
            color=color,
            size=size,
            hover_name=color if color else None,
            hover_data=self.df.columns,
            title=title,
            zoom=zoom,
            mapbox_style=mapbox_style
        )
        
        # 图表样式设置
        fig.update_layout(
            title_x=0.5,
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        if color:
            fig.update_layout(legend_title_text=color)
        
        return fig
    
    def create_density_mapbox(self, lat_column: str, lon_column: str, 
                             z_column: Optional[str] = None, title: str = "密度地图",
                             zoom: int = 1, radius: int = 10,
                             mapbox_style: str = "carto-positron") -> go.Figure:
        """创建密度地图
        
        Args:
            lat_column: 纬度列名
            lon_column: 经度列名
            z_column: 密度值列名，如不提供则使用点数
            title: 图表标题
            zoom: 缩放级别
            radius: 半径大小
            mapbox_style: 地图样式
            
        Returns:
            密度地图对象
        """
        if self.df is None:
            return go.Figure()
        
        # 创建密度地图
        fig = px.density_mapbox(
            self.df,
            lat=lat_column,
            lon=lon_column,
            z=z_column,
            radius=radius,
            title=title,
            zoom=zoom,
            mapbox_style=mapbox_style
        )
        
        # 图表样式设置
        fig.update_layout(
            title_x=0.5,
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def create_multi_chart_dashboard(self, charts: List[Tuple[go.Figure, int, int]],
                                   title: str = "多图表仪表盘", cols: int = 2) -> go.Figure:
        """创建多图表仪表盘
        
        Args:
            charts: 图表列表，每个元素是一个三元组(图表对象，行跨度，列跨度)
            title: 仪表盘标题
            cols: 列数
            
        Returns:
            组合图表对象
        """
        if not charts:
            return go.Figure()
        
        # 计算行数
        total_cells = sum(row_span * col_span for _, row_span, col_span in charts)
        rows = (total_cells + cols - 1) // cols  # 向上取整
        
        # 创建子图布局
        subplot_specs = [[{'colspan': 1, 'rowspan': 1} for _ in range(cols)] for _ in range(rows)]
        
        # 创建组合图表
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[chart.layout.title.text for chart, _, _ in charts],
            specs=subplot_specs,
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # 添加每个图表的轨迹
        current_row, current_col = 1, 1
        for chart, row_span, col_span in charts:
            # 添加所有轨迹
            for trace in chart.data:
                fig.add_trace(
                    trace,
                    row=current_row,
                    col=current_col
                )
            
            # 更新位置
            current_col += col_span
            if current_col > cols:
                current_col = 1
                current_row += 1
        
        # 图表样式设置
        fig.update_layout(
            title=title,
            title_x=0.5,
            height=250 * rows,
            showlegend=False
        )
        
        return fig
    
    def create_interactive_chart(self, chart_type: str, **kwargs) -> go.Figure:
        """创建交互式图表
        
        Args:
            chart_type: 图表类型，支持'bar', 'line', 'scatter', 'pie', 'box', 'histogram'等
            **kwargs: 传递给对应图表创建函数的参数
            
        Returns:
            图表对象
        """
        if self.df is None:
            return go.Figure()
        
        # 根据图表类型创建相应图表
        if chart_type == 'bar':
            return self.create_bar_chart(**kwargs)
        elif chart_type == 'line':
            return self.create_line_chart(**kwargs)
        elif chart_type == 'scatter':
            return self.create_scatter_chart(**kwargs)
        elif chart_type == 'pie':
            return self.create_pie_chart(**kwargs)
        elif chart_type == 'box':
            return self.create_box_plot(**kwargs)
        elif chart_type == 'histogram':
            return self.create_histogram(**kwargs)
        elif chart_type == 'heatmap':
            return self.create_heatmap(**kwargs)
        elif chart_type == 'bubble':
            return self.create_bubble_chart(**kwargs)
        elif chart_type == 'geo_map':
            return self.create_geo_map(**kwargs)
        elif chart_type == 'time_series':
            return self.create_time_series(**kwargs)
        else:
            raise ValueError(f"不支持的图表类型: {chart_type}")
    
    @staticmethod
    def add_annotations(fig: go.Figure, annotations: List[Dict]) -> go.Figure:
        """向图表添加注释
        
        Args:
            fig: 图表对象
            annotations: 注释列表，每个元素是一个包含x, y, text等键的字典
            
        Returns:
            添加注释后的图表对象
        """
        if not fig or not annotations:
            return fig
        
        # 添加每个注释
        for anno in annotations:
            fig.add_annotation(
                x=anno.get('x'),
                y=anno.get('y'),
                text=anno.get('text', ''),
                showarrow=anno.get('showarrow', True),
                arrowhead=anno.get('arrowhead', 1),
                ax=anno.get('ax', 0),
                ay=anno.get('ay', -40)
            )
        
        return fig 