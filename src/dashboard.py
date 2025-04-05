import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Optional, List, Dict, Any, Tuple
import json
import numpy as np
import os
import sqlite3

from src.data_processor import DataProcessor
from src.visualizer import Visualizer
from src.ml_model import MLModel  # 导入新增的机器学习模块
from src.data_loader import DataLoader  # 导入数据加载器


class Dashboard:
    """交互式数据可视化仪表盘类"""
    
    def __init__(self, title: str = "自动化数据可视化工具"):
        """初始化仪表盘
        
        Args:
            title: 仪表盘标题
        """
        self.title = title
        self.df = None
        self.processor = None
        self.visualizer = None
        self.ml_model = None  # 新增机器学习模型属性
    
    def run(self):
        """运行Streamlit应用"""
        # 页面配置已在app.py中设置，此处不再重复设置
        
        st.title(self.title)
        st.write("欢迎使用自动化数据可视化工具！上传您的数据集并探索洞察。")
        
        # 侧边栏 - 数据加载
        self._load_data_sidebar()
        
        # 如果数据已加载，显示主界面
        if self.df is not None:
            # 创建选项卡
            tabs = st.tabs(["📊 数据概览", "🧹 数据处理", "📈 数据可视化", "🤖 机器学习", "📋 数据报告"])
            
            # 数据概览选项卡
            with tabs[0]:
                self._render_data_overview()
            
            # 数据处理选项卡
            with tabs[1]:
                self._render_data_processing()
            
            # 数据可视化选项卡
            with tabs[2]:
                self._render_data_visualization()
            
            # 机器学习选项卡
            with tabs[3]:
                self._render_machine_learning()
            
            # 数据报告选项卡
            with tabs[4]:
                self._render_data_report()
    
    def _load_data_sidebar(self):
        """侧边栏 - 数据加载部分"""
        with st.sidebar:
            st.header("数据加载")
            
            # 创建数据加载器对象（如果还没有）
            if not hasattr(self, 'data_loader'):
                self.data_loader = DataLoader()
            
            # 数据源选择
            data_source = st.radio(
                "选择数据来源",
                ["文件上传", "示例数据", "数据库", "API"],
                captions=["上传本地CSV/Excel/JSON文件", "生成测试数据", "从SQLite数据库加载", "从REST API加载"]
            )
            
            try:
                # 1. 文件上传
                if data_source == "文件上传":
                    # 文件上传
                    uploaded_file = st.file_uploader(
                        "上传数据文件", 
                        type=["csv", "xlsx", "xls", "json"],
                        help="支持CSV、Excel和JSON格式"
                    )
                    
                    if uploaded_file is not None:
                        try:
                            with st.spinner("正在加载数据..."):
                                # 加载数据
                                self.df = self.data_loader.load_file(uploaded_file)
                                
                                # 显示数据信息
                                st.success(f"成功加载: {uploaded_file.name}")
                                st.caption(f"数据大小: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")
                                
                                # 初始化处理器和可视化器
                                self._init_components()
                        except Exception as e:
                            st.error(f"加载数据失败: {str(e)}")
                
                # 2. 示例数据
                elif data_source == "示例数据":
                    data_type = st.selectbox(
                        "示例数据类型",
                        ["销售数据", "股票数据", "问卷调查数据"],
                        format_func=lambda x: x
                    )
                    
                    rows = st.slider("样本数量", 20, 500, 100)
                    
                    # 映射中文类型到英文
                    data_type_map = {
                        "销售数据": "sales",
                        "股票数据": "stock",
                        "问卷调查数据": "survey"
                    }
                    
                    if st.button("加载示例数据", use_container_width=True):
                        try:
                            with st.spinner("正在生成示例数据..."):
                                # 加载示例数据
                                self.df = self.data_loader.generate_sample_data(
                                    data_type=data_type_map[data_type],
                                    rows=rows
                                )
                                
                                # 显示数据信息
                                st.success(f"成功加载{data_type}示例数据")
                                st.caption(f"数据大小: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")
                                
                                # 初始化处理器和可视化器
                                self._init_components()
                        except Exception as e:
                            st.error(f"生成示例数据失败: {str(e)}")
                
                # 3. 数据库
                elif data_source == "数据库":
                    # 使用文件上传或本地路径
                    database_option = st.radio(
                        "数据库选择方式",
                        ["上传数据库文件", "使用示例数据库"],
                        captions=["上传SQLite数据库文件", "使用示例销售数据库"]
                    )
                    
                    if database_option == "上传数据库文件":
                        db_file = st.file_uploader("上传SQLite数据库文件", type=["db", "sqlite", "sqlite3"])
                        
                        if db_file is not None:
                            # 保存上传的数据库文件到临时位置
                            db_path = f"temp_{db_file.name}"
                            with open(db_path, "wb") as f:
                                f.write(db_file.getbuffer())
                            st.success(f"成功上传数据库文件: {db_file.name}")
                        else:
                            db_path = None
                    else:
                        # 使用示例数据库
                        db_path = "examples/sample_database.db"
                        
                        # 如果示例数据库不存在，显示提示
                        if not os.path.exists(db_path):
                            st.warning("示例数据库未找到，请先创建示例数据库或上传自己的数据库文件。")
                    
                    # 如果数据库路径有效，允许输入查询
                    if db_path and os.path.exists(db_path):
                        # 显示可用的表
                        try:
                            conn = sqlite3.connect(db_path)
                            cursor = conn.cursor()
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                            tables = cursor.fetchall()
                            table_names = [table[0] for table in tables]
                            conn.close()
                            
                            if table_names:
                                st.success(f"数据库包含以下表: {', '.join(table_names)}")
                                
                                # 选择表
                                selected_table = st.selectbox("选择数据表", table_names)
                                
                                # 生成查询
                                st.subheader("SQL查询")
                                use_custom_query = st.checkbox("使用自定义SQL查询")
                                
                                if use_custom_query:
                                    query = st.text_area(
                                        "输入SQL查询", 
                                        value=f"SELECT * FROM {selected_table} LIMIT 100;"
                                    )
                                else:
                                    query = f"SELECT * FROM {selected_table} LIMIT 100;"
                                    st.code(query)
                                
                                # 执行查询
                                if st.button("执行查询", use_container_width=True):
                                    try:
                                        with st.spinner("正在执行查询..."):
                                            # 加载数据
                                            self.df = self.data_loader.load_database(db_path, query)
                                            
                                            # 显示数据信息
                                            st.success(f"成功执行查询")
                                            st.caption(f"查询结果: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")
                                            
                                            # 初始化处理器和可视化器
                                            self._init_components()
                                    except Exception as e:
                                        st.error(f"查询执行失败: {str(e)}")
                            else:
                                st.warning("数据库中没有表")
                        except Exception as e:
                            st.error(f"读取数据库表失败: {str(e)}")
                
                # 4. API
                elif data_source == "API":
                    # API配置
                    api_url = st.text_input(
                        "API URL",
                        value="https://jsonplaceholder.typicode.com/users",
                        help="输入API端点URL"
                    )
                    
                    # 响应格式
                    api_format = st.selectbox(
                        "响应格式",
                        ["JSON", "CSV"],
                        help="选择API响应的数据格式"
                    )
                    
                    # 高级选项
                    with st.expander("高级选项"):
                        # 参数
                        params_input = st.text_area(
                            "请求参数 (JSON格式)",
                            value="{}",
                            help="输入请求参数，JSON格式"
                        )
                        
                        # 头信息
                        headers_input = st.text_area(
                            "请求头 (JSON格式)",
                            value='{"Content-Type": "application/json"}',
                            help="输入请求头，JSON格式"
                        )
                    
                    # 执行API请求
                    if st.button("发送请求", use_container_width=True):
                        try:
                            # 解析参数和头信息
                            try:
                                params = json.loads(params_input)
                                headers = json.loads(headers_input)
                            except json.JSONDecodeError as e:
                                st.error(f"JSON解析错误: {str(e)}")
                                params = {}
                                headers = {"Content-Type": "application/json"}
                            
                            with st.spinner("正在发送API请求..."):
                                # 加载数据
                                self.df = self.data_loader.load_api(
                                    url=api_url,
                                    params=params,
                                    headers=headers,
                                    format=api_format.lower()
                                )
                                
                                # 显示数据信息
                                st.success(f"成功接收API响应")
                                st.caption(f"数据大小: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")
                                
                                # 初始化处理器和可视化器
                                self._init_components()
                        except Exception as e:
                            st.error(f"API请求失败: {str(e)}")
                
                # 如果数据已加载，显示数据信息和导出选项
                if self.df is not None:
                    st.divider()
                    
                    # 显示数据信息
                    with st.expander("数据信息", expanded=True):
                        info = self.data_loader.get_data_info()
                        for key, value in info.items():
                            if key not in ["状态", "数据源"]:
                                st.caption(f"{key}: {value}")
                    
                    # 导出选项
                    export_format = st.selectbox("导出格式", ["CSV", "Excel", "JSON"])
                    export_html = self.data_loader.create_download_link(export_format.lower())
                    st.markdown(export_html, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"出现未知错误: {str(e)}")
                st.error("请刷新页面并重试")
    
    def _init_components(self):
        """初始化数据处理器和可视化器"""
        if self.df is not None:
            self.processor = DataProcessor(self.df)
            self.visualizer = Visualizer(self.df)
            self.ml_model = None  # 重置ML模型
    
    def _render_data_overview(self):
        """数据概览选项卡内容"""
        st.header("数据概览")
        
        # 基本信息
        st.subheader("数据维度")
        col1, col2 = st.columns(2)
        col1.metric("行数", self.df.shape[0])
        col2.metric("列数", self.df.shape[1])
        
        # 数据预览
        st.subheader("数据预览")
        st.dataframe(self.df.head(10), use_container_width=True)
        
        # 列信息
        st.subheader("列信息")
        # 修复PyArrow兼容性问题：将dtypes转换为字符串而不是对象
        col_info = pd.DataFrame({
            "数据类型": [str(dtype) for dtype in self.df.dtypes.values],
            "非空值数": self.df.count().values,
            "缺失值数": self.df.isnull().sum().values,
            "缺失值百分比": (self.df.isnull().sum() / len(self.df) * 100).round(2).values,
            "唯一值数": [self.df[col].nunique() for col in self.df.columns],
        }, index=self.df.columns)
        st.dataframe(col_info, use_container_width=True)
        
        # 数值型数据统计
        st.subheader("数值型列统计")
        numeric_df = self.df.select_dtypes(include=["int64", "float64"])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe().T, use_container_width=True)
        else:
            st.info("没有数值型列")
        
        # 分类型数据统计
        st.subheader("分类型列概览")
        categorical_df = self.df.select_dtypes(include=["object", "category"])
        if not categorical_df.empty:
            for col in categorical_df.columns[:5]:  # 限制只显示前5个分类列，避免界面过长
                with st.expander(f"**{col}** (唯一值: {categorical_df[col].nunique()})"):
                    value_counts = self.df[col].value_counts().head(10).reset_index()
                    value_counts.columns = [col, '计数']
                    st.dataframe(value_counts, use_container_width=True)
                    
                    # 如果唯一值不超过10个，显示饼图
                    if categorical_df[col].nunique() <= 10:
                        fig = px.pie(value_counts, names=col, values='计数', title=f"{col}分布")
                        st.plotly_chart(fig, use_container_width=True)
            
            # 如果有更多分类列，显示查看更多的选项
            if len(categorical_df.columns) > 5:
                with st.expander(f"查看更多分类列 (共{len(categorical_df.columns)}个)"):
                    for col in categorical_df.columns[5:]:
                        st.write(f"**{col}** (唯一值: {categorical_df[col].nunique()})")
                        value_counts = self.df[col].value_counts().head(10).reset_index()
                        value_counts.columns = [col, '计数']
                        st.dataframe(value_counts, use_container_width=True)
        else:
            st.info("没有分类型列")
    
    def _render_data_processing(self):
        """数据处理选项卡内容"""
        st.header("数据处理")
        
        # 处理控制面板
        st.subheader("数据处理选项")
        processed_df = self.df.copy()
        
        # 缺失值处理
        st.write("**缺失值处理**")
        missing_cols = self.df.columns[self.df.isnull().any()]
        if len(missing_cols) > 0:
            missing_options = st.multiselect("选择要处理的列", missing_cols, default=missing_cols)
            if missing_options:
                strategy = st.selectbox("缺失值处理策略", ["mean", "median", "most_frequent", "drop"], index=0)
                if st.button("应用缺失值处理"):
                    with st.spinner("处理中..."):
                        processor = DataProcessor(processed_df)
                        processed_df = processor.handle_missing_values(strategy, missing_options).get_result()
                        st.success("缺失值处理完成")
        else:
            st.info("数据中没有缺失值")
        
        # 数据标准化/归一化
        st.write("**数据标准化/归一化**")
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            norm_options = st.multiselect("选择要标准化的列", numeric_cols)
            if norm_options:
                norm_strategy = st.selectbox("标准化方法", ["standard", "minmax"], index=0)
                if st.button("应用标准化"):
                    with st.spinner("处理中..."):
                        processor = DataProcessor(processed_df)
                        processed_df = processor.normalize_data(norm_strategy, norm_options).get_result()
                        st.success("数据标准化完成")
        else:
            st.info("数据中没有数值列")
        
        # 分类变量编码
        st.write("**分类变量编码**")
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            cat_options = st.multiselect("选择要编码的列", cat_cols)
            if cat_options:
                cat_strategy = st.selectbox("编码方法", ["onehot", "label"], index=0)
                if st.button("应用编码"):
                    with st.spinner("处理中..."):
                        processor = DataProcessor(processed_df)
                        processed_df = processor.encode_categorical(cat_options, cat_strategy).get_result()
                        st.success("分类变量编码完成")
        else:
            st.info("数据中没有分类列")
        
        # 显示处理后的数据预览
        st.subheader("处理后的数据预览")
        st.dataframe(processed_df.head(10), use_container_width=True)
        
        # 下载处理后的数据
        if st.button("下载处理后的数据"):
            processed_csv = processed_df.to_csv(index=False)
            st.download_button(
                label="下载为CSV",
                data=processed_csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )
    
    def _render_data_visualization(self):
        """数据可视化选项卡内容"""
        st.header("数据可视化")
        
        if self.df is None:
            st.warning("请先加载数据")
            return
        
        # 获取数据列
        numerical_cols = self.visualizer.get_numerical_columns()
        categorical_cols = self.visualizer.get_categorical_columns()
        datetime_cols = self.visualizer.get_datetime_columns()
        
        # 智能图表推荐
        with st.expander("✨ 智能图表推荐", expanded=True):
            st.info("根据数据特点，系统可以为您推荐以下可视化方式：")
            
            recommendations = self._generate_visualization_recommendations()
            
            if recommendations:
                for i, (rec_title, rec_type, rec_params) in enumerate(recommendations):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader(f"{i+1}. {rec_title}")
                    with col2:
                        if st.button("生成图表", key=f"rec_btn_{i}"):
                            try:
                                fig = self.visualizer.create_interactive_chart(rec_type, **rec_params)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"图表生成失败: {str(e)}")
            else:
                st.warning("暂无可推荐的图表，请尝试手动选择图表类型。")
        
        # 选择图表类型
        chart_types = {
            "条形图": "bar",
            "折线图": "line",
            "散点图": "scatter",
            "饼图": "pie",
            "热力图": "heatmap",
            "箱线图": "box",
            "直方图": "histogram",
            "气泡图": "bubble",
            "时间序列": "time_series",
            "地理地图": "geo_map"
        }
        
        chart_type = st.selectbox("选择图表类型", list(chart_types.keys()))
        
        # 根据图表类型显示不同的参数选择
        chart_id = chart_types[chart_type]
        
        with st.form(key="visualization_form"):
            if chart_id == "bar":
                title = st.text_input("图表标题", "条形图")
                col1, col2 = st.columns(2)
                with col1:
                    x = st.selectbox("X轴(分类)", categorical_cols)
                with col2:
                    y = st.selectbox("Y轴(数值)", numerical_cols)
                color = st.selectbox("颜色分组(可选)", ["无"] + categorical_cols)
                orientation = st.radio("方向", ["垂直", "水平"], horizontal=True)
                sort_values = st.checkbox("按值排序")
                
                if st.form_submit_button("生成图表"):
                    try:
                        fig = self.visualizer.create_bar_chart(
                            x=x, 
                            y=y, 
                            title=title,
                            color=None if color == "无" else color,
                            orientation='v' if orientation == "垂直" else 'h',
                            sort_values=sort_values
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"图表生成失败: {str(e)}")
            
            elif chart_id == "line":
                title = st.text_input("图表标题", "折线图")
                col1, col2 = st.columns(2)
                with col1:
                    x = st.selectbox("X轴", datetime_cols + numerical_cols)
                with col2:
                    # 可以选择多个Y轴
                    y_options = numerical_cols
                    y = st.multiselect("Y轴(可多选)", y_options, default=[y_options[0]] if y_options else [])
                
                color = st.selectbox("颜色分组(可选)", ["无"] + categorical_cols)
                mode = st.radio("显示模式", ["线+点", "仅线条", "仅点"], horizontal=True)
                
                mode_map = {"线+点": "lines+markers", "仅线条": "lines", "仅点": "markers"}
                
                if st.form_submit_button("生成图表"):
                    try:
                        if len(y) == 1:
                            y = y[0]  # 单个Y轴
                            
                        fig = self.visualizer.create_line_chart(
                            x=x, 
                            y=y, 
                            title=title,
                            color=None if color == "无" else color,
                            mode=mode_map[mode]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"图表生成失败: {str(e)}")
            
            elif chart_id == "scatter":
                title = st.text_input("图表标题", "散点图")
                col1, col2 = st.columns(2)
                with col1:
                    x = st.selectbox("X轴", numerical_cols)
                with col2:
                    y = st.selectbox("Y轴", numerical_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    color = st.selectbox("颜色分组(可选)", ["无"] + categorical_cols)
                with col2:
                    size = st.selectbox("点大小(可选)", ["无"] + numerical_cols)
                
                add_trend = st.checkbox("添加趋势线")
                
                if st.form_submit_button("生成图表"):
                    try:
                        fig = self.visualizer.create_scatter_chart(
                            x=x, 
                            y=y, 
                            title=title,
                            color=None if color == "无" else color,
                            size=None if size == "无" else size,
                            add_trend=add_trend
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"图表生成失败: {str(e)}")
            
            elif chart_id == "pie":
                title = st.text_input("图表标题", "饼图")
                col1, col2 = st.columns(2)
                with col1:
                    names = st.selectbox("分类", categorical_cols)
                with col2:
                    values = st.selectbox("数值", ["计数"] + numerical_cols)
                
                hole = st.slider("中心孔径(0为饼图，>0为环形图)", 0.0, 0.8, 0.0, 0.1)
                
                if st.form_submit_button("生成图表"):
                    try:
                        if values == "计数":
                            # 使用分类计数
                            value_counts = self.df[names].value_counts().reset_index()
                            value_counts.columns = [names, 'count']
                            fig = px.pie(
                                value_counts, 
                                names=names, 
                                values='count',
                                title=title,
                                hole=hole
                            )
                        else:
                            fig = self.visualizer.create_pie_chart(
                                names=names, 
                                values=values, 
                                title=title,
                                hole=hole
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"图表生成失败: {str(e)}")
            
            elif chart_id == "heatmap":
                title = st.text_input("图表标题", "相关性热力图")
                
                # 选择要包含在热力图中的列
                heatmap_cols = st.multiselect(
                    "选择要包含的数值列", 
                    numerical_cols,
                    default=numerical_cols[:min(len(numerical_cols), 8)]  # 默认选择前8个
                )
                
                if st.form_submit_button("生成图表"):
                    try:
                        if len(heatmap_cols) < 2:
                            st.error("热力图至少需要选择2个数值列")
                        else:
                            fig = self.visualizer.create_heatmap(
                                columns=heatmap_cols,
                                title=title
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"图表生成失败: {str(e)}")
            
            elif chart_id == "box":
                title = st.text_input("图表标题", "箱线图")
                col1, col2 = st.columns(2)
                with col1:
                    y = st.selectbox("数值列", numerical_cols)
                with col2:
                    x = st.selectbox("分组列(可选)", ["无"] + categorical_cols)
                
                color = st.selectbox("颜色分组(可选)", ["无"] + categorical_cols)
                
                if st.form_submit_button("生成图表"):
                    try:
                        fig = self.visualizer.create_box_plot(
                            x=None if x == "无" else x,
                            y=y,
                            title=title,
                            color=None if color == "无" else color
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"图表生成失败: {str(e)}")
            
            elif chart_id == "histogram":
                title = st.text_input("图表标题", "直方图")
                col1, col2 = st.columns(2)
                with col1:
                    column = st.selectbox("数值列", numerical_cols)
                with col2:
                    bins = st.slider("分组数量", 5, 100, 20)
                
                color = st.selectbox("颜色分组(可选)", ["无"] + categorical_cols)
                cumulative = st.checkbox("显示累积分布")
                
                if st.form_submit_button("生成图表"):
                    try:
                        fig = self.visualizer.create_histogram(
                            column=column,
                            bins=bins,
                            title=title,
                            color=None if color == "无" else color,
                            cumulative=cumulative
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"图表生成失败: {str(e)}")
            
            elif chart_id == "bubble":
                title = st.text_input("图表标题", "气泡图")
                col1, col2 = st.columns(2)
                with col1:
                    x = st.selectbox("X轴", numerical_cols)
                with col2:
                    y = st.selectbox("Y轴", numerical_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    size = st.selectbox("气泡大小", numerical_cols)
                with col2:
                    color = st.selectbox("颜色分组(可选)", ["无"] + categorical_cols)
                
                if st.form_submit_button("生成图表"):
                    try:
                        fig = self.visualizer.create_bubble_chart(
                            x=x,
                            y=y,
                            size=size,
                            title=title,
                            color=None if color == "无" else color
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"图表生成失败: {str(e)}")
            
            elif chart_id == "time_series":
                title = st.text_input("图表标题", "时间序列图")
                
                if not datetime_cols:
                    st.warning("未检测到日期时间列。您可以选择文本列，系统将尝试转换为日期格式。")
                    date_options = self.df.select_dtypes(include=['object']).columns.tolist()
                else:
                    date_options = datetime_cols
                
                col1, col2 = st.columns(2)
                with col1:
                    date_column = st.selectbox("日期列", date_options)
                with col2:
                    value_column = st.selectbox("数值列", numerical_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    color = st.selectbox("颜色分组(可选)", ["无"] + categorical_cols)
                with col2:
                    resample_freq = st.selectbox(
                        "重采样频率(可选)",
                        ["无", "日(D)", "周(W)", "月(M)", "季度(Q)", "年(Y)"]
                    )
                    freq_map = {"无": None, "日(D)": "D", "周(W)": "W", "月(M)": "M", "季度(Q)": "Q", "年(Y)": "Y"}
                
                if st.form_submit_button("生成图表"):
                    try:
                        fig = self.visualizer.create_time_series(
                            date_column=date_column,
                            value_column=value_column,
                            title=title,
                            freq=freq_map[resample_freq],
                            color=None if color == "无" else color
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"图表生成失败: {str(e)}")
            
            elif chart_id == "geo_map":
                title = st.text_input("图表标题", "地理地图")
                
                # 查找可能的地理列
                geo_cols = [col for col in self.df.columns if any(kw in col.lower() for kw in 
                                                       ['country', 'region', 'location', 'state', 'province', 'city', 
                                                        '国家', '地区', '省', '市'])]
                
                col1, col2 = st.columns(2)
                with col1:
                    location_column = st.selectbox("地区/国家列", geo_cols if geo_cols else categorical_cols)
                with col2:
                    value_column = st.selectbox("数值列(可选)", ["计数"] + numerical_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    scope = st.selectbox("地图范围", ["世界", "亚洲", "欧洲", "北美", "南美", "非洲", "中国"])
                    scope_map = {
                        "世界": "world", "亚洲": "asia", "欧洲": "europe", 
                        "北美": "north america", "南美": "south america", 
                        "非洲": "africa", "中国": "china"
                    }
                with col2:
                    color_scale = st.selectbox(
                        "颜色比例尺", 
                        ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", 
                         "Blues", "Greens", "Reds", "YlOrRd", "RdBu", "Spectral"]
                    )
                
                if st.form_submit_button("生成图表"):
                    try:
                        fig = self.visualizer.create_geo_map(
                            location_column=location_column,
                            value_column=None if value_column == "计数" else value_column,
                            title=title,
                            scope=scope_map[scope],
                            color_scale=color_scale
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"图表生成失败: {str(e)}")
    
    def _generate_visualization_recommendations(self) -> List[Tuple[str, str, Dict]]:
        """生成可视化推荐
        
        Returns:
            推荐列表，每项包含(标题, 图表类型, 参数字典)
        """
        if self.df is None or len(self.df) == 0:
            return []
        
        recommendations = []
        
        try:
            numerical_cols = self.visualizer.get_numerical_columns()
            categorical_cols = self.visualizer.get_categorical_columns()
            datetime_cols = self.visualizer.get_datetime_columns()
            
            # 1. 如果有日期列和数值列，推荐时间序列
            if datetime_cols and numerical_cols:
                date_col = datetime_cols[0]
                num_col = max(numerical_cols, key=lambda x: self.df[x].notna().sum())
                
                recommendations.append((
                    f"{num_col}随时间变化趋势",
                    "time_series",
                    {"date_column": date_col, "value_column": num_col, "title": f"{num_col}随时间变化趋势"}
                ))
            
            # 2. 如果有分类列，推荐条形图展示分布
            if categorical_cols:
                cat_col = max(categorical_cols, key=lambda x: len(self.df[x].unique()))
                
                # 如果类别太多，选择前10个最多的
                if len(self.df[cat_col].unique()) > 10:
                    recommendations.append((
                        f"{cat_col}的前10个类别分布",
                        "bar",
                        {"x": cat_col, "y": "count", "title": f"{cat_col}的前10个类别分布", "sort_values": True}
                    ))
                else:
                    recommendations.append((
                        f"{cat_col}的分布",
                        "bar",
                        {"x": cat_col, "y": "count", "title": f"{cat_col}的分布", "sort_values": True}
                    ))
            
            # 3. 如果有2个以上数值列，推荐相关性热力图
            if len(numerical_cols) >= 3:
                # 选择相关性可能较高的列(尽量排除ID列)
                non_id_cols = [col for col in numerical_cols 
                              if not any(kw in col.lower() for kw in ['id', 'code', 'key', 'index'])]
                
                if len(non_id_cols) >= 3:
                    selected_cols = non_id_cols[:min(8, len(non_id_cols))]
                    recommendations.append((
                        "主要数值特征相关性分析",
                        "heatmap",
                        {"columns": selected_cols, "title": "主要数值特征相关性分析"}
                    ))
            
            # 4. 如果有分类列和数值列，推荐箱线图
            if categorical_cols and numerical_cols:
                cat_col = min(categorical_cols, key=lambda x: len(self.df[x].unique()))
                num_col = max(numerical_cols, key=lambda x: self.df[x].var())
                
                # 确保类别数量适中
                if 2 <= len(self.df[cat_col].unique()) <= 10:
                    recommendations.append((
                        f"{cat_col}分组的{num_col}箱线图分析",
                        "box",
                        {"x": cat_col, "y": num_col, "title": f"{cat_col}分组的{num_col}箱线图分析"}
                    ))
            
            # 5. 如果有2个数值列，推荐散点图
            if len(numerical_cols) >= 2:
                # 选择方差较大的两列
                cols_by_var = sorted(numerical_cols, key=lambda x: -self.df[x].var())
                if len(cols_by_var) >= 2:
                    x_col, y_col = cols_by_var[0], cols_by_var[1]
                    
                    # 如果有合适的分类变量，添加为颜色
                    color_col = None
                    if categorical_cols:
                        for col in categorical_cols:
                            if 2 <= len(self.df[col].unique()) <= 7:
                                color_col = col
                                break
                    
                    recommendations.append((
                        f"{x_col}与{y_col}的关系分析",
                        "scatter",
                        {
                            "x": x_col, 
                            "y": y_col, 
                            "title": f"{x_col}与{y_col}的关系分析",
                            "color": color_col,
                            "add_trend": True
                        }
                    ))
            
            # 6. 如果有数值列，推荐直方图
            if numerical_cols:
                # 选择分布较宽的列
                num_col = max(numerical_cols, key=lambda x: self.df[x].std())
                recommendations.append((
                    f"{num_col}的分布",
                    "histogram",
                    {"column": num_col, "bins": 20, "title": f"{num_col}的分布"}
                ))
            
            # 7. 如果有地理相关列，推荐地图
            geo_cols = [col for col in self.df.columns if any(kw in col.lower() for kw in 
                                                  ['country', 'region', 'location', 'state', 'province', 
                                                   '国家', '地区', '省', '市'])]
            if geo_cols and numerical_cols:
                geo_col = geo_cols[0]
                num_col = numerical_cols[0]
                recommendations.append((
                    f"{geo_col}的{num_col}地理分布",
                    "geo_map",
                    {"location_column": geo_col, "value_column": num_col, 
                     "title": f"{geo_col}的{num_col}地理分布", "scope": "world"}
                ))
        
        except Exception as e:
            # 异常处理，确保推荐生成不会导致整个应用崩溃
            print(f"生成可视化推荐时出错: {str(e)}")
        
        return recommendations
    
    def _render_machine_learning(self):
        """机器学习选项卡内容"""
        st.header("机器学习")
        
        # 确保数据已加载
        if self.df is None or self.df.empty:
            st.error("请先加载数据")
            return
        
        # 创建两列布局
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("模型训练参数")
            
            # 选择目标变量
            target_column = st.selectbox(
                "选择目标变量", 
                self.df.columns,
                help="选择需要预测的目标变量"
            )
            
            # 检查目标变量是否有足够数据
            if self.df[target_column].isna().sum() > 0.5 * len(self.df):
                st.warning(f"选择的目标列'{target_column}'有超过50%的缺失值，可能导致预测不准确。")
            
            # 检查数据量是否足够
            if len(self.df) < 10:
                st.warning(f"⚠️ 警告：当前数据只有{len(self.df)}行，样本量较小。将自动使用留一法交叉验证，但模型预测可能不稳定。建议增加数据量以获得更可靠的结果。")
            
            # 选择特征变量
            numeric_cols = list(self.df.select_dtypes(include=["int64", "float64"]).columns)
            feature_columns = st.multiselect(
                "选择特征变量", 
                [col for col in numeric_cols if col != target_column],
                help="选择用于预测的特征变量，默认全选所有数值型列"
            )
            
            # 如果没有选择特征，默认使用所有数值列(除了目标变量)
            if not feature_columns:
                feature_columns = [col for col in numeric_cols if col != target_column]
                st.info(f"未选择特征，将使用所有数值列作为特征: {', '.join(feature_columns)}")
            
            # 模型类型选择
            problem_type = st.radio(
                "问题类型", 
                ["自动判断", "回归问题", "分类问题"],
                help="自动判断将根据目标变量的特征决定使用回归还是分类模型"
            )
            
            # 模型选择
            model_options = {
                "回归问题": ["线性回归", "随机森林回归", "梯度提升回归", "支持向量回归", "K近邻回归"],
                "分类问题": ["逻辑回归", "随机森林分类", "梯度提升分类", "支持向量分类", "K近邻分类"]
            }
            
            # 按问题类型选择模型
            if problem_type == "自动判断":
                model_type = st.selectbox(
                    "模型选择", 
                    ["自动选择最佳模型"] + model_options["回归问题"] + model_options["分类问题"]
                )
            elif problem_type == "回归问题":
                model_type = st.selectbox("模型选择", ["自动选择最佳模型"] + model_options["回归问题"])
            else:  # 分类问题
                model_type = st.selectbox("模型选择", ["自动选择最佳模型"] + model_options["分类问题"])
            
            # 高级选项
            with st.expander("高级选项"):
                test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
                random_state = st.number_input("随机种子", 0, 1000, 42)
                scale_method = st.selectbox(
                    "特征缩放方法", 
                    [None, "标准化(standard)", "归一化(minmax)"],
                    format_func=lambda x: "不缩放" if x is None else x
                )
                
                # 映射缩放方法
                if scale_method == "标准化(standard)":
                    scale_method = "standard"
                elif scale_method == "归一化(minmax)":
                    scale_method = "minmax"
            
            # 训练按钮
            train_button = st.button("训练模型", use_container_width=True)
        
        # 显示训练结果
        with col2:
            st.subheader("模型结果")
            
            if train_button:
                try:
                    with st.spinner("正在训练模型..."):
                        # 初始化模型
                        self.ml_model = MLModel(self.df, target_column)
                        
                        # 预处理数据
                        self.ml_model.preprocess_data(
                            feature_columns=feature_columns, 
                            test_size=test_size, 
                            random_state=random_state, 
                            scale_method=scale_method
                        )
                        
                        # 根据不同模型类型和选项训练模型
                        model_mapping = {
                            "线性回归": "linear",
                            "随机森林回归": "random_forest",
                            "梯度提升回归": "gradient_boosting",
                            "支持向量回归": "svr",
                            "K近邻回归": "knn",
                            "逻辑回归": "logistic",
                            "随机森林分类": "random_forest",
                            "梯度提升分类": "gradient_boosting",
                            "支持向量分类": "svc",
                            "K近邻分类": "knn"
                        }
                        
                        if model_type == "自动选择最佳模型":
                            metrics = self.ml_model.auto_train()
                            st.success(f"已自动选择最佳模型: {metrics.get('model_type', '未知')}")
                        else:
                            # 判断是分类还是回归
                            is_classification = self.ml_model._is_classification()
                            
                            if is_classification:
                                if model_type in model_options["分类问题"]:
                                    metrics = self.ml_model.train_classification_model(model_mapping[model_type])
                                    st.success(f"分类模型训练完成: {model_type}")
                                else:
                                    st.error(f"数据适合分类模型，但选择了回归模型: {model_type}")
                                    return
                            else:
                                if model_type in model_options["回归问题"]:
                                    metrics = self.ml_model.train_regression_model(model_mapping[model_type])
                                    st.success(f"回归模型训练完成: {model_type}")
                                else:
                                    st.error(f"数据适合回归模型，但选择了分类模型: {model_type}")
                                    return
                        
                        # 创建选项卡来展示结果
                        result_tabs = st.tabs(["📊 性能指标", "📈 可视化结果", "🔍 特征重要性"])
                        
                        # 性能指标选项卡
                        with result_tabs[0]:
                            # 判断是分类还是回归
                            is_classification = self.ml_model._is_classification()
                            
                            if is_classification:
                                # 分类指标
                                metrics_df = pd.DataFrame({
                                    "指标": ["准确率", "精确率", "召回率", "F1分数", "交叉验证准确率"],
                                    "数值": [
                                        metrics.get("accuracy", 0),
                                        metrics.get("precision", 0),
                                        metrics.get("recall", 0),
                                        metrics.get("f1", 0),
                                        metrics.get("cv_accuracy", 0)
                                    ]
                                })
                            else:
                                # 回归指标
                                metrics_df = pd.DataFrame({
                                    "指标": ["均方误差(MSE)", "均方根误差(RMSE)", "决定系数(R²)", "交叉验证RMSE"],
                                    "数值": [
                                        metrics.get("mse", 0),
                                        metrics.get("rmse", 0),
                                        metrics.get("r2", 0),
                                        metrics.get("cv_rmse", 0)
                                    ]
                                })
                            
                            st.dataframe(metrics_df, use_container_width=True)
                        
                        # 可视化结果选项卡
                        with result_tabs[1]:
                            if is_classification:
                                # 显示混淆矩阵
                                conf_fig = self.ml_model.plot_confusion_matrix()
                                if conf_fig is not None:
                                    st.plotly_chart(conf_fig, use_container_width=True)
                                else:
                                    st.info("无法生成混淆矩阵")
                            else:
                                # 显示回归结果
                                reg_fig = self.ml_model.plot_regression_results()
                                if reg_fig is not None:
                                    st.plotly_chart(reg_fig, use_container_width=True)
                                else:
                                    st.info("无法生成回归结果图")
                        
                        # 特征重要性选项卡
                        with result_tabs[2]:
                            # 显示特征重要性
                            importance_fig = self.ml_model.plot_feature_importance()
                            if importance_fig is not None:
                                st.plotly_chart(importance_fig, use_container_width=True)
                            else:
                                st.info("当前模型不支持特征重要性展示")
                    
                    # 添加模型预测功能
                    with st.expander("模型预测", expanded=True):
                        st.write("使用训练好的模型进行预测")
                        
                        # 创建输入表单
                        prediction_data = {}
                        for col in feature_columns:
                            if col in numeric_cols:
                                min_val = float(self.df[col].min())
                                max_val = float(self.df[col].max())
                                mean_val = float(self.df[col].mean())
                                
                                prediction_data[col] = st.slider(
                                    f"{col}", 
                                    min_val, 
                                    max_val, 
                                    mean_val,
                                    help=f"范围: [{min_val:.2f}, {max_val:.2f}], 平均值: {mean_val:.2f}"
                                )
                        
                        if st.button("预测", use_container_width=True):
                            try:
                                # 预测
                                input_df = pd.DataFrame([prediction_data])
                                prediction = self.ml_model.predict(input_df)
                                
                                # 显示预测结果
                                st.success(f"预测结果: {prediction[0]}")
                                
                                # 如果是分类，显示每个类别的概率(如果模型支持)
                                if is_classification and hasattr(self.ml_model.model, "predict_proba"):
                                    proba = self.ml_model.model.predict_proba(self.ml_model.scaler.transform(input_df) if self.ml_model.scaler else input_df)
                                    
                                    # 获取类别标签
                                    if self.ml_model.label_encoder is not None:
                                        class_labels = self.ml_model.label_encoder.classes_
                                    else:
                                        class_labels = [f"类别 {i}" for i in range(proba.shape[1])]
                                    
                                    # 显示概率
                                    proba_df = pd.DataFrame({
                                        "类别": class_labels,
                                        "概率": proba[0]
                                    })
                                    st.dataframe(proba_df, use_container_width=True)
                                    
                                    # 绘制概率条形图
                                    fig = px.bar(
                                        proba_df, 
                                        x="类别", 
                                        y="概率", 
                                        title="各类别预测概率",
                                        text="概率"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"预测失败: {str(e)}")
                
                except Exception as e:
                    st.error(f"模型训练失败: {str(e)}")
                    st.error("请检查数据格式和选择的特征是否合适")
                    import traceback
                    st.code(traceback.format_exc())
    
    def _render_data_report(self):
        """数据报告选项卡内容"""
        st.header("数据报告")
        
        if st.button("生成数据分析报告"):
            with st.spinner("正在生成报告..."):
                # 基本信息
                st.subheader("1. 数据概览")
                st.write(f"- 数据集大小: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")
                st.write(f"- 数值型特征: {len(self.df.select_dtypes(include=['int64', 'float64']).columns)} 个")
                st.write(f"- 分类型特征: {len(self.df.select_dtypes(include=['object', 'category']).columns)} 个")
                st.write(f"- 缺失值: {self.df.isnull().sum().sum()} 个")
                
                # 数据质量
                st.subheader("2. 数据质量分析")
                missing_data = self.df.isnull().sum()[self.df.isnull().sum() > 0]
                if not missing_data.empty:
                    st.write("**缺失值情况**")
                    missing_df = pd.DataFrame({
                        '缺失值数量': missing_data,
                        '缺失值比例': (missing_data / len(self.df) * 100).round(2)
                    })
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.write("数据集中没有缺失值，数据完整度良好。")
                
                # 特征分布
                st.subheader("3. 特征分布分析")
                numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
                
                st.write("**数值特征统计**")
                stats_df = self.df[numeric_cols].describe().T
                stats_df['变异系数'] = (stats_df['std'] / stats_df['mean']).abs().round(4)
                st.dataframe(stats_df, use_container_width=True)
                
                # 相关性分析
                if len(numeric_cols) > 1:
                    st.subheader("4. 相关性分析")
                    st.write("**相关系数热力图**")
                    corr_fig = self.visualizer.correlation_heatmap(numeric_cols)
                    st.pyplot(corr_fig)
                    
                    # 高相关变量
                    corr_matrix = self.df[numeric_cols].corr().abs()
                    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    high_corr = [(col1, col2, corr_matrix.loc[col1, col2]) 
                                for col1 in corr_matrix.columns 
                                for col2 in corr_matrix.columns 
                                if corr_matrix.loc[col1, col2] > 0.7 and col1 != col2 and col1 < col2]
                    
                    if high_corr:
                        st.write("**高相关特征对** (相关系数 > 0.7)")
                        high_corr_df = pd.DataFrame(high_corr, columns=['特征1', '特征2', '相关系数']).sort_values('相关系数', ascending=False)
                        st.dataframe(high_corr_df, use_container_width=True)
                
                # 分类特征分析
                cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                    st.subheader("5. 分类特征分析")
                    
                    for col in cat_cols[:3]:  # 限制显示前3个分类特征以避免报告过长
                        st.write(f"**{col}** 的分布")
                        counts = self.df[col].value_counts().reset_index()
                        counts.columns = [col, '计数']
                        counts['占比'] = (counts['计数'] / counts['计数'].sum() * 100).round(2)
                        st.dataframe(counts.head(10), use_container_width=True)
                        
                        fig = self.visualizer.plotly_bar(col)
                        st.plotly_chart(fig, use_container_width=True)
                
                # 报告总结
                st.subheader("6. 报告总结")
                st.write("**数据特点**")
                st.write(f"- 该数据集包含 {self.df.shape[0]} 条记录和 {self.df.shape[1]} 个特征")
                
                if self.df.isnull().sum().sum() > 0:
                    missing_pct = (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]) * 100).round(2)
                    st.write(f"- 数据集缺失值比例为 {missing_pct}%，建议进行适当的缺失值处理")
                else:
                    st.write("- 数据集完整度良好，无缺失值")
                
                if len(high_corr) > 0 if 'high_corr' in locals() else False:
                    st.write(f"- 发现 {len(high_corr)} 对高相关特征，可能存在特征冗余")
                
                st.write("**建议**")
                if self.df.isnull().sum().sum() > 0:
                    st.write("- 对缺失值较多的特征进行填充或考虑删除")
                
                if len(high_corr) > 0 if 'high_corr' in locals() else False:
                    st.write("- 考虑对高度相关的特征进行特征选择或降维处理")
                
                st.write("- 对于数值特征，考虑进行标准化或归一化处理")
                st.write("- 对于分类特征，考虑进行适当的编码转换")
                
                st.success("数据分析报告生成完成！") 