import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Optional, List, Dict, Any
import json
import numpy as np

from src.data_processor import DataProcessor
from src.visualizer import Visualizer
from src.ml_model import MLModel  # 导入新增的机器学习模块


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
        st.set_page_config(page_title=self.title, page_icon="📊", layout="wide")
        
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
            
            # 文件上传
            uploaded_file = st.file_uploader("上传数据文件", type=["csv", "xlsx", "xls", "json"])
            
            if uploaded_file is not None:
                try:
                    # 根据文件类型加载数据
                    file_type = uploaded_file.name.split(".")[-1].lower()
                    
                    if file_type == "csv":
                        self.df = pd.read_csv(uploaded_file)
                    elif file_type in ["xlsx", "xls"]:
                        self.df = pd.read_excel(uploaded_file)
                    elif file_type == "json":
                        self.df = pd.read_json(uploaded_file)
                    
                    # 初始化处理器和可视化器
                    self.processor = DataProcessor(self.df)
                    self.visualizer = Visualizer(self.df)
                    
                    st.success(f"成功加载数据: {uploaded_file.name}")
                    st.write(f"数据大小: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")
                except Exception as e:
                    st.error(f"加载数据时出错: {str(e)}")
            
            # 使用示例数据的选项
            if st.button("加载示例数据"):
                # 加载示例数据
                try:
                    import seaborn as sns
                    self.df = sns.load_dataset("tips")
                    
                    # 初始化处理器和可视化器
                    self.processor = DataProcessor(self.df)
                    self.visualizer = Visualizer(self.df)
                    
                    st.success("成功加载示例数据 (tips)")
                except Exception as e:
                    st.error(f"加载示例数据时出错: {str(e)}")
    
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
        col_info = pd.DataFrame({
            "数据类型": self.df.dtypes,
            "非空值数": self.df.count(),
            "缺失值数": self.df.isnull().sum(),
            "缺失值百分比": (self.df.isnull().sum() / len(self.df) * 100).round(2),
            "唯一值数": [self.df[col].nunique() for col in self.df.columns],
        })
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
            for col in categorical_df.columns:
                st.write(f"**{col}** (Top 10)")
                st.dataframe(self.df[col].value_counts().head(10).reset_index(), use_container_width=True)
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
        
        # 图表类型选择
        chart_type = st.selectbox(
            "选择图表类型",
            ["散点图", "条形图", "折线图", "饼图", "直方图", "箱线图", "相关系数热力图", "成对关系图"]
        )
        
        # 根据图表类型提供不同的选项
        if chart_type == "散点图":
            x_col = st.selectbox("X轴", self.df.select_dtypes(include=["int64", "float64"]).columns)
            y_col = st.selectbox("Y轴", self.df.select_dtypes(include=["int64", "float64"]).columns, 
                                index=min(1, len(self.df.select_dtypes(include=["int64", "float64"]).columns)-1))
            
            cat_cols = list(self.df.select_dtypes(include=["object", "category"]).columns)
            color_col = st.selectbox("颜色分组 (可选)", ["None"] + cat_cols) if cat_cols else "None"
            color_col = None if color_col == "None" else color_col
            
            if st.button("生成散点图"):
                fig = self.visualizer.plotly_scatter(x_col, y_col, color_col)
                st.plotly_chart(fig, use_container_width=True)
                
        elif chart_type == "条形图":
            cat_cols = list(self.df.select_dtypes(include=["object", "category"]).columns)
            num_cols = list(self.df.select_dtypes(include=["int64", "float64"]).columns)
            
            x_col = st.selectbox("X轴 (分类变量)", cat_cols if cat_cols else self.df.columns)
            y_col = st.selectbox("Y轴 (数值变量，可选)", ["计数"] + num_cols)
            y_col = None if y_col == "计数" else y_col
            
            if st.button("生成条形图"):
                fig = self.visualizer.plotly_bar(x_col, y_col)
                st.plotly_chart(fig, use_container_width=True)
                
        elif chart_type == "折线图":
            x_col = st.selectbox("X轴", self.df.columns)
            y_col = st.selectbox("Y轴", self.df.select_dtypes(include=["int64", "float64"]).columns)
            
            cat_cols = list(self.df.select_dtypes(include=["object", "category"]).columns)
            color_col = st.selectbox("颜色分组 (可选)", ["None"] + cat_cols) if cat_cols else "None"
            color_col = None if color_col == "None" else color_col
            
            if st.button("生成折线图"):
                fig = self.visualizer.plotly_line(x_col, y_col, color_col)
                st.plotly_chart(fig, use_container_width=True)
                
        elif chart_type == "饼图":
            cat_cols = list(self.df.select_dtypes(include=["object", "category"]).columns)
            if cat_cols:
                col = st.selectbox("选择变量", cat_cols)
                if st.button("生成饼图"):
                    fig = self.visualizer.plotly_pie(col)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("未发现适合饼图的分类变量")
                
        elif chart_type == "直方图":
            num_cols = list(self.df.select_dtypes(include=["int64", "float64"]).columns)
            if num_cols:
                col = st.selectbox("选择变量", num_cols)
                bins = st.slider("分箱数量", 5, 100, 30)
                kde = st.checkbox("显示密度曲线", True)
                
                if st.button("生成直方图"):
                    fig = self.visualizer.distribution_plot(col, bins, kde)
                    st.pyplot(fig)
            else:
                st.warning("未发现适合直方图的数值变量")
                
        elif chart_type == "箱线图":
            num_cols = list(self.df.select_dtypes(include=["int64", "float64"]).columns)
            if num_cols:
                col = st.selectbox("选择数值变量", num_cols)
                
                cat_cols = list(self.df.select_dtypes(include=["object", "category"]).columns)
                group_col = st.selectbox("分组变量 (可选)", ["None"] + cat_cols) if cat_cols else "None"
                group_col = None if group_col == "None" else group_col
                
                if st.button("生成箱线图"):
                    fig = self.visualizer.boxplot(col, group_col)
                    st.pyplot(fig)
            else:
                st.warning("未发现适合箱线图的数值变量")
                
        elif chart_type == "相关系数热力图":
            num_cols = list(self.df.select_dtypes(include=["int64", "float64"]).columns)
            if len(num_cols) > 1:
                selected_cols = st.multiselect("选择变量 (默认全选)", num_cols, default=num_cols)
                
                if st.button("生成相关系数热力图") and selected_cols:
                    fig = self.visualizer.correlation_heatmap(selected_cols)
                    st.pyplot(fig)
            else:
                st.warning("至少需要两个数值变量来计算相关系数")
                
        elif chart_type == "成对关系图":
            num_cols = list(self.df.select_dtypes(include=["int64", "float64"]).columns)
            if len(num_cols) > 1:
                max_cols = min(5, len(num_cols))
                selected_cols = st.multiselect("选择变量 (建议选择2-5个)", num_cols, 
                                             default=num_cols[:max_cols])
                
                cat_cols = list(self.df.select_dtypes(include=["object", "category"]).columns)
                hue_col = st.selectbox("颜色分组 (可选)", ["None"] + cat_cols) if cat_cols else "None"
                hue_col = None if hue_col == "None" else hue_col
                
                if st.button("生成成对关系图") and len(selected_cols) >= 2:
                    with st.spinner("正在生成成对关系图，这可能需要一些时间..."):
                        fig = self.visualizer.pair_plot(selected_cols, hue_col)
                        st.pyplot(fig)
            else:
                st.warning("至少需要两个数值变量来创建成对关系图")
    
    def _render_machine_learning(self):
        """机器学习选项卡内容"""
        st.header("机器学习分析")
        
        # 显示警告，如果数据量太小
        if len(self.df) < 20:
            st.warning(f"当前数据集仅包含 {len(self.df)} 条记录，这对机器学习模型来说样本量偏小。建议增加样本量以获得更可靠的结果。", icon="⚠️")
            if len(self.df) < 10:
                st.info("对于小样本数据集，系统将自动调整训练测试集比例和交叉验证方法，但预测结果可能不够稳定。", icon="ℹ️")
        
        # 选择分析类型
        analysis_type = st.radio(
            "选择分析类型", 
            ["预测分析", "聚类分析"],
            horizontal=True
        )
        
        if analysis_type == "预测分析":
            st.subheader("预测模型训练")
            
            # 选择目标变量
            target_column = st.selectbox("选择目标变量", self.df.columns)
            
            # 选择特征变量
            feature_candidates = [col for col in self.df.columns if col != target_column]
            selected_features = st.multiselect(
                "选择特征变量 (默认使用所有数值型特征)", 
                feature_candidates
            )
            
            # 模型训练参数
            col1, col2 = st.columns(2)
            with col1:
                # 根据数据集大小调整测试集比例选项
                if len(self.df) < 10:
                    min_test_size = max(0.1, 1/len(self.df))
                    test_size = st.slider("测试集比例", min_test_size, 0.5, min_test_size, 0.05)
                    st.caption(f"由于样本量较小，测试集至少包含1个样本")
                else:
                    test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
            with col2:
                scale_method = st.selectbox("特征缩放方法", ["不缩放", "标准化", "归一化"])
                if scale_method == "不缩放":
                    scale_method = None
                elif scale_method == "标准化":
                    scale_method = "standard"
                else:
                    scale_method = "minmax"
            
            # 判断目标变量类型
            is_categorical = False
            if target_column:
                target_values = self.df[target_column].dropna().unique()
                is_categorical = (len(target_values) <= 10 or 
                                  self.df[target_column].dtype == 'object' or 
                                  self.df[target_column].dtype.name == 'category')
            
            # 确定模型类型和可选模型
            if is_categorical:
                st.info(f"目标变量 '{target_column}' 被识别为分类变量，将使用分类模型")
                model_type = st.selectbox(
                    "选择分类模型", 
                    ["自动选择最佳模型", "逻辑回归", "随机森林", "梯度提升树", "支持向量机", "K近邻"]
                )
                
                if model_type == "自动选择最佳模型":
                    model_name = "auto"
                elif model_type == "逻辑回归":
                    model_name = "logistic"
                elif model_type == "随机森林":
                    model_name = "random_forest"
                elif model_type == "梯度提升树":
                    model_name = "gradient_boosting"
                elif model_type == "支持向量机":
                    model_name = "svc"
                else:
                    model_name = "knn"
            else:
                st.info(f"目标变量 '{target_column}' 被识别为连续变量，将使用回归模型")
                model_type = st.selectbox(
                    "选择回归模型",
                    ["自动选择最佳模型", "线性回归", "随机森林", "梯度提升树", "支持向量回归", "K近邻"]
                )
                
                if model_type == "自动选择最佳模型":
                    model_name = "auto"
                elif model_type == "线性回归":
                    model_name = "linear"
                elif model_type == "随机森林":
                    model_name = "random_forest"
                elif model_type == "梯度提升树":
                    model_name = "gradient_boosting"
                elif model_type == "支持向量回归":
                    model_name = "svr"
                else:
                    model_name = "knn"
            
            # 训练模型按钮
            if st.button("训练模型"):
                with st.spinner("模型训练中..."):
                    try:
                        # 初始化机器学习模型
                        self.ml_model = MLModel(self.df, target_column)
                        
                        # 数据预处理
                        feature_columns = selected_features if selected_features else None
                        self.ml_model.preprocess_data(
                            feature_columns=feature_columns, 
                            test_size=test_size,
                            scale_method=scale_method
                        )
                        
                        # 显示划分后的训练集和测试集大小
                        col1, col2 = st.columns(2)
                        col1.metric("训练集样本数", len(self.ml_model.y_train))
                        col2.metric("测试集样本数", len(self.ml_model.y_test))
                        
                        # 如果样本太少，显示警告
                        if len(self.ml_model.y_train) < 5:
                            st.warning(f"训练集仅包含 {len(self.ml_model.y_train)} 个样本，模型性能可能不稳定。", icon="⚠️")
                        
                        # 训练模型
                        if model_name == "auto":
                            metrics = self.ml_model.auto_train()
                            st.success(f"自动选择了最佳模型: {metrics.get('model_type', '未知')}")
                        elif is_categorical:
                            metrics = self.ml_model.train_classification_model(model_name)
                            st.success(f"分类模型训练完成")
                        else:
                            metrics = self.ml_model.train_regression_model(model_name)
                            st.success(f"回归模型训练完成")
                        
                        # 显示模型评估结果
                        st.subheader("模型评估")
                        
                        # 显示不同的评估指标
                        if is_categorical:
                            metrics_data = [
                                ['准确率', metrics.get('accuracy', 0)],
                                ['精确率', metrics.get('precision', '不适用') if metrics.get('precision') is not None else '不适用'],
                                ['召回率', metrics.get('recall', '不适用') if metrics.get('recall') is not None else '不适用'],
                                ['F1得分', metrics.get('f1', '不适用') if metrics.get('f1') is not None else '不适用'],
                                ['交叉验证准确率', metrics.get('cv_accuracy', '不适用') if metrics.get('cv_accuracy') is not None else '不适用']
                            ]
                            metrics_df = pd.DataFrame(metrics_data, columns=['指标', '值'])
                            st.dataframe(metrics_df, use_container_width=True)
                            
                            # 混淆矩阵
                            if metrics.get('confusion_matrix') is not None:
                                st.subheader("混淆矩阵")
                                confusion_fig = self.ml_model.plot_confusion_matrix()
                                if confusion_fig is not None:
                                    st.plotly_chart(confusion_fig, use_container_width=True)
                            else:
                                st.info("样本量不足，无法生成混淆矩阵")
                        else:
                            metrics_data = [
                                ['均方误差(MSE)', metrics.get('mse', 0)],
                                ['均方根误差(RMSE)', metrics.get('rmse', 0)],
                                ['决定系数(R²)', metrics.get('r2', 0)],
                                ['交叉验证RMSE', metrics.get('cv_rmse', '不适用') if metrics.get('cv_rmse') is not None else '不适用']
                            ]
                            metrics_df = pd.DataFrame(metrics_data, columns=['指标', '值'])
                            st.dataframe(metrics_df, use_container_width=True)
                            
                            # 回归结果可视化
                            st.subheader("回归结果")
                            regression_fig = self.ml_model.plot_regression_results()
                            if regression_fig is not None:
                                st.plotly_chart(regression_fig, use_container_width=True)
                        
                        # 特征重要性
                        importance_fig = self.ml_model.plot_feature_importance()
                        if importance_fig is not None:
                            st.subheader("特征重要性")
                            st.plotly_chart(importance_fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"模型训练过程中出错: {str(e)}")
                        st.error("如果报错与交叉验证相关，请尝试减少特征数量或增加训练集样本量")
                        # 显示详细的异常信息
                        import traceback
                        st.expander("查看详细错误信息").code(traceback.format_exc())
            
            # 模型预测部分
            if self.ml_model is not None and self.ml_model.model is not None:
                st.subheader("使用模型进行预测")
                
                # 创建预测数据输入表单
                st.write("输入特征值进行预测:")
                
                col_count = min(3, len(self.ml_model.feature_columns))
                cols = st.columns(col_count)
                
                input_data = {}
                for i, feature in enumerate(self.ml_model.feature_columns):
                    col_idx = i % col_count
                    with cols[col_idx]:
                        # 根据特征类型设置不同的输入控件
                        if self.df[feature].dtype in ['int64', 'int32']:
                            input_data[feature] = st.number_input(
                                f"{feature}", 
                                value=int(self.df[feature].mean()),
                                step=1
                            )
                        elif self.df[feature].dtype in ['float64', 'float32']:
                            input_data[feature] = st.number_input(
                                f"{feature}", 
                                value=float(self.df[feature].mean()),
                                format="%.2f"
                            )
                        else:
                            # 对于分类特征，提供唯一值列表
                            options = self.df[feature].dropna().unique().tolist()
                            input_data[feature] = st.selectbox(f"{feature}", options)
                
                if st.button("预测"):
                    with st.spinner("正在预测..."):
                        try:
                            # 创建预测数据框
                            pred_df = pd.DataFrame([input_data])
                            
                            # 进行预测
                            prediction = self.ml_model.predict(pred_df)
                            
                            # 显示预测结果
                            st.success("预测完成!")
                            if is_categorical:
                                st.metric("预测分类", prediction[0])
                            else:
                                st.metric("预测值", f"{prediction[0]:.4f}")
                        
                        except Exception as e:
                            st.error(f"预测过程中出错: {str(e)}")
        
        else:  # 聚类分析
            st.subheader("聚类分析")
            
            # 选择特征变量
            all_numeric_cols = list(self.df.select_dtypes(include=['int64', 'float64']).columns)
            selected_features = st.multiselect(
                "选择用于聚类的特征 (默认使用所有数值型特征)", 
                self.df.columns,
                default=all_numeric_cols[:min(5, len(all_numeric_cols))]
            )
            
            # 聚类方法选择
            cluster_method = st.selectbox(
                "选择聚类方法",
                ["K均值聚类", "DBSCAN密度聚类", "层次聚类"]
            )
            
            # 根据聚类方法设置参数
            if cluster_method == "K均值聚类":
                method = "kmeans"
                n_clusters = st.slider("聚类数量", 2, 10, 3)
                params = {}
            
            elif cluster_method == "DBSCAN密度聚类":
                method = "dbscan"
                eps = st.slider("邻域半径(eps)", 0.1, 2.0, 0.5, 0.1)
                min_samples = st.slider("最小样本数", 2, 20, 5)
                params = {"eps": eps, "min_samples": min_samples}
                n_clusters = None
            
            else:  # 层次聚类
                method = "hierarchical"
                n_clusters = st.slider("聚类数量", 2, 10, 3)
                linkage = st.selectbox("连接方式", ["ward", "complete", "average", "single"])
                params = {"linkage": linkage}
            
            # 执行聚类
            if st.button("执行聚类"):
                with st.spinner("正在进行聚类分析..."):
                    try:
                        # 初始化聚类模型
                        self.ml_model = MLModel(self.df)
                        if selected_features:
                            self.ml_model.feature_columns = selected_features
                        
                        # 执行聚类
                        if n_clusters:
                            result = self.ml_model.perform_clustering(method, n_clusters, params)
                        else:
                            result = self.ml_model.perform_clustering(method, params=params)
                        
                        # 显示聚类结果
                        st.success(f"聚类完成: 识别出 {len(set(result['labels'])) - (1 if -1 in result['labels'] else 0)} 个聚类")
                        
                        # 聚类评估
                        if 'silhouette_score' in result:
                            st.metric("轮廓系数", f"{result['silhouette_score']:.4f}")
                            
                        # 可视化聚类结果
                        st.subheader("聚类结果可视化")
                        
                        viz_method = st.radio(
                            "可视化方法",
                            ["PCA降维", "使用原始特征"],
                            horizontal=True
                        )
                        
                        cluster_viz = self.ml_model.visualize_clusters(
                            'pca' if viz_method == "PCA降维" else 'original'
                        )
                        st.plotly_chart(cluster_viz, use_container_width=True)
                        
                        # 聚类统计
                        st.subheader("聚类统计")
                        cluster_counts = self.ml_model.df['cluster'].value_counts().reset_index()
                        cluster_counts.columns = ['聚类编号', '样本数量']
                        st.dataframe(cluster_counts, use_container_width=True)
                        
                        # 各聚类特征分布
                        st.subheader("各聚类特征分布")
                        cluster_stats = self.ml_model.df.groupby('cluster')[selected_features].mean()
                        st.dataframe(cluster_stats, use_container_width=True)
                        
                        # 下载聚类结果
                        csv = self.ml_model.df.to_csv(index=False)
                        st.download_button(
                            "下载聚类结果数据",
                            csv,
                            "clustered_data.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    
                    except Exception as e:
                        st.error(f"聚类分析过程中出错: {str(e)}")
    
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