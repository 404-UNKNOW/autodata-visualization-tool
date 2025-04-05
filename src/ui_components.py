import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Optional, List, Dict, Any, Tuple, Callable
import numpy as np

from src.data_processor import DataProcessor
from src.visualizer import Visualizer
from src.ml_model import MLModel


class UIComponents:
    """UI组件类，用于创建和管理可复用的UI组件"""
    
    @staticmethod
    def render_data_overview_tab(df: pd.DataFrame, visualizer: Visualizer):
        """渲染数据概览选项卡
        
        Args:
            df: 数据框
            visualizer: 可视化器对象
        """
        st.header("数据概览")
        
        # 基本信息
        st.subheader("数据维度")
        col1, col2 = st.columns(2)
        col1.metric("行数", df.shape[0])
        col2.metric("列数", df.shape[1])
        
        # 数据预览
        st.subheader("数据预览")
        st.dataframe(df.head(10), use_container_width=True)
        
        # 列信息
        st.subheader("列信息")
        # 修复PyArrow兼容性问题：将dtypes转换为字符串而不是对象
        col_info = pd.DataFrame({
            "数据类型": [str(dtype) for dtype in df.dtypes.values],
            "非空值数": df.count().values,
            "缺失值数": df.isnull().sum().values,
            "缺失值百分比": (df.isnull().sum() / len(df) * 100).round(2).values,
            "唯一值数": [df[col].nunique() for col in df.columns],
        }, index=df.columns)
        st.dataframe(col_info, use_container_width=True)
        
        # 数值型数据统计
        st.subheader("数值型列统计")
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe().T, use_container_width=True)
        else:
            st.info("没有数值型列")
        
        # 分类型数据统计
        st.subheader("分类型列概览")
        categorical_df = df.select_dtypes(include=["object", "category"])
        if not categorical_df.empty:
            for col in categorical_df.columns[:3]:  # 限制显示前3个分类列
                with st.expander(f"{col} - 唯一值数: {df[col].nunique()}"):
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = [col, "计数"]
                    st.dataframe(value_counts.head(10), use_container_width=True)
        else:
            st.info("没有分类型列")
    
    @staticmethod
    def render_data_processing_tab(df: pd.DataFrame, processor: DataProcessor, on_data_changed: Callable):
        """渲染数据处理选项卡
        
        Args:
            df: 数据框
            processor: 数据处理器对象
            on_data_changed: 数据变更后的回调函数
        """
        st.header("数据处理")
        
        # 处理控制面板
        st.subheader("数据处理选项")
        processed_df = df.copy()
        
        # 缺失值处理
        with st.expander("缺失值处理", expanded=True):
            missing_cols = df.columns[df.isnull().any()]
            if len(missing_cols) > 0:
                missing_options = st.multiselect("选择要处理的列", missing_cols, default=missing_cols)
                if missing_options:
                    strategy = st.selectbox("缺失值处理策略", ["mean", "median", "most_frequent", "drop"], index=0)
                    if st.button("应用缺失值处理"):
                        with st.spinner("处理中..."):
                            processor = DataProcessor(processed_df)
                            processed_df = processor.handle_missing_values(strategy, missing_options).get_result()
                            st.success("缺失值处理完成")
                            on_data_changed(processed_df)
            else:
                st.info("数据中没有缺失值")
        
        # 数据标准化/归一化
        with st.expander("数据标准化/归一化", expanded=False):
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
            if len(numeric_cols) > 0:
                norm_options = st.multiselect("选择要标准化的列", numeric_cols)
                if norm_options:
                    norm_strategy = st.selectbox("标准化方法", ["standard", "minmax"], index=0)
                    if st.button("应用标准化"):
                        with st.spinner("处理中..."):
                            processor = DataProcessor(processed_df)
                            processed_df = processor.normalize_data(norm_strategy, norm_options).get_result()
                            st.success("数据标准化完成")
                            on_data_changed(processed_df)
            else:
                st.info("数据中没有数值列")
        
        # 分类变量编码
        with st.expander("分类变量编码", expanded=False):
            cat_cols = df.select_dtypes(include=["object", "category"]).columns
            if len(cat_cols) > 0:
                cat_options = st.multiselect("选择要编码的分类列", cat_cols)
                if cat_options:
                    encode_strategy = st.selectbox("编码方法", ["onehot", "label"], index=0)
                    if st.button("应用编码"):
                        with st.spinner("处理中..."):
                            processor = DataProcessor(processed_df)
                            processed_df = processor.encode_categorical(cat_options, encode_strategy).get_result()
                            st.success("分类变量编码完成")
                            on_data_changed(processed_df)
            else:
                st.info("数据中没有分类列")
        
        # 数据过滤
        with st.expander("数据过滤", expanded=False):
            filter_col = st.selectbox("选择要过滤的列", df.columns)
            if filter_col:
                if df[filter_col].dtype in ['int64', 'float64']:
                    # 数值列过滤
                    min_val = float(df[filter_col].min())
                    max_val = float(df[filter_col].max())
                    range_val = st.slider(f"选择{filter_col}的范围", min_val, max_val, (min_val, max_val))
                    if st.button("应用数值过滤"):
                        processed_df = processed_df[(processed_df[filter_col] >= range_val[0]) & 
                                                  (processed_df[filter_col] <= range_val[1])]
                        st.success(f"过滤完成，剩余{len(processed_df)}行数据")
                        on_data_changed(processed_df)
                else:
                    # 分类列过滤
                    categories = df[filter_col].unique()
                    selected_cats = st.multiselect(f"选择要保留的{filter_col}类别", categories, default=categories)
                    if st.button("应用分类过滤"):
                        processed_df = processed_df[processed_df[filter_col].isin(selected_cats)]
                        st.success(f"过滤完成，剩余{len(processed_df)}行数据")
                        on_data_changed(processed_df)
    
    @staticmethod
    def render_data_visualization_tab(df: pd.DataFrame, visualizer: Visualizer):
        """渲染数据可视化选项卡
        
        Args:
            df: 数据框
            visualizer: 可视化器对象
        """
        st.header("数据可视化")
        
        # 图表类型选择
        viz_type = st.selectbox(
            "选择可视化类型",
            ["散点图", "柱状图", "折线图", "饼图", "直方图", "箱线图", "相关性热力图", "对图", "小提琴图", "地图"]
        )
        
        # 图表设置
        if viz_type == "散点图":
            UIComponents._render_scatter_plot(df, visualizer)
        elif viz_type == "柱状图":
            UIComponents._render_bar_chart(df, visualizer)
        elif viz_type == "折线图":
            UIComponents._render_line_plot(df, visualizer)
        elif viz_type == "饼图":
            UIComponents._render_pie_chart(df, visualizer)
        elif viz_type == "直方图":
            UIComponents._render_histogram(df, visualizer)
        elif viz_type == "箱线图":
            UIComponents._render_boxplot(df, visualizer)
        elif viz_type == "相关性热力图":
            UIComponents._render_heatmap(df, visualizer)
        elif viz_type == "对图":
            UIComponents._render_pairplot(df, visualizer)
        elif viz_type == "小提琴图":
            UIComponents._render_violinplot(df, visualizer)
        elif viz_type == "地图":
            UIComponents._render_map(df, visualizer)
    
    @staticmethod
    def _render_scatter_plot(df, visualizer):
        """渲染散点图设置"""
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X轴", numeric_cols, index=0)
            y_col = st.selectbox("Y轴", numeric_cols, index=min(1, len(numeric_cols)-1))
            
            color_col = st.selectbox("颜色分组 (可选)", ["无"] + list(df.columns), index=0)
            color = None if color_col == "无" else color_col
            
            size_col = st.selectbox("大小变量 (可选)", ["无"] + list(numeric_cols), index=0)
            size = None if size_col == "无" else size_col
            
            if st.button("生成散点图"):
                with st.spinner("绘图中..."):
                    fig = visualizer.plotly_scatter(x_col, y_col, color=color, size=size)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("散点图需要至少两个数值列")
    
    @staticmethod
    def _render_bar_chart(df, visualizer):
        """渲染柱状图设置"""
        x_col = st.selectbox("分类变量 (X轴)", df.columns)
        y_options = ["计数"] + list(df.select_dtypes(include=["int64", "float64"]).columns)
        y_col = st.selectbox("数值变量 (Y轴, 可选)", y_options, index=0)
        y = None if y_col == "计数" else y_col
        
        color_col = st.selectbox("颜色分组 (可选)", ["无"] + list(df.columns), index=0)
        color = None if color_col == "无" else color_col
        
        if st.button("生成柱状图"):
            with st.spinner("绘图中..."):
                fig = visualizer.plotly_bar(x_col, y, color=color)
                st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_line_plot(df, visualizer):
        """渲染折线图设置"""
        x_col = st.selectbox("X轴 (通常为时间或序列)", df.columns)
        y_col = st.selectbox("Y轴 (数值)", df.select_dtypes(include=["int64", "float64"]).columns)
        
        color_col = st.selectbox("颜色分组 (可选)", ["无"] + list(df.columns), index=0)
        color = None if color_col == "无" else color_col
        
        if st.button("生成折线图"):
            with st.spinner("绘图中..."):
                fig = visualizer.plotly_line(x_col, y_col, color=color)
                st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_pie_chart(df, visualizer):
        """渲染饼图设置"""
        col = st.selectbox("分类变量", df.select_dtypes(include=["object", "category"]).columns)
        
        if st.button("生成饼图"):
            with st.spinner("绘图中..."):
                fig = visualizer.plotly_pie(col)
                st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_histogram(df, visualizer):
        """渲染直方图设置"""
        col = st.selectbox("数值变量", df.select_dtypes(include=["int64", "float64"]).columns)
        bins = st.slider("分箱数", 5, 100, 30)
        
        if st.button("生成直方图"):
            with st.spinner("绘图中..."):
                fig = visualizer.plotly_histogram(col, bins=bins)
                st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_boxplot(df, visualizer):
        """渲染箱线图设置"""
        y_col = st.selectbox("Y轴 (数值)", df.select_dtypes(include=["int64", "float64"]).columns)
        
        x_options = ["无"] + list(df.select_dtypes(include=["object", "category"]).columns)
        x_col = st.selectbox("X轴 (分组, 可选)", x_options, index=0)
        x = None if x_col == "无" else x_col
        
        if st.button("生成箱线图"):
            with st.spinner("绘图中..."):
                fig = visualizer.plotly_box(y_col, x=x)
                st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_heatmap(df, visualizer):
        """渲染热力图设置"""
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        selected_cols = st.multiselect("选择要包含的数值列", numeric_cols, default=list(numeric_cols)[:5])
        
        if selected_cols and len(selected_cols) > 1:
            if st.button("生成热力图"):
                with st.spinner("绘图中..."):
                    fig = visualizer.plotly_heatmap(selected_cols)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("热力图需要至少选择两个数值列")
    
    @staticmethod
    def _render_pairplot(df, visualizer):
        """渲染对图设置"""
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        selected_cols = st.multiselect("选择要包含的数值列", numeric_cols, 
                                     default=list(numeric_cols)[:min(4, len(numeric_cols))])
        
        color_options = ["无"] + list(df.select_dtypes(include=["object", "category"]).columns)
        color_col = st.selectbox("颜色分组 (可选)", color_options, index=0)
        color = None if color_col == "无" else color_col
        
        if selected_cols and len(selected_cols) > 1:
            if st.button("生成对图"):
                with st.spinner("绘图中... (可能需要较长时间)"):
                    fig = visualizer.plotly_scatter_matrix(selected_cols, color=color)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("对图需要至少选择两个数值列")
    
    @staticmethod
    def _render_violinplot(df, visualizer):
        """渲染小提琴图设置"""
        y_col = st.selectbox("Y轴 (数值)", df.select_dtypes(include=["int64", "float64"]).columns)
        
        x_options = list(df.select_dtypes(include=["object", "category"]).columns)
        if x_options:
            x_col = st.selectbox("X轴 (分组)", x_options, index=0)
            
            if st.button("生成小提琴图"):
                with st.spinner("绘图中..."):
                    fig = visualizer.plotly_violin(x_col, y_col)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("小提琴图需要至少一个分类列作为分组")
    
    @staticmethod
    def _render_map(df, visualizer):
        """渲染地图设置"""
        # 检查是否有经纬度列
        lat_options = ["无"] + [col for col in df.columns if "lat" in col.lower()]
        lon_options = ["无"] + [col for col in df.columns if "lon" in col.lower()]
        
        lat_col = st.selectbox("纬度列", lat_options, 
                              index=min(1, len(lat_options)-1) if len(lat_options) > 1 else 0)
        lon_col = st.selectbox("经度列", lon_options,
                              index=min(1, len(lon_options)-1) if len(lon_options) > 1 else 0)
        
        if lat_col != "无" and lon_col != "无":
            color_options = ["无"] + list(df.columns)
            color_col = st.selectbox("颜色变量 (可选)", color_options, index=0)
            color = None if color_col == "无" else color_col
            
            size_options = ["无"] + list(df.select_dtypes(include=["int64", "float64"]).columns)
            size_col = st.selectbox("大小变量 (可选)", size_options, index=0)
            size = None if size_col == "无" else size_col
            
            if st.button("生成地图"):
                with st.spinner("绘图中..."):
                    fig = visualizer.plotly_map(lat_col, lon_col, color=color, size=size)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("地图需要指定经纬度列。如果您的数据中没有经纬度信息，请尝试其他图表类型。")
    
    @staticmethod
    def render_machine_learning_tab(df: pd.DataFrame, ml_model: Optional[MLModel] = None):
        """渲染机器学习选项卡
        
        Args:
            df: 数据框
            ml_model: 机器学习模型对象
        """
        st.header("机器学习")
        
        # 模型类型选择
        model_task = st.radio("选择任务类型", ["预测分析", "聚类分析"])
        
        if model_task == "预测分析":
            UIComponents._render_prediction_section(df, ml_model)
        else:
            UIComponents._render_clustering_section(df, ml_model)
    
    @staticmethod
    def _render_prediction_section(df, ml_model):
        """渲染预测分析部分"""
        # 特征和目标选择
        st.subheader("模型配置")
        
        # 选择目标变量
        target_column = st.selectbox("选择目标变量", df.columns)
        
        # 检测特征列的数据类型
        categorical_columns = list(df.select_dtypes(include=["object", "category"]).columns)
        
        # 选择特征
        feature_options = [col for col in df.columns if col != target_column]
        feature_columns = st.multiselect("选择特征列", feature_options, default=feature_options)
        
        # 显示特征类型警告
        categorical_features = [col for col in feature_columns if col in categorical_columns]
        if categorical_features:
            st.warning(f"注意：您选择了以下分类特征: {', '.join(categorical_features)}。这些特征需要编码才能用于模型训练。")
            
            # 提供分类变量处理选项
            encoding_method = st.selectbox(
                "分类变量编码方法", 
                ["自动编码", "无(使用前请先在数据处理选项卡中手动编码)"], 
                index=0
            )

        # 数据预处理选项
        with st.expander("数据预处理选项"):
            test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
            scale_method = st.selectbox("特征缩放", ["None", "standard", "minmax"], index=0)
            scale_method = None if scale_method == "None" else scale_method
        
        # 模型选择
        st.subheader("模型选择")
        
        # 根据目标变量类型推断问题类型
        if df[target_column].dtype in ['int64', 'float64'] and df[target_column].nunique() > 10:
            problem_type = "回归问题"
            model_options = {
                "回归问题": ["自动选择最佳模型", "线性回归", "随机森林回归", "梯度提升回归", "支持向量回归", "K近邻回归", "XGBoost回归", "LightGBM回归"]
            }
            model_mapping = {
                "线性回归": "linear",
                "随机森林回归": "random_forest",
                "梯度提升回归": "gradient_boosting",
                "支持向量回归": "svr",
                "K近邻回归": "knn",
                "XGBoost回归": "xgboost",
                "LightGBM回归": "lightgbm"
            }
        else:
            problem_type = "分类问题"
            model_options = {
                "分类问题": ["自动选择最佳模型", "逻辑回归", "随机森林分类", "梯度提升分类", "支持向量机", "K近邻分类", "XGBoost分类", "LightGBM分类"]
            }
            model_mapping = {
                "逻辑回归": "logistic",
                "随机森林分类": "random_forest",
                "梯度提升分类": "gradient_boosting",
                "支持向量机": "svc",
                "K近邻分类": "knn",
                "XGBoost分类": "xgboost",
                "LightGBM分类": "lightgbm"
            }
        
        st.write(f"检测到的问题类型: **{problem_type}**")
        model_type = st.selectbox("选择算法", model_options[problem_type])
        
        # 添加模型超参数设置
        with st.expander("高级模型参数", expanded=False):
            st.info("这里可以添加模型的超参数设置")
            # 此处可以添加各模型的超参数设置
        
        # 训练模型
        if st.button("训练模型", use_container_width=True):
            if not feature_columns:
                st.error("请至少选择一个特征列")
            else:
                with st.spinner("正在训练模型..."):
                    # 准备数据
                    train_df = df.copy()
                    
                    # 处理分类变量
                    if categorical_features and 'encoding_method' in locals() and encoding_method == "自动编码":
                        with st.spinner("正在对分类特征进行编码..."):
                            # 对目标列进行编码(如果是分类)
                            if train_df[target_column].dtype in ['object', 'category']:
                                st.info(f"目标变量 '{target_column}' 为分类型，进行自动标签编码")
                                from sklearn.preprocessing import LabelEncoder
                                le = LabelEncoder()
                                train_df[target_column] = le.fit_transform(train_df[target_column])
                            
                            # 对特征列进行编码
                            for col in categorical_features:
                                # 对分类特征进行独热编码
                                st.info(f"特征 '{col}' 为分类型，进行独热编码")
                                # 获取唯一值数量
                                n_values = train_df[col].nunique()
                                if n_values > 10:
                                    st.warning(f"特征 '{col}' 具有 {n_values} 个唯一值，独热编码可能导致维度过大")
                                
                                # 使用pandas的get_dummies进行独热编码
                                dummies = pd.get_dummies(train_df[col], prefix=col, drop_first=False)
                                train_df = pd.concat([train_df.drop(col, axis=1), dummies], axis=1)
                            
                            # 更新特征列
                            feature_columns = [col for col in train_df.columns if col != target_column]
                            st.success(f"分类特征编码完成，编码后特征数量: {len(feature_columns)}")
                    
                    # 创建模型对象
                    if ml_model is None:
                        ml_model = MLModel(train_df, target_column=target_column)
                    
                    try:
                        # 数据预处理
                        ml_model.preprocess_data(
                            feature_columns=feature_columns,
                            test_size=test_size,
                            scale_method=scale_method
                        )
                        
                        # 训练模型
                        if model_type == "自动选择最佳模型":
                            metrics = ml_model.auto_train()
                            st.success(f"已自动选择最佳模型: {metrics.get('model_type', '未知')}")
                        else:
                            # 判断是分类还是回归
                            is_classification = ml_model._is_classification()
                            
                            if is_classification:
                                if model_type in model_options["分类问题"]:
                                    metrics = ml_model.train_classification_model(model_mapping[model_type])
                                    st.success(f"分类模型训练完成: {model_type}")
                                else:
                                    st.error(f"数据适合分类模型，但选择了回归模型: {model_type}")
                                    return
                            else:
                                if model_type in model_options["回归问题"]:
                                    metrics = ml_model.train_regression_model(model_mapping[model_type])
                                    st.success(f"回归模型训练完成: {model_type}")
                                else:
                                    st.error(f"数据适合回归模型，但选择了分类模型: {model_type}")
                                    return
                        
                        # 显示模型性能指标
                        st.subheader("模型性能")
                        metrics_df = pd.DataFrame({
                            "指标": list(metrics.keys()),
                            "值": list(metrics.values())
                        })
                        # 仅显示数值型指标
                        metrics_to_show = ["mse", "rmse", "r2", "cv_rmse", "accuracy", "precision", "recall", "f1", "cv_accuracy"]
                        metrics_df = metrics_df[metrics_df["指标"].isin(metrics_to_show)]
                        if not metrics_df.empty:
                            st.dataframe(metrics_df, use_container_width=True)
                        
                        # 显示模型解释
                        st.subheader("模型解释")
                        with st.spinner("正在生成模型解释..."):
                            # 特征重要性
                            if "feature_importance" in metrics and metrics["feature_importance"]:
                                importance_df = pd.DataFrame({
                                    "特征": list(metrics["feature_importance"].keys()),
                                    "重要性": list(metrics["feature_importance"].values())
                                }).sort_values("重要性", ascending=False)
                                
                                st.bar_chart(importance_df.set_index("特征"))
                            
                            # 生成SHAP值解释
                            try:
                                shap_explanation = ml_model.generate_shap_explanation()
                                if shap_explanation is not None:
                                    st.success("成功生成SHAP值解释")
                                    
                                    # 显示SHAP图表
                                    st.subheader("SHAP值分析 - 特征重要性")
                                    from src.visualizer import Visualizer
                                    visualizer = Visualizer(train_df)
                                    shap_fig = visualizer.plot_shap_values(ml_model.model, shap_explanation["data"])
                                    if shap_fig is not None:
                                        st.pyplot(shap_fig)
                            except Exception as e:
                                st.warning(f"SHAP值解释生成失败: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"模型训练失败: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
    
    @staticmethod
    def _render_clustering_section(df, ml_model):
        """渲染聚类分析部分"""
        st.subheader("聚类分析")
        st.info("聚类分析功能正在开发中...")
    
    @staticmethod
    def render_data_report_tab(df: pd.DataFrame, visualizer: Visualizer):
        """渲染数据报告选项卡
        
        Args:
            df: 数据框
            visualizer: 可视化器对象
        """
        st.header("数据报告")
        
        # 报告类型选择
        report_type = st.radio("报告类型", ["基础数据报告", "高级分析报告"])
        
        if report_type == "基础数据报告":
            # 基础数据报告
            st.subheader("数据概要")
            st.write(f"数据集包含 **{df.shape[0]}** 行和 **{df.shape[1]}** 列。")
            
            # 数据完整性
            st.subheader("1. 数据完整性")
            missing_data = df.isnull().sum()
            missing_df = pd.DataFrame({
                '列名': missing_data.index,
                '缺失值数': missing_data.values,
                '缺失值比例': (missing_data / len(df) * 100).round(2)
            })
            if missing_data.sum() > 0:
                st.write("数据集中存在缺失值：")
                st.dataframe(missing_df[missing_df['缺失值数'] > 0], use_container_width=True)
            else:
                st.write("数据集中没有缺失值，数据完整度良好。")
            
            # 数据类型分布
            st.subheader("2. 数据类型分布")
            dtype_counts = df.dtypes.value_counts().reset_index()
            dtype_counts.columns = ['数据类型', '列数']
            st.dataframe(dtype_counts, use_container_width=True)
            
            # 生成报告
            if st.button("生成完整报告", use_container_width=True):
                with st.spinner("正在生成报告..."):
                    # 此处添加更详细的报告生成逻辑
                    st.success("报告生成完成！")
                    # 提供下载链接或直接显示更多内容
        
        else:
            # 高级分析报告
            st.subheader("高级分析选项")
            
            analysis_options = st.multiselect(
                "选择要包含的分析",
                ["数据分布分析", "相关性分析", "异常值检测", "时间序列分析", "文本分析"],
                default=["数据分布分析", "相关性分析"]
            )
            
            if st.button("生成高级分析报告", use_container_width=True):
                with st.spinner("正在进行高级分析..."):
                    # 数据分布分析
                    if "数据分布分析" in analysis_options:
                        st.subheader("数据分布分析")
                        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
                        for col in numeric_cols[:3]:  # 限制显示前3个数值列
                            st.write(f"**{col}** 的分布")
                            fig = visualizer.plotly_histogram(col)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # 相关性分析
                    if "相关性分析" in analysis_options:
                        st.subheader("相关性分析")
                        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
                        if len(numeric_cols) > 1:
                            fig = visualizer.plotly_heatmap(numeric_cols)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # 其他分析选项可以在这里实现
                    
                    st.success("高级分析报告生成完成！") 