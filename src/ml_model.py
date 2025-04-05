import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, LeaveOneOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


class MLModel:
    """机器学习模型类，提供预测和聚类功能"""
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        """初始化机器学习模型
        
        Args:
            df: 数据源DataFrame对象
            target_column: 目标变量列名
        """
        self.df = df.copy()
        self.target_column = target_column
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns = None
        self.scaler = None
        self.label_encoder = None
        self.metrics = {}
    
    def preprocess_data(self, feature_columns: Optional[List[str]] = None, test_size: float = 0.2, 
                       random_state: int = 42, scale_method: Optional[str] = None) -> None:
        """数据预处理
        
        Args:
            feature_columns: 特征列名列表，如果为None则使用除目标列外的所有数值列
            test_size: 测试集比例
            random_state: 随机种子
            scale_method: 缩放方法，可选'standard'、'minmax'或None
        """
        if self.target_column is None:
            raise ValueError("目标列名不能为空")
        
        # 筛选特征列
        if feature_columns is None:
            # 默认使用所有数值列作为特征
            numeric_cols = list(self.df.select_dtypes(include=['int64', 'float64']).columns)
            self.feature_columns = [col for col in numeric_cols if col != self.target_column]
        else:
            self.feature_columns = feature_columns
        
        # 分割特征和目标变量
        X = self.df[self.feature_columns]
        y = self.df[self.target_column]
        
        # 对目标变量进行编码(如果是分类问题)
        if y.dtype == 'object' or y.dtype.name == 'category':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        
        # 分割训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 特征缩放
        if scale_method == 'standard':
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        elif scale_method == 'minmax':
            self.scaler = MinMaxScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
    
    def _is_classification(self) -> bool:
        """判断是否为分类问题
        
        Returns:
            是否为分类问题
        """
        if self.label_encoder is not None:
            return True
        
        if isinstance(self.y_train, np.ndarray):
            unique_values = np.unique(self.y_train)
        else:
            unique_values = self.y_train.unique()
        
        # 如果唯一值数量小于总样本的10%，则认为是分类问题
        return len(unique_values) < len(self.y_train) * 0.1
    
    def _get_cv(self) -> Union[int, object]:
        """根据训练集大小确定交叉验证折数
        
        Returns:
            int或LeaveOneOut对象: 交叉验证折数或留一法交叉验证对象
        """
        n_samples = len(self.y_train)
        
        if n_samples < 5:
            # 样本数小于5时使用留一法交叉验证
            print(f"训练集样本量过小 ({n_samples}), 使用留一法交叉验证")
            return LeaveOneOut()
        elif n_samples < 10:
            # 样本数小于10时使用最多3折交叉验证
            cv = min(3, n_samples)
            print(f"训练集样本量较小 ({n_samples}), 使用{cv}折交叉验证")
            return cv
        else:
            # 样本数大于等于10时使用5折交叉验证
            cv = min(5, n_samples // 2)
            return cv
    
    def train_regression_model(self, model_type: str = 'linear', 
                              params: Optional[Dict] = None) -> Dict[str, Any]:
        """训练回归模型
        
        Args:
            model_type: 模型类型，可选'linear'、'random_forest'、'gradient_boosting'、'svr'、'knn'
            params: 模型参数
            
        Returns:
            训练结果指标
        """
        if self._is_classification():
            raise ValueError("数据适合分类问题，请使用train_classification_model方法")
        
        if params is None:
            params = {}
        
        # 创建模型
        if model_type == 'linear':
            self.model = LinearRegression(**params)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(**params)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**params)
        elif model_type == 'svr':
            self.model = SVR(**params)
        elif model_type == 'knn':
            self.model = KNeighborsRegressor(**params)
        else:
            raise ValueError(f"不支持的回归模型类型: {model_type}")
        
        # 训练模型
        self.model.fit(self.X_train, self.y_train)
        
        # 预测
        y_pred = self.model.predict(self.X_test)
        
        # 评估模型
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                   cv=self._get_cv(), scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        # 特征重要性(如果模型支持)
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        # 保存评估结果
        self.metrics = {
            'model_type': model_type,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'cv_rmse': cv_rmse,
            'feature_importance': feature_importance
        }
        
        return self.metrics
    
    def train_classification_model(self, model_type: str = 'logistic', 
                                 params: Optional[Dict] = None) -> Dict[str, Any]:
        """训练分类模型
        
        Args:
            model_type: 模型类型，可选'logistic'、'random_forest'、'gradient_boosting'、'svc'、'knn'
            params: 模型参数
            
        Returns:
            训练结果指标
        """
        if not self._is_classification():
            raise ValueError("数据适合回归问题，请使用train_regression_model方法")
        
        if params is None:
            params = {}
        
        # 创建模型
        if model_type == 'logistic':
            self.model = LogisticRegression(**params)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(**params)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(**params)
        elif model_type == 'svc':
            self.model = SVC(**params, probability=True)
        elif model_type == 'knn':
            self.model = KNeighborsClassifier(**params)
        else:
            raise ValueError(f"不支持的分类模型类型: {model_type}")
        
        # 训练模型
        self.model.fit(self.X_train, self.y_train)
        
        # 预测
        y_pred = self.model.predict(self.X_test)
        
        # 评估模型
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=self._get_cv(), scoring='accuracy')
        cv_accuracy = cv_scores.mean()
        
        # 特征重要性(如果模型支持)
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        # 保存评估结果
        self.metrics = {
            'model_type': model_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'cv_accuracy': cv_accuracy,
            'feature_importance': feature_importance
        }
        
        return self.metrics
    
    def auto_train(self) -> Dict[str, Any]:
        """自动选择并训练模型
        
        Returns:
            训练结果指标
        """
        if self._is_classification():
            # 尝试不同的分类模型
            results = {}
            models = ['logistic', 'random_forest', 'gradient_boosting', 'knn']
            
            for model_type in models:
                try:
                    result = self.train_classification_model(model_type)
                    results[model_type] = result['accuracy']
                except Exception as e:
                    results[model_type] = 0
            
            # 选择最佳模型
            best_model = max(results, key=results.get)
            self.train_classification_model(best_model)
            return self.metrics
        else:
            # 尝试不同的回归模型
            results = {}
            models = ['linear', 'random_forest', 'gradient_boosting', 'knn']
            
            for model_type in models:
                try:
                    result = self.train_regression_model(model_type)
                    results[model_type] = result['r2']
                except Exception as e:
                    results[model_type] = 0
            
            # 选择最佳模型
            best_model = max(results, key=results.get)
            self.train_regression_model(best_model)
            return self.metrics
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """使用训练好的模型进行预测
        
        Args:
            data: 预测数据
            
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 转换数据格式
        if isinstance(data, pd.DataFrame):
            # 确保输入数据包含所有特征列
            missing_cols = set(self.feature_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"输入数据缺少特征列: {missing_cols}")
            
            data = data[self.feature_columns]
        
        # 特征缩放
        if self.scaler is not None:
            data = self.scaler.transform(data)
        
        # 预测
        predictions = self.model.predict(data)
        
        # 如果是分类问题且使用了标签编码，则转换回原始标签
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def perform_clustering(self, method: str = 'kmeans', n_clusters: int = 3, 
                         params: Optional[Dict] = None) -> Dict[str, Any]:
        """执行聚类分析
        
        Args:
            method: 聚类方法，可选'kmeans'、'dbscan'、'hierarchical'
            n_clusters: 聚类数量(kmeans和hierarchical需要)
            params: 其他参数
            
        Returns:
            聚类结果
        """
        if params is None:
            params = {}
        
        # 确保数据已加载
        if self.feature_columns is None:
            # 默认使用所有数值列
            self.feature_columns = list(self.df.select_dtypes(include=['int64', 'float64']).columns)
            
            if self.target_column in self.feature_columns:
                self.feature_columns.remove(self.target_column)
        
        # 获取特征数据
        data = self.df[self.feature_columns].values
        
        # 特征缩放
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # 执行聚类
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, **params)
            labels = model.fit_predict(scaled_data)
            centers = model.cluster_centers_
            inertia = model.inertia_
            score = silhouette_score(scaled_data, labels) if len(np.unique(labels)) > 1 else 0
            
            result = {
                'labels': labels,
                'centers': centers,
                'inertia': inertia,
                'silhouette_score': score
            }
        elif method == 'dbscan':
            model = DBSCAN(**params)
            labels = model.fit_predict(scaled_data)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            score = silhouette_score(scaled_data, labels) if n_clusters > 1 and len(np.unique(labels)) > 1 else 0
            
            result = {
                'labels': labels,
                'n_clusters': n_clusters,
                'silhouette_score': score
            }
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters, **params)
            labels = model.fit_predict(scaled_data)
            score = silhouette_score(scaled_data, labels) if len(np.unique(labels)) > 1 else 0
            
            result = {
                'labels': labels,
                'n_clusters': n_clusters,
                'silhouette_score': score
            }
        else:
            raise ValueError(f"不支持的聚类方法: {method}")
        
        # 将聚类结果添加到数据中
        self.df['cluster'] = labels
        
        return result
    
    def visualize_clusters(self, method: str = 'pca') -> go.Figure:
        """可视化聚类结果
        
        Args:
            method: 降维方法，可选'pca'或'original'(使用前两个特征)
            
        Returns:
            plotly图形对象
        """
        if 'cluster' not in self.df.columns:
            raise ValueError("请先执行聚类操作")
        
        data = self.df[self.feature_columns].values
        
        if method == 'pca':
            # 使用PCA降维到2D
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(data)
            x_label = 'PCA维度1'
            y_label = 'PCA维度2'
            explained_var = pca.explained_variance_ratio_
            title = f"聚类结果 (PCA降维, 解释方差: {explained_var[0]:.2f}, {explained_var[1]:.2f})"
        else:
            # 使用前两个特征
            if len(self.feature_columns) < 2:
                raise ValueError("特征数量不足2个，无法直接可视化")
            
            reduced_data = data[:, :2]
            x_label = self.feature_columns[0]
            y_label = self.feature_columns[1]
            title = f"聚类结果 (特征: {x_label}, {y_label})"
        
        # 创建可视化
        viz_df = pd.DataFrame({
            'x': reduced_data[:, 0],
            'y': reduced_data[:, 1],
            'cluster': self.df['cluster']
        })
        
        fig = px.scatter(
            viz_df, x='x', y='y', color='cluster',
            title=title,
            labels={'x': x_label, 'y': y_label}
        )
        
        return fig
    
    def plot_feature_importance(self) -> Optional[go.Figure]:
        """可视化特征重要性
        
        Returns:
            plotly图形对象或None(如果模型不支持特征重要性)
        """
        if self.metrics is None or 'feature_importance' not in self.metrics:
            return None
        
        feature_importance = self.metrics['feature_importance']
        if not feature_importance:
            return None
        
        # 特征重要性数据准备
        importance_df = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        }).sort_values('importance', ascending=False)
        
        # 创建条形图
        fig = px.bar(
            importance_df, x='importance', y='feature', orientation='h',
            title='特征重要性',
            labels={'importance': '重要性', 'feature': '特征'},
            color='importance'
        )
        
        return fig
    
    def plot_regression_results(self) -> Optional[go.Figure]:
        """可视化回归结果
        
        Returns:
            plotly图形对象或None(如果不是回归问题)
        """
        if self.model is None or self._is_classification():
            return None
        
        # 获取测试集预测结果
        y_pred = self.model.predict(self.X_test)
        
        # 创建散点图
        fig = go.Figure()
        
        # 添加散点图
        fig.add_trace(go.Scatter(
            x=self.y_test,
            y=y_pred,
            mode='markers',
            name='测试集预测',
            marker=dict(color='royalblue')
        ))
        
        # 添加对角线(理想情况)
        min_val = min(min(self.y_test), min(y_pred))
        max_val = max(max(self.y_test), max(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='理想预测',
            line=dict(color='red', dash='dash')
        ))
        
        # 设置图表标题和轴标签
        model_type = self.metrics.get('model_type', 'unknown')
        r2 = self.metrics.get('r2', 0)
        
        fig.update_layout(
            title=f'{model_type.capitalize()}回归模型结果 (R² = {r2:.4f})',
            xaxis_title='实际值',
            yaxis_title='预测值',
            legend=dict(x=0, y=1)
        )
        
        return fig
    
    def plot_confusion_matrix(self) -> Optional[go.Figure]:
        """绘制混淆矩阵
        
        Returns:
            go.Figure: 混淆矩阵图
        """
        if 'confusion_matrix' not in self.metrics or self.metrics['confusion_matrix'] is None:
            return None
        
        conf_matrix = self.metrics['confusion_matrix']
        
        # 获取标签
        if self.label_encoder is not None and hasattr(self.label_encoder, 'classes_'):
            labels = self.label_encoder.classes_
            # 确保标签长度与混淆矩阵的维度一致
            if len(labels) != conf_matrix.shape[0]:
                labels = [str(i) for i in range(conf_matrix.shape[0])]
        else:
            labels = [str(i) for i in range(conf_matrix.shape[0])]
        
        # 创建热力图
        try:
            fig = px.imshow(
                conf_matrix,
                labels=dict(x="预测类别", y="真实类别", color="数量"),
                x=labels,
                y=labels,
                color_continuous_scale='Blues',
                title='混淆矩阵'
            )
            
            # 在每个单元格上显示数值
            annotations = []
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    annotations.append(dict(
                        x=j,  # 使用索引而不是标签文本
                        y=i,  # 使用索引而不是标签文本
                        text=str(conf_matrix[i, j]),
                        showarrow=False,
                        font=dict(color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')
                    ))
            
            fig.update_layout(annotations=annotations)
            
            return fig
        except ValueError as e:
            print(f"创建混淆矩阵图时出错: {str(e)}")
            print(f"混淆矩阵形状: {conf_matrix.shape}，标签长度: {len(labels)}")
            
            # 使用更简单的备选方案 - 直接绘制无标签的热力图
            fig = px.imshow(
                conf_matrix,
                labels=dict(x="预测类别", y="真实类别", color="数量"),
                color_continuous_scale='Blues',
                title='混淆矩阵'
            )
            
            return fig 