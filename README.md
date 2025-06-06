# autodata-visualization-tool

这是一个强大的自动化数据可视化和分析工具，可以帮助用户快速导入、处理、可视化数据，并进行机器学习分析。

## 主要功能

- **数据导入与导出**：支持CSV、Excel、JSON等多种格式数据的导入和导出
- **数据处理**：缺失值处理、标准化、归一化、分类变量编码等数据预处理功能
- **数据可视化**：提供散点图、条形图、折线图、饼图、直方图、箱线图、热力图等多种可视化选项
- **机器学习分析**：
  - **预测分析**：支持回归和分类任务，包括线性回归、随机森林、梯度提升树等多种模型
  - **聚类分析**：支持K均值、DBSCAN、层次聚类等方法
  - **小样本数据集支持**：智能调整交叉验证策略，支持极小样本量的模型训练
- **数据库连接**：支持SQLite等数据库的连接和查询
- **数据报告**：自动生成数据分析报告，包括基本统计信息、相关性分析等

## 安装方法

1. 克隆仓库:
```bash
git clone https://github.com/你的用户名/autodata-visualization-tool.git
cd autodata-visualization-tool
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

## 使用方法

运行主应用程序:
```bash
streamlit run app.py
```

应用程序会在浏览器中自动打开，然后可以：
1. 上传数据文件或使用示例数据
2. 在不同选项卡中进行数据处理、可视化和分析
3. 根据需要训练机器学习模型和生成报告

## 项目结构

```
├── app.py                  # 主入口文件
├── requirements.txt        # 项目依赖
├── examples/               # 示例数据和文件
│   └── sample_database.db  # 示例SQLite数据库
└── src/                    # 源代码
    ├── dashboard.py        # 仪表盘界面实现
    ├── data_loader.py      # 数据加载模块
    ├── data_processor.py   # 数据处理模块
    ├── ml_model.py         # 机器学习模型
    └── visualizer.py       # 数据可视化模块
```

## 依赖库

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- sqlite3

## 特性

- 响应式布局，适合不同设备使用
- 易于扩展的模块化设计
- 智能处理小样本数据集的特殊算法
- 丰富的数据可视化选项
- 详细的模型评估指标和可视化结果

## 贡献指南

欢迎贡献代码、报告问题或提出新功能建议。请通过GitHub Issues或Pull Requests参与项目开发。
