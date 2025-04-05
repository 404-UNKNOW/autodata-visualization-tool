import streamlit as st
from src.dashboard import Dashboard

# 创建并运行仪表盘
dashboard = Dashboard(title="自动化数据可视化工具")
dashboard.run() 