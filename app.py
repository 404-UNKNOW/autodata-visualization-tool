import streamlit as st
from src.dashboard import Dashboard
import time

# 设置页面配置
st.set_page_config(
    page_title="数据可视化与分析工具",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置应用主题和全局配置
if 'theme' not in st.session_state:
    st.session_state.theme = 'blue'  # 默认主题

# 可选的主题颜色
themes = {
    'blue': {'primary': '#1E88E5', 'secondary': '#424242', 'background': '#F5F7FA'},
    'green': {'primary': '#4CAF50', 'secondary': '#2E7D32', 'background': '#F1F8E9'},
    'purple': {'primary': '#673AB7', 'secondary': '#512DA8', 'background': '#F3E5F5'},
    'red': {'primary': '#E53935', 'secondary': '#C62828', 'background': '#FFEBEE'},
    'dark': {'primary': '#607D8B', 'secondary': '#455A64', 'background': '#ECEFF1'}
}

# 获取当前主题颜色
current_theme = themes[st.session_state.theme]

# 自定义CSS样式
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        color: {current_theme['primary']};
        text-align: center;
        margin-bottom: 1rem;
    }}
    .sub-header {{
        font-size: 1.2rem;
        color: {current_theme['secondary']};
        text-align: center;
        margin-bottom: 2rem;
    }}
    .stButton>button {{
        background-color: {current_theme['primary']};
        color: white;
    }}
    .card {{
        border-radius: 5px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .metric-card {{
        text-align: center;
        background-color: {current_theme['background']};
        border-left: 5px solid {current_theme['primary']};
    }}
    .chart-container {{
        padding: 1rem;
        border-radius: 5px;
        background-color: white;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }}
    /* 自定义选项卡样式 */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        padding: 0 20px;
        font-weight: 500;
    }}
    /* 自定义滚动条 */
    ::-webkit-scrollbar {{
        width: 10px;
        background: #f0f2f6;
    }}
    ::-webkit-scrollbar-thumb {{
        background: {current_theme['primary']}; 
        border-radius: 5px;
    }}
</style>
""", unsafe_allow_html=True)

# 添加一个系统设置菜单到侧边栏底部
with st.sidebar:
    st.sidebar.divider()
    with st.expander("⚙️ 系统设置"):
        selected_theme = st.selectbox(
            "选择主题色",
            options=list(themes.keys()),
            format_func=lambda x: {
                'blue': '蓝色主题', 
                'green': '绿色主题', 
                'purple': '紫色主题',
                'red': '红色主题',
                'dark': '暗色主题'
            }[x],
            index=list(themes.keys()).index(st.session_state.theme)
        )
        
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()
        
        # 性能设置
        st.subheader("性能设置")
        cache_option = st.checkbox("启用数据缓存", value=True, 
                                  help="启用后可加快数据加载速度，但会占用更多内存")
        
        # 数据设置
        st.subheader("数据设置")
        max_upload_size = st.slider("最大上传文件大小 (MB)", 10, 200, 50, 
                                   help="设置允许上传的最大文件大小")
        
        # 图表设置
        st.subheader("图表设置")
        default_chart_type = st.selectbox(
            "默认图表类型",
            ["自动推荐", "柱状图", "折线图", "饼图", "散点图", "热力图"]
        )
        
        # 导出报告设置
        show_code = st.checkbox("在报告中显示代码", value=False,
                              help="启用后，导出的报告将包含生成图表的代码")
        
        # 语言设置
        language = st.selectbox("界面语言", ["简体中文", "English"], index=0)

# 应用标题
st.markdown('<div class="main-header">📊 自动化数据可视化与分析工具</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">快速导入、分析、可视化数据并构建机器学习模型</div>', unsafe_allow_html=True)

# 初始化进度条
with st.spinner("正在初始化应用程序..."):
    # 模拟加载过程
    progress_bar = st.progress(0)
    for i in range(100):
        # 更新进度条
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    progress_bar.empty()

# 启动仪表盘
dashboard = Dashboard()
dashboard.run() 