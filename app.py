import streamlit as st
from src.dashboard import Dashboard
import time
import json
import os

# 多语言支持
if 'language' not in st.session_state:
    st.session_state.language = 'zh'  # 默认语言：中文

# 加载语言文件
def load_language_data():
    try:
        # 尝试从语言文件加载
        lang_file_path = os.path.join('src', 'languages.json')
        if os.path.exists(lang_file_path):
            with open(lang_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 使用内置的语言数据
            return {
                "zh": {
                    "title": "数据可视化与分析工具",
                    "main_header": "📊 自动化数据可视化与分析工具",
                    "sub_header": "快速导入、分析、可视化数据并构建机器学习模型",
                    "init_msg": "正在初始化应用程序...",
                    "theme_setting": "系统设置",
                    "theme_select": "选择主题色",
                    "themes": {
                        "blue": "蓝色主题",
                        "green": "绿色主题",
                        "purple": "紫色主题",
                        "red": "红色主题",
                        "dark": "暗色主题"
                    },
                    "perf_setting": "性能设置",
                    "enable_cache": "启用数据缓存",
                    "cache_help": "启用后可加快数据加载速度，但会占用更多内存",
                    "data_setting": "数据设置",
                    "max_upload": "最大上传文件大小 (MB)",
                    "max_upload_help": "设置允许上传的最大文件大小",
                    "chart_setting": "图表设置",
                    "default_chart": "默认图表类型",
                    "chart_types": ["自动推荐", "柱状图", "折线图", "饼图", "散点图", "热力图"],
                    "report_setting": "导出报告设置",
                    "show_code": "在报告中显示代码",
                    "show_code_help": "启用后，导出的报告将包含生成图表的代码",
                    "lang_setting": "界面语言"
                },
                "en": {
                    "title": "Data Visualization & Analysis Tool",
                    "main_header": "📊 Automated Data Visualization & Analysis Tool",
                    "sub_header": "Quickly import, analyze, visualize data and build machine learning models",
                    "init_msg": "Initializing application...",
                    "theme_setting": "System Settings",
                    "theme_select": "Select Theme",
                    "themes": {
                        "blue": "Blue Theme",
                        "green": "Green Theme",
                        "purple": "Purple Theme",
                        "red": "Red Theme",
                        "dark": "Dark Theme"
                    },
                    "perf_setting": "Performance Settings",
                    "enable_cache": "Enable Data Cache",
                    "cache_help": "Speeds up data loading but uses more memory",
                    "data_setting": "Data Settings",
                    "max_upload": "Max Upload Size (MB)",
                    "max_upload_help": "Set maximum allowed file upload size",
                    "chart_setting": "Chart Settings",
                    "default_chart": "Default Chart Type",
                    "chart_types": ["Auto Recommend", "Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Heatmap"],
                    "report_setting": "Report Export Settings",
                    "show_code": "Show Code in Reports",
                    "show_code_help": "When enabled, exported reports will include chart generation code",
                    "lang_setting": "Interface Language"
                }
            }
    except Exception as e:
        st.error(f"Error loading language data: {str(e)}")
        # 返回一个基本的语言包作为后备
        return {"zh": {"title": "数据可视化工具"}, "en": {"title": "Data Visualization Tool"}}

# 获取语言数据
lang_data = load_language_data()
current_lang = lang_data[st.session_state.language]

# 设置页面配置
st.set_page_config(
    page_title=current_lang["title"],
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
    with st.expander(f"⚙️ {current_lang['theme_setting']}"):
        selected_theme = st.selectbox(
            current_lang["theme_select"],
            options=list(themes.keys()),
            format_func=lambda x: current_lang["themes"][x],
            index=list(themes.keys()).index(st.session_state.theme)
        )
        
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()
        
        # 性能设置
        st.subheader(current_lang["perf_setting"])
        cache_option = st.checkbox(current_lang["enable_cache"], value=True, 
                                  help=current_lang["cache_help"])
        
        # 数据设置
        st.subheader(current_lang["data_setting"])
        max_upload_size = st.slider(current_lang["max_upload"], 10, 200, 50, 
                                   help=current_lang["max_upload_help"])
        
        # 图表设置
        st.subheader(current_lang["chart_setting"])
        default_chart_type = st.selectbox(
            current_lang["default_chart"],
            current_lang["chart_types"]
        )
        
        # 导出报告设置
        show_code = st.checkbox(current_lang["show_code"], value=False,
                              help=current_lang["show_code_help"])
        
        # 语言设置
        language_options = {"zh": "简体中文", "en": "English"}
        selected_language = st.selectbox(
            current_lang["lang_setting"], 
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=list(language_options.keys()).index(st.session_state.language)
        )
        
        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            st.rerun()

# 应用标题
st.markdown(f'<div class="main-header">{current_lang["main_header"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{current_lang["sub_header"]}</div>', unsafe_allow_html=True)

# 初始化进度条
with st.spinner(current_lang["init_msg"]):
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