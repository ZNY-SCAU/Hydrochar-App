import streamlit as st
import pandas as pd
from logic import ModelBackend

# ================= 1. 网页配置 =================
st.set_page_config(page_title="Hydrochar Optimization", layout="wide")

# 注入 CSS (SCI 风格 + 紧凑布局)
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Times New Roman', serif; }
    
    /* 顶部留白 */
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    
    /* 标题微调 */
    h1 { font-size: 1.6rem; color: #1A5276; margin-bottom: 0px; }
    h4 { font-size: 1.05rem; color: #2C3E50; border-bottom: 2px solid #eee; margin-bottom: 8px; padding-bottom: 3px; font-weight: bold; }
    
    /* 紧凑间距 */
    div[data-testid="stVerticalBlock"] > div { gap: 0.2rem; }
    .stNumberInput, .stSelectbox { margin-bottom: 0px; }
    /* 按钮样式微调 */
    .stButton button { padding: 0rem 0.5rem; line-height: 1.2; min-height: 36px; }
    
    /* 字体与颜色 */
    .caption-text { font-size: 0.8em; color: #7F8C8D; }
    .result-text { font-weight: bold; color: #27AE60; font-size: 0.9em; }
    .lock-text { color: #95A5A6; font-style: italic; font-size: 0.85em; text-align: right;}
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ================= 2. 初始化模型 =================
if 'backend' not in st.session_state:
    st.session_state.backend = ModelBackend()
    success, msg = st.session_state.backend.load_model()
    if not success:
        st.error(f"Failed to load model: {msg}")
        st.stop()

# 默认值
USER_DEFAULTS = {
    'H(%)': 6.08, 'N(%)': 0.98, 'S(%)': 0.09, '(O+N)/C': 1.106, 'H/C': 0.136,
    'hydrothermal-T(℃)': 230.0, 'hydrothermal-time(h)': 0.5, 'hydrothermal-SLR(g/ml)': 0.167,
    'activation-SLR(g/L)': 0.0, 'activator-concentration(mol/L)': 0.0, 'activation-time(h)': 0.0,
    'adsorption-SLR(g/L)': 10.0, 'RPM(r/min)': 200.0, 'adsorption-time(h)': 6.23,
    'pH': 6.98, 'initial-NH4+-N(mg/L)': 1323.94, 'adsorption-T(℃)': 25.0,
    'C(%)': 44.56, 'O(%)': 48.29
}

#
