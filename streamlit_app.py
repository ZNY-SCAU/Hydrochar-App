import streamlit as st
import pandas as pd
from logic import ModelBackend

# ================= 1. 网页配置 =================
st.set_page_config(page_title="Hydrochar Optimization", layout="wide")

# 注入紧凑版 CSS (减少留白，Times New Roman 字体)
st.markdown("""
<style>
    /* 全局字体 */
    html, body, [class*="css"] { font-family: 'Times New Roman', serif; }
    
    /* 极致紧凑的标题和留白 */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    h1 { font-size: 1.8rem; margin-bottom: 0.5rem; color: #1A5276; }
    h3 { font-size: 1.1rem; border-bottom: 1px solid #ccc; padding-bottom: 0.2rem; margin-top: 0.5rem; color: #2C3E50;}
    
    /* 输入框紧凑化 */
    .stNumberInput, .stSelectbox { margin-bottom: -15px; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
    
    /* 状态文字 */
    .success-text { color: #27AE60; font-weight: bold; font-size: 0.9em; }
    .lock-text { color: #95A5A6; font-style: italic; font-size: 0.8em; }
    
    /* 隐藏多余元素 */
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

# User Defaults
USER_DEFAULTS = {
    'H(%)': 6.08, 'N(%)': 0.98, 'S(%)': 0.09, '(O+N)/C': 1.106, 'H/C': 0.136,
    'hydrothermal-T(℃)': 230.0, 'hydrothermal-time(h)': 0.5, 'hydrothermal-SLR(g/ml)': 0.167,
    'activation-SLR(g/L)': 0.0, 'activator-concentration(mol/L)': 0.0, 'activation-time(h)': 0.0,
    'adsorption-SLR(g/L)': 10.0, 'RPM(r/min)': 200.0, 'adsorption-time(h)': 6.23,
    'pH': 6.98, 'initial-NH4+-N(mg/L)': 1323.94, 'adsorption-T(℃)': 25.0,
    'C(%)': 44.56, 'O(%)': 48.29
}

# 初始化参数
if 'params' not in st.session_state:
    st.session_state.params = {}
    for feat in st.session_state.backend.ui_numeric_cols:
        # 🔥【过滤】只初始化模型真正用到的特征
        if feat in st.session_state.backend.model_features:
            val = USER_DEFAULTS.get(feat, 0.0)
            st.session_state.params[feat] = val

if 'results' not in st.session_state:
    st.session_state.results = {}

# ================= 3. 核心逻辑：活化联动 =================

def check_activation_logic():
    """活化参数归零 -> 方法重置为'0'"""
    slr = st.session_state.get('in_activation-SLR(g/L)', 0.0)
    conc = st.session_state.get('in_activator-concentration(mol/L)', 0.0)
    time = st.session_state.get('in_activation-time(h)', 0.0)
    current_method = st.session_state.get('activation-method', '')
    
    if (slr <= 0.001 or conc <= 0.001 or time <= 0.001):
        opts = st.session_state.backend.cat_options.get('activation-method', [])
        target_opt = '0'
        for opt in opts:
            if str(opt) == '0' or '基准' in str(opt) or 'Base' in str(opt):
                target_opt = opt; break
        
        if str(current_method) != str(target_opt):
            st.session_state['activation-method'] = target_opt
            st.session_state.params['activation-SLR(g/L)'] = 0.0
            st.session_state.params['activator-concentration(mol/L)'] = 0.0
            st.session_state.params['activation-time(h)'] = 0.0
            st.session_state.params['activation-T(℃)'] = 0.0

def is_activation_locked():
    """方法为'0' -> 锁定活化参数"""
    method = str(st.session_state.get('activation-method', ''))
    if method == '0' or '(基准)' in method or method == '' or 'Base' in method:
        return True
    return False

# ================= 4. 界面布局 (紧凑版) =================

st.title("Hydrochar Process Optimization System")

# --- 顶部区域：实验条件 & 优化目标 (并排显示) ---
top_c1, top_c2 = st.columns([1, 1])

with top_c1:
    st.markdown("### 1. Conditions")
    if st.session_state.backend.ui_cat_cols:
        # 使用更紧凑的列
        c_cats = st.columns(2) 
        for i, cat in enumerate(st.session_state.backend.ui_cat_cols):
            opts = st.session_state.backend.cat_options.get(cat, [])
            c_cats[i % 2].selectbox(cat, opts, key=cat)

with top_c2:
    st.markdown("### 2. Targets")
    t_c1, t_c2 = st.columns(2)
    with t_c1:
        use_ads = st.checkbox("Ads. (mg/g)")
        target_ads = st.number_input("Tgt Ads", disabled=not use_ads, label_visibility="collapsed")
    with t_c2:
        use_rem = st.checkbox("Rem. Rate (%)")
        target_rem = st.number_input("Tgt Rem", disabled=not use_rem, label_visibility="collapsed")

# --- 中部区域：工艺参数 (使用 Tabs 选项卡节省空间) ---
st.markdown("### 3. Parameters")

structure_groups = {
    'Raw Material': ['H(%)', 'N(%)', 'S(%)', '(O+N)/C', 'H/C', 'C(%)', 'O(%)'],
    'Hydrothermal': ['hydrothermal-T(℃)', 'hydrothermal-time(h)', 'hydrothermal-SLR(g/ml)'],
    'Activation': ['activation-SLR(g/L)', 'activator-concentration(mol/L)', 'activation-time(h)', 'activation-T(℃)'],
    'Adsorption': ['adsorption-SLR(g/L)', 'RPM(r/min)', 'adsorption-time(h)', 'pH', 'initial-NH4+-N(mg/L)', 'adsorption-T(℃)']
}
activation_feats = structure_groups['Activation']

# 创建 4 个选项卡
tabs = st.tabs(list(structure_groups.keys()))

for tab, (g_name, g_feats) in zip(tabs, structure_groups.items()):
    with tab:
        # 🔥 严格过滤：只显示模型真正用到的特征
        valid_feats = [
            f for f in g_feats 
            if f in st.session_state.backend.ui_numeric_cols 
            and f in st.session_state.backend.model_features
        ]
        
        if not valid_feats:
            st.info("No parameters used in this group.")
        else:
            # 内部使用 3 列布局，更紧凑
            cols = st.columns(3)
            for i, feat in enumerate(valid_feats):
                with cols[i % 3]:
                    stat = st.session_state.backend.stats.get(feat, {'min':0, 'max':100})
                    
                    # 标题行：Checkbox + 范围
                    sub_c1, sub_c2 = st.columns([2, 1])
                    is_opt = sub_c1.checkbox(feat, key=f"chk_{feat}")
                    sub_c2.caption(f"{stat['min']:.0f}-{stat['max']:.0f}")
                    
                    # 锁定逻辑
                    should_lock = is_opt
                    if feat in activation_feats and is_activation_locked():
                        should_lock = True
                        display_val = 0.0
                    else:
                        display_val = st.session_state.params.get(feat, 0.0)

                    # 输入框
                    new_val = st.number_input(
                        label=feat,
                        value=float(display_val),
                        label_visibility="collapsed",
                        disabled=should_lock,
                        key=f"in_{feat}",
                        on_change=check_activation_logic if feat in activation_feats else None
                    )
                    
                    # 状态回显
                    if should_lock and feat in activation_feats and is_activation_locked():
                        st.session_state.params[feat] = 0.0
                        st.markdown("<span class='lock-text'>🔒 Method=0</span>", unsafe_allow_html=True)
                    elif not should_lock:
                        st.session_state.params[feat] = new_val
                        if feat in st.session_state.results:
                            st.markdown(f"<span class='success-text'>✅ {st.session_state.results[feat]:.3f}</span>", unsafe_allow_html=True)

# --- 底部区域：运行 & 结果 ---
st.markdown("---")
b_col1, b_col2 = st.columns([1, 3])

with b_col1:
    if st.button("🚀 RUN OPTIMIZATION", type="primary", use_container_width=True):
        inputs = {}
        for cat in st.session_state.backend.ui_cat_cols:
            inputs[cat] = st.session_state[cat]
        
        for feat in st.session_state.backend.ui_numeric_cols:
            if feat not in st.session_state.backend.model_features: continue
            
            val = st.session_state.params.get(feat, 0.0)
            if feat in activation_feats and is_activation_locked(): val = 0.0
            is_predict = st.session_state.get(f"chk_{feat}", False)
            inputs[feat] = {'value': val, 'is_predict': is_predict}
        
        targets = {
            'ads': {'value': target_ads, 'is_constraint': use_ads},
            'rem': {'value': target_rem, 'is_constraint': use_rem}
        }
        
        with st.spinner("Calculating..."):
            res = st.session_state.backend.run_task(inputs, targets)
        
        if res['success']:
            st.session_state.pred_ads = res['ads']
            st.session_state.pred_rem = res['rem']
            st.session_state.verify = res.get('verification', {})
            st.session_state.results = {}
            if res['mode'] == 'reverse':
                for k, v in res['optimized_params'].items():
                    st.session_state.results[k] = v
            st.toast("Calculation Completed!", icon="✅")
            st.rerun()
        else:
            st.error(res['error'])

with b_col2:
    # 结果显示区
    if 'pred_ads' in st.session_state:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted Ads.", f"{st.session_state.pred_ads:.2f}", "mg/g")
        r2.metric("Predicted Rem.", f"{st.session_state.pred_rem:.2f}", "%")
        
        # 验证信息
        v = st.session_state.verify
        mb_err = v.get('mass_balance_error', 0)
        r3.metric("Mass Balance Err", f"{mb_err:.2f}%", delta_color="inverse" if mb_err < 5 else "normal")
        r4.metric("Elem. Sum", v.get('elemental_msg', 'N/A'))
