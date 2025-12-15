import streamlit as st
import pandas as pd
from logic import ModelBackend

# ================= 1. 网页配置与样式 =================
st.set_page_config(page_title="Hydrochar Optimization", layout="wide")

# 注入 CSS 以复刻 SCI 风格 (Times New Roman, 颜色等)
st.markdown("""
<style>
    /* 全局字体 */
    html, body, [class*="css"] {
        font-family: 'Times New Roman', serif;
    }
    /* 标题颜色 */
    h1 { color: #1A5276; font-weight: bold; }
    h3 { color: #2C3E50; font-weight: bold; border-bottom: 2px solid #E0E0E0; padding-bottom: 5px; }
    h4 { color: #5D6D7E; font-size: 1.1rem; }
    
    /* 输入框样式微调 */
    .stTextInput input { font-family: 'Times New Roman'; }
    
    /* 成功/警告文字颜色 */
    .success-text { color: #27AE60; font-weight: bold; }
    .warning-text { color: #E74C3C; font-weight: bold; }
    .lock-text { color: #95A5A6; font-style: italic; }
    
    /* 隐藏右上角的菜单，进一步保护代码 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ================= 2. 初始化后台与状态 =================
if 'backend' not in st.session_state:
    st.session_state.backend = ModelBackend()
    success, msg = st.session_state.backend.load_model()
    if not success:
        st.error(f"Failed to load model: {msg}")
        st.stop()

# 定义初始值 (User Defaults)
USER_DEFAULTS = {
    'H(%)': 6.08, 'N(%)': 0.98, 'S(%)': 0.09, '(O+N)/C': 1.106, 'H/C': 0.136,
    'hydrothermal-T(℃)': 230.0, 'hydrothermal-time(h)': 0.5, 'hydrothermal-SLR(g/ml)': 0.167,
    'activation-SLR(g/L)': 0.0, 'activator-concentration(mol/L)': 0.0, 'activation-time(h)': 0.0,
    'adsorption-SLR(g/L)': 10.0, 'RPM(r/min)': 200.0, 'adsorption-time(h)': 6.23,
    'pH': 6.98, 'initial-NH4+-N(mg/L)': 1323.94, 'adsorption-T(℃)': 25.0,
    'C(%)': 44.56, 'O(%)': 48.29
}

# 初始化 Session State
if 'params' not in st.session_state:
    st.session_state.params = {}
    for feat in st.session_state.backend.ui_numeric_cols:
        val = USER_DEFAULTS.get(feat, 0.0)
        st.session_state.params[feat] = val

if 'results' not in st.session_state:
    st.session_state.results = {}

# ================= 3. 核心联动逻辑函数 =================
def on_activation_change(changed_key):
    val = st.session_state.params[changed_key]
    if val <= 0.0001:
        st.session_state['activation-method'] = '0' 
        targets = ['activation-SLR(g/L)', 'activator-concentration(mol/L)', 'activation-time(h)', 'activation-T(℃)']
        for t in targets:
            st.session_state.params[t] = 0.0

def is_locked(feat):
    targets = ['activation-SLR(g/L)', 'activator-concentration(mol/L)', 'activation-time(h)', 'activation-T(℃)']
    if feat not in targets: return False
    method = str(st.session_state.get('activation-method', ''))
    if method == '0' or '(基准)' in method or method == '':
        return True
    return False

# ================= 4. 界面布局 =================

# --- Header ---
st.title("Hydrochar Process Prediction & Optimization System")
st.markdown("*Machine Learning Based Dual-Target Analysis*")

# --- 1. Experimental Conditions ---
st.markdown("### 1. Experimental Conditions")
cols_cat = st.columns(len(st.session_state.backend.ui_cat_cols))
for i, cat in enumerate(st.session_state.backend.ui_cat_cols):
    opts = st.session_state.backend.cat_options.get(cat, [])
    key = cat if cat != 'activation-method' else 'activation-method'
    st.session_state[key] = cols_cat[i].selectbox(cat, opts, key=key)

# --- 2. Process Parameters ---
st.markdown("### 2. Process Parameters")

groups = {
    'Raw Material': ['H(%)', 'N(%)', 'S(%)', '(O+N)/C', 'H/C', 'C(%)', 'O(%)'],
    'Hydrothermal': ['hydrothermal-T(℃)', 'hydrothermal-time(h)', 'hydrothermal-SLR(g/ml)'],
    'Activation': ['activation-SLR(g/L)', 'activator-concentration(mol/L)', 'activation-time(h)', 'activation-T(℃)'],
    'Adsorption': ['adsorption-SLR(g/L)', 'RPM(r/min)', 'adsorption-time(h)', 'pH', 'initial-NH4+-N(mg/L)', 'adsorption-T(℃)']
}

row1 = st.columns(2)
row2 = st.columns(2)
group_locations = [row1[0], row1[1], row2[0], row2[1]]

for (g_name, g_cols), loc in zip(groups.items(), group_locations):
    with loc:
        st.markdown(f"#### {g_name}")
        for feat in g_cols:
            if feat in st.session_state.backend.ui_numeric_cols:
                stat = st.session_state.backend.stats.get(feat, {'min':0, 'max':100})
                c1, c2 = st.columns([2, 1])
                
                is_opt = c1.checkbox(feat, key=f"chk_{feat}")
                disabled = is_opt or is_locked(feat)
                display_val = 0.0 if is_locked(feat) else st.session_state.params[feat]
                
                new_val = c1.number_input(
                    label="Value", 
                    value=float(display_val),
                    label_visibility="collapsed",
                    disabled=disabled,
                    key=f"in_{feat}",
                    on_change=on_activation_change if feat in groups['Activation'] else None,
                    args=(feat,) if feat in groups['Activation'] else None
                )
                
                if not disabled:
                    st.session_state.params[feat] = new_val

                with c2:
                    st.caption(f"[{stat['min']:.3f}-{stat['max']:.3f}]")
                    if feat in st.session_state.results:
                        st.markdown(f"<span class='success-text'>✅ {st.session_state.results[feat]:.4f}</span>", unsafe_allow_html=True)
                    if is_locked(feat):
                        st.markdown("<span class='lock-text'>🔒 0</span>", unsafe_allow_html=True)

# --- 3. Optimization Targets ---
st.markdown("### 3. Optimization Targets")
t1, t2 = st.columns(2)

with t1:
    use_ads = st.checkbox("Set Goal: Adsorption-NH₄⁺-N(mg/g)")
    target_ads = st.number_input("Ads Target", disabled=not use_ads, label_visibility="collapsed")
    if 'pred_ads' in st.session_state:
        st.markdown(f"→ Predicted: <span class='success-text'>{st.session_state.pred_ads:.2f}</span>", unsafe_allow_html=True)

with t2:
    use_rem = st.checkbox("Set Goal: Removal Rate (%)")
    target_rem = st.number_input("Rem Target", disabled=not use_rem, label_visibility="collapsed")
    if 'pred_rem' in st.session_state:
        st.markdown(f"→ Predicted: <span class='success-text'>{st.session_state.pred_rem:.2f}</span>", unsafe_allow_html=True)

# --- 4. Check ---
st.markdown("### 4. Check")
chk1, chk2 = st.columns(2)
if 'verify' in st.session_state:
    v = st.session_state.verify
    mb_err = v.get('mass_balance_error', 0)
    mb_color = "success-text" if mb_err < 5.0 else "warning-text"
    chk1.markdown(f"**Mass Balance Error:** <span class='{mb_color}'>{mb_err:.2f}%</span>", unsafe_allow_html=True)
    chk1.caption("Formula: Removal% ≈ (Ads × SLR × 100) / Initial_Conc")
    
    el_err = v.get('elemental_error', 0)
    el_msg = v.get('elemental_msg', 'N/A')
    el_color = "success-text" if el_err < 0.5 else "warning-text"
    chk2.markdown(f"**Elemental Sum:** <span class='{el_color}'>{el_msg}</span>", unsafe_allow_html=True)
    chk2.caption("Formula: Total = C% + H% + O% + N% + S% ≈ 100%")
else:
    chk1.markdown("Mass Balance: N/A")
    chk2.markdown("Elemental Sum: N/A")

# --- Run Button ---
st.markdown("---")
if st.button("RUN OPTIMIZATION", type="primary", use_container_width=True):
    inputs = {}
    for cat in st.session_state.backend.ui_cat_cols:
        inputs[cat] = st.session_state[cat if cat != 'activation-method' else 'activation-method']
    for feat in st.session_state.backend.ui_numeric_cols:
        val = 0.0 if is_locked(feat) else st.session_state.params[feat]
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
        st.session_state.verify = res['verification']
        st.session_state.results = {}
        if res['mode'] == 'reverse':
            for k, v in res['optimized_params'].items():
                st.session_state.results[k] = v
        st.success("Calculation Completed Successfully!")
        st.rerun()
    else:
        st.error(f"Error: {res['error']}")
