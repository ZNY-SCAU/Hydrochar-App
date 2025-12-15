import streamlit as st
import pandas as pd
from logic import ModelBackend

# ================= 1. 网页配置 =================
st.set_page_config(page_title="Hydrochar Optimization", layout="wide")

# 注入 CSS (极致紧凑 + SCI 风格)
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Times New Roman', serif; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    h1 { font-size: 1.6rem; color: #1A5276; margin-bottom: 0px; }
    h4 { font-size: 1.0rem; color: #2C3E50; border-bottom: 1px solid #ddd; margin-bottom: 10px; padding-bottom: 5px; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.2rem; }
    .stNumberInput { margin-bottom: 0px; }
    .success-text { color: #27AE60; font-weight: bold; font-size: 0.8em; }
    .lock-text { color: #95A5A6; font-style: italic; font-size: 0.8em; }
    .check-pass { color: #27AE60; font-weight: bold; }
    .check-fail { color: #E74C3C; font-weight: bold; }
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

# 初始化 (带过滤)
if 'params' not in st.session_state:
    st.session_state.params = {}
    for feat in st.session_state.backend.ui_numeric_cols:
        # 🔥 过滤：只初始化模型真正用到的特征
        if feat in st.session_state.backend.model_features:
            val = USER_DEFAULTS.get(feat, 0.0)
            st.session_state.params[feat] = val

if 'results' not in st.session_state:
    st.session_state.results = {}

# ================= 3. 核心逻辑：活化联动 =================

def check_activation_logic():
    """参数归零 -> 方法重置为'0'"""
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

# ================= 4. 界面布局 =================

st.title("Hydrochar Process Optimization System")

# --- 区域1: 实验条件 & 目标 ---
with st.container():
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown("#### 1. Conditions")
        if st.session_state.backend.ui_cat_cols:
            cols_cat = st.columns(2)
            for i, cat in enumerate(st.session_state.backend.ui_cat_cols):
                opts = st.session_state.backend.cat_options.get(cat, [])
                cols_cat[i % 2].selectbox(cat, opts, key=cat, label_visibility="collapsed")
    with c2:
        st.markdown("#### 2. Targets")
        use_ads = st.checkbox("Ads. (mg/g)")
        target_ads = st.number_input("Tgt Ads", disabled=not use_ads, label_visibility="collapsed")
    with c3:
        st.markdown("&nbsp;")
        use_rem = st.checkbox("Rem. Rate (%)")
        target_rem = st.number_input("Tgt Rem", disabled=not use_rem, label_visibility="collapsed")

st.markdown("---")

# --- 区域2: 工艺参数 (4列全景) ---
structure_groups = {
    'Raw Material': ['H(%)', 'N(%)', 'S(%)', '(O+N)/C', 'H/C', 'C(%)', 'O(%)'],
    'Hydrothermal': ['hydrothermal-T(℃)', 'hydrothermal-time(h)', 'hydrothermal-SLR(g/ml)'],
    'Activation': ['activation-SLR(g/L)', 'activator-concentration(mol/L)', 'activation-time(h)', 'activation-T(℃)'],
    'Adsorption': ['adsorption-SLR(g/L)', 'RPM(r/min)', 'adsorption-time(h)', 'pH', 'initial-NH4+-N(mg/L)', 'adsorption-T(℃)']
}
activation_feats = structure_groups['Activation']

cols_main = st.columns(4)
group_names = list(structure_groups.keys())

for i, g_name in enumerate(group_names):
    with cols_main[i]:
        st.markdown(f"#### {g_name}")
        g_feats = structure_groups[g_name]
        
        # 🔥 严格过滤
        valid_feats = [
            f for f in g_feats 
            if f in st.session_state.backend.ui_numeric_cols 
            and f in st.session_state.backend.model_features
        ]
        
        if not valid_feats:
            st.caption("-(N/A)-")
        else:
            for feat in valid_feats:
                stat = st.session_state.backend.stats.get(feat, {'min':0, 'max':100})
                
                # 第一行：勾选 + 范围
                sub_c1, sub_c2 = st.columns([1, 1])
                is_opt = sub_c1.checkbox(feat, key=f"chk_{feat}")
                sub_c2.caption(f"{stat['min']:.0f}~{stat['max']:.0f}")
                
                should_lock = is_opt
                if feat in activation_feats and is_activation_locked():
                    should_lock = True
                    display_val = 0.0
                else:
                    display_val = st.session_state.params.get(feat, 0.0)

                # 第二行：输入
                new_val = st.number_input(
                    label=feat,
                    value=float(display_val),
                    label_visibility="collapsed",
                    disabled=should_lock,
                    key=f"in_{feat}",
                    on_change=check_activation_logic if feat in activation_feats else None
                )
                
                # 数据回写
                if should_lock and feat in activation_feats and is_activation_locked():
                    st.session_state.params[feat] = 0.0
                    st.markdown("<div style='text-align:right; color:#999; font-size:0.8em'>🔒 Locked</div>", unsafe_allow_html=True)
                elif not should_lock:
                    st.session_state.params[feat] = new_val
                    if feat in st.session_state.results:
                        res_v = st.session_state.results[feat]
                        st.markdown(f"<div style='text-align:right; color:#27AE60; font-weight:bold'>✅ {res_v:.3f}</div>", unsafe_allow_html=True)

# --- 区域3: 底部控制台 (运行 + 结果 + 校验) ---
st.markdown("---")

# 创建底部两列：左边是按钮，右边是详细仪表盘
col_btn, col_dash = st.columns([1, 4])

with col_btn:
    st.write("") 
    st.write("") 
    btn_run = st.button("🚀 RUN", type="primary", use_container_width=True)

with col_dash:
    # 结果容器
    res_container = st.container()

# 运行逻辑
if btn_run:
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
        # 保存验证数据
        st.session_state.verify = res.get('verification', {})
        st.session_state.results = {}
        if res['mode'] == 'reverse':
            for k, v in res['optimized_params'].items():
                st.session_state.results[k] = v
        st.rerun()
    else:
        st.error(res['error'])

# 结果显示逻辑 (包含 Check 模块)
if 'pred_ads' in st.session_state:
    with res_container:
        # 使用 4 列布局显示所有关键指标
        r1, r2, r3, r4 = st.columns(4)
        
        # 1. 预测结果
        r1.metric("Predicted Ads.", f"{st.session_state.pred_ads:.2f}", "mg/g")
        r2.metric("Predicted Rem.", f"{st.session_state.pred_rem:.2f}", "%")
        
        # 2. Check 纠错验证模块
        v = st.session_state.verify
        
        # 质量平衡
        mb = v.get('mass_balance_error', 0)
        mb_label = f"{mb:.2f}%"
        # 颜色逻辑：误差<5%为绿，否则为红
        r3.metric("Mass Balance Err", mb_label, delta="✔ Pass" if mb < 5 else "❌ High Error", delta_color="normal" if mb < 5 else "inverse")
        
        # 元素平衡
        el_msg = v.get('elemental_msg', 'N/A')
        el_err = v.get('elemental_error', 0)
        # 颜色逻辑：误差<0.5%为绿
        r4.metric("Elem. Sum", el_msg, delta="✔ Pass" if el_err < 0.5 else "❌ Check Inputs", delta_color="normal" if el_err < 0.5 else "inverse")
