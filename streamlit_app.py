import streamlit as st
import pandas as pd
from logic import ModelBackend

# ================= 1. 网页配置 =================
st.set_page_config(page_title="Hydrochar Optimization", layout="wide")

# 注入 CSS (SCI 风格 + 紧凑布局)
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Times New Roman', serif; }
    
    /* 调整顶部留白 */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    
    /* 标题样式 (完整显示，不缩写) */
    h1 { font-size: 1.6rem; color: #1A5276; margin-bottom: 0px; }
    h3 { font-size: 1.2rem; color: #2C3E50; border-bottom: 2px solid #ddd; margin-top: 10px; margin-bottom: 10px; padding-bottom: 5px; font-weight: bold; }
    h4 { font-size: 1.0rem; color: #2C3E50; border-bottom: 1px solid #eee; margin-bottom: 8px; padding-bottom: 4px; font-weight: bold; }
    
    /* 紧凑间距 */
    div[data-testid="stVerticalBlock"] > div { gap: 0.3rem; }
    .stNumberInput { margin-bottom: 0px; }
    
    /* 状态文字 */
    .success-text { color: #27AE60; font-weight: bold; font-size: 0.85em; }
    .lock-text { color: #95A5A6; font-style: italic; font-size: 0.85em; }
    
    /* 隐藏菜单 */
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

# 初始化参数 (严格过滤)
if 'params' not in st.session_state:
    st.session_state.params = {}
    for feat in st.session_state.backend.ui_numeric_cols:
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

# ================= 4. 界面布局 (全景版) =================

st.title("Hydrochar Process Prediction & Optimization System")

# --- 顶部区域：实验条件 & 优化目标 ---
# 使用两列大布局，确保标题完整不换行
top_c1, top_c2 = st.columns([1, 1])

with top_c1:
    st.markdown("### 1. Experimental Conditions")
    if st.session_state.backend.ui_cat_cols:
        # 内部再分两列
        sub_c1, sub_c2 = st.columns(2)
        for i, cat in enumerate(st.session_state.backend.ui_cat_cols):
            opts = st.session_state.backend.cat_options.get(cat, [])
            # 奇数列放左边，偶数列放右边
            curr_col = sub_c1 if i % 2 == 0 else sub_c2
            curr_col.selectbox(cat, opts, key=cat, label_visibility="visible")

with top_c2:
    st.markdown("### 3. Optimization Targets") # 对应 main.py 的编号
    t_c1, t_c2 = st.columns(2)
    with t_c1:
        use_ads = st.checkbox("Adsorption-NH₄⁺-N (mg/g)")
        target_ads = st.number_input("Target Value", disabled=not use_ads, label_visibility="collapsed", key="tgt_ads")
    with t_c2:
        use_rem = st.checkbox("Removal Rate (%)")
        target_rem = st.number_input("Target Value", disabled=not use_rem, label_visibility="collapsed", key="tgt_rem")

# --- 中部区域：工艺参数 (4列全景) ---
st.markdown("### 2. Process Parameters")

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
                sub_c1, sub_c2 = st.columns([1.2, 0.8])
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

st.markdown("---")

# --- 底部区域：运行按钮 & 结果面板 & 4. Check ---
col_btn, col_dash = st.columns([1, 4])

with col_btn:
    st.write("") 
    st.write("") 
    btn_run = st.button("🚀 RUN OPTIMIZATION", type="primary", use_container_width=True)

with col_dash:
    res_container = st.container()

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
        st.markdown("### 4. Check & Results") # 恢复 main.py 的编号
        r1, r2, r3, r4 = st.columns(4)
        
        # 结果
        r1.metric("Predicted Ads. (mg/g)", f"{st.session_state.pred_ads:.2f}")
        r2.metric("Predicted Rem. (%)", f"{st.session_state.pred_rem:.2f}")
        
        # Check 模块
        v = st.session_state.verify
        mb = v.get('mass_balance_error', 0)
        # 质量平衡
        r3.metric("Mass Balance Err", f"{mb:.2f}%", 
                 delta="✔ Pass" if mb < 5 else "❌ Check", delta_color="normal" if mb < 5 else "inverse")
        
        # 元素平衡
        el_msg = v.get('elemental_msg', 'N/A')
        el_err = v.get('elemental_error', 0)
        r4.metric("Elemental Sum", el_msg, 
                 delta="✔ Pass" if el_err < 0.5 else "❌ Check", delta_color="normal" if el_err < 0.5 else "inverse")
