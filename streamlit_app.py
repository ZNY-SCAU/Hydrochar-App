import streamlit as st
import pandas as pd
from logic import ModelBackend

# ================= 1. 网页配置 =================
st.set_page_config(page_title="Hydrochar Optimization", layout="wide")

# 注入 CSS (紧凑 SCI 风格)
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Times New Roman', serif; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    
    /* 标题微调 */
    h1 { font-size: 1.6rem; color: #1A5276; margin-bottom: 5px; }
    h4 { font-size: 1.05rem; color: #2C3E50; border-bottom: 2px solid #eee; margin-bottom: 8px; padding-bottom: 3px; font-weight: bold; }
    
    /* 极度紧凑的输入框间距 */
    div[data-testid="stVerticalBlock"] > div { gap: 0.2rem; }
    .stNumberInput, .stSelectbox, .stCheckbox { margin-bottom: 0px; }
    div[data-testid="column"] { padding: 0rem; }
    
    /* 字体调整 */
    .caption-text { font-size: 0.8em; color: #7F8C8D; }
    .result-text { font-weight: bold; color: #27AE60; font-size: 0.9em; }
    
    /* 隐藏无关元素 */
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

# 初始化参数
if 'params' not in st.session_state:
    st.session_state.params = {}
    for feat in st.session_state.backend.ui_numeric_cols:
        # 🔥 过滤：只初始化模型真正用到的特征
        if feat in st.session_state.backend.model_features:
            val = USER_DEFAULTS.get(feat, 0.0)
            st.session_state.params[feat] = val

if 'results' not in st.session_state:
    st.session_state.results = {}
if 'pred_ads' not in st.session_state:
    st.session_state.pred_ads = 0.0
if 'pred_rem' not in st.session_state:
    st.session_state.pred_rem = 0.0
if 'verify' not in st.session_state:
    st.session_state.verify = {}

# ================= 3. 核心逻辑 =================

def is_activation_locked():
    """只有当用户显式选择 '0' 或 '基准' 方法时才锁定输入框"""
    # 获取 selectbox 的当前值
    method = st.session_state.get('activation-method', '')
    if str(method) == '0' or '(基准)' in str(method) or str(method) == '' or 'Base' in str(method):
        return True
    return False

# ================= 4. 界面布局 =================

st.title("Hydrochar Process Prediction & Optimization System")

# --- Top: 实验条件 & 目标 ---
c_top1, c_top2 = st.columns([1.2, 0.8])

with c_top1:
    st.markdown("#### 1. Experimental Conditions")
    if st.session_state.backend.ui_cat_cols:
        cols_cat = st.columns(2)
        for i, cat in enumerate(st.session_state.backend.ui_cat_cols):
            opts = st.session_state.backend.cat_options.get(cat, [])
            # 这里的 key=cat 会自动绑定到 st.session_state[cat]
            cols_cat[i % 2].selectbox(cat, opts, key=cat, label_visibility="visible")

with c_top2:
    st.markdown("#### 3. Targets") # 保持与 main.py 编号一致
    tc1, tc2 = st.columns(2)
    with tc1:
        use_ads = st.checkbox("Ads. (mg/g)")
        target_ads = st.number_input("Tgt Ads", disabled=not use_ads, label_visibility="collapsed", key="tgt_ads")
    with tc2:
        use_rem = st.checkbox("Rem. Rate (%)")
        target_rem = st.number_input("Tgt Rem", disabled=not use_rem, label_visibility="collapsed", key="tgt_rem")

# --- Middle: 工艺参数 (4列全景) ---
st.markdown("#### 2. Process Parameters")

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
        st.caption(f"**{g_name}**") # 小标题
        g_feats = structure_groups[g_name]
        
        # 🔥 过滤
        valid_feats = [f for f in g_feats if f in st.session_state.backend.ui_numeric_cols and f in st.session_state.backend.model_features]
        
        if not valid_feats:
            st.caption("-(N/A)-")
        else:
            for feat in valid_feats:
                stat = st.session_state.backend.stats.get(feat, {'min':0, 'max':100})
                
                # 第一行：勾选 + 范围 (保留3位小数)
                sc1, sc2 = st.columns([1, 1.2])
                is_opt = sc1.checkbox(feat, key=f"chk_{feat}")
                # 🔥 修正：使用 .3f 避免 0-0 问题
                sc2.markdown(f"<span class='caption-text'>[{stat['min']:.3f}-{stat['max']:.3f}]</span>", unsafe_allow_html=True)
                
                # 锁定判断 (仅由 Method 决定，不再受输入值 0 的影响，防止锁死)
                should_lock = is_opt
                if feat in activation_feats and is_activation_locked():
                    should_lock = True
                    display_val = 0.0
                else:
                    display_val = st.session_state.params.get(feat, 0.0)

                # 第二行：输入框
                # 🔥 修正：移除了 on_change 回调，防止输入 0 时瞬间锁死
                new_val = st.number_input(
                    label=feat,
                    value=float(display_val),
                    label_visibility="collapsed",
                    disabled=should_lock,
                    key=f"in_{feat}",
                    format="%.4f" 
                )
                
                # 数据同步
                if should_lock and feat in activation_feats and is_activation_locked():
                    st.session_state.params[feat] = 0.0
                    st.markdown("<div style='text-align:right; color:#999; font-size:0.8em'>🔒 Locked</div>", unsafe_allow_html=True)
                elif not should_lock:
                    st.session_state.params[feat] = new_val
                    # 结果回显 (绿色文字)
                    if feat in st.session_state.results:
                        res_v = st.session_state.results[feat]
                        st.markdown(f"<div style='text-align:right' class='result-text'>✅ {res_v:.4f}</div>", unsafe_allow_html=True)

st.markdown("---")

# --- Bottom: 运行 & 结果 ---
c_btn, c_res = st.columns([1, 5])

with c_btn:
    st.write("")
    st.write("")
    btn_run = st.button("🚀 RUN", type="primary", use_container_width=True)

# 运行逻辑 (放在这里，不使用 st.rerun，确保一次执行到底)
if btn_run:
    inputs = {}
    for cat in st.session_state.backend.ui_cat_cols:
        inputs[cat] = st.session_state[cat]
    
    for feat in st.session_state.backend.ui_numeric_cols:
        if feat not in st.session_state.backend.model_features: continue
        
        # 此时才做数据清洗：如果 Method 是 0，强制传 0 给模型
        val = st.session_state.params.get(feat, 0.0)
        if feat in activation_feats and is_activation_locked():
            val = 0.0
        
        is_predict = st.session_state.get(f"chk_{feat}", False)
        inputs[feat] = {'value': val, 'is_predict': is_predict}
    
    targets = {
        'ads': {'value': target_ads, 'is_constraint': use_ads},
        'rem': {'value': target_rem, 'is_constraint': use_rem}
    }
    
    with st.spinner("Calculating..."):
        res = st.session_state.backend.run_task(inputs, targets)
    
    if res['success']:
        # 更新状态
        st.session_state.pred_ads = res['ads']
        st.session_state.pred_rem = res['rem']
        st.session_state.verify = res.get('verification', {})
        st.session_state.results = {}
        if res['mode'] == 'reverse':
            for k, v in res['optimized_params'].items():
                st.session_state.results[k] = v
        # 🔥 修正：前向计算不需要 rerun，直接往下走即可显示结果
        # 只有在反向优化(改变了输入参数)时才需要 rerun 刷新上面的输入框
        if res['mode'] == 'reverse':
             st.rerun()
    else:
        st.error(res['error'])

# 结果显示 (始终显示最新状态)
with c_res:
    # 4. Check & Results
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Predicted Ads. (mg/g)", f"{st.session_state.pred_ads:.2f}")
    r2.metric("Predicted Rem. (%)", f"{st.session_state.pred_rem:.2f}")
    
    v = st.session_state.verify
    mb = v.get('mass_balance_error', 0)
    r3.metric("Mass Balance Err", f"{mb:.2f}%", delta="✔" if mb < 5 else "❌ High", delta_color="inverse")
    
    el_msg = v.get('elemental_msg', 'N/A')
    el_err = v.get('elemental_error', 0)
    r4.metric("Elem. Sum", el_msg, delta="✔" if el_err < 0.5 else "❌ Check", delta_color="inverse")
