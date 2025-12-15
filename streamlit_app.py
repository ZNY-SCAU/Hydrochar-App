import streamlit as st
import pandas as pd
from logic import ModelBackend

# ================= 1. 网页配置 =================
st.set_page_config(page_title="Hydrochar Optimization", layout="wide")

# 注入 CSS (SCI 风格)
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Times New Roman', serif; }
    h1 { color: #1A5276; font-weight: bold; }
    h3 { color: #2C3E50; border-bottom: 2px solid #E0E0E0; padding-bottom: 5px; }
    .success-text { color: #27AE60; font-weight: bold; }
    .lock-text { color: #95A5A6; font-style: italic; font-size: 0.9em;}
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

# User Defaults (仅作为缺省值参考)
USER_DEFAULTS = {
    'H(%)': 6.08, 'N(%)': 0.98, 'S(%)': 0.09, '(O+N)/C': 1.106, 'H/C': 0.136,
    'hydrothermal-T(℃)': 230.0, 'hydrothermal-time(h)': 0.5, 'hydrothermal-SLR(g/ml)': 0.167,
    'activation-SLR(g/L)': 0.0, 'activator-concentration(mol/L)': 0.0, 'activation-time(h)': 0.0,
    'adsorption-SLR(g/L)': 10.0, 'RPM(r/min)': 200.0, 'adsorption-time(h)': 6.23,
    'pH': 6.98, 'initial-NH4+-N(mg/L)': 1323.94, 'adsorption-T(℃)': 25.0,
    'C(%)': 44.56, 'O(%)': 48.29
}

# 初始化参数状态
if 'params' not in st.session_state:
    st.session_state.params = {}
    # 遍历模型真正需要的特征进行初始化
    for feat in st.session_state.backend.ui_numeric_cols:
        val = USER_DEFAULTS.get(feat, 0.0)
        st.session_state.params[feat] = val

if 'results' not in st.session_state:
    st.session_state.results = {}

# ================= 3. 核心逻辑：活化部分联动 =================

def check_activation_logic():
    """
    逻辑复刻自 main.py:
    1. 如果 活化剂用量、浓度、时间 任意一个接近 0 -> 强制方法选 '0'
    2. 如果 方法选了 '0' -> 强制数值归零
    """
    # 获取当前值
    slr = st.session_state.params.get('activation-SLR(g/L)', 0.0)
    conc = st.session_state.params.get('activator-concentration(mol/L)', 0.0)
    time = st.session_state.params.get('activation-time(h)', 0.0)
    
    # 获取方法 (注意：selectbox 的 key 是 'activation-method')
    method = st.session_state.get('activation-method', '')
    
    # 判定 A: 数值归零 -> 方法变 '0'
    if slr <= 0.001 or conc <= 0.001 or time <= 0.001:
        opts = st.session_state.backend.cat_options.get('activation-method', [])
        target_opt = '0'
        for opt in opts:
            if opt == '0' or '基准' in opt:
                target_opt = opt
                break
        
        if method != target_opt:
            st.session_state['activation-method'] = target_opt
            st.session_state.params['activation-SLR(g/L)'] = 0.0
            st.session_state.params['activator-concentration(mol/L)'] = 0.0
            st.session_state.params['activation-time(h)'] = 0.0
            st.session_state.params['activation-T(℃)'] = 0.0

def is_activation_locked():
    """判断活化参数是否应该锁定 (灰色不可选)"""
    method = str(st.session_state.get('activation-method', ''))
    if method == '0' or '(基准)' in method or method == '':
        return True
    return False

# 统一回调：数值变化时触发检查
def on_numeric_change():
    check_activation_logic()

# ================= 4. 界面布局 =================

st.title("Hydrochar Process Prediction & Optimization System")
st.markdown("*Machine Learning Based Dual-Target Analysis*")

# --- 1. Experimental Conditions ---
st.markdown("### 1. Experimental Conditions")

# 动态生成分类输入框
if st.session_state.backend.ui_cat_cols:
    cols_cat = st.columns(len(st.session_state.backend.ui_cat_cols))
    for i, cat in enumerate(st.session_state.backend.ui_cat_cols):
        opts = st.session_state.backend.cat_options.get(cat, [])
        
        # 🔥🔥🔥【已修正】直接调用，不要赋值给 session_state，Streamlit 会自动处理 🔥🔥🔥
        cols_cat[i].selectbox(
            cat, 
            opts, 
            key=cat, 
            # 活化方法的改变是单向控制，不需要回调
        )

# --- 2. Process Parameters ---
st.markdown("### 2. Process Parameters")

structure_groups = {
    'Raw Material': ['H(%)', 'N(%)', 'S(%)', '(O+N)/C', 'H/C', 'C(%)', 'O(%)'],
    'Hydrothermal': ['hydrothermal-T(℃)', 'hydrothermal-time(h)', 'hydrothermal-SLR(g/ml)'],
    'Activation': ['activation-SLR(g/L)', 'activator-concentration(mol/L)', 'activation-time(h)', 'activation-T(℃)'],
    'Adsorption': ['adsorption-SLR(g/L)', 'RPM(r/min)', 'adsorption-time(h)', 'pH', 'initial-NH4+-N(mg/L)', 'adsorption-T(℃)']
}

activation_feats = structure_groups['Activation']

row1 = st.columns(2)
row2 = st.columns(2)
locs = [row1[0], row1[1], row2[0], row2[1]]

for (g_name, g_feats), loc in zip(structure_groups.items(), locs):
    with loc:
        st.markdown(f"#### {g_name}")
        
        valid_feats = [f for f in g_feats if f in st.session_state.backend.ui_numeric_cols]
        
        if not valid_feats:
            st.caption("*No parameters in this group used by model*")
            continue

        for feat in valid_feats:
            stat = st.session_state.backend.stats.get(feat, {'min':0, 'max':100})
            c1, c2 = st.columns([2, 1])
            
            is_opt = c1.checkbox(feat, key=f"chk_{feat}")
            
            should_lock = is_opt
            if feat in activation_feats and is_activation_locked():
                should_lock = True
                current_val = 0.0
            else:
                current_val = st.session_state.params.get(feat, 0.0)

            new_val = c1.number_input(
                label=feat,
                value=float(current_val),
                label_visibility="collapsed",
                disabled=should_lock,
                key=f"in_{feat}",
                on_change=on_numeric_change if feat in activation_feats else None
            )
            
            if should_lock and feat in activation_feats and is_activation_locked():
                st.session_state.params[feat] = 0.0
            elif not should_lock:
                st.session_state.params[feat] = new_val

            with c2:
                if feat in activation_feats and is_activation_locked():
                    st.markdown("<span class='lock-text'>🔒 Locked (Method=0)</span>", unsafe_allow_html=True)
                else:
                    st.caption(f"[{stat['min']:.2f}-{stat['max']:.2f}]")
                    if feat in st.session_state.results:
                        res_val = st.session_state.results[feat]
                        st.markdown(f"<span class='success-text'>✅ {res_val:.4f}</span>", unsafe_allow_html=True)

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

# --- 4. Run ---
st.markdown("---")
if st.button("RUN OPTIMIZATION", type="primary", use_container_width=True):
    inputs = {}
    for cat in st.session_state.backend.ui_cat_cols:
        inputs[cat] = st.session_state[cat]
    
    for feat in st.session_state.backend.ui_numeric_cols:
        if feat in activation_feats and is_activation_locked():
            val = 0.0
        else:
            val = st.session_state.params.get(feat, 0.0)
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
        st.session_state.results = {}
        if res['mode'] == 'reverse':
            for k, v in res['optimized_params'].items():
                st.session_state.results[k] = v
        st.success("Calculation Completed Successfully!")
        st.rerun()
    else:
        st.error(f"Error: {res['error']}")
