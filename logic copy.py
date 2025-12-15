import joblib
import pandas as pd
import numpy as np
import random
import traceback
import warnings
import os

try:
    from scipy.optimize import differential_evolution
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings('ignore')

class ModelBackend:
    def __init__(self, model_path="GUI_Model_Package.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.ui_numeric_cols = []
        self.ui_cat_cols = []
        self.stats = {}       
        self.cat_options = {} 
        self.model_features = [] 

    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                return False, "Model file not found: GUI_Model_Package.pkl"

            package = joblib.load(self.model_path)
            self.model = package['model']
            self.scaler = package['scaler']
            
            # 优先读取模型真实的特征名
            if hasattr(self.model, "feature_names_in_"):
                self.model_features = list(self.model.feature_names_in_)
            else:
                self.model_features = package['model_features']
            
            self.ui_numeric_cols = package.get('ui_numeric_cols', [])
            self.ui_cat_cols = package.get('ui_cat_cols', [])
            self.stats = package.get('ui_numeric_stats', {})
            self.cat_options = package.get('ui_cat_options', {})
            
            return True, "Loaded successfully"
        except Exception as e:
            return False, f"Load failed: {str(e)}"

    def _build_input_df(self, params_dict):
        df = pd.DataFrame(0.0, index=[0], columns=self.model_features)
        
        for col in self.ui_numeric_cols:
            val = params_dict.get(col)
            if val is not None and col in self.model_features:
                df[col] = float(val)
        
        for cat_col in self.ui_cat_cols:
            selected_val = params_dict.get(cat_col)
            if selected_val:
                options = self.cat_options.get(cat_col, [])
                is_baseline = (options and selected_val == options[0]) or ("(基准)" in selected_val) or ("0" == selected_val)
                if not is_baseline and selected_val in self.model_features:
                    df[selected_val] = 1.0

        if self.scaler:
            try:
                if hasattr(self.scaler, "feature_names_in_"):
                    scaler_cols = list(self.scaler.feature_names_in_)
                    if all(c in df.columns for c in scaler_cols):
                        sub_df = df[scaler_cols]
                        df[scaler_cols] = self.scaler.transform(sub_df)
            except: pass
        return df

    def run_task(self, inputs, targets):
        try:
            # 1. 物理硬限位
            BASE_HARD_LIMITS = {
                'activation-SLR(g/L)': {'min': 0.0, 'max': 100.0},
                'activator-concentration(mol/L)': {'min': 0.0, 'max': 12.0},
                'activation-time(h)': {'min': 0.0, 'max': 55.0},
                'hydrothermal-T(℃)': {'min': 180, 'max': 300},
                'hydrothermal-time(h)': {'min': 0.5, 'max': 6.0},
                'hydrothermal-SLR(g/ml)': {'min': 0.001, 'max': 0.2},
                'adsorption-SLR(g/L)': {'min': 0.0, 'max': 50.0},
                'adsorption-time(h)': {'min': 0.0, 'max': 24.0},
                'pH': {'min': 5.0, 'max': 9.0},
                'RPM(r/min)': {'min': 100.0, 'max': 300.0},
                'adsorption-T(℃)': {'min': 20.0, 'max': 50.0},
                'S(%)': {'min': 0.0, 'max': 3.0},
                'N(%)': {'min': 0.0, 'max': 28.0},
                'H(%)': {'min': 3.0, 'max': 10.0},
                'C(%)': {'min': 20.0, 'max': 80.0},
                'O(%)': {'min': 5.0, 'max': 60.0},
                'H/C': {'min': 0.0, 'max': 4.0}, 
                '(O+N)/C': {'min': 0.0, 'max': 4.0},
            }

            fixed_params = {}
            optimize_vars = []
            optimize_bounds = []
            
            for k, v in inputs.items():
                if isinstance(v, dict):
                    if v.get('is_predict', False):
                        optimize_vars.append(k)
                        if k in BASE_HARD_LIMITS:
                            lb, ub = BASE_HARD_LIMITS[k]['min'], BASE_HARD_LIMITS[k]['max']
                        else:
                            stat = self.stats.get(k, {'min':0, 'max':100})
                            lb, ub = stat['min'], stat['max']
                        optimize_bounds.append((lb, ub))
                        fixed_params[k] = (lb + ub) / 2
                    else:
                        fixed_params[k] = v['value']
                else:
                    fixed_params[k] = v

            target_ads = targets['ads']['value'] if targets['ads']['is_constraint'] else None
            target_rem = targets['rem']['value'] if targets['rem']['is_constraint'] else None

            # 逻辑强校验
            def enforce_logic(params):
                k_method = 'activation-method'
                k_slr = 'activation-SLR(g/L)'
                k_conc = 'activator-concentration(mol/L)'
                k_time = 'activation-time(h)'

                method = str(params.get(k_method, '')).strip()
                slr = params.get(k_slr, 0.0)
                conc = params.get(k_conc, 0.0)
                time = params.get(k_time, 0.0)
                threshold = 0.001 

                is_method_zero = (method == '0' or '基准' in method or method == '')
                is_any_num_zero = (slr < threshold) or (conc < threshold) or (time < threshold)

                if is_method_zero or is_any_num_zero:
                    params[k_slr] = 0.0
                    params[k_conc] = 0.0
                    params[k_time] = 0.0
                else:
                    min_phys = 0.1 
                    if params[k_slr] < min_phys: params[k_slr] = min_phys
                    if params[k_conc] < min_phys: params[k_conc] = min_phys
                    if params[k_time] < 1.0: params[k_time] = 1.0

                return params

            # 🔥🔥🔥【智能反推】物理校验计算器 (Smart Verification) 🔥🔥🔥
            def calc_verification_metrics(params, ads, rem):
                # 1. 质量守恒误差
                mb_err = 0.0
                mb_msg = "N/A"
                try:
                    c0 = params.get('initial-NH4+-N(mg/L)', 0)
                    slr = params.get('adsorption-SLR(g/L)', 0)
                    if c0 > 1.0 and slr > 0:
                        theo_rem = (ads * slr * 100) / c0
                        mb_err = abs(rem - theo_rem)
                        mb_msg = f"{mb_err:.2f}%"
                except: pass

                # 2. 元素平衡 (智能反推版)
                elem_err = 0.0
                elem_msg = "N/A"
                try:
                    # 读取基础元素
                    h = params.get('H(%)', 0)
                    n = params.get('N(%)', 0)
                    s = params.get('S(%)', 0)
                    # 尝试读取 C 和 O (可能为0)
                    c = params.get('C(%)', 0)
                    o = params.get('O(%)', 0)
                    
                    # 💡 智能反推逻辑：如果 C 或 O 缺失，尝试用比值特征反推
                    # C = H / (H/C)
                    if c <= 0.001:
                        hc_ratio = params.get('H/C', 0)
                        if hc_ratio > 0 and h > 0:
                            c = h / hc_ratio # 反推 C
                            params['C(%)'] = c # 临时存入，方便后续使用

                    # O = (O+N)/C * C - N
                    if o <= 0.001:
                        onc_ratio = params.get('(O+N)/C', 0)
                        if onc_ratio > 0 and c > 0:
                            o = (onc_ratio * c) - n
                            if o < 0: o = 0 
                            params['O(%)'] = o # 临时存入

                    total = c + h + o + n + s
                    
                    # 只有当总和有物理意义时才显示
                    if total > 5.0: 
                        elem_err = abs(total - 100.0)
                        elem_msg = f"{total:.2f}% (Err: {elem_err:.2f}%)"
                    else:
                        elem_msg = "Insufficient Data"
                        
                except: pass
                
                return {
                    'mass_balance_error': mb_err,
                    'mass_balance_msg': mb_msg,
                    'elemental_error': elem_err,
                    'elemental_msg': elem_msg
                }

            # --- 模式 A: 正向预测 ---
            if not optimize_vars:
                fixed_params = enforce_logic(fixed_params) 
                df = self._build_input_df(fixed_params)
                pred = self.model.predict(df)[0]
                verify = calc_verification_metrics(fixed_params, pred[0], pred[1])
                return {'success': True, 'mode': 'forward', 'ads': pred[0], 'rem': pred[1], 'verification': verify}

            # --- 模式 B: 逆向优化 ---
            def objective(x):
                current = fixed_params.copy()
                for i, var in enumerate(optimize_vars):
                    current[var] = x[i]
                current = enforce_logic(current)
                
                df = self._build_input_df(current)
                pred = self.model.predict(df)[0]
                p_ads, p_rem = pred[0], pred[1]
                
                loss = 0
                if target_ads: loss += abs(p_ads - target_ads) / (target_ads + 1e-6)
                if target_rem: loss += abs(p_rem - target_rem) / (target_rem + 1e-6)
                if not target_ads and not target_rem: loss = - (p_ads + p_rem) 
                
                # 物理惩罚
                metrics = calc_verification_metrics(current, p_ads, p_rem)
                if metrics['mass_balance_error'] > 5.0: loss += metrics['mass_balance_error'] * 0.1
                # 元素误差也加入惩罚
                if metrics['elemental_error'] > 2.0: loss += metrics['elemental_error'] * 0.1

                return loss

            best_vals = []
            if HAS_SCIPY:
                result = differential_evolution(objective, bounds=optimize_bounds, maxiter=20, popsize=10, seed=42)
                best_vals = result.x
            else:
                best_score = float('inf')
                for _ in range(500):
                    x_try = [random.uniform(b[0], b[1]) for b in optimize_bounds]
                    sc = objective(x_try)
                    if sc < best_score: best_score, best_vals = sc, x_try

            final_res_params = fixed_params.copy()
            for i, var in enumerate(optimize_vars):
                final_res_params[var] = best_vals[i]
            
            final_res_params = enforce_logic(final_res_params)
            
            final_df = self._build_input_df(final_res_params)
            final_pred = self.model.predict(final_df)[0]
            
            # 最终校验 & 补全缺失元素
            verify = calc_verification_metrics(final_res_params, final_pred[0], final_pred[1])
            
            # 🔥 关键：确保 C(%) 和 O(%) 即使不在输入里，也能通过 optimized_params 传回给界面显示
            if 'C(%)' in final_res_params:
                if 'C(%)' not in optimize_vars: optimize_vars.append('C(%)')
            if 'O(%)' in final_res_params:
                if 'O(%)' not in optimize_vars: optimize_vars.append('O(%)')

            return {
                'success': True, 'mode': 'reverse',
                'ads': final_pred[0], 'rem': final_pred[1],
                'optimized_params': {k: final_res_params[k] for k in optimize_vars},
                'verification': verify
            }

        except Exception as e:
            return {'success': False, 'error': f"Logic Error: {str(e)}\n{traceback.format_exc()}"}