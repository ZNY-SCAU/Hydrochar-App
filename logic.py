import joblib
import pandas as pd
import numpy as np
import random
import traceback
import warnings
import os

# 过滤警告
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
        """加载模型包，保持原有逻辑不变"""
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
        """构建输入DataFrame，确保特征顺序与训练时一致"""
        df = pd.DataFrame(0.0, index=[0], columns=self.model_features)
        
        # 填充数值特征
        for col in self.ui_numeric_cols:
            val = params_dict.get(col)
            if val is not None and col in self.model_features:
                df[col] = float(val)
        
        # 填充分类特征（One-Hot逻辑）
        for cat_col in self.ui_cat_cols:
            selected_val = params_dict.get(cat_col)
            if selected_val:
                options = self.cat_options.get(cat_col, [])
                is_baseline = (options and selected_val == options[0]) or ("(基准)" in selected_val) or ("0" == selected_val)
                if not is_baseline and selected_val in self.model_features:
                    df[selected_val] = 1.0

        # 标准化处理
        if self.scaler:
            try:
                if hasattr(self.scaler, "feature_names_in_"):
                    scaler_cols = list(self.scaler.feature_names_in_)
                    # 只对存在的列进行标准化
                    if all(c in df.columns for c in scaler_cols):
                        sub_df = df[scaler_cols]
                        df[scaler_cols] = self.scaler.transform(sub_df)
            except: pass
        return df

    # ================= 🧬 自定义遗传算法 (Genetic Algorithm) =================
    def _run_genetic_algorithm(self, objective_func, bounds, pop_size=50, generations=40, mutation_rate=0.1):
        """
        轻量级实数编码遗传算法，专为 Streamlit 优化
        :param objective_func: 目标函数（损失函数）
        :param bounds: 变量范围 [(min, max), ...]
        :param pop_size: 种群大小 (默认50)
        :param generations: 迭代代数 (默认40)
        :param mutation_rate: 变异率
        """
        # 1. 初始化
        n_vars = len(bounds)
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        
        # 随机生成初始种群 [pop_size, n_vars]
        population = lb + (ub - lb) * np.random.rand(pop_size, n_vars)
        
        best_solution = None
        best_fitness = float('inf')

        # 开始进化迭代
        for gen in range(generations):
            # 2. 评估适应度 (Fitness Evaluation)
            # 注意：这里的 objective_func 越小越好（损失函数）
            fitness = np.array([objective_func(ind) for ind in population])
            
            # 记录本代最优
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_solution = population[min_idx].copy()
            
            # 3. 选择 (Selection) - 锦标赛选择法
            # 随机选两组，两两PK，保留胜者
            idx1 = np.random.randint(0, pop_size, pop_size)
            idx2 = np.random.randint(0, pop_size, pop_size)
            mask = fitness[idx1] < fitness[idx2] # 谁损失小谁赢
            winners_idx = np.where(mask, idx1, idx2)
            parents = population[winners_idx]

            # 4. 交叉 (Crossover) - 简单算术交叉
            offspring = parents.copy()
            # 将父代打乱
            np.random.shuffle(offspring)
            
            # 划分为两半进行配对
            cut_point = pop_size // 2
            p1 = offspring[:cut_point]
            p2 = offspring[cut_point : 2*cut_point]
            
            # 随机生成交叉比例 alpha
            alpha = np.random.rand(cut_point, n_vars)
            
            # 生成子代
            c1 = alpha * p1 + (1 - alpha) * p2
            c2 = (1 - alpha) * p1 + alpha * p2
            
            # 更新子代种群
            offspring[:cut_point] = c1
            offspring[cut_point : 2*cut_point] = c2

            # 5. 变异 (Mutation) - 动态高斯变异
            # 随机选择个体进行变异
            mutation_mask = np.random.rand(pop_size, n_vars) < mutation_rate
            
            # 变异强度：范围的 10%
            sigma = 0.1 * (ub - lb)
            noise = np.random.normal(0, 1, (pop_size, n_vars)) * sigma
            
            offspring = offspring + mutation_mask * noise
            
            # 6. 边界处理 (Clip) - 确保不超出物理限制
            offspring = np.clip(offspring, lb, ub)
            
            # 7. 精英策略 (Elitism) - 强制保留历史最优
            # 将本代最好的一定放回去，替换掉子代中随机一个
            offspring[0] = best_solution
            
            population = offspring

        return best_solution

    def run_task(self, inputs, targets):
        """执行预测或反推任务"""
        try:
            # 1. 物理硬限位 (与训练代码保持一致)
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
            
            # 解析输入，区分固定值和需要优化的变量
            for k, v in inputs.items():
                if isinstance(v, dict):
                    if v.get('is_predict', False):
                        # 如果勾选了"Predict" (Check)，则该变量需要反推
                        optimize_vars.append(k)
                        # 确定优化边界
                        if k in BASE_HARD_LIMITS:
                            lb, ub = BASE_HARD_LIMITS[k]['min'], BASE_HARD_LIMITS[k]['max']
                        else:
                            stat = self.stats.get(k, {'min':0, 'max':100})
                            lb, ub = stat['min'], stat['max']
                        optimize_bounds.append((lb, ub))
                        fixed_params[k] = (lb + ub) / 2 # 初始值给个中间值
                    else:
                        # 否则固定为用户输入的值
                        fixed_params[k] = v['value']
                else:
                    fixed_params[k] = v

            target_ads = targets['ads']['value'] if targets['ads']['is_constraint'] else None
            target_rem = targets['rem']['value'] if targets['rem']['is_constraint'] else None

            # 逻辑强校验函数 (确保物理逻辑，如未活化则相关参数归零)
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

            # 智能校验函数 (Smart Verification)
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

                # 2. 元素平衡 (智能反推)
                elem_err = 0.0
                elem_msg = "N/A"
                try:
                    h = params.get('H(%)', 0)
                    n = params.get('N(%)', 0)
                    s = params.get('S(%)', 0)
                    c = params.get('C(%)', 0)
                    o = params.get('O(%)', 0)
                    
                    # 补全逻辑
                    if c <= 0.001:
                        hc_ratio = params.get('H/C', 0)
                        if hc_ratio > 0 and h > 0:
                            c = h / hc_ratio
                            params['C(%)'] = c 

                    if o <= 0.001:
                        onc_ratio = params.get('(O+N)/C', 0)
                        if onc_ratio > 0 and c > 0:
                            o = (onc_ratio * c) - n
                            if o < 0: o = 0 
                            params['O(%)'] = o 

                    total = c + h + o + n + s
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

            # --- 模式 A: 正向预测 (Predict Mode) ---
            # 如果没有变量被勾选为"反推"，则直接计算
            if not optimize_vars:
                fixed_params = enforce_logic(fixed_params) 
                df = self._build_input_df(fixed_params)
                pred = self.model.predict(df)[0]
                verify = calc_verification_metrics(fixed_params, pred[0], pred[1])
                return {'success': True, 'mode': 'forward', 'ads': pred[0], 'rem': pred[1], 'verification': verify}

            # --- 模式 B: 逆向优化 (Reverse / Optimization Mode) ---
            
            # 定义目标函数 (Loss Function)
            def objective(x):
                current = fixed_params.copy()
                # 将优化的变量值填入参数字典
                for i, var in enumerate(optimize_vars):
                    current[var] = x[i]
                current = enforce_logic(current)
                
                # 预测当前参数下的结果
                df = self._build_input_df(current)
                pred = self.model.predict(df)[0]
                p_ads, p_rem = pred[0], pred[1]
                
                loss = 0
                # 计算与目标的差距
                if target_ads: loss += abs(p_ads - target_ads) / (target_ads + 1e-6)
                if target_rem: loss += abs(p_rem - target_rem) / (target_rem + 1e-6)
                # 如果没有设定目标值，则默认最大化吸附量和去除率 (最小化负值)
                if not target_ads and not target_rem: loss = - (p_ads + p_rem) 
                
                # 物理约束惩罚 (Soft Constraints)
                metrics = calc_verification_metrics(current, p_ads, p_rem)
                if metrics['mass_balance_error'] > 5.0: loss += metrics['mass_balance_error'] * 0.1
                if metrics['elemental_error'] > 2.0: loss += metrics['elemental_error'] * 0.1

                return loss

            best_vals = []
            
            # 🔥 调用自定义遗传算法 🔥
            try:
                best_vals = self._run_genetic_algorithm(
                    objective, 
                    optimize_bounds, 
                    pop_size=50,       # 种群大小
                    generations=40,    # 迭代次数
                    mutation_rate=0.1  # 变异率
                )
            except Exception as e:
                # 兜底：如果GA运算出错，退化为随机搜索
                print(f"Genetic Algorithm failed: {e}, using Random Search instead.")
                best_score = float('inf')
                for _ in range(500):
                    x_try = [random.uniform(b[0], b[1]) for b in optimize_bounds]
                    sc = objective(x_try)
                    if sc < best_score: best_score, best_vals = sc, x_try

            # 整理最终结果
            final_res_params = fixed_params.copy()
            for i, var in enumerate(optimize_vars):
                final_res_params[var] = best_vals[i]
            
            final_res_params = enforce_logic(final_res_params)
            
            final_df = self._build_input_df(final_res_params)
            final_pred = self.model.predict(final_df)[0]
            
            verify = calc_verification_metrics(final_res_params, final_pred[0], final_pred[1])
            
            # 补全可能被反推的关联元素 (如C/O)
            if 'C(%)' in final_res_params and 'C(%)' not in optimize_vars:
                 optimize_vars.append('C(%)')
            if 'O(%)' in final_res_params and 'O(%)' not in optimize_vars:
                 optimize_vars.append('O(%)')

            return {
                'success': True, 
                'mode': 'reverse',
                'ads': final_pred[0], 
                'rem': final_pred[1],
                'optimized_params': {k: final_res_params[k] for k in optimize_vars},
                'verification': verify
            }

        except Exception as e:
            return {'success': False, 'error': f"Logic Error: {str(e)}\n{traceback.format_exc()}"}
