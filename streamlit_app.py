import tkinter as tk
from tkinter import ttk, messagebox, font
import threading
from logic import ModelBackend

# ================= SCI 风格配置 =================
THEME = {
    'bg_main': '#FFFFFF',          
    'bg_block': '#F9F9F9',         
    'fg_text': '#000000',          
    'fg_title': '#2C3E50',         
    'fg_success': '#27AE60',       
    'fg_warning': '#E74C3C',       
    'fg_disabled': '#E0E0E0',      
    'font_main': ('Times New Roman', 12),
    'font_bold': ('Times New Roman', 12, 'bold'),
    'font_title': ('Times New Roman', 16, 'bold'),
    'font_big': ('Times New Roman', 22, 'bold'),
}

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hydrochar Optimization System")
        self.root.state('zoomed')
        self.root.configure(bg=THEME['bg_main'])
        
        self.backend = ModelBackend()
        self.cat_vars = {}
        self.param_vars = {} 
        self.target_vars = {}
        
        self.group_frames = {}
        
        self._build_layout()
        self.root.after(200, self.load_data)

    def _build_layout(self):
        # --- 顶部标题 ---
        header = tk.Frame(self.root, bg=THEME['bg_main'], pady=15)
        header.pack(fill='x')
        tk.Label(header, text="Hydrochar Process Prediction & Optimization System", 
                 font=THEME['font_big'], bg=THEME['bg_main'], fg="black").pack()
        tk.Label(header, text="Machine Learning Based Dual-Target Analysis", 
                 font=('Times New Roman', 12, 'italic'), bg=THEME['bg_main'], fg="gray").pack()

        # ==================== 1. 实验条件 ====================
        self.frame_cat = tk.LabelFrame(self.root, text=" 1. Experimental Conditions ", 
                                     font=THEME['font_bold'], bg=THEME['bg_main'], fg=THEME['fg_title'],
                                     bd=2, relief='groove', padx=15, pady=10)
        self.frame_cat.pack(fill='x', padx=20, pady=5)

        # ==================== 2. 工艺参数 ====================
        self.frame_num = tk.LabelFrame(self.root, text=" 2. Process Parameters ", 
                                     font=THEME['font_bold'], bg=THEME['bg_main'], fg=THEME['fg_title'],
                                     bd=2, relief='groove', padx=15, pady=10)
        self.frame_num.pack(fill='both', expand=True, padx=20, pady=5)
        
        # 分组
        self.group_frames['Raw Material'] = tk.LabelFrame(self.frame_num, text="Raw Material Parameters", bg=THEME['bg_main'], font=('Times New Roman', 11, 'bold'))
        self.group_frames['Hydrothermal'] = tk.LabelFrame(self.frame_num, text="Hydrothermal Parameters", bg=THEME['bg_main'], font=('Times New Roman', 11, 'bold'))
        self.group_frames['Activation'] = tk.LabelFrame(self.frame_num, text="Activation Parameters", bg=THEME['bg_main'], font=('Times New Roman', 11, 'bold'))
        self.group_frames['Adsorption'] = tk.LabelFrame(self.frame_num, text="Adsorption Parameters", bg=THEME['bg_main'], font=('Times New Roman', 11, 'bold'))

        self.group_frames['Raw Material'].grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self.group_frames['Hydrothermal'].grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        self.group_frames['Activation'].grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        self.group_frames['Adsorption'].grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        
        self.frame_num.columnconfigure(0, weight=1)
        self.frame_num.columnconfigure(1, weight=1)
        self.frame_num.rowconfigure(0, weight=1)
        self.frame_num.rowconfigure(1, weight=1)

        # ==================== 3. 目标 ====================
        self.frame_target = tk.LabelFrame(self.root, text=" 3. Optimization Targets ", 
                                        font=THEME['font_bold'], bg=THEME['bg_main'], fg=THEME['fg_title'],
                                        bd=2, relief='groove', padx=15, pady=10)
        self.frame_target.pack(fill='x', padx=20, pady=5)

        # ==================== 4. Check ====================
        self.frame_dash = tk.LabelFrame(self.root, text=" 4. Check ", 
                                      font=THEME['font_bold'], bg=THEME['bg_main'], fg=THEME['fg_title'],
                                      bd=2, relief='groove', padx=15, pady=10)
        self.frame_dash.pack(fill='x', padx=20, pady=5)
        
        f_mb = tk.Frame(self.frame_dash, bg=THEME['bg_main'])
        f_mb.pack(side='left', padx=20, fill='x', expand=True)
        tk.Label(f_mb, text="Mass Balance Formula: Removal% ≈ (Ads × SLR × 100) / Initial_Conc", 
                 font=('Times New Roman', 10, 'italic'), fg='gray', bg=THEME['bg_main']).pack(anchor='w')
        self.lbl_mb_status = tk.Label(f_mb, text="(Error: N/A)", font=THEME['font_bold'], bg=THEME['bg_main'], fg="gray")
        self.lbl_mb_status.pack(anchor='w')

        f_el = tk.Frame(self.frame_dash, bg=THEME['bg_main'])
        f_el.pack(side='left', padx=20, fill='x', expand=True)
        tk.Label(f_el, text="Elemental Sum Formula: Total = C% + H% + O% + N% + S% ≈ 100%", 
                 font=('Times New Roman', 10, 'italic'), fg='gray', bg=THEME['bg_main']).pack(anchor='w')
        self.lbl_elem_status = tk.Label(f_el, text="(Sum: N/A)", font=THEME['font_bold'], bg=THEME['bg_main'], fg="gray")
        self.lbl_elem_status.pack(anchor='w')

        # --- 底部按钮 ---
        self.frame_btn = tk.Frame(self.root, bg=THEME['bg_main'], pady=10)
        self.frame_btn.pack(fill='x')
        self.btn_run = tk.Button(self.frame_btn, text="RUN OPTIMIZATION", 
                                font=('Times New Roman', 14, 'bold'),
                                bg="#3498DB", fg="white", activebackground="#2980B9", activeforeground="white",
                                relief='flat', padx=40, pady=8, cursor='hand2',
                                command=self.run)
        self.btn_run.pack()

    def load_data(self):
        success, msg = self.backend.load_model()
        if not success:
            messagebox.showerror("Model Error", msg)
            return

        USER_DEFAULTS = {
            'H(%)': 6.08, 'N(%)': 0.98, 'S(%)': 0.09, '(O+N)/C': 1.106, 'H/C': 0.136,
            'hydrothermal-T(℃)': 230.0, 'hydrothermal-time(h)': 0.5, 'hydrothermal-SLR(g/ml)': 0.167,
            'activation-SLR(g/L)': 0.0, 'activator-concentration(mol/L)': 0.0, 'activation-time(h)': 0.0,
            'adsorption-SLR(g/L)': 10.0, 'RPM(r/min)': 200.0, 'adsorption-time(h)': 6.23,
            'pH': 6.98, 'initial-NH4+-N(mg/L)': 1323.94, 'adsorption-T(℃)': 25.0,
            'C(%)': 44.56, 'O(%)': 48.29
        }

        # 1. 加载分类变量
        for i, cat in enumerate(self.backend.ui_cat_cols):
            sub = tk.Frame(self.frame_cat, bg=THEME['bg_main'])
            sub.pack(side='left', padx=20, fill='y')
            tk.Label(sub, text=cat, font=THEME['font_bold'], bg=THEME['bg_main']).pack(anchor='w')
            opts = self.backend.cat_options.get(cat, [])
            var = tk.StringVar(value=opts[0] if opts else "")
            cb = ttk.Combobox(sub, textvariable=var, values=opts, state="readonly", font=THEME['font_main'], width=25)
            cb.pack(pady=5)
            self.cat_vars[cat] = var
            if "activation-method" in cat: cb.bind("<<ComboboxSelected>>", self.on_method_change)

        # 2. 加载数值变量
        cols = self.backend.ui_numeric_cols
        groups = {
            'Raw Material': ['H(%)', 'N(%)', 'S(%)', '(O+N)/C', 'H/C', 'C(%)', 'O(%)'],
            'Hydrothermal': ['hydrothermal-T(℃)', 'hydrothermal-time(h)', 'hydrothermal-SLR(g/ml)'],
            'Activation': ['activation-SLR(g/L)', 'activator-concentration(mol/L)', 'activation-time(h)', 'activation-T(℃)'],
            'Adsorption': ['adsorption-SLR(g/L)', 'RPM(r/min)', 'adsorption-time(h)', 'pH', 'initial-NH4+-N(mg/L)', 'adsorption-T(℃)']
        }
        col_counters = {k: 0 for k in groups}
        
        for feat in cols:
            target_group = 'Adsorption'
            for g_name, g_cols in groups.items():
                if feat in g_cols: target_group = g_name; break
            
            parent_frame = self.group_frames[target_group]
            r, c = divmod(col_counters[target_group], 2)
            col_counters[target_group] += 1
            
            card = tk.Frame(parent_frame, bg=THEME['bg_block'], bd=1, relief='solid', padx=5, pady=5)
            card.grid(row=r, column=c, sticky='nsew', padx=5, pady=5)
            parent_frame.columnconfigure(c, weight=1)
            
            stats = self.backend.stats.get(feat, {'min':0, 'max':100, 'mean':0})
            
            # Row 1
            row1 = tk.Frame(card, bg=THEME['bg_block'])
            row1.pack(fill='x')
            predict_var = tk.BooleanVar(value=False)
            chk = tk.Checkbutton(row1, text=feat, variable=predict_var, 
                                bg=THEME['bg_block'], font=('Times New Roman', 10, 'bold'),
                                activebackground=THEME['bg_block'])
            chk.pack(side='left')
            range_str = f"[{stats['min']:.3f}-{stats['max']:.3f}]"
            tk.Label(row1, text=range_str, bg=THEME['bg_block'], fg='#7F8C8D', font=('Times New Roman', 9)).pack(side='right')
            
            # Row 2
            default_val = USER_DEFAULTS.get(feat, stats['mean'])
            entry_var = tk.DoubleVar(value=round(default_val, 3))
            entry = tk.Entry(card, textvariable=entry_var, font=('Times New Roman', 11), width=10, relief='solid', bd=1)
            entry.pack(fill='x', pady=2)
            
            # Row 3
            res_var = tk.StringVar(value="")
            res_lbl = tk.Label(card, textvariable=res_var, bg=THEME['bg_block'], fg=THEME['fg_success'], font=('Times New Roman', 10, 'bold'))
            res_lbl.pack(anchor='e')
            
            self.param_vars[feat] = {'val': entry_var, 'predict': predict_var, 'res': res_var, 'entry': entry, 'chk': chk, 'min': stats['min'], 'max': stats['max']}
            
            entry_var.trace('w', lambda *a, f=feat: self.validate_range(f))
            
            # 🔥🔥🔥【修复】将 trace 改为 FocusOut，解决“输入0.0被卡死”的问题 🔥🔥🔥
            if feat in ['activation-SLR(g/L)', 'activator-concentration(mol/L)', 'activation-time(h)']:
                # 只有当用户输入完成并移开鼠标时，才触发检查
                entry.bind('<FocusOut>', lambda e, f=feat: self.check_activation_trigger(f))

            def toggle_state(v=predict_var, e=entry, r=res_var, n=feat):
                if v.get(): e.configure(state='disabled', bg=THEME['fg_disabled']); r.set("⏳")
                else:
                    if not self.is_locked_by_method(n): e.configure(state='normal', bg='white'); r.set(""); self.validate_range(n)
            predict_var.trace('w', lambda *args, v=predict_var, e=entry, r=res_var, n=feat: toggle_state(v, e, r, n))

        # 3. 目标
        self._create_target_row("Adsorption-NH₄⁺-N(mg/g)", 'ads')
        self._create_target_row("Removal Rate (%)", 'rem')
        self.on_method_change()

    def _create_target_row(self, label, key):
        f = tk.Frame(self.frame_target, bg=THEME['bg_main'])
        f.pack(fill='x', padx=20, pady=5)
        tk.Label(f, text=label, width=28, anchor='w', font=THEME['font_main'], bg=THEME['bg_main']).pack(side='left')
        val_var = tk.StringVar()
        chk_var = tk.BooleanVar()
        entry = tk.Entry(f, textvariable=val_var, width=10, font=THEME['font_main'], state='disabled')
        entry.pack(side='left', padx=10)
        def toggle(): entry.configure(state='normal' if chk_var.get() else 'disabled')
        tk.Checkbutton(f, text="Set Goal", variable=chk_var, command=toggle, bg=THEME['bg_main'], font=THEME['font_main']).pack(side='left')
        res = tk.StringVar()
        tk.Label(f, textvariable=res, fg=THEME['fg_success'], bg=THEME['bg_main'], font=THEME['font_bold']).pack(side='left', padx=20)
        self.target_vars[key] = {'val': val_var, 'check': chk_var, 'res': res}

    def validate_range(self, feat):
        ui = self.param_vars[feat]
        if ui['predict'].get() or str(ui['entry']['state']) == 'disabled': return
        try:
            val = float(ui['val'].get())
            if val < ui['min'] or val > ui['max']: ui['entry'].configure(bg='#FDEDEC', fg=THEME['fg_warning'])
            else: ui['entry'].configure(bg='white', fg='black')
        except: pass

    # 🔥🔥🔥 核心联动：数值 -> 方法 (离手生效) 🔥🔥🔥
    def check_activation_trigger(self, changed_feat):
        try:
            val = float(self.param_vars[changed_feat]['val'].get())
        except:
            return

        # 只有确实是0的时候才触发
        if val <= 0.0001:
            # 1. 联动其他数值
            targets = ['activation-SLR(g/L)', 'activator-concentration(mol/L)', 'activation-time(h)', 'activation-T(℃)']
            for t in targets:
                if t != changed_feat and t in self.param_vars:
                    self.param_vars[t]['val'].set(0.0)

            # 2. 联动方法 (一票否决)
            for name, var in self.cat_vars.items():
                if "activation-method" in name:
                    if str(var.get()) != '0':
                        var.set('0') 
                        self.on_method_change() # 触发变灰
                    break

    def is_locked_by_method(self, feat_name):
        target_params = ['activation-SLR(g/L)', 'activator-concentration(mol/L)', 'activation-time(h)', 'activation-T(℃)']
        if feat_name not in target_params: return False
        current_method = ""
        for name, var in self.cat_vars.items():
            if "activation-method" in name: current_method = str(var.get()).strip(); break
        if current_method == '0': return True
        if '(基准)' in current_method: return True
        if 'Base' in current_method: return True
        if current_method == '': return True 
        return False

    def on_method_change(self, event=None):
        should_lock = self.is_locked_by_method('activation-SLR(g/L)')
        target_params = ['activation-SLR(g/L)', 'activator-concentration(mol/L)', 'activation-time(h)', 'activation-T(℃)']
        for p in target_params:
            if p in self.param_vars:
                ui = self.param_vars[p]
                if should_lock:
                    ui['val'].set(0.0); ui['entry'].configure(state='disabled', bg=THEME['fg_disabled'])
                    ui['predict'].set(False); ui['chk'].configure(state='disabled'); ui['res'].set("🔒 0")
                else:
                    ui['chk'].configure(state='normal')
                    if not ui['predict'].get(): ui['entry'].configure(state='normal', bg='white'); ui['res'].set(""); self.validate_range(p)

    def run(self):
        inputs = {}
        for cat, var in self.cat_vars.items(): inputs[cat] = var.get()
        for feat, ui in self.param_vars.items():
            try: v = float(ui['val'].get())
            except: v = 0.0
            inputs[feat] = {'value': v, 'is_predict': ui['predict'].get()}
        targets = {
            'ads': {'value': float(self.target_vars['ads']['val'].get() or 0), 'is_constraint': self.target_vars['ads']['check'].get()},
            'rem': {'value': float(self.target_vars['rem']['val'].get() or 0), 'is_constraint': self.target_vars['rem']['check'].get()}
        }
        
        def task():
            try:
                res = self.backend.run_task(inputs, targets)
                def update_ui():
                    if not res['success']: messagebox.showerror("Error", res['error']); return
                    
                    self.target_vars['ads']['res'].set(f"→ Predicted: {res['ads']:.2f}")
                    self.target_vars['rem']['res'].set(f"→ Predicted: {res['rem']:.2f}")
                    
                    if res['mode'] == 'reverse':
                        for k, v in res['optimized_params'].items():
                            if k in self.param_vars:
                                self.param_vars[k]['res'].set(f"✅ {v:.4f}")
                                self.validate_range(k)
                    else:
                        for feat, ui in self.param_vars.items():
                            if "Fixed" not in ui['res'].get(): ui['res'].set("")
                    
                    verify = res.get('verification', {})
                    mb_err = verify.get('mass_balance_error', 0)
                    mb_col = THEME['fg_success'] if mb_err < 5.0 else THEME['fg_warning']
                    self.lbl_mb_status.config(text=f"(Error: {mb_err:.2f}%)", fg=mb_col)
                    
                    elem_err = verify.get('elemental_error', 0)
                    elem_col = THEME['fg_success'] if elem_err < 0.5 else THEME['fg_warning']
                    self.lbl_elem_status.config(text=f"({verify.get('elemental_msg', 'N/A')})", fg=elem_col)

                    messagebox.showinfo("Done", "Calculation Completed Successfully.")
                self.root.after(0, update_ui)
            except Exception as e: self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        self.btn_run.configure(text="Computing...", state='disabled')
        threading.Thread(target=task, daemon=True).start()
        self.root.after(2000, lambda: self.btn_run.configure(text="RUN OPTIMIZATION", state='normal'))

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()