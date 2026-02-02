#!/usr/bin/env python3
"""
δ理論 疲労予測 Web App (Streamlit)
v6.8 Structure Presets + A_int/A_ext 分離

デモURL: streamlit run delta_fatigue_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# v6.8 Core (embedded)
# ===============================
eV_to_J = 1.602176634e-19
k_B = 1.380649e-23
PI = np.pi

STRUCTURE_PRESETS = {
    "BCC": {"r_th": 0.65, "n_cl": 10, "desc": "Body-Centered Cubic (Fe, W) - 明確な疲労限度"},
    "FCC": {"r_th": 0.02, "n_cl": 7, "desc": "Face-Centered Cubic (Cu, Al, Ni) - 疲労限度なし"},
    "HCP": {"r_th": 0.40, "n_cl": 8, "desc": "Hexagonal Close-Packed (Ti, Mg) - 中間的挙動"},
}

MATERIAL_DB = {
    "Fe": {"structure": "BCC", "E_bond_eV": 4.28, "f_d": 1.5, "a_lat_m": 2.92e-10, 
           "alpha0": 0.289, "T_m_K": 1811, "delta_L": 0.18, "M_taylor": 3.0, 
           "tau_sigma_ratio": 0.565, "color": "#1f77b4"},
    "W":  {"structure": "BCC", "E_bond_eV": 8.90, "f_d": 1.3, "a_lat_m": 3.16e-10, 
           "alpha0": 0.289, "T_m_K": 3695, "delta_L": 0.18, "M_taylor": 3.0, 
           "tau_sigma_ratio": 0.565, "color": "#17becf"},
    "Cu": {"structure": "FCC", "E_bond_eV": 3.49, "f_d": 2.0, "a_lat_m": 3.61e-10, 
           "alpha0": 0.289, "T_m_K": 1358, "delta_L": 0.18, "M_taylor": 3.06, 
           "tau_sigma_ratio": 0.565, "color": "#d62728"},
    "Al": {"structure": "FCC", "E_bond_eV": 3.39, "f_d": 1.0, "a_lat_m": 4.05e-10, 
           "alpha0": 0.289, "T_m_K": 933, "delta_L": 0.18, "M_taylor": 3.06, 
           "tau_sigma_ratio": 0.565, "color": "#2ca02c"},
    "Ni": {"structure": "FCC", "E_bond_eV": 4.44, "f_d": 1.8, "a_lat_m": 3.52e-10, 
           "alpha0": 0.289, "T_m_K": 1728, "delta_L": 0.18, "M_taylor": 3.06, 
           "tau_sigma_ratio": 0.565, "color": "#ff7f0e"},
    "Ti": {"structure": "HCP", "E_bond_eV": 4.85, "f_d": 1.2, "a_lat_m": 2.95e-10, 
           "alpha0": 0.289, "T_m_K": 1941, "delta_L": 0.18, "M_taylor": 4.0, 
           "tau_sigma_ratio": 0.50, "color": "#9467bd"},
}

def get_burgers(mat):
    struct, a = mat["structure"], mat["a_lat_m"]
    if struct == "BCC": return a * np.sqrt(3) / 2
    elif struct == "FCC": return a / np.sqrt(2)
    else: return a

def get_V_act(mat):
    return get_burgers(mat) ** 3

def calc_A_int(mat, T_K=300.0):
    E_bond = mat["E_bond_eV"] * eV_to_J
    E_eff = E_bond * mat["alpha0"] * mat["f_d"]
    V_act = get_V_act(mat)
    T_m = mat["T_m_K"]
    tau_sigma = mat["tau_sigma_ratio"]
    A_raw = tau_sigma * E_eff / (V_act * k_B * T_m)
    # Normalize to Fe
    Fe = MATERIAL_DB["Fe"]
    E_Fe = Fe["E_bond_eV"] * eV_to_J * Fe["alpha0"] * Fe["f_d"]
    V_Fe = get_V_act(Fe)
    A_Fe = Fe["tau_sigma_ratio"] * E_Fe / (V_Fe * k_B * Fe["T_m_K"])
    return A_raw / A_Fe

def sigma_yield(mat, T_K, d_m, beta_hp=3.5e-3):
    E_bond = mat["E_bond_eV"] * eV_to_J
    E_eff = E_bond * mat["alpha0"] * mat["f_d"]
    V_act = get_V_act(mat)
    delta_L = mat["delta_L"]
    M = mat["M_taylor"]
    T_m = mat["T_m_K"]
    HP = max(1e-9, 1.0 - T_K / T_m)
    A0 = (E_eff / V_act) * delta_L / (2 * PI * M)
    sigma0 = A0 * HP
    R_block = 1.0 + beta_hp / np.sqrt(d_m)
    return sigma0 * R_block

def N_fail_CL(sigma_max, R_ratio, T_K, d_m, mat, A_ext, D_req=0.78):
    struct = mat["structure"]
    preset = STRUCTURE_PRESETS[struct]
    r_th, n_cl = preset["r_th"], preset["n_cl"]
    A_int = calc_A_int(mat, T_K)
    A_total = A_int * A_ext
    
    sigma_amp = 0.5 * sigma_max * (1 - R_ratio)
    sigma_y = sigma_yield(mat, T_K, d_m)
    r = sigma_amp / sigma_y
    
    if r <= r_th:
        return float("inf"), r, r_th, n_cl, A_int, sigma_y
    
    k = A_total * (r - r_th) ** n_cl
    if k <= 0:
        return float("inf"), r, r_th, n_cl, A_int, sigma_y
    
    N_fail = D_req / k
    return N_fail, r, r_th, n_cl, A_int, sigma_y

def calibrate_A_ext(sigma_max, R_ratio, N_target, T_K, d_m, mat, D_req=0.78):
    struct = mat["structure"]
    preset = STRUCTURE_PRESETS[struct]
    r_th, n_cl = preset["r_th"], preset["n_cl"]
    A_int = calc_A_int(mat, T_K)
    
    sigma_amp = 0.5 * sigma_max * (1 - R_ratio)
    sigma_y = sigma_yield(mat, T_K, d_m)
    r = sigma_amp / sigma_y
    
    if r <= r_th:
        return None, r, r_th, n_cl, A_int, sigma_y, "Error: r ≤ r_th (疲労限度以下)"
    
    A_total_need = D_req / (N_target * (r - r_th) ** n_cl)
    A_ext = A_total_need / A_int
    return A_ext, r, r_th, n_cl, A_int, sigma_y, "OK"

# ===============================
# Streamlit App
# ===============================

st.set_page_config(
    page_title="δ理論 疲労予測 v6.8",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 δ理論 疲労予測システム v6.8")
st.markdown("""
**結晶構造から疲労限度を予測** | フィッティングパラメータ: **0.5個**（1点校正のみ）
""")

# Sidebar
st.sidebar.header("⚙️ 設定")

tab1, tab2, tab3 = st.tabs(["📊 S-N曲線予測", "🎯 A_ext キャリブレーション", "📚 理論説明"])

# ========== Tab 1: S-N Prediction ==========
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("パラメータ設定")
        
        materials = st.multiselect(
            "材料選択",
            options=list(MATERIAL_DB.keys()),
            default=["Fe", "Cu", "Al", "Ni"]
        )
        
        T_K = st.slider("温度 T [K]", 200, 800, 300)
        d_um = st.slider("粒径 d [μm]", 1, 100, 30)
        A_ext = st.number_input("A_ext (校正値)", value=2.46e-4, format="%.2e")
        R_ratio = st.slider("応力比 R", -1.0, 0.5, -1.0, 0.1)
        
        st.divider()
        sigma_min = st.slider("σ_min [MPa]", 10, 200, 30)
        sigma_max = st.slider("σ_max [MPa]", 100, 500, 300)
        n_points = st.slider("計算点数", 10, 50, 25)
    
    with col2:
        if st.button("🚀 S-N曲線を計算", type="primary"):
            d_m = d_um * 1e-6
            sigma_list = np.linspace(sigma_min * 1e6, sigma_max * 1e6, n_points)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            
            results_data = []
            
            for mat_name in materials:
                mat = MATERIAL_DB.get(mat_name)
                if mat is None:
                    continue
                
                N_list = []
                for sig in sigma_list:
                    N, r, r_th, n_cl, A_int, sigma_y = N_fail_CL(sig, R_ratio, T_K, d_m, mat, A_ext)
                    N_list.append(N)
                
                # Filter finite values
                valid = [(s, N) for s, N in zip(sigma_list, N_list) if N < 1e20]
                
                if valid:
                    x = [s/1e6 for s, N in valid]
                    y = [N for s, N in valid]
                    ax.loglog(x, y, 'o-', color=mat.get("color", "black"), 
                             linewidth=2.5, markersize=6, 
                             label=f'{mat_name} ({mat["structure"]}, r_th={STRUCTURE_PRESETS[mat["structure"]]["r_th"]:.2f})')
                
                struct = mat["structure"]
                A_int_val = calc_A_int(mat, T_K)
                results_data.append({
                    "材料": mat_name,
                    "構造": struct,
                    "r_th": STRUCTURE_PRESETS[struct]["r_th"],
                    "n_cl": STRUCTURE_PRESETS[struct]["n_cl"],
                    "A_int": f"{A_int_val:.3f}",
                    "有限寿命点": f"{len(valid)}/{n_points}",
                    "疲労限度": "✅ あり" if STRUCTURE_PRESETS[struct]["r_th"] > 0.1 else "❌ なし"
                })
            
            ax.set_xlabel('Maximum Stress σ_max [MPa]', fontsize=12)
            ax.set_ylabel('Cycles to Failure N_f', fontsize=12)
            ax.set_title(f'δ理論 S-N曲線予測 (v6.8)\nT={T_K}K, d={d_um}μm, R={R_ratio}', fontsize=14)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, which='both', alpha=0.3)
            ax.set_ylim(1e2, 1e15)
            
            st.pyplot(fig)
            
            st.subheader("予測結果サマリー")
            st.dataframe(pd.DataFrame(results_data), use_container_width=True)
            
            st.success(f"""
            **計算完了！**  
            - 材料数: {len(materials)}  
            - A_ext: {A_ext:.2e}（1点校正パラメータ）  
            - r_th, n_cl は構造プリセットから自動設定（フィッティングなし）
            """)

# ========== Tab 2: Calibration ==========
with tab2:
    st.subheader("A_ext の1点校正")
    st.markdown("""
    実験データ1点から **A_ext** を逆算します。  
    この値を使って他の材料・条件の S-N 曲線を予測できます。
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        cal_material = st.selectbox("材料", list(MATERIAL_DB.keys()), index=0)
        cal_T_K = st.slider("温度 T [K]", 200, 800, 300, key="cal_T")
        cal_d_um = st.slider("粒径 d [μm]", 1, 100, 30, key="cal_d")
    
    with col2:
        cal_sigma_max = st.number_input("σ_max [MPa]", value=244.0)
        cal_R_ratio = st.slider("応力比 R", -1.0, 0.5, -1.0, 0.1, key="cal_R")
        cal_N_target = st.number_input("N_target [cycles]", value=7.25e7, format="%.2e")
    
    if st.button("🔧 A_ext を計算", type="primary"):
        mat = MATERIAL_DB[cal_material].copy()
        d_m = cal_d_um * 1e-6
        
        A_ext_cal, r, r_th, n_cl, A_int, sigma_y, status = calibrate_A_ext(
            cal_sigma_max * 1e6, cal_R_ratio, cal_N_target, cal_T_K, d_m, mat
        )
        
        if status == "OK":
            st.success(f"### A_ext = {A_ext_cal:.6e}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("σ_y (δ理論)", f"{sigma_y/1e6:.1f} MPa")
                st.metric("r = σ_a/σ_y", f"{r:.4f}")
            with col2:
                st.metric("r_th (プリセット)", f"{r_th:.2f}")
                st.metric("n_cl (プリセット)", f"{n_cl}")
            with col3:
                st.metric("A_int (δスケール)", f"{A_int:.4f}")
                st.metric("A_total", f"{A_int * A_ext_cal:.6e}")
        else:
            st.error(status)

# ========== Tab 3: Theory ==========
with tab3:
    st.subheader("δ理論とは")
    
    st.markdown("""
    ### 核心方程式
    """)
    
    st.latex(r"\Lambda = \frac{K}{|V|_{\text{eff}}}")
    
    st.markdown("""
    - **K**: 破壊駆動エネルギー（外力）
    - **|V|_eff**: 有効凝集エネルギー（材料抵抗）
    - **Λ = 1**: 臨界条件（破壊発生）
    
    ---
    
    ### v6.8 疲労モデルの特徴
    
    **ダメージ蓄積（非マルコフ的）:**
    """)
    
    st.latex(r"\frac{dD}{dN} = A_{\text{cl}} \times (r - r_{th})^{n_{cl}} \times (1-D)^{m_{cl}}")
    
    st.markdown("""
    - **r = σ_a / σ_y**: 正規化応力振幅
    - **r_th**: 疲労限度閾値（構造依存）
    - **r ≤ r_th → dD/dN = 0**: 疲労限度が自然に出る！
    
    ---
    
    ### 構造プリセット
    """)
    
    preset_df = pd.DataFrame([
        {"構造": "BCC", "r_th": 0.65, "n_cl": 10, "疲労限度": "✅ 明確にあり", "例": "Fe, W, Mo"},
        {"構造": "FCC", "r_th": 0.02, "n_cl": 7, "疲労限度": "❌ なし/曖昧", "例": "Cu, Al, Ni, Au"},
        {"構造": "HCP", "r_th": 0.40, "n_cl": 8, "疲労限度": "△ 中間的", "例": "Ti, Mg, Zn"},
    ])
    st.dataframe(preset_df, use_container_width=True)
    
    st.markdown("""
    ---
    
    ### A_int / A_ext 分離
    
    | パラメータ | 意味 | 導出方法 |
    |-----------|------|---------|
    | **r_th** | 疲労限度閾値 | 構造プリセット（BCC/FCC/HCP） |
    | **n_cl** | Basquin指数 | 構造プリセット |
    | **A_int** | 内部スケール | δパラメータから計算 |
    | **A_ext** | 外部要因 | **1点校正のみ** |
    
    **結果: フィッティングパラメータ = 0.5個（A_ext の1点校正だけ）**
    
    ---
    
    ### Author
    **飯泉真道 & 環**  
    Version 6.8 | 2026-01-31
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
δ理論 疲労予測システム v6.8 | Masamichi Iizumi & Tamaki | 2026
</div>
""", unsafe_allow_html=True)
