"""
δ-theory 統合材料データベース (v10.0 SSOC)

v10.0 変更点:
  - δ_Lフリー式への移行: √(E_coh·k_B·T_m) が δ_L を置換
  - SSOC (Structure-Selective Orbital Coupling) パラメータ追加
    FCC用: mu_GPa, gamma_isf
    BCC用: group, sel
    HCP用: R_crss
    共通:  n_d, period
  - f_d_elec は後方互換のため残存（SSOCではssoc.pyが計算）
  - delta_L は後方互換のため残存（v10.0では不使用）
  - COEFF_V10 = 8√5/(5π) を定数として追加

v7.0 → v10.0 公式の変更:
  旧: σ_y = (E_bond × α × (b/d)² × f_d_elec / V_act) × (δ_L × HP / 2πM)
  新: σ_y = (8√5/5πMZ) × α₀ × (b/d)² × f_de × √(E_coh·k_B·T_m) / V_act × HP

  Key insight: δ_L ∝ √(k_B·T_m / E_coh) なので、
    E_bond × δ_L ∝ √(E_coh · k_B · T_m)
  v10.0 はこの関係を直接使い、δ_Lへの依存を排除。

Author: 環 & ご主人さま (飯泉真道)
Date: 2026-02-08
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Literal, Optional
import numpy as np


# ==============================================================================
# 物理定数
# ==============================================================================
eV_to_J: float = 1.602176634e-19   # 1 eV in Joule
k_B: float = 1.380649e-23          # Boltzmann constant [J/K]
PI: float = np.pi

# (b/d)^2 = 3/2: universal slip-plane geometric constant.
BD_RATIO_SQ: float = 1.5  # (b/d)^2 = 3/2

# v10.0 結晶幾何係数: 8√5/(5π)
COEFF_V10: float = 8 * np.sqrt(5) / (5 * PI)  # ≈ 1.1384


# ==============================================================================
# 構造プリセット（BCC/FCC/HCP）
# ==============================================================================
@dataclass(frozen=True)
class StructurePreset:
    """結晶構造に依存するプリセット値"""
    name: str
    Z_bulk: int              # バルク配位数
    alpha0: float            # 幾何係数 (δ理論)
    M_taylor: float          # Taylor因子
    r_th: float              # 疲労限度閾値
    n_cl: float              # Basquin指数
    desc: str                # 説明


STRUCTURE_PRESETS: Dict[str, StructurePreset] = {
    "BCC": StructurePreset(
        name="BCC",
        Z_bulk=8,
        alpha0=0.289,
        M_taylor=3.0,
        r_th=0.65,
        n_cl=10.0,
        desc="Body-Centered Cubic - 明確な疲労限度",
    ),
    "FCC": StructurePreset(
        name="FCC",
        Z_bulk=12,
        alpha0=0.250,
        M_taylor=3.0,
        r_th=0.02,
        n_cl=7.0,
        desc="Face-Centered Cubic - 疲労限度なし",
    ),
    "HCP": StructurePreset(
        name="HCP",
        Z_bulk=12,
        alpha0=0.350,
        M_taylor=3.0,
        r_th=0.20,
        n_cl=9.0,
        desc="Hexagonal Close-Packed - 中間的挙動",
    ),
}


# ==============================================================================
# 統合材料データクラス
# ==============================================================================
@dataclass(frozen=True)
class Material:
    """
    統合材料データベース (v10.0 SSOC)
    
    全てのδ理論モジュールで使用する材料パラメータを一元管理
    
    v10.0 SSOC:
      σ_y = (COEFF/M·Z) × α₀ × BD2 × f_de × √(E_coh·k_B·T_m) / V_act × HP
      
      f_de は ssoc.py で計算（material.py はデータのみ保持）
      
      パラメータ分類:
        [データ層] material.py — 物質固有の測定値・量子数
        [計算層] ssoc.py — SSOC f_de計算ロジック
        [応用層] unified_yield_fatigue — σ_y → S-N統合
    
    カテゴリ:
      [基本]   結晶構造、格子定数、弾性定数、熱物性
      [SSOC]   n_d, period, group, sel, mu_GPa, gamma_isf, R_crss
      [τ/σ]    せん断/引張比、双晶因子、圧縮/引張比
      [粒界]   分離仕事、偏析エネルギー (DBT用)
      [疲労]   熱軟化、Born崩壊、A_int
      [後方互換] delta_L, f_d_elec (v7.0用、v10.0では不使用)
    """
    
    # ==========================================================================
    # [必須] 基本物性
    # ==========================================================================
    name: str                           # 材料名 (Fe, Cu, Al, ...)
    structure: Literal["BCC", "FCC", "HCP"]  # 結晶構造
    a: float                            # 格子定数 @ 300K [m]
    T_m: float                          # 融点 [K]
    E: float                            # ヤング率 [Pa]
    nu: float                           # ポアソン比
    rho: float                          # 密度 [kg/m³]
    M_amu: float                        # 原子質量 [amu]
    E_bond_eV: float                    # 結合エネルギー (= E_coh) [eV]
    
    # ==========================================================================
    # [SSOC] 構造選択的軌道結合パラメータ
    # ==========================================================================
    n_d: int = 0                        # d電子数 (0-10)
    period: int = 4                     # 周期 (3-6)
    
    # BCC SCC用
    group: int = 0                      # 族番号 (5,6,8 for BCC)
    sel: int = 2                        # 選択子 (1 or 2, BCC Group5用)
    
    # FCC PCC用
    mu_GPa: float = 0.0                 # 剪断弾性率 [GPa]
    gamma_isf: float = 100.0            # 積層欠陥エネルギー [mJ/m²]
    
    # HCP PCC用
    R_crss: float = 1.0                 # CRSS異方性比 (prismatic/basal)
    
    # ==========================================================================
    # [後方互換] v7.0パラメータ（v10.0では不使用）
    # ==========================================================================
    delta_L: float = 0.10               # Lindemann閾値（v10.0で不要だが互換用に残存）
    f_d_elec: float = 1.0              # 電子方向性係数（v10.0ではssoc.pyが計算）
    
    # ==========================================================================
    # [オプション] デフォルト値あり
    # ==========================================================================
    # 熱物性
    alpha_thermal: float = 1.5e-5       # 線膨張係数 [1/K]
    
    # τ/σ (v6.9用)
    c_a: float = 1.633                  # HCP c/a比 (FCC/BCCは使わない)
    T_twin: float = 1.0                 # 双晶因子 (引張)
    R_comp: float = 1.0                 # 圧縮/引張 比 (σ_c/σ_t)
    A_texture: float = 1.0              # 集合組織係数
    
    # 粒界 (DBT用)
    W_sep0: float = 2.0                 # 基準分離仕事 [J/m²]
    delta_sep: float = 0.2e-9           # 分離変位 [m]
    E_seg_eV: float = 0.45              # 偏析エネルギー [eV]
    
    # 疲労
    lambda_base: float = 25.0           # 熱軟化パラメータ
    kappa: float = 1.0                  # 非線形熱軟化係数
    fG: float = 0.10                    # Born崩壊係数
    A_int: float = 1.0                  # 内部スケール (Fe=1.0基準)
    
    # 表示
    color: str = "#333333"              # プロット色
    
    # ==========================================================================
    # 計算プロパティ
    # ==========================================================================
    
    @property
    def preset(self) -> StructurePreset:
        return STRUCTURE_PRESETS[self.structure]
    
    @property
    def Z_bulk(self) -> int:
        return self.preset.Z_bulk
    
    @property
    def alpha0(self) -> float:
        return self.preset.alpha0
    
    @property
    def M_taylor(self) -> float:
        return self.preset.M_taylor
    
    @property
    def r_th(self) -> float:
        return self.preset.r_th
    
    @property
    def n_cl(self) -> float:
        return self.preset.n_cl
    
    @property
    def G(self) -> float:
        """剛性率 [Pa]"""
        return self.E / (2 * (1 + self.nu))
    
    @property
    def b(self) -> float:
        """バーガースベクトル [m]"""
        if self.structure == "BCC":
            return self.a * np.sqrt(3) / 2
        elif self.structure == "FCC":
            return self.a / np.sqrt(2)
        else:  # HCP
            return self.a
    
    @property
    def V_act(self) -> float:
        """活性化体積 [m³]"""
        return self.b ** 3
    
    @property
    def f_d(self) -> float:
        """後方互換: 旧 f_d = BD_RATIO_SQ × f_d_elec"""
        return BD_RATIO_SQ * self.f_d_elec
    
    @property
    def E_eff_v7(self) -> float:
        """v7.0 有効結合エネルギー [J] (後方互換)
        E_eff = E_bond × α₀ × (b/d)² × f_d_elec
        """
        return self.E_bond_eV * eV_to_J * self.alpha0 * BD_RATIO_SQ * self.f_d_elec
    
    @property
    def E_eff(self) -> float:
        """有効結合エネルギー [J] — v7.0互換エイリアス"""
        return self.E_eff_v7
    
    @property
    def sqrt_EkT(self) -> float:
        """v10.0 コアエネルギー項: √(E_coh · k_B · T_m) [J]
        
        δ_Lフリーの核心: E_bond × δ_L ∝ √(E_coh · k_B · T_m)
        """
        return np.sqrt(self.E_bond_eV * eV_to_J * k_B * self.T_m)
    
    # ==========================================================================
    # 表示
    # ==========================================================================
    
    def __str__(self) -> str:
        return f"Material({self.name}, {self.structure}, T_m={self.T_m}K)"
    
    def summary(self) -> str:
        ssoc_info = ""
        if self.structure == "FCC":
            ssoc_info = f"""    μ          = {self.mu_GPa} GPa
    γ_isf      = {self.gamma_isf} mJ/m²"""
        elif self.structure == "BCC":
            ssoc_info = f"""    group      = {self.group}
    sel        = {self.sel}"""
        elif self.structure == "HCP":
            ssoc_info = f"""    R_crss     = {self.R_crss}
    c/a        = {self.c_a}"""
        
        return f"""
{'='*60}
Material: {self.name} ({self.structure})  [v10.0 SSOC]
{'='*60}
  [基本]
    a          = {self.a*1e10:.3f} Å
    T_m        = {self.T_m:.0f} K
    E          = {self.E/1e9:.0f} GPa
    G          = {self.G/1e9:.1f} GPa
    ν          = {self.nu}
    ρ          = {self.rho} kg/m³
  
  [SSOC]  σ_y = (8√5/5πMZ) × α₀ × (3/2) × f_de × √(E·kT) / V × HP
    n_d        = {self.n_d}
    period     = {self.period}
    E_bond     = {self.E_bond_eV} eV
    √(E·kT)   = {self.sqrt_EkT:.4e} J
    α₀         = {self.alpha0}  [preset]
    b          = {self.b*1e10:.3f} Å
{ssoc_info}
  
  [v7.0互換]
    δ_L        = {self.delta_L}  (v10.0では不使用)
    f_d_elec   = {self.f_d_elec:.4f}  (v10.0ではssocが計算)
  
  [τ/σ]
    T_twin     = {self.T_twin}
    R_comp     = {self.R_comp}
    A_texture  = {self.A_texture}
  
  [疲労]
    r_th       = {self.r_th} (preset)
    n_cl       = {self.n_cl} (preset)
    A_int      = {self.A_int}
{'='*60}
"""


# ==============================================================================
# 材料データベース
# ==============================================================================

# --------------------------------------------------------------------------
# BCC 金属
# --------------------------------------------------------------------------

Fe = Material(
    name="Fe",
    structure="BCC",
    a=2.92e-10,
    T_m=1811,
    E=211e9,
    nu=0.29,
    rho=7870,
    M_amu=55.845,
    E_bond_eV=4.28,
    # SSOC (SCC)
    n_d=6, period=4, group=8, sel=2,
    # v7.0互換
    delta_L=0.18,
    f_d_elec=1.0,
    # τ/σ
    T_twin=1.0,
    R_comp=1.0,
    # 粒界
    W_sep0=2.0,
    delta_sep=0.2e-9,
    E_seg_eV=0.45,
    # 疲労
    alpha_thermal=1.50e-5,
    lambda_base=49.2,
    kappa=0.573,
    fG=0.027,
    A_int=1.0,
    color="#1f77b4",
)

W = Material(
    name="W",
    structure="BCC",
    a=3.16e-10,
    T_m=3695,
    E=411e9,
    nu=0.28,
    rho=19300,
    M_amu=183.84,
    E_bond_eV=8.90,
    # SSOC (SCC)
    n_d=4, period=6, group=6, sel=2,
    # v7.0互換
    delta_L=0.16,
    f_d_elec=47/15,
    # τ/σ
    T_twin=1.0,
    R_comp=1.0,
    # 疲労
    alpha_thermal=4.51e-6,
    lambda_base=10.9,
    kappa=2.759,
    fG=0.021,
    A_int=0.85,
    color="#17becf",
)

V_metal = Material(
    name="V",
    structure="BCC",
    a=3.03e-10,
    T_m=2183,
    E=128e9,
    nu=0.37,
    rho=6100,
    M_amu=50.942,
    E_bond_eV=5.31,
    # SSOC (SCC)
    n_d=3, period=4, group=5, sel=2,
    # v7.0互換
    delta_L=0.12,
    f_d_elec=1.0,
    # 疲労
    alpha_thermal=8.4e-6,
    A_int=0.90,
    color="#aec7e8",
)

Cr = Material(
    name="Cr",
    structure="BCC",
    a=2.91e-10,
    T_m=2180,
    E=279e9,
    nu=0.21,
    rho=7190,
    M_amu=51.996,
    E_bond_eV=4.10,
    # SSOC (SCC)
    n_d=5, period=4, group=6, sel=1,
    # v7.0互換
    delta_L=0.12,
    f_d_elec=1.0,
    # 疲労
    alpha_thermal=4.9e-6,
    A_int=0.90,
    color="#98df8a",
)

Nb = Material(
    name="Nb",
    structure="BCC",
    a=3.30e-10,
    T_m=2750,
    E=105e9,
    nu=0.40,
    rho=8570,
    M_amu=92.906,
    E_bond_eV=7.57,
    # SSOC (SCC)
    n_d=4, period=5, group=5, sel=1,
    # v7.0互換
    delta_L=0.13,
    f_d_elec=1.0,
    # 疲労
    alpha_thermal=7.3e-6,
    A_int=0.90,
    color="#c5b0d5",
)

Mo = Material(
    name="Mo",
    structure="BCC",
    a=3.15e-10,
    T_m=2896,
    E=329e9,
    nu=0.31,
    rho=10220,
    M_amu=95.95,
    E_bond_eV=6.82,
    # SSOC (SCC)
    n_d=5, period=5, group=6, sel=1,
    # v7.0互換
    delta_L=0.14,
    f_d_elec=1.0,
    # 疲労
    alpha_thermal=4.8e-6,
    A_int=0.90,
    color="#ffbb78",
)

Ta = Material(
    name="Ta",
    structure="BCC",
    a=3.30e-10,
    T_m=3290,
    E=186e9,
    nu=0.34,
    rho=16650,
    M_amu=180.948,
    E_bond_eV=8.10,
    # SSOC (SCC)
    n_d=3, period=6, group=5, sel=2,
    # v7.0互換
    delta_L=0.14,
    f_d_elec=1.0,
    # 疲労
    alpha_thermal=6.3e-6,
    A_int=0.90,
    color="#c49c94",
)

# --------------------------------------------------------------------------
# FCC 金属
# --------------------------------------------------------------------------

Cu = Material(
    name="Cu",
    structure="FCC",
    a=3.61e-10,
    T_m=1357,
    E=130e9,
    nu=0.34,
    rho=8960,
    M_amu=63.546,
    E_bond_eV=3.49,
    # SSOC (PCC)
    n_d=10, period=4, mu_GPa=48.0, gamma_isf=45.0,
    # v7.0互換
    delta_L=0.10,
    f_d_elec=4/3,
    # τ/σ
    T_twin=1.0,
    R_comp=1.0,
    # 疲労
    alpha_thermal=1.70e-5,
    lambda_base=26.3,
    kappa=1.713,
    fG=0.101,
    A_int=1.41,
    color="#d62728",
)

Al = Material(
    name="Al",
    structure="FCC",
    a=4.05e-10,
    T_m=933,
    E=70e9,
    nu=0.35,
    rho=2700,
    M_amu=26.982,
    E_bond_eV=3.39,
    # SSOC (PCC)
    n_d=0, period=3, mu_GPa=26.0, gamma_isf=166.0,
    # v7.0互換
    delta_L=0.10,
    f_d_elec=16/15,
    # τ/σ
    T_twin=1.0,
    R_comp=1.0,
    # 疲労
    alpha_thermal=2.30e-5,
    lambda_base=27.3,
    kappa=4.180,
    fG=0.101,
    A_int=0.71,
    color="#2ca02c",
)

Ni = Material(
    name="Ni",
    structure="FCC",
    a=3.52e-10,
    T_m=1728,
    E=200e9,
    nu=0.31,
    rho=8900,
    M_amu=58.693,
    E_bond_eV=4.44,
    # SSOC (PCC)
    n_d=8, period=4, mu_GPa=76.0, gamma_isf=125.0,
    # v7.0互換
    delta_L=0.11,
    f_d_elec=26/15,
    # τ/σ
    T_twin=1.0,
    R_comp=1.0,
    # 疲労
    alpha_thermal=1.30e-5,
    lambda_base=22.6,
    kappa=0.279,
    fG=0.092,
    A_int=1.37,
    color="#ff7f0e",
)

Au = Material(
    name="Au",
    structure="FCC",
    a=4.08e-10,
    T_m=1337,
    E=79e9,
    nu=0.44,
    rho=19300,
    M_amu=196.967,
    E_bond_eV=3.81,
    # SSOC (PCC)
    n_d=10, period=6, mu_GPa=27.0, gamma_isf=32.0,
    # v7.0互換
    delta_L=0.10,
    f_d_elec=11/15,
    # τ/σ
    T_twin=1.0,
    R_comp=1.0,
    # 疲労
    alpha_thermal=1.42e-5,
    lambda_base=25.0,
    kappa=1.5,
    fG=0.101,
    A_int=1.0,
    color="#e377c2",
)

Ag = Material(
    name="Ag",
    structure="FCC",
    a=4.09e-10,
    T_m=1235,
    E=83e9,
    nu=0.37,
    rho=10490,
    M_amu=107.868,
    E_bond_eV=2.95,
    # SSOC (PCC)
    n_d=10, period=5, mu_GPa=30.0, gamma_isf=16.0,
    # v7.0互換
    delta_L=0.10,
    f_d_elec=4/3,
    # τ/σ
    T_twin=1.0,
    R_comp=1.0,
    # 疲労
    alpha_thermal=1.89e-5,
    lambda_base=24.0,
    kappa=1.8,
    fG=0.101,
    A_int=1.0,
    color="#bcbd22",
)

Pt = Material(
    name="Pt",
    structure="FCC",
    a=3.924e-10,
    T_m=2041,
    E=168e9,
    nu=0.38,
    rho=21450,
    M_amu=195.084,
    E_bond_eV=5.84,
    # SSOC (PCC)
    n_d=9, period=6, mu_GPa=61.0, gamma_isf=322.0,
    # v7.0互換
    delta_L=0.10,
    f_d_elec=1.0,
    # 疲労
    alpha_thermal=8.8e-6,
    A_int=1.0,
    color="#c7c7c7",
)

Pd = Material(
    name="Pd",
    structure="FCC",
    a=3.891e-10,
    T_m=1828,
    E=121e9,
    nu=0.39,
    rho=12020,
    M_amu=106.42,
    E_bond_eV=3.89,
    # SSOC (PCC)
    n_d=10, period=5, mu_GPa=44.0, gamma_isf=180.0,
    # v7.0互換
    delta_L=0.10,
    f_d_elec=1.0,
    # 疲労
    alpha_thermal=1.18e-5,
    A_int=1.0,
    color="#dbdb8d",
)

Ir = Material(
    name="Ir",
    structure="FCC",
    a=3.839e-10,
    T_m=2719,
    E=528e9,
    nu=0.26,
    rho=22560,
    M_amu=192.217,
    E_bond_eV=6.94,
    # SSOC (PCC)
    n_d=7, period=6, mu_GPa=210.0, gamma_isf=300.0,
    # v7.0互換
    delta_L=0.10,
    f_d_elec=1.0,
    # 疲労
    alpha_thermal=6.4e-6,
    A_int=1.0,
    color="#9edae5",
)

Rh = Material(
    name="Rh",
    structure="FCC",
    a=3.803e-10,
    T_m=2237,
    E=380e9,
    nu=0.26,
    rho=12410,
    M_amu=102.906,
    E_bond_eV=5.75,
    # SSOC (PCC)
    n_d=8, period=5, mu_GPa=150.0, gamma_isf=350.0,
    # v7.0互換
    delta_L=0.10,
    f_d_elec=1.0,
    # 疲労
    alpha_thermal=8.2e-6,
    A_int=1.0,
    color="#f7b6d2",
)

Pb = Material(
    name="Pb",
    structure="FCC",
    a=4.951e-10,
    T_m=600,
    E=16e9,
    nu=0.44,
    rho=11340,
    M_amu=207.2,
    E_bond_eV=2.03,
    # SSOC (PCC)
    n_d=0, period=6, mu_GPa=5.6, gamma_isf=300.0,
    # v7.0互換
    delta_L=0.12,
    f_d_elec=1.0,
    # 疲労
    alpha_thermal=2.89e-5,
    A_int=0.50,
    color="#c49c94",
)


# --------------------------------------------------------------------------
# HCP 金属
# --------------------------------------------------------------------------

Ti = Material(
    name="Ti",
    structure="HCP",
    a=2.95e-10,
    T_m=1941,
    E=116e9,
    nu=0.32,
    rho=4500,
    M_amu=47.867,
    E_bond_eV=4.85,
    # SSOC (PCC)
    n_d=2, period=4, R_crss=0.65,
    c_a=1.587,
    # v7.0互換
    delta_L=0.10,
    f_d_elec=19/5,
    # τ/σ
    T_twin=1.0,
    R_comp=1.0,
    # 疲労
    alpha_thermal=8.60e-6,
    lambda_base=43.1,
    kappa=0.771,
    fG=0.101,
    A_int=1.10,
    color="#9467bd",
)

Mg = Material(
    name="Mg",
    structure="HCP",
    a=3.21e-10,
    T_m=923,
    E=45e9,
    nu=0.29,
    rho=1740,
    M_amu=24.305,
    E_bond_eV=1.51,
    # SSOC (PCC)
    n_d=0, period=3, R_crss=75.0,
    c_a=1.624,
    # v7.0互換
    delta_L=0.117,
    f_d_elec=82/15,
    # τ/σ
    T_twin=0.6,
    R_comp=0.6,
    # 疲労
    alpha_thermal=2.70e-5,
    lambda_base=7.5,
    kappa=37.568,
    fG=0.082,
    A_int=0.60,
    color="#8c564b",
)

Zn = Material(
    name="Zn",
    structure="HCP",
    a=2.66e-10,
    T_m=693,
    E=108e9,
    nu=0.25,
    rho=7140,
    M_amu=65.38,
    E_bond_eV=1.35,
    # SSOC (PCC)
    n_d=10, period=4, R_crss=2.0,
    c_a=1.856,
    # v7.0互換
    delta_L=0.12,
    f_d_elec=4/3,
    # τ/σ
    T_twin=0.9,
    R_comp=1.2,
    # 疲労
    alpha_thermal=3.02e-5,
    lambda_base=15.0,
    kappa=5.0,
    fG=0.075,
    A_int=0.75,
    color="#7f7f7f",
)

Zr = Material(
    name="Zr",
    structure="HCP",
    a=3.23e-10,
    T_m=2128,
    E=88e9,
    nu=0.34,
    rho=6520,
    M_amu=91.224,
    E_bond_eV=6.25,
    # SSOC (PCC)
    n_d=2, period=5, R_crss=0.50,
    c_a=1.593,
    # v7.0互換
    delta_L=0.10,
    f_d_elec=1.0,
    # 疲労
    alpha_thermal=5.7e-6,
    A_int=1.0,
    color="#e7969c",
)

Hf = Material(
    name="Hf",
    structure="HCP",
    a=3.19e-10,
    T_m=2506,
    E=78e9,
    nu=0.37,
    rho=13310,
    M_amu=178.49,
    E_bond_eV=6.44,
    # SSOC (PCC)
    n_d=2, period=6, R_crss=0.55,
    c_a=1.581,
    # v7.0互換
    delta_L=0.10,
    f_d_elec=1.0,
    # 疲労
    alpha_thermal=5.9e-6,
    A_int=1.0,
    color="#ce6dbd",
)

Re = Material(
    name="Re",
    structure="HCP",
    a=2.76e-10,
    T_m=3459,
    E=463e9,
    nu=0.30,
    rho=21020,
    M_amu=186.207,
    E_bond_eV=8.03,
    # SSOC (PCC)
    n_d=5, period=6, R_crss=5.0,
    c_a=1.615,
    # v7.0互換
    delta_L=0.10,
    f_d_elec=1.0,
    # 疲労
    alpha_thermal=6.2e-6,
    A_int=1.0,
    color="#de9ed6",
)

Cd = Material(
    name="Cd",
    structure="HCP",
    a=2.98e-10,
    T_m=594,
    E=50e9,
    nu=0.30,
    rho=8650,
    M_amu=112.414,
    E_bond_eV=1.16,
    # SSOC (PCC)
    n_d=10, period=5, R_crss=1.5,
    c_a=1.886,
    # v7.0互換
    delta_L=0.12,
    f_d_elec=1.0,
    # 疲労
    alpha_thermal=3.08e-5,
    A_int=0.60,
    color="#637939",
)

Ru = Material(
    name="Ru",
    structure="HCP",
    a=2.71e-10,
    T_m=2607,
    E=447e9,
    nu=0.30,
    rho=12370,
    M_amu=101.07,
    E_bond_eV=6.74,
    # SSOC (PCC)
    n_d=7, period=5, R_crss=3.0,
    c_a=1.582,
    # v7.0互換
    delta_L=0.10,
    f_d_elec=1.0,
    # 疲労
    alpha_thermal=6.4e-6,
    A_int=1.0,
    color="#8ca252",
)


# ==============================================================================
# 材料データベース辞書
# ==============================================================================

MATERIALS: Dict[str, Material] = {
    # BCC
    "Fe": Fe, "Iron": Fe, "SECD": Fe,
    "W": W, "Tungsten": W,
    "V": V_metal, "Vanadium": V_metal,
    "Cr": Cr, "Chromium": Cr,
    "Nb": Nb, "Niobium": Nb,
    "Mo": Mo, "Molybdenum": Mo,
    "Ta": Ta, "Tantalum": Ta,
    # FCC
    "Cu": Cu, "Copper": Cu, "FCC_Cu": Cu,
    "Al": Al, "Aluminum": Al,
    "Ni": Ni, "Nickel": Ni,
    "Au": Au, "Gold": Au,
    "Ag": Ag, "Silver": Ag,
    "Pt": Pt, "Platinum": Pt,
    "Pd": Pd, "Palladium": Pd,
    "Ir": Ir, "Iridium": Ir,
    "Rh": Rh, "Rhodium": Rh,
    "Pb": Pb, "Lead": Pb,
    # HCP
    "Ti": Ti, "Titanium": Ti,
    "Mg": Mg, "Magnesium": Mg,
    "Zn": Zn, "Zinc": Zn,
    "Zr": Zr, "Zirconium": Zr,
    "Hf": Hf, "Hafnium": Hf,
    "Re": Re, "Rhenium": Re,
    "Cd": Cd, "Cadmium": Cd,
    "Ru": Ru, "Ruthenium": Ru,
}


def get_material(name: str) -> Material:
    """名前から材料を取得"""
    if name not in MATERIALS:
        available = list_materials()
        raise ValueError(f"Unknown material: {name}. Available: {available}")
    return MATERIALS[name]


def list_materials() -> List[str]:
    """主要材料一覧（重複なし）"""
    seen = set()
    result = []
    for name, mat in MATERIALS.items():
        if mat.name not in seen:
            seen.add(mat.name)
            result.append(mat.name)
    return result


def list_by_structure(structure: str) -> List[str]:
    """構造別材料一覧"""
    return [name for name in list_materials()
            if MATERIALS[name].structure == structure]


# ==============================================================================
# テスト
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("統合材料データベース テスト (v10.0 SSOC)")
    print("=" * 60)
    
    print(f"\n[利用可能な材料] ({len(list_materials())}種)")
    print(f"  BCC ({len(list_by_structure('BCC'))}): {list_by_structure('BCC')}")
    print(f"  FCC ({len(list_by_structure('FCC'))}): {list_by_structure('FCC')}")
    print(f"  HCP ({len(list_by_structure('HCP'))}): {list_by_structure('HCP')}")
    
    print("\n[SSОCパラメータ確認]")
    print(f"  {'Metal':<4} {'Str':>3} {'n_d':>3} {'per':>3} {'grp':>3} {'sel':>3} "
          f"{'μ':>5} {'γ':>5} {'R':>5} {'√EkT':>10}")
    print(f"  {'-'*55}")
    for name in list_materials():
        mat = MATERIALS[name]
        print(f"  {name:<4} {mat.structure:>3} {mat.n_d:>3} {mat.period:>3} "
              f"{mat.group:>3} {mat.sel:>3} "
              f"{mat.mu_GPa:>5.1f} {mat.gamma_isf:>5.0f} {mat.R_crss:>5.2f} "
              f"{mat.sqrt_EkT:>10.4e}")
    
    print("\n✅ テスト完了!")
