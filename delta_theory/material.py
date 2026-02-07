"""
δ-theory 統合材料データベース (v7.0)

v7.0 変更点:
  - BD_RATIO_SQ = (b/d)² = 3/2 を純粋結晶幾何定数として分離
  - f_d → f_d_elec (電子構造寄与のみ)
  - 旧 f_d = BD_RATIO_SQ × f_d_elec（後方互換プロパティで提供）
  
  自己整合性の注記:
    E_bond（蒸発熱）と δ_L（Debye-Waller）はどちらも欠損を含む
    実物多結晶の測定値。欠損効果は部分的に相殺するため、
    別途の多結晶補正因子 α_c は不要。

Author: 環 & ご主人さま (飯泉真道)
Date: 2026-02-07
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
# d/b = sqrt(2/3) for BCC, FCC, HCP — identical across crystal structures.
# This is a pure crystallographic constant, NOT a fitting parameter.
# Physical meaning: ratio of slip-plane spacing (d) to Burgers vector (b).
BD_RATIO_SQ: float = 1.5  # (b/d)^2 = 3/2


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
        M_taylor=3.06,
        r_th=0.02,
        n_cl=7.0,
        desc="Face-Centered Cubic - 疲労限度なし",
    ),
    "HCP": StructurePreset(
        name="HCP",
        Z_bulk=12,
        alpha0=0.350,
        M_taylor=4.0,
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
    統合材料データベース (v7.0)
    
    全てのδ理論モジュールで使用する材料パラメータを一元管理
    
    v7.0: f_d を BD_RATIO_SQ × f_d_elec に分離
      - BD_RATIO_SQ = (b/d)² = 3/2 — 純粋結晶幾何定数
      - f_d_elec — 電子構造寄与のみ（d電子方向性）
    
    σ_y の全構成要素の分類:
      【純粋幾何】   α, (b/d)², V_act=b³, HP, M, 2π
      【実測セット】 E_bond(蒸発熱), δ_L(Debye-Waller) — 欠損込みで自己整合
      【電子物理】   f_d_elec のみ
    
    カテゴリ:
      [基本] 結晶構造、格子定数、弾性定数、熱物性
      [δ理論] Lindemann閾値、結合エネルギー、電子方向性
      [τ/σ] せん断/引張比、双晶因子、圧縮/引張比
      [粒界] 分離仕事、偏析エネルギー (DBT用)
      [疲労] 熱軟化、Born崩壊、A_int (疲労用)
      [表示] プロット色
    """
    
    # ==========================================================================
    # [必須] デフォルト値なし（先に定義）
    # ==========================================================================
    name: str                           # 材料名 (Fe, Cu, Al, ...)
    structure: Literal["BCC", "FCC", "HCP"]  # 結晶構造
    a: float                            # 格子定数 @ 300K [m]
    T_m: float                          # 融点 [K]
    E: float                            # ヤング率 [Pa]
    nu: float                           # ポアソン比
    rho: float                          # 密度 [kg/m³]
    M_amu: float                        # 原子質量 [amu]
    delta_L: float                      # Lindemann閾値（実測, Debye-Waller）
    E_bond_eV: float                    # 結合エネルギー [eV]（実測, 蒸発熱）
    f_d_elec: float                     # 電子方向性係数（d電子寄与のみ）
                                        # NOTE: 旧 f_d = BD_RATIO_SQ * f_d_elec
    
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
    E_seg_eV: float = 0.45              # 偏析エネルギー [eV] (Fe-P系デフォルト)
    
    # 疲労
    lambda_base: float = 25.0           # 熱軟化パラメータ
    kappa: float = 1.0                  # 非線形熱軟化係数
    fG: float = 0.10                    # Born崩壊係数 (融点での G/G₀)
    A_int: float = 1.0                  # 内部スケール (Fe=1.0基準)
    
    # 表示
    color: str = "#333333"              # プロット色
    
    # ==========================================================================
    # 計算プロパティ
    # ==========================================================================
    
    @property
    def preset(self) -> StructurePreset:
        """構造プリセットを取得"""
        return STRUCTURE_PRESETS[self.structure]
    
    @property
    def Z_bulk(self) -> int:
        """バルク配位数"""
        return self.preset.Z_bulk
    
    @property
    def alpha0(self) -> float:
        """幾何係数"""
        return self.preset.alpha0
    
    @property
    def M_taylor(self) -> float:
        """Taylor因子"""
        return self.preset.M_taylor
    
    @property
    def r_th(self) -> float:
        """疲労限度閾値"""
        return self.preset.r_th
    
    @property
    def n_cl(self) -> float:
        """Basquin指数"""
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
    def E_eff(self) -> float:
        """有効結合エネルギー [J]
        
        v7.0: E_eff = E_bond × α₀ × (b/d)² × f_d_elec
        
        構成要素の分類:
          E_bond   — 実測（蒸発熱, 欠損込み）
          α₀       — 純粋幾何（結合ベクトル射影）
          (b/d)²   — 純粋幾何（すべり面定数 = 3/2）
          f_d_elec — 電子物理（d電子方向性）
        """
        return self.E_bond_eV * eV_to_J * self.alpha0 * BD_RATIO_SQ * self.f_d_elec
    
    # ==========================================================================
    # 表示
    # ==========================================================================
    
    def __str__(self) -> str:
        return f"Material({self.name}, {self.structure}, T_m={self.T_m}K)"
    
    def summary(self) -> str:
        """詳細サマリー"""
        return f"""
{'='*60}
Material: {self.name} ({self.structure})  [v7.0 geometric factorization]
{'='*60}
  [基本]
    a          = {self.a*1e10:.3f} Å
    T_m        = {self.T_m:.0f} K
    E          = {self.E/1e9:.0f} GPa
    G          = {self.G/1e9:.1f} GPa
    ν          = {self.nu}
    ρ          = {self.rho} kg/m³
  
  [δ理論]  σ_y = (E_bond × α × (b/d)² × f_d_elec / V_act) × (δ_L × HP / 2πM)
    δ_L        = {self.delta_L}        [実測: Debye-Waller, 欠損込み]
    E_bond     = {self.E_bond_eV} eV   [実測: 蒸発熱, 欠損込み]
    (b/d)²     = {BD_RATIO_SQ}         [幾何: すべり面定数]
    f_d_elec   = {self.f_d_elec:.4f}   [電子: d電子方向性]
    f_d(旧)    = {self.f_d:.4f}        [= (b/d)² × f_d_elec, 後方互換]
    α₀         = {self.alpha0}         [幾何: preset]
    b          = {self.b*1e10:.3f} Å
  
  [τ/σ]
    T_twin     = {self.T_twin}
    R_comp     = {self.R_comp}
    A_texture  = {self.A_texture}
  
  [疲労]
    r_th       = {self.r_th} (preset)
    n_cl       = {self.n_cl} (preset)
    A_int      = {self.A_int}
    λ_base     = {self.lambda_base}
    fG         = {self.fG}
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
    alpha_thermal=1.50e-5,
    E=211e9,
    nu=0.29,
    rho=7870,
    M_amu=55.845,
    # δ理論
    delta_L=0.18,
    E_bond_eV=4.28,
    f_d_elec=1.0,          # was f_d=1.5 → 1.5/1.5 = 1.0 (基準点!)
    # τ/σ
    T_twin=1.0,
    R_comp=1.0,
    # 粒界
    W_sep0=2.0,
    delta_sep=0.2e-9,
    E_seg_eV=0.45,
    # 疲労
    lambda_base=49.2,
    kappa=0.573,
    fG=0.027,
    A_int=1.0,
    # 表示
    color="#1f77b4",
)

W = Material(
    name="W",
    structure="BCC",
    a=3.16e-10,
    T_m=3695,
    alpha_thermal=4.51e-6,
    E=411e9,
    nu=0.28,
    rho=19300,
    M_amu=183.84,
    # δ理論
    delta_L=0.16,
    E_bond_eV=8.90,
    f_d_elec=47/15,        # was f_d=4.7 → 4.7/1.5 = 47/15
    # τ/σ
    T_twin=1.0,
    R_comp=1.0,
    # 疲労
    lambda_base=10.9,
    kappa=2.759,
    fG=0.021,
    A_int=0.85,
    # 表示
    color="#17becf",
)

# --------------------------------------------------------------------------
# FCC 金属
# --------------------------------------------------------------------------

Cu = Material(
    name="Cu",
    structure="FCC",
    a=3.61e-10,
    T_m=1357,
    alpha_thermal=1.70e-5,
    E=130e9,
    nu=0.34,
    rho=8960,
    M_amu=63.546,
    # δ理論
    delta_L=0.10,
    E_bond_eV=3.49,
    f_d_elec=4/3,          # was f_d=2.0 → 2.0/1.5 = 4/3
    # τ/σ
    T_twin=1.0,
    R_comp=1.0,
    # 疲労
    lambda_base=26.3,
    kappa=1.713,
    fG=0.101,
    A_int=1.41,
    # 表示
    color="#d62728",
)

Al = Material(
    name="Al",
    structure="FCC",
    a=4.05e-10,
    T_m=933,
    alpha_thermal=2.30e-5,
    E=70e9,
    nu=0.35,
    rho=2700,
    M_amu=26.982,
    # δ理論
    delta_L=0.10,
    E_bond_eV=3.39,
    f_d_elec=16/15,        # was f_d=1.6 → 1.6/1.5 = 16/15
    # τ/σ
    T_twin=1.0,
    R_comp=1.0,
    # 疲労
    lambda_base=27.3,
    kappa=4.180,
    fG=0.101,
    A_int=0.71,
    # 表示
    color="#2ca02c",
)

Ni = Material(
    name="Ni",
    structure="FCC",
    a=3.52e-10,
    T_m=1728,
    alpha_thermal=1.30e-5,
    E=200e9,
    nu=0.31,
    rho=8900,
    M_amu=58.693,
    # δ理論
    delta_L=0.11,
    E_bond_eV=4.44,
    f_d_elec=26/15,        # was f_d=2.6 → 2.6/1.5 = 26/15
    # τ/σ
    T_twin=1.0,
    R_comp=1.0,
    # 疲労
    lambda_base=22.6,
    kappa=0.279,
    fG=0.092,
    A_int=1.37,
    # 表示
    color="#ff7f0e",
)

Au = Material(
    name="Au",
    structure="FCC",
    a=4.08e-10,
    T_m=1337,
    alpha_thermal=1.42e-5,
    E=79e9,
    nu=0.44,
    rho=19300,
    M_amu=196.967,
    # δ理論
    delta_L=0.10,
    E_bond_eV=3.81,
    f_d_elec=11/15,        # was f_d=1.1 → 1.1/1.5 = 11/15
    # τ/σ
    T_twin=1.0,
    R_comp=1.0,
    # 疲労
    lambda_base=25.0,
    kappa=1.5,
    fG=0.101,
    A_int=1.0,
    # 表示
    color="#e377c2",
)

Ag = Material(
    name="Ag",
    structure="FCC",
    a=4.09e-10,
    T_m=1235,
    alpha_thermal=1.89e-5,
    E=83e9,
    nu=0.37,
    rho=10490,
    M_amu=107.868,
    # δ理論
    delta_L=0.10,
    E_bond_eV=2.95,
    f_d_elec=4/3,          # was f_d=2.0 → 2.0/1.5 = 4/3
    # τ/σ
    T_twin=1.0,
    R_comp=1.0,
    # 疲労
    lambda_base=24.0,
    kappa=1.8,
    fG=0.101,
    A_int=1.0,
    # 表示
    color="#bcbd22",
)

# --------------------------------------------------------------------------
# HCP 金属
# --------------------------------------------------------------------------

Ti = Material(
    name="Ti",
    structure="HCP",
    a=2.95e-10,
    T_m=1941,
    alpha_thermal=8.60e-6,
    E=116e9,
    nu=0.32,
    rho=4500,
    M_amu=47.867,
    # δ理論
    delta_L=0.10,
    E_bond_eV=4.85,
    f_d_elec=19/5,         # was f_d=5.7 → 5.7/1.5 = 19/5
    # τ/σ
    c_a=1.587,
    T_twin=1.0,
    R_comp=1.0,
    # 疲労
    lambda_base=43.1,
    kappa=0.771,
    fG=0.101,
    A_int=1.10,
    # 表示
    color="#9467bd",
)

Mg = Material(
    name="Mg",
    structure="HCP",
    a=3.21e-10,
    T_m=923,
    alpha_thermal=2.70e-5,
    E=45e9,
    nu=0.29,
    rho=1740,
    M_amu=24.305,
    # δ理論
    delta_L=0.117,
    E_bond_eV=1.51,
    f_d_elec=82/15,        # was f_d=8.2 → 8.2/1.5 = 82/15
    # τ/σ
    c_a=1.624,
    T_twin=0.6,   # 双晶活性
    R_comp=0.6,   # 圧縮異方性
    # 疲労
    lambda_base=7.5,
    kappa=37.568,
    fG=0.082,
    A_int=0.60,
    # 表示
    color="#8c564b",
)

Zn = Material(
    name="Zn",
    structure="HCP",
    a=2.66e-10,
    T_m=693,
    alpha_thermal=3.02e-5,
    E=108e9,
    nu=0.25,
    rho=7140,
    M_amu=65.38,
    # δ理論
    delta_L=0.12,
    E_bond_eV=1.35,
    f_d_elec=4/3,          # was f_d=2.0 → 2.0/1.5 = 4/3
    # τ/σ
    c_a=1.856,
    T_twin=0.9,
    R_comp=1.2,
    # 疲労
    lambda_base=15.0,
    kappa=5.0,
    fG=0.075,
    A_int=0.75,
    # 表示
    color="#7f7f7f",
)


# ==============================================================================
# 材料データベース辞書
# ==============================================================================

MATERIALS: Dict[str, Material] = {
    # BCC
    "Fe": Fe, "Iron": Fe, "SECD": Fe,
    "W": W, "Tungsten": W,
    # FCC
    "Cu": Cu, "Copper": Cu, "FCC_Cu": Cu,
    "Al": Al, "Aluminum": Al,
    "Ni": Ni, "Nickel": Ni,
    "Au": Au, "Gold": Au,
    "Ag": Ag, "Silver": Ag,
    # HCP
    "Ti": Ti, "Titanium": Ti,
    "Mg": Mg, "Magnesium": Mg,
    "Zn": Zn, "Zinc": Zn,
}


def get_material(name: str) -> Material:
    """名前から材料を取得"""
    if name not in MATERIALS:
        available = list_materials()
        raise ValueError(f"Unknown material: {name}. Available: {available}")
    return MATERIALS[name]


def list_materials() -> List[str]:
    """主要材料一覧"""
    return ["Fe", "W", "Cu", "Al", "Ni", "Au", "Ag", "Ti", "Mg", "Zn"]


def list_by_structure(structure: str) -> List[str]:
    """構造別材料一覧"""
    return [name for name in list_materials() 
            if MATERIALS[name].structure == structure]


# ==============================================================================
# 後方互換: MaterialGPU クラス (既存コードとの互換用)
# ==============================================================================

class MaterialGPU:
    """
    後方互換用ラッパー
    
    既存の materials.py を使用しているコードのために、
    同じインターフェースを提供
    """
    
    @classmethod
    def Fe(cls) -> Material:
        return Fe
    
    @classmethod
    def W(cls) -> Material:
        return W
    
    @classmethod
    def Cu(cls) -> Material:
        return Cu
    
    @classmethod
    def Al(cls) -> Material:
        return Al
    
    @classmethod
    def Ni(cls) -> Material:
        return Ni
    
    @classmethod
    def Au(cls) -> Material:
        return Au
    
    @classmethod
    def Ag(cls) -> Material:
        return Ag
    
    @classmethod
    def Ti(cls) -> Material:
        return Ti
    
    @classmethod
    def Mg(cls) -> Material:
        return Mg
    
    @classmethod
    def Zn(cls) -> Material:
        return Zn
    
    # エイリアス
    @classmethod
    def SECD(cls) -> Material:
        return Fe
    
    @classmethod
    def Iron(cls) -> Material:
        return Fe
    
    @classmethod
    def Copper(cls) -> Material:
        return Cu
    
    @classmethod
    def Aluminum(cls) -> Material:
        return Al
    
    @classmethod
    def FCC_Cu(cls) -> Material:
        return Cu
    
    @classmethod
    def list_materials(cls) -> List[str]:
        return list_materials()
    
    @classmethod
    def get(cls, name: str) -> Material:
        return get_material(name)


# ==============================================================================
# テスト
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("統合材料データベース テスト (v7.0)")
    print("=" * 60)
    
    print(f"\n[v7.0] BD_RATIO_SQ = (b/d)² = {BD_RATIO_SQ}")
    
    print("\n[利用可能な材料]")
    print(f"  全体: {list_materials()}")
    print(f"  BCC:  {list_by_structure('BCC')}")
    print(f"  FCC:  {list_by_structure('FCC')}")
    print(f"  HCP:  {list_by_structure('HCP')}")
    
    print("\n[構造プリセット]")
    for name, preset in STRUCTURE_PRESETS.items():
        print(f"  {name}: r_th={preset.r_th}, n_cl={preset.n_cl}, α₀={preset.alpha0}")
    
    # v7.0 検算: f_d_elec × BD_RATIO_SQ == 旧f_d
    print("\n[v7.0 検算: f_d_elec × (b/d)² = 旧f_d]")
    old_f_d = {'Fe':1.5, 'W':4.7, 'Cu':2.0, 'Al':1.6, 'Ni':2.6,
               'Au':1.1, 'Ag':2.0, 'Ti':5.7, 'Mg':8.2, 'Zn':2.0}
    
    print(f"  {'Metal':<5} {'f_d_elec':>10} {'f_d(prop)':>10} {'f_d(old)':>10} {'match':>6}")
    print(f"  {'-'*45}")
    all_ok = True
    for name in list_materials():
        mat = get_material(name)
        fd_prop = mat.f_d  # 後方互換プロパティ
        fd_old = old_f_d[name]
        ok = abs(fd_prop - fd_old) < 1e-10
        all_ok &= ok
        print(f"  {name:<5} {mat.f_d_elec:>10.6f} {fd_prop:>10.4f} {fd_old:>10.4f} {'✓' if ok else '✗':>6}")
    print(f"\n  後方互換: {'✓ ALL PASS' if all_ok else '✗ FAIL'}")
    
    # σ_base 計算テスト
    print("\n[σ_base 計算テスト (300K)]")
    print(f"  {'Metal':<5} {'E_bond':>6} {'α':>6} {'(b/d)²':>6} {'f_d_e':>6} {'δ_L':>6} {'HP':>6} {'σ_y':>8}")
    print(f"  {'-'*50}")
    for name in list_materials():
        mat = get_material(name)
        HP = max(0, 1 - 300/mat.T_m)
        sigma = (mat.E_eff / mat.V_act) * mat.delta_L * HP / (2 * PI * mat.M_taylor) / 1e6
        print(f"  {name:<5} {mat.E_bond_eV:>6.2f} {mat.alpha0:>6.3f} {BD_RATIO_SQ:>6.1f} "
              f"{mat.f_d_elec:>6.3f} {mat.delta_L:>6.3f} {HP:>6.3f} {sigma:>8.1f}")
    
    print("\n[材料サマリー: Fe]")
    print(Fe.summary())
    
    print("\n✅ テスト完了!")
