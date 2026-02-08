"""
SSOC (Structure-Selective Orbital Coupling) — f_de計算モジュール

δ_Lフリー統一降伏応力の核心: f_de の構造依存計算ロジック

設計原則:
  - material.py = データ層（物質固有パラメータ）
  - ssoc.py     = 計算層（SSOC f_de + σ_base）  ← このモジュール
  - unified_yield_fatigue = 応用層（σ_y → S-N統合）

統一式:
  σ_y = (8√5/5πMZ) × α₀ × (b/d)² × f_de × √(E_coh·k_B·T_m) / V_act × HP

  f_de^(s) = (X_s / X_ref)^(2/3 · g_d) × f_aux^(s)
    2/3 = 面→体積の幾何的次元変換指数（全構造共通）

  PCC (FCC, HCP): 入力場と応答が分離可能
  SCC (BCC):      場と応答が不可分（SCF解の縮約）

Author: 飯泉真道
Date: 2026-02-08
================================================================================
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from .material import Material, BD_RATIO_SQ, COEFF_V10, eV_to_J, k_B, PI

# ==============================================================================
# SSOC 共通定数
# ==============================================================================
P_DIM: float = 2.0 / 3.0    # 面(2D)→体積(3D) 次元変換 = 2/3（全構造共通）

# v10.0ではMを全構造統一 = 3.0
# （構造差はf_deのSSOCが吸収するため、Mは多結晶平均の汎用値）
# material.pyのM_taylorはv7.0互換で構造依存値を保持
M_SSOC: float = 3.0


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FCC — PCC (Perturbative-Coupled Channel)                          ║
# ║  チャンネル X_FCC = μ (剪断弾性率)                                   ║
# ║  f_de = f_μ × f_shell × f_core                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

FCC_MU_REF    = 43.6      # GPa, Cu基準
FCC_GAMMA_REF = 48.2      # mJ/m², Cu基準
FCC_SHELL_A0, FCC_SHELL_W0 = 0.25, 0.35
FCC_SHELL_A1, FCC_SHELL_W1 = 0.28, 0.40


def fcc_gate(n_d: int) -> float:
    """FCC d軌道ゲート g_d (PCC: 離散二値)
    d-metals (n_d ≥ 2): g_d = 1
    sp-metals (n_d < 2): g_d = 0
    """
    return 1.0 if n_d >= 2 else 0.0


def fcc_f_mu(mu_GPa: float, g_d: float) -> float:
    """FCC メインチャンネル: 剪断弾性率スケーリング
    f_μ = (μ / μ_ref)^(P_DIM × g_d)
    """
    return (mu_GPa / FCC_MU_REF) ** (P_DIM * g_d)


def fcc_f_shell(n_d: int, period: int) -> float:
    """FCC 補助チャンネル①: 殻閉殻補正"""
    holes = 10 - n_d
    if period <= 4 or n_d == 0:
        return 1.0
    return max(0.4, 1.0
               - FCC_SHELL_A0 * np.exp(-holes**2 / (2 * FCC_SHELL_W0**2))
               - FCC_SHELL_A1 * np.exp(-(holes - 1)**2 / (2 * FCC_SHELL_W1**2)))


def fcc_f_core(gamma_isf: float, g_d: float) -> float:
    """FCC 補助チャンネル②: 積層欠陥エネルギー補正
    γ_isf < γ_ref → 部分転位拡張 → コア抵抗増加
    """
    if gamma_isf < FCC_GAMMA_REF:
        return (FCC_GAMMA_REF / gamma_isf) ** (P_DIM * g_d)
    return 1.0


def fcc_f_de(mat: Material) -> float:
    """FCC SSOC f_de — PCC 3層モデル
    f_de = f_μ(main) × f_shell(aux) × f_core(aux)
    """
    g_d = fcc_gate(mat.n_d)
    return (fcc_f_mu(mat.mu_GPa, g_d)
            * fcc_f_shell(mat.n_d, mat.period)
            * fcc_f_core(mat.gamma_isf, g_d))


def fcc_f_de_detail(mat: Material) -> Dict[str, float]:
    """FCC f_de の内訳を返す（診断用）"""
    g_d = fcc_gate(mat.n_d)
    f_mu = fcc_f_mu(mat.mu_GPa, g_d)
    f_sh = fcc_f_shell(mat.n_d, mat.period)
    f_co = fcc_f_core(mat.gamma_isf, g_d)
    return {
        'g_d': g_d,
        'f_mu': f_mu,
        'f_shell': f_sh,
        'f_core': f_co,
        'f_de': f_mu * f_sh * f_co,
    }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  BCC — SCC (Self-Consistent Channel)                               ║
# ║  チャンネル X_BCC = ΔE_P (Peierls障壁, 自己生成)                    ║
# ║  f_de = f_JT × f_5d × f_lattice                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

BCC_DELTA_JT = 0.9    # d⁴ Jahn-Teller anomaly
BCC_F_5D = 1.5         # 5d相対論補正

BCC_LATTICE = {
    6: {'f_lat': 1.05},
    8: {'f_lat': 1.05},
    5: {'sel1': 0.33,
        'sel2_base': 0.85,
        'sel2_5d': 0.70},
}


def bcc_f_jt(n_d: int) -> float:
    """BCC SCC Layer 1: d⁴ Jahn-Teller anomaly"""
    return 1.0 + BCC_DELTA_JT if n_d == 4 else 1.0


def bcc_f_5d(n_d: int, period: int) -> float:
    """BCC SCC Layer 2: 相対論的軌道拡張"""
    if period == 6 and n_d not in (0, 10):
        return BCC_F_5D
    return 1.0


def bcc_f_lattice(group: int, sel: int, period: int) -> float:
    """BCC SCC Layer 3: s-d混成チャンネル"""
    if group in (6, 8):
        return BCC_LATTICE[group]['f_lat']
    if group == 5:
        g5 = BCC_LATTICE[5]
        if sel == 1:
            return g5['sel1']
        base = g5['sel2_base']
        if period == 6:
            base *= g5['sel2_5d']
        return base
    return 1.0


def bcc_f_de(mat: Material) -> float:
    """BCC SSOC f_de — SCC 3層モデル
    f_de = f_JT × f_5d × f_lat
    """
    return (bcc_f_jt(mat.n_d)
            * bcc_f_5d(mat.n_d, mat.period)
            * bcc_f_lattice(mat.group, mat.sel, mat.period))


def bcc_f_de_detail(mat: Material) -> Dict[str, float]:
    """BCC f_de の内訳を返す（診断用）"""
    f_jt = bcc_f_jt(mat.n_d)
    f_5d = bcc_f_5d(mat.n_d, mat.period)
    f_lat = bcc_f_lattice(mat.group, mat.sel, mat.period)
    return {
        'f_jt': f_jt,
        'f_5d': f_5d,
        'f_lat': f_lat,
        'f_de': f_jt * f_5d * f_lat,
    }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HCP — PCC (Perturbative-Coupled Channel)                          ║
# ║  チャンネル X_HCP = R (CRSS異方性比)                                 ║
# ║  f_de = f_elec × f_aniso(R) × f_ca × f_5d                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

HCP_CA_IDEAL = 1.633
HCP_CA_COEFF = -4.00

HCP_B_PRIS   = 0.20
HCP_A_GATE   = 1.93
HCP_KG_GATE  = 18
HCP_X0_GATE  = 0.29

HCP_5D_K0    = 0.495
HCP_5D_K1    = -0.010

# d電子基底テーブル (BCC旧テーブルと共用)
_HCP_F_ELEC_BASE = {
    0: 1.0, 1: 1.0, 2: 2.5, 3: 1.0, 4: 1.9, 5: 1.0,
    6: 1.0, 7: 1.0, 8: 1.5, 9: 1.0, 10: 1.0,
}


def hcp_f_aniso(R: float) -> float:
    """HCP PCC メインチャンネル: CRSS異方性ゲート"""
    if R < 1:
        return 1.0 + HCP_B_PRIS * (1.0 - R)
    lr = np.log10(R)
    gate = 1.0 / (1.0 + np.exp(-HCP_KG_GATE * (lr - HCP_X0_GATE)))
    return 1.0 + HCP_A_GATE * gate * lr


def hcp_f_ca(ca: float) -> float:
    """HCP PCC 補助チャンネル①: c/a片側拘束"""
    dca = ca - HCP_CA_IDEAL
    return 1.0 + HCP_CA_COEFF * dca if dca < 0 else 1.0


def hcp_f_5d(n_d: int, period: int) -> float:
    """HCP PCC 補助チャンネル②: 5d相対論補正"""
    if period == 6:
        return max(0.1, HCP_5D_K0 + HCP_5D_K1 * n_d)
    return 1.0


def hcp_f_elec(n_d: int, period: int) -> float:
    """HCP d電子因子"""
    b = _HCP_F_ELEC_BASE.get(n_d, 1.0)
    if period == 6 and n_d not in (0, 10):
        b *= 1.5
    return b


def hcp_f_de(mat: Material) -> float:
    """HCP SSOC f_de — PCC 3層モデル
    f_de = f_elec × f_aniso(R) × f_ca × f_5d
    """
    return (hcp_f_elec(mat.n_d, mat.period)
            * hcp_f_aniso(mat.R_crss)
            * hcp_f_ca(mat.c_a)
            * hcp_f_5d(mat.n_d, mat.period))


def hcp_f_de_detail(mat: Material) -> Dict[str, float]:
    """HCP f_de の内訳を返す（診断用）"""
    f_e = hcp_f_elec(mat.n_d, mat.period)
    f_R = hcp_f_aniso(mat.R_crss)
    f_c = hcp_f_ca(mat.c_a)
    f_5d = hcp_f_5d(mat.n_d, mat.period)
    return {
        'f_elec': f_e,
        'f_aniso': f_R,
        'f_ca': f_c,
        'f_5d': f_5d,
        'f_de': f_e * f_R * f_c * f_5d,
    }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  統一ディスパッチ                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

def calc_f_de(mat: Material) -> float:
    """材料のf_deを構造に応じて計算（統一エントリポイント）"""
    if mat.structure == 'FCC':
        return fcc_f_de(mat)
    elif mat.structure == 'BCC':
        return bcc_f_de(mat)
    elif mat.structure == 'HCP':
        return hcp_f_de(mat)
    else:
        raise ValueError(f"Unknown structure: {mat.structure}")


def calc_f_de_detail(mat: Material) -> Dict[str, float]:
    """f_deの内訳を返す（診断用）"""
    if mat.structure == 'FCC':
        return fcc_f_de_detail(mat)
    elif mat.structure == 'BCC':
        return bcc_f_de_detail(mat)
    elif mat.structure == 'HCP':
        return hcp_f_de_detail(mat)
    else:
        raise ValueError(f"Unknown structure: {mat.structure}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  σ_base (v10.0 SSOC)                                               ║
# ║  σ = (COEFF/M·Z) × α₀ × BD2 × f_de × √(E·kT) / V × HP           ║
# ╚══════════════════════════════════════════════════════════════════════╝

def sigma_base_v10(mat: Material, T_K: float = 300.0) -> float:
    """v10.0 SSOC 基底降伏応力 [MPa]
    
    σ_y = (8√5/5πMZ) × α₀ × (b/d)² × f_de × √(E_coh·k_B·T_m) / V_act × HP
    
    Note: M = M_SSOC = 3.0 (全構造統一、v10.0仕様)
    """
    HP = max(0.0, 1.0 - T_K / mat.T_m)
    f_de = calc_f_de(mat)
    Z = mat.Z_bulk
    
    sigma = (COEFF_V10 / (M_SSOC * Z)
             * mat.alpha0 * BD_RATIO_SQ * f_de
             * mat.sqrt_EkT
             / mat.V_act * HP)
    
    return sigma / 1e6


def sigma_base_v10_with_fde(
    mat: Material,
    f_de: float,
    T_K: float = 300.0,
) -> float:
    """外部からf_deを渡す版（テスト・検証用）"""
    HP = max(0.0, 1.0 - T_K / mat.T_m)
    Z = mat.Z_bulk
    
    sigma = (COEFF_V10 / (M_SSOC * Z)
             * mat.alpha0 * BD_RATIO_SQ * f_de
             * mat.sqrt_EkT
             / mat.V_act * HP)
    
    return sigma / 1e6


def inverse_f_de(mat: Material, sigma_exp_MPa: float, T_K: float = 300.0) -> float:
    """実験値からf_deを逆算"""
    HP = max(0.0, 1.0 - T_K / mat.T_m)
    if HP <= 0:
        return float('inf')
    
    Z = mat.Z_bulk
    
    return (sigma_exp_MPa * 1e6 * mat.V_act * M_SSOC * Z
            / (COEFF_V10 * mat.alpha0 * BD_RATIO_SQ * mat.sqrt_EkT * HP))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  テスト                                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    from .material import MATERIALS, list_materials, list_by_structure
    
    print("=" * 95)
    print("  SSOC f_de計算モジュール テスト")
    print("=" * 95)
    
    for struct in ['FCC', 'BCC', 'HCP']:
        metals = list_by_structure(struct)
        mode = 'PCC' if struct != 'BCC' else 'SCC'
        print(f"\n[{struct} — {mode}]")
        print(f"  {'El':<4} {'f_de':>8} {'σ_calc':>8} MPa")
        print(f"  {'-'*25}")
        for name in metals:
            mat = MATERIALS[name]
            f_de = calc_f_de(mat)
            sig = sigma_base_v10(mat)
            print(f"  {name:<4} {f_de:>8.4f} {sig:>8.1f}")
    
    print("\n✅ テスト完了!")
