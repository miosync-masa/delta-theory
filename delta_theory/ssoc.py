"""
SSOC (Structure-Selective Orbital Coupling) — f_de計算モジュール v10.1

δ_Lフリー統一降伏応力の核心: f_de の構造依存計算ロジック

v10.1: 37金属対応ゲート拡張
  ① f_elec[1]: d¹方向性ゲート (period依存)
  ② sp共有結合ゲート (Be: period≤2 軽元素)
  ③ BCC Group 7 + 複雑構造ゲート (α-Mn)
  ④ sp金属ゲート統合 (p-block d¹⁰: Sn)
  ⑤ ランタノイド 4f結晶場ゲート (Ce, Nd)
  ⑥ FCC p-block ゲート (In)

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

Author: 飯泉真道 & 環
Date: 2026-02-13
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
M_SSOC: float = 3.0


# ==============================================================================
# 共通ユーティリティ (v10.1)
# ==============================================================================

def _get_n_f(mat: Material) -> int:
    """Material から n_f を安全に取得（後方互換）"""
    return getattr(mat, 'n_f', 0)


def _get_group(mat: Material) -> int | None:
    """Material から group を安全に取得（後方互換）"""
    return getattr(mat, 'group', None)


def _get_n_atoms_cell(mat: Material) -> int:
    """Material から n_atoms_cell を安全に取得（後方互換）"""
    return getattr(mat, 'n_atoms_cell', 2)


def is_p_block(group: int | None) -> bool:
    """p-block判定: group 13-16"""
    return group is not None and group >= 13


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FCC — PCC (Perturbative-Coupled Channel)                          ║
# ║  チャンネル X_FCC = μ (剪断弾性率)                                   ║
# ║  v10.0: f_de = f_μ × f_shell × f_core                              ║
# ║  v10.1: f_de = f_μ × f_shell × f_core × f_lanthanide               ║
# ╚══════════════════════════════════════════════════════════════════════╝

FCC_MU_REF    = 43.6      # GPa, Cu基準
FCC_GAMMA_REF = 48.2      # mJ/m², Cu基準
FCC_SHELL_A0, FCC_SHELL_W0 = 0.25, 0.35
FCC_SHELL_A1, FCC_SHELL_W1 = 0.28, 0.40

# --- v10.1 ランタノイド定数 ---
LANT_K_4F    = 0.4226     # f_4f = 1 + k_f × n_f_eff (Nd校正)
LANT_F_5D1   = 2.5769     # 5d¹電子の方向性寄与 (Ce校正)


def fcc_gate(n_d: int, n_f: int = 0, group: int | None = None) -> float:
    """FCC d軌道ゲート g_d (PCC: 離散二値)

    v10.0: d-metals (n_d ≥ 2) → 1, sp-metals (n_d < 2) → 0
    v10.1: + p-block d¹⁰閉殻 → 0 (方向性なし)
           + ランタノイド 4f → 別チャンネル (g_d=0のまま)
    """
    if is_p_block(group) and n_d == 10:
        return 0.0
    return 1.0 if n_d >= 2 else 0.0


def fcc_f_mu(mu_GPa: float, g_d: float) -> float:
    """FCC メインチャンネル: 剪断弾性率スケーリング
    f_μ = (μ / μ_ref)^(P_DIM × g_d)
    """
    return (mu_GPa / FCC_MU_REF) ** (P_DIM * g_d)


def fcc_f_shell(n_d: int, period: int, group: int | None = None) -> float:
    """FCC 補助チャンネル①: 殻閉殻補正

    v10.1: p-block → 1.0 (d殻影響なし)
    """
    holes = 10 - n_d
    if period <= 4 or n_d == 0:
        return 1.0
    if is_p_block(group):
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


def fcc_f_lanthanide(n_f: int, n_d: int) -> float:
    """v10.1: ランタノイド 4f結晶場ゲート (FCC)

    f_4f = 1 + k_f × n_f_eff  (n_f_eff = min(n_f, 14-n_f))
    f_d_contrib: 5d電子がある場合の追加方向性寄与

    ランタノイドでない場合 (n_f=0) → 1.0 (既存金属に影響なし)
    """
    if n_f == 0:
        return 1.0
    n_f_eff = n_f if n_f <= 7 else 14 - n_f
    f_4f = 1.0 + LANT_K_4F * n_f_eff
    f_d = LANT_F_5D1 if n_d >= 1 else 1.0
    return f_4f * f_d


def fcc_f_de(mat: Material) -> float:
    """FCC SSOC f_de — PCC v10.1
    f_de = f_μ(main) × f_shell(aux) × f_core(aux) × f_lanthanide(aux)
    """
    n_f = _get_n_f(mat)
    group = _get_group(mat)

    g_d = fcc_gate(mat.n_d, n_f=n_f, group=group)
    return (fcc_f_mu(mat.mu_GPa, g_d)
            * fcc_f_shell(mat.n_d, mat.period, group=group)
            * fcc_f_core(mat.gamma_isf, g_d)
            * fcc_f_lanthanide(n_f, mat.n_d))


def fcc_f_de_detail(mat: Material) -> Dict[str, float]:
    """FCC f_de の内訳を返す（診断用）"""
    n_f = _get_n_f(mat)
    group = _get_group(mat)

    g_d = fcc_gate(mat.n_d, n_f=n_f, group=group)
    f_mu = fcc_f_mu(mat.mu_GPa, g_d)
    f_sh = fcc_f_shell(mat.n_d, mat.period, group=group)
    f_co = fcc_f_core(mat.gamma_isf, g_d)
    f_la = fcc_f_lanthanide(n_f, mat.n_d)
    return {
        'g_d': g_d,
        'f_mu': f_mu,
        'f_shell': f_sh,
        'f_core': f_co,
        'f_lanthanide': f_la,
        'f_de': f_mu * f_sh * f_co * f_la,
    }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  BCC — SCC (Self-Consistent Channel)                               ║
# ║  チャンネル X_BCC = ΔE_P (Peierls障壁, 自己生成)                    ║
# ║  v10.0: f_de = f_JT × f_5d × f_lattice                             ║
# ║  v10.1: f_de = f_JT × f_5d × f_lattice × f_complex                 ║
# ║         + sp統合 (p-block d¹⁰)                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

BCC_DELTA_JT = 0.9         # d⁴ Jahn-Teller anomaly
BCC_F_5D = 1.5             # 5d相対論補正

BCC_F_SP_BASE  = 0.10      # 純sp (Li, Na)
BCC_F_SP_MIN   = 0.05
BCC_F_SP_MAX   = 0.20
BCC_F_P_BLOCK  = 0.80      # v10.1: d¹⁰ p-block (Sn)
BCC_P_COMPLEX  = 0.25      # v10.1: 複雑構造指数

BCC_LATTICE = {
    5: {'sel1': 0.33,
        'sel2_base': 0.85,
        'sel2_5d': 0.70},
    6: {'f_lat': 1.05},
    7: {'f_lat': 1.05},    # v10.1: Group 7 (Mn)
    8: {'f_lat': 1.05},
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
    if group in BCC_LATTICE:
        entry = BCC_LATTICE[group]
        if 'f_lat' in entry:
            return entry['f_lat']
        if group == 5:
            if sel == 1:
                return entry['sel1']
            base = entry['sel2_base']
            if period == 6:
                base *= entry['sel2_5d']
            return base
    return 1.0


def bcc_f_complex(n_atoms_cell: int) -> float:
    """v10.1: 複雑構造ゲート

    α-Mn: 58 atoms/cell → f = (58/2)^0.25 ≈ 2.32
    通常BCC: 2 atoms/cell → f = 1.0
    """
    if n_atoms_cell <= 2:
        return 1.0
    return (n_atoms_cell / 2) ** BCC_P_COMPLEX


def bcc_f_sp(n_d: int, period: int = 0,
             group: int | None = None) -> float:
    """BCC sp-metal branch (v10.1 統合版)

    純sp (Li, Na):         n_d < 2           → 0.10
    p-block d¹⁰ (Sn):     n_d=10, group≥13  → 0.80
    d-metal:               otherwise         → 1.0 (SCC本流)
    """
    if n_d < 2:
        return float(np.clip(BCC_F_SP_BASE, BCC_F_SP_MIN, BCC_F_SP_MAX))
    if n_d == 10 and is_p_block(group):
        return BCC_F_P_BLOCK
    return 1.0


def bcc_f_de(mat: Material) -> float:
    """BCC SSOC f_de — SCC v10.1
    sp branch → 縮約因子で直接返す
    d branch  → f_JT × f_5d × f_lattice × f_complex
    """
    group = _get_group(mat)
    f_sp = bcc_f_sp(mat.n_d, mat.period, group=group)
    if f_sp < 1.0:
        return f_sp

    n_atoms = _get_n_atoms_cell(mat)
    return (bcc_f_jt(mat.n_d)
            * bcc_f_5d(mat.n_d, mat.period)
            * bcc_f_lattice(mat.group, mat.sel, mat.period)
            * bcc_f_complex(n_atoms))


def bcc_f_de_detail(mat: Material) -> Dict[str, float]:
    """BCC f_de の内訳を返す（診断用）"""
    group = _get_group(mat)
    f_sp = bcc_f_sp(mat.n_d, mat.period, group=group)

    if f_sp < 1.0:
        return {
            'branch': 'sp',
            'f_sp': f_sp,
            'f_jt': None,
            'f_5d': None,
            'f_lat': None,
            'f_complex': None,
            'f_de': f_sp,
        }

    n_atoms = _get_n_atoms_cell(mat)
    f_jt = bcc_f_jt(mat.n_d)
    f_5d = bcc_f_5d(mat.n_d, mat.period)
    f_lat = bcc_f_lattice(mat.group, mat.sel, mat.period)
    f_cx = bcc_f_complex(n_atoms)
    return {
        'branch': 'scc',
        'f_jt': f_jt,
        'f_5d': f_5d,
        'f_lat': f_lat,
        'f_complex': f_cx,
        'f_sp': None,
        'f_de': f_jt * f_5d * f_lat * f_cx,
    }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HCP — PCC (Perturbative-Coupled Channel)                          ║
# ║  チャンネル X_HCP = R (CRSS異方性比)                                 ║
# ║  v10.0: f_de = f_elec × f_aniso(R) × f_ca × f_5d                   ║
# ║  v10.1: f_de = f_elec × f_aniso × f_ca × f_5d                      ║
# ║               × f_lanthanide × f_sp_cov                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

HCP_CA_IDEAL = 1.633
HCP_CA_COEFF = -4.00

HCP_B_PRIS   = 0.20
HCP_A_GATE   = 1.93
HCP_KG_GATE  = 18
HCP_X0_GATE  = 0.29

HCP_5D_K0    = 0.495
HCP_5D_K1    = -0.010

# v10.1: sp共有結合ゲート
HCP_F_SP_COV = 1.905      # Be: period≤2, n_d=0


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


def hcp_f_5d(n_d: int, period: int, n_f: int = 0) -> float:
    """HCP PCC 補助チャンネル②: 5d相対論補正

    v10.1: ランタノイド (n_f > 0) には適用しない
    """
    if n_f > 0:
        return 1.0
    if period == 6:
        return max(0.1, HCP_5D_K0 + HCP_5D_K1 * n_d)
    return 1.0


def hcp_f_elec(n_d: int, period: int, n_f: int = 0) -> float:
    """HCP d電子因子 (v10.1: d¹ period依存ゲート追加)

    n_d=1: d¹ e_g方向性ロッキング
      period≤4: 3.0 (3d compact)
      period=5:  1.5 (4d diffuse)
      period=6:  1.0 (5d → 別途5d補正あり)
    """
    F_ELEC_BASE = {
        0: 1.0, 1: 1.0, 2: 2.5, 3: 1.0, 4: 1.9, 5: 1.0,
        6: 1.0, 7: 1.0, 8: 1.5, 9: 1.0, 10: 1.0,
    }

    if n_d == 1:
        if period <= 4:
            b = 3.0
        elif period == 5:
            b = 1.5
        else:
            b = 1.0
    else:
        b = F_ELEC_BASE.get(n_d, 1.0)

    # 5d遷移金属 (非ランタノイド) の追加補正
    if period == 6 and n_d not in (0, 10) and n_f == 0:
        b *= 1.5
    return b


def hcp_f_lanthanide(n_f: int, n_d: int) -> float:
    """v10.1: ランタノイド 4f結晶場ゲート (HCP)

    f_4f = 1 + k_f × n_f_eff
    HCPランタノイドでは d電子寄与は f_elec で処理済み

    ランタノイドでない場合 (n_f=0) → 1.0 (既存金属に影響なし)
    """
    if n_f == 0:
        return 1.0
    n_f_eff = n_f if n_f <= 7 else 14 - n_f
    return 1.0 + LANT_K_4F * n_f_eff


def hcp_f_sp_cov(n_d: int, period: int) -> float:
    """v10.1: sp共有結合ゲート

    Be: period≤2, n_d=0 → f = 1.905
    軽元素のsp共有結合性が強度を増大させる

    該当しない場合 → 1.0 (既存金属に影響なし)
    """
    if n_d == 0 and period <= 2:
        return HCP_F_SP_COV
    return 1.0


def hcp_f_de(mat: Material) -> float:
    """HCP SSOC f_de — PCC v10.1
    f_de = f_elec × f_aniso(R) × f_ca × f_5d × f_lanthanide × f_sp_cov
    """
    n_f = _get_n_f(mat)
    return (hcp_f_elec(mat.n_d, mat.period, n_f=n_f)
            * hcp_f_aniso(mat.R_crss)
            * hcp_f_ca(mat.c_a)
            * hcp_f_5d(mat.n_d, mat.period, n_f=n_f)
            * hcp_f_lanthanide(n_f, mat.n_d)
            * hcp_f_sp_cov(mat.n_d, mat.period))


def hcp_f_de_detail(mat: Material) -> Dict[str, float]:
    """HCP f_de の内訳を返す（診断用）"""
    n_f = _get_n_f(mat)
    f_e = hcp_f_elec(mat.n_d, mat.period, n_f=n_f)
    f_R = hcp_f_aniso(mat.R_crss)
    f_c = hcp_f_ca(mat.c_a)
    f_5d = hcp_f_5d(mat.n_d, mat.period, n_f=n_f)
    f_la = hcp_f_lanthanide(n_f, mat.n_d)
    f_cov = hcp_f_sp_cov(mat.n_d, mat.period)
    return {
        'f_elec': f_e,
        'f_aniso': f_R,
        'f_ca': f_c,
        'f_5d': f_5d,
        'f_lanthanide': f_la,
        'f_sp_cov': f_cov,
        'f_de': f_e * f_R * f_c * f_5d * f_la * f_cov,
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
# ║  σ_base (v10.1 SSOC)                                               ║
# ║  σ = (COEFF/M·Z) × α₀ × BD2 × f_de × √(E·kT) / V × HP           ║
# ╚══════════════════════════════════════════════════════════════════════╝

def sigma_base_v10(mat: Material, T_K: float = 300.0) -> float:
    """v10.1 SSOC 基底降伏応力 [MPa]

    σ_y = (8√5/5πMZ) × α₀ × (b/d)² × f_de × √(E_coh·k_B·T_m) / V_act × HP

    Note: M = M_SSOC = 3.0 (全構造統一)
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
    print("  SSOC v10.1 f_de計算モジュール テスト")
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

    print("\n✅ v10.1 テスト完了!")
