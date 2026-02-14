"""
creep.py — δ-theory 拡散・クリープモジュール

SSOCルールによる拡散係数とクリープ速度の幾何的導出

核心思想:
  Q_self = k_B × T_m × Q_base(struct) × g_ssoc(pattern)

  - Q_base: 配位数Zで決まる構造の幾何定数
  - g_ssoc: f_de と同じ分岐パターン（係数は空孔鞍点用）
  - 新規フィッティングパラメータ: 0

  D = ν × a² × exp(-Q_self / k_B T)
    ν: デバイ振動数 ≈ √(E_coh / M) / a  → material.py から導出
    a: 格子定数                           → material.py に既存

  ε̇ = A × D × σ × Ω / (k_B × T × d²)
    Ω = b³ = |V|_eff                     → material.py に既存
    d: 結晶粒径                           → ユーザー入力

SSOC拡散ルール (g_ssoc):
  f_de と同じ分岐条件を使い、係数だけ空孔鞍点の物理に合わせる。
  金属ごとの個別値は存在しない。ルールに該当するか否かだけ。

  BCC (SCC channel):
    n_d = 0 (sp):     0.86  ← d方向性による拘束なし
    n_d = 5 (半充填): 1.19  ← 全5軌道が鞍点をブロック
    n_d ≥ 6 (ペア):   0.96  ← ペアリングで1方向解放
    else:              1.00  ← ベース

  FCC (PCC channel):
    n_d = 0 (sp):     0.96  ← d方向性による拘束なし
    Period 6 (5d):     0.93  ← 相対論的s安定化
    n_d < 10 (holes):  1 + 0.075 × holes  ← open-shell方向性
    n_d = 10:          1.01  ← 閉殻

  HCP (PCC channel):
    all:               1.00  ← ベースで十分

物理的意味:
  Q_base × g_ssoc ≈ 15 は「1原子が隣に移動するのに壊す拘束の数」
  T_m × k_B は「全拘束が壊れるエネルギー」
  → Q_self = 「壊す拘束の割合」×「全拘束エネルギー」

Author: 飯泉真道 & 環
Date: 2026-02-14
================================================================================
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np

from .material import Material, k_B, eV_to_J

# ==============================================================================
# 物理定数
# ==============================================================================
k_B_eV: float = 8.617e-5          # eV/K
N_A: float = 6.022e23             # アボガドロ数
AMU_TO_KG: float = 1.6605e-27     # amu → kg

# ==============================================================================
# 空孔形成エネルギー比 (universal)
# ==============================================================================
C_F: float = 0.35                 # Q_form / E_coh ≈ 0.35 (全金属共通)

# ==============================================================================
# Q_base: 構造幾何定数
# ==============================================================================
# 配位数Zで決まる「拘束の総数」に比例
# BCC Z=8:  14.4,  FCC Z=12: 16.6,  HCP Z=12: 16.5
Q_BASE: Dict[str, float] = {
    'BCC': 14.4,
    'FCC': 16.6,
    'HCP': 16.5,
}


# ==============================================================================
# g_ssoc: SSOC ルール分配 (f_de と同じ分岐パターン)
# ==============================================================================

def g_ssoc_bcc(n_d: int, period: int, group: int) -> float:
    """BCC SCC チャンネル: 空孔鞍点の拘束度

    d⁵半充填: 5軌道×各1電子 → 全方向ブロック
    sp金属:   d方向性なし → 拘束減少
    d≥6:      ペアリングで1方向解放

    分岐条件は f_de の BCC SCC と同一。
    """
    if n_d < 2:
        return 0.86
    if n_d == 5:
        return 1.19
    if n_d >= 6:
        return 0.96
    return 1.00


def g_ssoc_fcc(n_d: int, period: int, group: int) -> float:
    """FCC PCC チャンネル: 空孔鞍点の拘束度

    sp金属:     d方向性なし
    5d (P=6):   相対論的s安定化で鞍点再配分コスト変化
    open-shell: d殻に穴 → 方向性残存 → 拘束増加
    閉殻d¹⁰:   方向性なし → ベース

    分岐条件は f_de の FCC PCC と同一。
    """
    if n_d == 0:
        return 0.96
    if period == 6:
        return 0.93
    if n_d < 10:
        holes = 10 - n_d
        return 1.0 + 0.075 * holes
    return 1.01


def g_ssoc_hcp(n_d: int, period: int, group: int) -> float:
    """HCP PCC チャンネル: 空孔鞍点の拘束度

    HCPは Q_base だけで十分な精度。
    追加拘束ルールなし。
    """
    return 1.00


def g_ssoc(mat: Material) -> float:
    """材料のg_ssocを構造に応じて計算（統一エントリポイント）"""
    n_d = mat.n_d
    period = mat.period
    group = getattr(mat, 'group', 0) or 0

    if mat.structure == 'BCC':
        return g_ssoc_bcc(n_d, period, group)
    elif mat.structure == 'FCC':
        return g_ssoc_fcc(n_d, period, group)
    elif mat.structure == 'HCP':
        return g_ssoc_hcp(n_d, period, group)
    else:
        return 1.00


# ==============================================================================
# Q_self: 自己拡散活性化エネルギー
# ==============================================================================

def Q_self_eV(mat: Material) -> float:
    """自己拡散活性化エネルギー [eV]

    Q_self = k_B × T_m × Q_base(struct) × g_ssoc(pattern)

    全て既存変数から導出。新規パラメータなし。
    """
    q_base = Q_BASE.get(mat.structure, 16.0)
    g = g_ssoc(mat)
    return k_B_eV * mat.T_m * q_base * g


def Q_form_eV(mat: Material) -> float:
    """空孔形成エネルギー [eV]

    Q_form = C_f × E_coh
    C_f ≈ 0.35 (universal)
    """
    return C_F * mat.E_bond_eV


def Q_mig_eV(mat: Material) -> float:
    """空孔移動エネルギー [eV]

    Q_mig = Q_self - Q_form
    """
    return Q_self_eV(mat) - Q_form_eV(mat)


# ==============================================================================
# デバイ振動数
# ==============================================================================

def debye_frequency(mat: Material) -> float:
    """デバイ振動数 ν [Hz]

    ν ≈ √(E_coh / M) / a

    E_coh: 結合エネルギー [eV → J]
    M:     原子質量 [amu → kg]
    a:     格子定数 [m]
    """
    E_J = mat.E_bond_eV * eV_to_J
    M_kg = mat.M_amu * AMU_TO_KG
    return np.sqrt(E_J / M_kg) / mat.a


# ==============================================================================
# 拡散係数
# ==============================================================================

def diffusion_coeff(mat: Material, T_K: float) -> float:
    """自己拡散係数 D [m²/s]

    D = ν × a² × exp(-Q_self / k_B T)

    全変数がmaterial.pyとSSOCルールから導出される。
    """
    if T_K <= 0:
        return 0.0

    nu = debye_frequency(mat)
    Q = Q_self_eV(mat)

    return nu * mat.a**2 * np.exp(-Q / (k_B_eV * T_K))


def D0_prefactor(mat: Material) -> float:
    """拡散係数の前指数因子 D₀ [m²/s]

    D₀ = ν × a²
    """
    return debye_frequency(mat) * mat.a**2


# ==============================================================================
# クリープ速度
# ==============================================================================

def creep_rate_NH(
    mat: Material,
    T_K: float,
    sigma_MPa: float,
    d_grain_m: float,
) -> float:
    """Nabarro-Herring クリープ速度 [1/s]

    ε̇_NH = A_NH × D_v × σ × Ω / (k_B × T × d²)

    D_v:   体積拡散係数（self-diffusion）
    σ:     応力 [Pa]
    Ω:     原子体積 ≈ b³ = |V|_eff  [m³]
    d:     結晶粒径 [m]
    A_NH:  幾何定数 ≈ 14 (多結晶)
    """
    A_NH = 14.0

    D = diffusion_coeff(mat, T_K)
    sigma_Pa = sigma_MPa * 1e6
    Omega = mat.b ** 3              # |V|_eff = b³
    kT = k_B_eV * eV_to_J * T_K    # k_B T [J]

    if kT <= 0 or d_grain_m <= 0:
        return 0.0

    return A_NH * D * sigma_Pa * Omega / (kT * d_grain_m**2)


def creep_rate_Coble(
    mat: Material,
    T_K: float,
    sigma_MPa: float,
    d_grain_m: float,
    delta_gb_m: float = 5e-10,
) -> float:
    """Coble クリープ速度 [1/s]

    ε̇_Coble = A_C × D_gb × σ × Ω × δ_gb / (k_B × T × d³)

    D_gb ≈ D_v × exp(+Q_mig × 0.4 / k_BT)
    （粒界拡散は活性化エネルギーが体積拡散の約0.5-0.6倍）
    δ_gb:  粒界厚さ ≈ 5Å
    A_C:   幾何定数 ≈ 50
    """
    A_C = 50.0

    D_v = diffusion_coeff(mat, T_K)
    # 粒界拡散: Q_gb ≈ 0.6 × Q_self → D_gb = D_v × exp(0.4 × Q_self / kT)
    Q = Q_self_eV(mat)
    D_gb = D_v * np.exp(0.4 * Q / (k_B_eV * T_K)) if T_K > 0 else 0.0

    sigma_Pa = sigma_MPa * 1e6
    Omega = mat.b ** 3
    kT = k_B_eV * eV_to_J * T_K

    if kT <= 0 or d_grain_m <= 0:
        return 0.0

    return A_C * D_gb * sigma_Pa * Omega * delta_gb_m / (kT * d_grain_m**3)


def creep_rate_total(
    mat: Material,
    T_K: float,
    sigma_MPa: float,
    d_grain_m: float,
) -> Dict[str, float]:
    """拡散クリープ速度の統合出力

    Returns:
        dict with keys:
            'NH':    Nabarro-Herring [1/s]
            'Coble': Coble [1/s]
            'total': NH + Coble [1/s]
            'dominant': 'NH' or 'Coble'
    """
    nh = creep_rate_NH(mat, T_K, sigma_MPa, d_grain_m)
    coble = creep_rate_Coble(mat, T_K, sigma_MPa, d_grain_m)
    total = nh + coble

    return {
        'NH': nh,
        'Coble': coble,
        'total': total,
        'dominant': 'NH' if nh >= coble else 'Coble',
    }


# ==============================================================================
# クリープ寿命推定
# ==============================================================================

def time_to_strain(
    mat: Material,
    T_K: float,
    sigma_MPa: float,
    d_grain_m: float,
    target_strain: float = 0.01,
) -> float:
    """目標ひずみに到達する時間 [hours]

    定常クリープを仮定: ε = ε̇ × t
    → t = ε_target / ε̇_total
    """
    rates = creep_rate_total(mat, T_K, sigma_MPa, d_grain_m)
    if rates['total'] <= 0:
        return float('inf')
    return target_strain / rates['total'] / 3600.0


# ==============================================================================
# 診断出力
# ==============================================================================

def diffusion_detail(mat: Material, T_K: float = None) -> Dict[str, float]:
    """拡散パラメータの全内訳を返す（診断用）

    T_K = None の場合、温度非依存のパラメータのみ返す
    """
    q_base = Q_BASE.get(mat.structure, 16.0)
    g = g_ssoc(mat)
    Q = Q_self_eV(mat)
    Q_f = Q_form_eV(mat)
    Q_m = Q_mig_eV(mat)
    nu = debye_frequency(mat)
    D0 = D0_prefactor(mat)

    result = {
        'structure': mat.structure,
        'n_d': mat.n_d,
        'period': mat.period,
        'Q_base': q_base,
        'g_ssoc': g,
        'Q_ratio': q_base * g,         # Q_self / (k_B T_m)
        'Q_self_eV': Q,
        'Q_form_eV': Q_f,
        'Q_mig_eV': Q_m,
        'C_f': C_F,
        'C_m': Q_m / mat.E_bond_eV,
        'C_tot': Q / mat.E_bond_eV,
        'nu_Hz': nu,
        'D0_m2s': D0,
        'E_coh_eV': mat.E_bond_eV,
        'T_m_K': mat.T_m,
        'a_m': mat.a,
        'b_m': mat.b,
        'Omega_m3': mat.b**3,
    }

    if T_K is not None and T_K > 0:
        D = diffusion_coeff(mat, T_K)
        result['T_K'] = T_K
        result['T_Tm'] = T_K / mat.T_m
        result['D_m2s'] = D
        result['log10_D'] = np.log10(D) if D > 0 else float('-inf')

    return result


# ==============================================================================
# テスト
# ==============================================================================

if __name__ == '__main__':
    from .material import MATERIALS, list_by_structure

    print("=" * 95)
    print("  δ-theory creep.py — SSOC-based Diffusion & Creep Module")
    print("  Q_self = k_B × T_m × Q_base(struct) × g_ssoc(pattern)")
    print("  Fitting parameters: 0")
    print("=" * 95)

    # Q_self 予測
    for struct in ['BCC', 'FCC', 'HCP']:
        metals = list_by_structure(struct)
        print(f"\n  [{struct}]")
        print(f"  {'El':<4} {'n_d':>3} {'P':>2} {'g_ssoc':>6} "
              f"{'Q/kTm':>7} {'Q_self':>6} {'Q_f':>5} {'Q_m':>5} "
              f"{'ν[THz]':>8} {'D₀[m²/s]':>10}")
        print(f"  {'-'*75}")
        for name in metals:
            mat = MATERIALS[name]
            detail = diffusion_detail(mat)
            print(f"  {name:<4} {mat.n_d:>3} {mat.period:>2} "
                  f"{detail['g_ssoc']:>6.2f} "
                  f"{detail['Q_ratio']:>7.1f} "
                  f"{detail['Q_self_eV']:>6.2f} "
                  f"{detail['Q_form_eV']:>5.2f} "
                  f"{detail['Q_mig_eV']:>5.2f} "
                  f"{detail['nu_Hz']/1e12:>8.2f} "
                  f"{detail['D0_m2s']:>10.2e}")

    # クリープ速度の例: Fe at 600°C, 100 MPa, d=50μm
    print(f"\n{'='*95}")
    print(f"  Example: Creep prediction")
    print(f"{'='*95}")

    for name in ['Fe', 'Ni', 'W', 'Al', 'Ti']:
        if name not in MATERIALS:
            continue
        mat = MATERIALS[name]
        T = 0.5 * mat.T_m  # T/Tm = 0.5
        sigma = 50.0        # MPa
        d_grain = 50e-6     # 50 μm

        rates = creep_rate_total(mat, T, sigma, d_grain)
        t_1pct = time_to_strain(mat, T, sigma, d_grain, 0.01)
        D = diffusion_coeff(mat, T)

        print(f"\n  {name} at T/Tm=0.5 ({T:.0f}K), σ={sigma}MPa, d=50μm:")
        print(f"    D = {D:.2e} m²/s")
        print(f"    ε̇_NH    = {rates['NH']:.2e} /s")
        print(f"    ε̇_Coble = {rates['Coble']:.2e} /s")
        print(f"    dominant: {rates['dominant']}")
        print(f"    Time to 1% strain: {t_1pct:.1f} hours")

    print(f"\n✅ creep.py テスト完了!")
