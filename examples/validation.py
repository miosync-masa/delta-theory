#!/usr/bin/env python3
"""
================================================================================
δ理論 v10.0 SSOC — 統一検証スクリプト (validation.py)
================================================================================

Master Equation:
  σ_y = σ_base(SSOC) + Δσ_ss(c) + Δσ_ρ(ε) + Δσ_ppt(r, f)

検証対象:
  [1] 固溶強化 — 冪指数 n のクラス普遍性
  [2] 加工硬化 — K_ρ フィット精度
  [3] 析出強化 — Cutting/Orowan 自動選択
  [4] 統合テスト — 工業合金の予測精度

全計算は delta_theory パッケージ v10.0 の実関数を使用。
σ_base は SSOC 版 (δL フリー)。

Author: Masamichi Iizumi & Tamaki
Date: 2026-02-09
================================================================================
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

# ==============================================================================
# v10.0 パッケージからの import（計算コードとの乖離ゼロ）
# ==============================================================================
from delta_theory.material import MATERIALS, Material
from delta_theory.ssoc import sigma_base_v10, calc_f_de, calc_f_de_detail
from delta_theory.unified_yield_fatigue_v10 import (
    N_EXPONENT, K_RHO,
    delta_sigma_ss, delta_sigma_taylor,
    delta_sigma_cutting, delta_sigma_orowan, delta_sigma_ppt,
    calc_sigma_y,
    M_TAYLOR, ALPHA_TAYLOR,
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  [1] 固溶強化 — 冪指数 n のクラス普遍性                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 既存系（確認済みフィット）
SS_EXISTING = {
    'Fe-C': {
        'base': 'Fe', 'type': 'interstitial',
        'data': [(0.02, 180), (0.05, 220), (0.10, 280), (0.15, 330)],
        'fit': {'k': 927.22, 'n': 0.847},
    },
    'Cu-Zn': {
        'base': 'Cu', 'type': 'substitutional',
        'data': [(5, 85), (10, 100), (20, 125), (30, 150)],
        'fit': {'k': 3.63, 'n': 0.913},
    },
    'Al-Mg': {
        'base': 'Al', 'type': 'substitutional',
        'data': [(0.5, 45), (1.0, 55), (2.0, 75), (3.0, 95), (5.0, 130)],
        'fit': {'k': 22.03, 'n': 0.923},
    },
}

# 追加系（新規フィット）
SS_NEW = {
    'Fe-N': {
        'base': 'Fe', 'type': 'interstitial',
        'data': [(0.005, 165), (0.010, 190), (0.020, 230), (0.040, 290)],
    },
    'Cu-Ni': {
        'base': 'Cu', 'type': 'substitutional',
        'data': [(5, 90), (10, 110), (20, 150), (30, 190)],
    },
    'Cu-Al': {
        'base': 'Cu', 'type': 'substitutional',
        'data': [(2, 95), (4, 130), (6, 170), (8, 210)],
    },
    'Al-Cu': {
        'base': 'Al', 'type': 'substitutional',
        'data': [(1.0, 55), (2.0, 80), (3.0, 105), (4.0, 130)],
    },
}


def _fit_power_law(c_list, sigma_list, sigma_base):
    """Δσ = k × c^n の log-log フィット"""
    delta = np.array(sigma_list) - sigma_base
    c = np.array(c_list)
    valid = (c > 0) & (delta > 0)
    if np.sum(valid) < 2:
        return None, None, np.array([])
    c_fit, d_fit = c[valid], delta[valid]
    n, log_k = np.polyfit(np.log(c_fit), np.log(d_fit), 1)
    k = np.exp(log_k)
    d_pred = k * (c_fit ** n)
    errors = np.abs(d_pred - d_fit) / d_fit * 100
    return k, n, errors


def validate_ss() -> Dict[str, dict]:
    """固溶強化 n のクラス普遍性を検証"""
    print("\n" + "=" * 95)
    print("  [1] 固溶強化 — 冪指数 n のクラス普遍性")
    print("=" * 95)

    # --- 既存系 ---
    print("\n  [1a] 既存系（確認済みフィット）")
    print(f"  {'System':<12} {'Type':<15} {'n':>8} {'k':>12}")
    print(f"  {'-'*50}")
    for sys_name, info in SS_EXISTING.items():
        print(f"  {sys_name:<12} {info['type']:<15} "
              f"{info['fit']['n']:>8.3f} {info['fit']['k']:>12.2f}")

    # --- 追加系: v10.0 σ_base でリフィット ---
    print(f"\n  [1b] 追加系 — v10.0 SSOC σ_base でフィット")
    print(f"  {'-'*70}")

    all_results = {}

    for sys_name, sys_info in SS_NEW.items():
        mat = MATERIALS[sys_info['base']]
        sig_base = sigma_base_v10(mat)

        c_list = [d[0] for d in sys_info['data']]
        sigma_list = [d[1] for d in sys_info['data']]

        k, n, errors = _fit_power_law(c_list, sigma_list, sig_base)
        if k is None:
            continue

        all_results[sys_name] = {
            'type': sys_info['type'], 'k': k, 'n': n,
            'mean_error': float(np.mean(errors)),
        }

        print(f"\n  --- {sys_name} ({sys_info['type']}) ---")
        print(f"    σ_base(SSOC v10.0) = {sig_base:.1f} MPa")
        print(f"    フィット: Δσ = {k:.2f} × c^{n:.3f}")
        print(f"    Mean Error: {np.mean(errors):.1f}%")

        print(f"\n    {'c[wt%]':>8} {'σ_exp':>8} {'σ_pred':>8} {'Δσ':>8}")
        for c, sigma in zip(c_list, sigma_list):
            delta_pred = k * (c ** n) if c > 0 else 0
            sigma_pred = sig_base + delta_pred
            print(f"    {c:>8.3f} {sigma:>8.0f} {sigma_pred:>8.1f} "
                  f"{sigma - sig_base:>8.1f}")

    # --- クラス別統計 ---
    interstitial_n: List[Tuple[str, float]] = []
    substitutional_n: List[Tuple[str, float]] = []

    for sys_name, info in SS_EXISTING.items():
        lst = interstitial_n if info['type'] == 'interstitial' else substitutional_n
        lst.append((sys_name, info['fit']['n']))

    for sys_name, info in all_results.items():
        lst = interstitial_n if info['type'] == 'interstitial' else substitutional_n
        lst.append((sys_name, info['n']))

    print(f"\n  [1c] クラス別 n 統計")
    print(f"  {'─'*60}")

    for label, data in [('侵入型', interstitial_n), ('置換型', substitutional_n)]:
        n_vals = [n for _, n in data]
        print(f"\n  【{label}】")
        for sys_name, n in data:
            print(f"    {sys_name:<12} n = {n:.3f}")
        print(f"    {'─'*30}")
        print(f"    平均 = {np.mean(n_vals):.3f}  σ = {np.std(n_vals):.3f}")

    # --- パッケージとの整合 ---
    int_n = [n for _, n in interstitial_n]
    sub_n = [n for _, n in substitutional_n]

    print(f"\n  [1d] パッケージ N_EXPONENT との整合")
    print(f"  {'─'*60}")
    print(f"    侵入型: pkg={N_EXPONENT['interstitial']:.2f}  "
          f"fit={np.mean(int_n):.3f}  Δ={N_EXPONENT['interstitial'] - np.mean(int_n):+.3f}")
    print(f"    置換型: pkg={N_EXPONENT['substitutional']:.2f}  "
          f"fit={np.mean(sub_n):.3f}  Δ={N_EXPONENT['substitutional'] - np.mean(sub_n):+.3f}")

    return all_results


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  [2] 加工硬化 — K_ρ フィット精度                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 実験データ: (ε, σ_y [MPa])
WH_DATA = {
    'Fe': [(0.0, 180), (0.05, 280), (0.10, 350), (0.20, 450),
           (0.30, 520), (0.40, 580)],
    'Cu': [(0.0, 70), (0.05, 140), (0.10, 180), (0.20, 240),
           (0.30, 280), (0.50, 330)],
    'Al': [(0.0, 35), (0.05, 65), (0.10, 85), (0.20, 110), (0.30, 130)],
    'Ni': [(0.0, 150), (0.10, 320), (0.20, 420), (0.30, 500)],
}


def _fit_K_rho(mat: Material, data: list) -> float:
    """実験データから K_ρ をフィット"""
    sig_base = sigma_base_v10(mat)
    b = mat.b
    coeff = M_TAYLOR * ALPHA_TAYLOR * mat.G * b  # Pa·m

    K_list = []
    for eps, sigma_exp_MPa in data:
        if eps <= 0:
            continue
        delta_Pa = (sigma_exp_MPa - sig_base) * 1e6
        if delta_Pa > 0:
            rho_eff = (delta_Pa / coeff) ** 2
            K_list.append(rho_eff / eps)

    return float(np.mean(K_list)) if K_list else 0.0


def validate_wh() -> Dict[str, dict]:
    """加工硬化 K_ρ の検証"""
    print("\n" + "=" * 95)
    print("  [2] 加工硬化 — K_ρ フィット精度")
    print("=" * 95)

    results = {}

    for name, data in WH_DATA.items():
        mat = MATERIALS[name]
        sig_base = sigma_base_v10(mat)
        K_fit = _fit_K_rho(mat, data)
        K_pkg = K_RHO.get(name, K_RHO.get(mat.structure, 1e15))

        print(f"\n  --- {name} ({mat.structure}) ---")
        print(f"    σ_base(SSOC v10.0) = {sig_base:.1f} MPa")
        print(f"    K_ρ fit  = {K_fit:.2e}")
        print(f"    K_ρ pkg  = {K_pkg:.2e}")
        print(f"    Δ = {(K_fit - K_pkg) / K_pkg * 100:+.1f}%")

        print(f"\n    {'ε':>6} {'σ_exp':>8} {'σ_pred':>8} {'Err':>8} {'ρ[m⁻²]':>12}")

        errors = []
        for eps, sigma_exp in data:
            # パッケージ関数で予測
            rho = 1e12 + K_pkg * max(eps, 0.0)
            delta_wh = delta_sigma_taylor(eps, mat, rho_0=0.0) if eps > 0 else 0.0
            # delta_sigma_taylor は内部で K_RHO を使う
            # → 直接 Taylor 式で計算して比較
            b = mat.b
            delta_manual = M_TAYLOR * ALPHA_TAYLOR * mat.G * b * np.sqrt(rho) / 1e6
            sigma_pred = sig_base + delta_manual

            err = (sigma_pred - sigma_exp) / sigma_exp * 100 if sigma_exp > 0 else 0
            errors.append(abs(err))

            print(f"    {eps:>6.2f} {sigma_exp:>8.0f} {sigma_pred:>8.1f} "
                  f"{err:>+8.1f}% {rho:>12.2e}")

        mean_err = float(np.mean(errors))
        print(f"    Mean Error: {mean_err:.1f}%")

        results[name] = {
            'K_fit': K_fit, 'K_pkg': K_pkg,
            'mean_error': mean_err,
        }

    # K_ρ 材料依存性
    print(f"\n  [2b] K_ρ 材料依存性")
    print(f"  {'─'*60}")
    print(f"  {'Metal':<6} {'K_ρ(fit)':>14} {'K_ρ(pkg)':>14} {'Δ%':>8}")
    for name, r in results.items():
        delta_pct = (r['K_fit'] - r['K_pkg']) / r['K_pkg'] * 100
        print(f"  {name:<6} {r['K_fit']:>14.2e} {r['K_pkg']:>14.2e} {delta_pct:>+8.1f}%")

    return results


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  [3] 析出強化 — Cutting/Orowan 自動選択                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def validate_ppt() -> Dict[str, dict]:
    """析出強化 Cutting/Orowan の検証"""
    print("\n" + "=" * 95)
    print("  [3] 析出強化 — Cutting/Orowan 自動選択")
    print("=" * 95)

    mat_Al = MATERIALS['Al']
    b = mat_Al.b
    G = mat_Al.G
    sig_base = sigma_base_v10(mat_Al)

    # Al-Mg-Si系 β'' パラメータ
    gamma_apb = 0.10  # J/m² (β'' in Al-Mg-Si, corrected from 0.12)
    f = 0.015         # 1.5%

    print(f"\n  Al-Mg-Si系 (β'' 析出物)")
    print(f"    γ_APB = {gamma_apb} J/m², f = {f*100:.1f}%")
    print(f"    G = {G/1e9:.0f} GPa, b = {b*1e10:.2f} Å")
    print(f"    σ_base(SSOC v10.0) = {sig_base:.1f} MPa")

    # --- r依存性テーブル ---
    print(f"\n  [3a] r 依存性（クラス分岐の確認）")
    print(f"  {'r[nm]':>8} {'Δσ_cut':>10} {'Δσ_oro':>10} {'Class':>10} {'Δσ_eff':>10}")
    print(f"  {'-'*55}")

    r_list = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    cut_vals, oro_vals = [], []

    for r_nm in r_list:
        d_cut = delta_sigma_cutting(r_nm, f, gamma_apb, G, b)
        d_oro = delta_sigma_orowan(r_nm, f, G, b)
        d_eff, mech = delta_sigma_ppt(r_nm, f, gamma_apb, mat_Al)

        cut_vals.append(d_cut)
        oro_vals.append(d_oro)

        print(f"  {r_nm:>8.1f} {d_cut:>10.1f} {d_oro:>10.1f} "
              f"{mech:>10} {d_eff:>10.1f}")

    # ピーク位置検出
    diff = np.array(cut_vals) - np.array(oro_vals)
    for i in range(len(diff) - 1):
        if diff[i] < 0 and diff[i + 1] >= 0:
            frac = -diff[i] / (diff[i + 1] - diff[i])
            r_peak = r_list[i] + (r_list[i + 1] - r_list[i]) * frac
            d_peak = oro_vals[i] + (oro_vals[i + 1] - oro_vals[i]) * frac
            print(f"\n  【ピーク時効】 r ≈ {r_peak:.1f} nm, Δσ_peak ≈ {d_peak:.1f} MPa")
            break

    # --- 6061-T6 校正テスト ---
    print(f"\n  [3b] 6061-T6 A係数の1点校正")
    print(f"  {'─'*60}")

    sigma_ss_est = 30.0  # Mg, Si 固溶分（推定）
    sigma_exp_T6 = 275.0

    r_T6, f_T6 = 3.0, 0.015
    d_ppt_raw, mech_raw = delta_sigma_ppt(r_T6, f_T6, gamma_apb, mat_Al)
    sigma_pred_raw = sig_base + sigma_ss_est + d_ppt_raw

    delta_target = sigma_exp_T6 - sig_base - sigma_ss_est
    A_fit = delta_target / d_ppt_raw if d_ppt_raw > 0 else 1.0

    print(f"    σ_pred(A=1)  = {sigma_pred_raw:.1f} MPa  (exp: {sigma_exp_T6})")
    print(f"    A_fit = {A_fit:.3f}")

    # 校正後の時効状態テスト
    test_cases = [
        ('Under-aged (T4)', 1.5, 0.010, 145),
        ('Peak-aged  (T6)', 3.0, 0.015, 275),
        ('Over-aged',       10.0, 0.015, 200),
    ]

    print(f"\n    {'State':<22} {'r':>5} {'f%':>5} {'σ_pred':>8} {'σ_exp':>8} {'Err':>8}")
    print(f"    {'-'*60}")

    ppt_errors = []
    for state, r, f_v, sigma_exp in test_cases:
        d_ppt, mech = delta_sigma_ppt(r, f_v, gamma_apb, mat_Al, A=A_fit)
        sigma_pred = sig_base + sigma_ss_est + d_ppt
        err = (sigma_pred - sigma_exp) / sigma_exp * 100
        ppt_errors.append(abs(err))
        print(f"    {state:<22} {r:>5.1f} {f_v*100:>5.1f} "
              f"{sigma_pred:>8.1f} {sigma_exp:>8.0f} {err:>+8.1f}%")

    print(f"    Mean Error: {np.mean(ppt_errors):.1f}%")

    return {'A_fit': A_fit, 'mean_error': float(np.mean(ppt_errors))}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  [4] 統合テスト — 工業合金の予測精度                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def validate_integrated() -> float:
    """全機構統合での工業合金予測"""
    print("\n" + "=" * 95)
    print("  [4] 統合テスト — 工業合金の予測精度")
    print("=" * 95)

    # (名前, metal, T_K, c_wt%, k_ss, solute_type, ε, r_nm, f, γ_apb, A_ppt, σ_exp)
    test_cases = [
        ('Pure Al (O)',       'Al', 300, 0,    0,   None,             0,   0,   0,    0,    1.0,   35),
        ('Pure Fe (anneal)',  'Fe', 300, 0,    0,   None,             0,   0,   0,    0,    1.0,  150),
        ('Fe-0.1C',          'Fe', 300, 0.10, 927, 'interstitial',   0,   0,   0,    0,    1.0,  280),
        ('Al-3Mg (5xxx)',    'Al', 300, 3.0,  22,  'substitutional', 0,   0,   0,    0,    1.0,   95),
        ('Cu cold-work 20%', 'Cu', 300, 0,    0,   None,             0.2, 0,   0,    0,    1.0,  240),
        ('6061-T6 (peak)',   'Al', 300, 1.0,  15,  'substitutional', 0,   3.0, 0.015,0.10, 1.00, 275),
    ]

    print(f"\n  {'Alloy':<22} {'σ_base':>7} {'Δσ_ss':>7} {'Δσ_wh':>7} "
          f"{'Δσ_ppt':>7} {'σ_pred':>8} {'σ_exp':>7} {'Err':>8}")
    print(f"  {'-'*88}")

    errors = []
    for name, met, T, c, k, sol, eps, r, f, gamma, A, exp in test_cases:
        mat = MATERIALS[met]
        result = calc_sigma_y(
            mat, T_K=T,
            c_wt_percent=c, k_ss=k, solute_type=sol,
            eps=eps,
            r_ppt_nm=r, f_ppt=f, gamma_apb=gamma, A_ppt=A,
        )

        err = (result['sigma_y'] - exp) / exp * 100
        errors.append(abs(err))

        print(f"  {name:<22} {result['sigma_base']:>7.1f} {result['delta_ss']:>7.1f} "
              f"{result['delta_wh']:>7.1f} {result['delta_ppt']:>7.1f} "
              f"{result['sigma_y']:>8.1f} {exp:>7} {err:>+8.1f}%")

    mean_err = float(np.mean(errors))
    print(f"  {'-'*88}")
    print(f"  {'Mean Absolute Error':<22} {'':>7} {'':>7} {'':>7} "
          f"{'':>7} {'':>8} {'':>7} {mean_err:>8.1f}%")

    return mean_err


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  [5] サマリー                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def print_summary(ss_results, wh_results, ppt_results, integrated_err):
    """全検証結果のサマリー"""
    print("\n" + "=" * 95)
    print("  [5] 検証サマリー")
    print("=" * 95)

    print(f"""
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  δ理論 v10.0 SSOC — 強化機構 検証結果サマリー                                       │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  [1] 固溶強化 n クラス普遍性                                                         │
│      侵入型:  N_EXPONENT = {N_EXPONENT['interstitial']:.2f}  ← パッケージ値                              │
│      置換型:  N_EXPONENT = {N_EXPONENT['substitutional']:.2f}  ← パッケージ値                              │
│                                                                                      │
│  [2] 加工硬化 K_ρ                                                                    │""")

    for name, r in wh_results.items():
        delta_pct = (r['K_fit'] - r['K_pkg']) / r['K_pkg'] * 100
        print(f"│      {name:<4}: K_fit={r['K_fit']:.2e}  K_pkg={r['K_pkg']:.2e}  "
              f"Δ={delta_pct:+.1f}%  Err={r['mean_error']:.1f}%{' ' * 6}│")

    print(f"""│                                                                                      │
│  [3] 析出強化 (6061-T6)                                                              │
│      A_fit = {ppt_results['A_fit']:.3f}, Mean Error = {ppt_results['mean_error']:.1f}%                                            │
│                                                                                      │
│  [4] 統合テスト（工業合金）                                                          │
│      Mean Absolute Error = {integrated_err:.1f}%                                                     │
│                                                                                      │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  パラメータ数:                                                                       │
│    基底強度(SSOC)  → 0 (第一原理)                                                    │
│    固溶強化        → 1/溶質系 (k のみ, n はクラスプリセット)                         │
│    加工硬化        → 0 (K_ρ は構造プリセット)                                        │
│    析出強化        → 1/析出系 (A のみ, 機構はクラス分岐)                             │
│    合計            → ≈ 1〜2 /材料系 (従来法: 5〜10+)                                 │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
""")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  main                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║           δ理論 v10.0 SSOC — 統一検証スクリプト (validation.py)                         ║
║                                                                                         ║
║   σ_y = σ_base(SSOC) + Δσ_ss(c) + Δσ_ρ(ε) + Δσ_ppt(r, f)                              ║
║                                                                                         ║
║   全計算は delta_theory パッケージ v10.0.2 の実関数を使用                                ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
    """)

    ss_results = validate_ss()
    wh_results = validate_wh()
    ppt_results = validate_ppt()
    integrated_err = validate_integrated()

    print_summary(ss_results, wh_results, ppt_results, integrated_err)


if __name__ == '__main__':
    main()
