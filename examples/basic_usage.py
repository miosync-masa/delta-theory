#!/usr/bin/env python3
"""
δ-Theory Basic Usage Examples
=============================

このスクリプトは、δ理論の基本的な使い方を示します。
"""

import sys
sys.path.insert(0, '..')

import numpy as np

# =============================================================================
# Example 1: 降伏応力計算
# =============================================================================
print("=" * 70)
print("Example 1: Yield Stress Calculation")
print("=" * 70)

from core.unified_yield_fatigue_v6_9 import calc_sigma_y, MATERIALS

# 純鉄の降伏応力
mat_fe = MATERIALS['Fe']
y_fe = calc_sigma_y(mat_fe, T_K=300)
print(f"\nPure Fe at 300K:")
print(f"  σ_base = {y_fe['sigma_base']:.1f} MPa")
print(f"  σ_y    = {y_fe['sigma_y']:.1f} MPa")

# Fe-0.1wt%C（炭素固溶）
y_fe_c = calc_sigma_y(
    mat_fe, 
    T_K=300,
    c_wt_percent=0.1,
    k_ss=400,  # interstitial C in Fe
    solute_type='interstitial'
)
print(f"\nFe-0.1wt%C at 300K:")
print(f"  σ_base = {y_fe_c['sigma_base']:.1f} MPa")
print(f"  Δσ_ss  = {y_fe_c['delta_ss']:.1f} MPa")
print(f"  σ_y    = {y_fe_c['sigma_y']:.1f} MPa")

# 高温での降伏応力
print(f"\nFe yield stress vs Temperature:")
for T in [300, 500, 700, 900]:
    y = calc_sigma_y(mat_fe, T_K=T)
    print(f"  T={T}K: σ_y = {y['sigma_y']:.1f} MPa")

# =============================================================================
# Example 2: 疲労寿命予測
# =============================================================================
print("\n" + "=" * 70)
print("Example 2: Fatigue Life Prediction")
print("=" * 70)

from core.unified_yield_fatigue_v6_9 import fatigue_life_const_amp, FATIGUE_CLASS_PRESET

# BCC Fe（明確な疲労限度）
print("\nBCC Fe (clear fatigue limit):")
print(f"  r_th = {FATIGUE_CLASS_PRESET['BCC']['r_th']}")

y_fe = calc_sigma_y(MATERIALS['Fe'], T_K=300)
for sigma_a in [50, 100, 150, 200]:
    result = fatigue_life_const_amp(
        MATERIALS['Fe'],
        sigma_a_MPa=sigma_a,
        sigma_y_tension_MPa=y_fe['sigma_y'],
        A_ext=2.46e-4,
    )
    if np.isfinite(result['N_fail']):
        print(f"  σ_a={sigma_a:3d} MPa: N_fail = {result['N_fail']:.2e} (r={result['r']:.3f})")
    else:
        print(f"  σ_a={sigma_a:3d} MPa: Infinite life (r={result['r']:.3f} ≤ r_th)")

# FCC Cu（疲労限度なし）
print("\nFCC Cu (no fatigue limit):")
print(f"  r_th = {FATIGUE_CLASS_PRESET['FCC']['r_th']}")

mat_cu = MATERIALS['Cu']
y_cu = calc_sigma_y(mat_cu, T_K=300)
for sigma_a in [20, 40, 60, 80]:
    result = fatigue_life_const_amp(
        mat_cu,
        sigma_a_MPa=sigma_a,
        sigma_y_tension_MPa=y_cu['sigma_y'],
        A_ext=2.46e-4,
    )
    if np.isfinite(result['N_fail']):
        print(f"  σ_a={sigma_a:3d} MPa: N_fail = {result['N_fail']:.2e} (r={result['r']:.3f})")
    else:
        print(f"  σ_a={sigma_a:3d} MPa: Infinite life (r={result['r']:.3f} ≤ r_th)")

# =============================================================================
# Example 3: DBT/DBTT予測
# =============================================================================
print("\n" + "=" * 70)
print("Example 3: Ductile-Brittle Transition Temperature")
print("=" * 70)

from core.dbt_unified import DBTUnified

model = DBTUnified()

# 単点計算
d = 30e-6  # 30 μm
c = 0.005  # 0.5 at%
T = 300    # K

summary = model.summary(d, c, T)
print(f"\nCondition: d={d*1e6:.0f}μm, c={c*100:.1f}at%, T={T}K")
print(f"  σ_y  = {summary['sigma_y_MPa']:.1f} MPa")
print(f"  σ_f  = {summary['sigma_f_MPa']:.1f} MPa")
print(f"  θ    = {summary['theta']:.4f}")
print(f"  Mode = {summary['mode'].upper()}")

# DBTT探索
print("\nDBTT search:")
for c_pct in [0.0, 0.2, 0.5, 1.0]:
    result = model.temp_view.find_DBTT(d=30e-6, c=c_pct/100)
    if np.isfinite(result['T_star']):
        print(f"  c={c_pct:.1f}%: DBTT = {result['T_star']:.0f} K ({result['T_star']-273:.0f}°C)")
    else:
        print(f"  c={c_pct:.1f}%: {result['status']}")

# =============================================================================
# Example 4: 材料データベース
# =============================================================================
print("\n" + "=" * 70)
print("Example 4: Material Database")
print("=" * 70)

print("\nAvailable materials in unified model:")
for name, mat in MATERIALS.items():
    print(f"  {name:3s} ({mat.structure}): T_m={mat.T_m:.0f}K, E_b={mat.Eb:.2f}eV, δ_L={mat.dL:.2f}")

from core.materials import MaterialGPU
print("\nDetailed MaterialGPU:")
fe = MaterialGPU.Fe()
print(fe.summary())

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
