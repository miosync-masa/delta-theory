#!/usr/bin/env python3
"""
δ-theory AM Alloy Fatigue Module
=================================
AM合金のS-N寿命予測（構造プリセット方式）

Physical model:
  N = min( N_init + N_prop,  N_static )

  N_prop   = 0.5 / (A × r^n)           — き裂伝播（Paris則ベース）
  N_init   = C × r^(-m)                — き裂核生成
  N_static = D × (1 - r/r_u)^p         — 静的破壊上限

  r    = σ_a / σ_y    （応力比, 温度依存: σ_y(T)経由）
  r_u  = σ_uts / σ_y  （延性余裕, per-sample）

Parameters:
  A, n     — 伝播: 構造プリセット（BCC/FCC/HCP）
  C, m     — 核生成: 構造プリセット
  D, p     — 静的破壊: 構造プリセット
  σ_y(T)   — unified_yield_fatigue_v10 から取得
  σ_uts    — 入力（実験値 or DB）

Temperature enters via σ_y(T):
  T↑ → σ_y↓ → r↑ → N↓  (自動的に温度効果が入る)

Author: 飯泉真道 & 環
Date: 2026-02
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Structure presets (validated against FatigueData-AM2022)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AMFatiguePreset:
    """AM alloy fatigue parameters per crystal structure."""
    A: float       # propagation coefficient
    n: float       # propagation exponent
    C: float       # initiation coefficient
    m: float       # initiation exponent
    D: float       # static cap coefficient
    p: float       # static cap exponent

# fmt: off
AM_PRESETS: Dict[str, AMFatiguePreset] = {
    'BCC': AMFatiguePreset(A=2.29e-3, n=6.50, C=7.943e+03, m=2.35, D=3.162e+05, p=2.50),
    'FCC': AMFatiguePreset(A=6.03e-4, n=8.50, C=8.913e+04, m=1.05, D=1.000e+06, p=3.00),
    'HCP': AMFatiguePreset(A=1.10e-3, n=6.50, C=5.623e+04, m=2.25, D=5.623e+05, p=2.80),
}
# fmt: on

# AM alloy → crystal structure mapping
ALLOY_STRUCTURE: Dict[str, str] = {
    # FCC
    'AlSi10Mg': 'FCC', 'IN718': 'FCC', '316L': 'FCC', 'IN625': 'FCC',
    'Scalmalloy': 'FCC', '304L': 'FCC', '304': 'FCC', 'Al-Mg-Sc-Zr': 'FCC',
    'NiTi': 'FCC', 'CuAl9Ni5Fe4': 'FCC', 'Mar-M-509': 'FCC',
    'GH4169': 'FCC', 'IN738LC': 'FCC', 'Fe40Mn20Co20Cr15Si5': 'FCC',
    'QuesTek Al': 'FCC', 'AIF357': 'FCC', 'AD1': 'FCC',
    'AlSi10Mg0.4': 'FCC', 'AlNiCu': 'FCC',
    # BCC
    '17-4 PH': 'BCC', '18Ni300': 'BCC', 'Maraging Steel': 'BCC',
    'Maraging Steel MS1': 'BCC',
    'CL 92PH stainless steel': 'BCC',
    '420J1 martensitic stainless steel': 'BCC',
    # HCP
    'Ti-6Al-4V': 'HCP', 'TA2 + TA15': 'HCP', 'WE43': 'HCP',
    'Ti-TiB': 'HCP', 'Ti-5Al-2.5Sn': 'HCP',
}


# ---------------------------------------------------------------------------
# Core prediction functions
# ---------------------------------------------------------------------------

def am_fatigue_life(
    sigma_a: float,
    sigma_y: float,
    sigma_uts: float,
    structure: Literal['BCC', 'FCC', 'HCP'],
    preset: Optional[AMFatiguePreset] = None,
) -> Dict[str, float]:
    """
    Predict AM alloy fatigue life.

    Parameters
    ----------
    sigma_a : float
        Stress amplitude [MPa].
    sigma_y : float
        Yield stress [MPa]. Use σ_y(T) for temperature dependence.
    sigma_uts : float
        Ultimate tensile stress [MPa].
    structure : str
        Crystal structure ('BCC', 'FCC', 'HCP').
    preset : AMFatiguePreset, optional
        Custom preset. If None, uses AM_PRESETS[structure].

    Returns
    -------
    dict with keys:
        N_f       — predicted fatigue life [cycles]
        r         — stress ratio σ_a/σ_y
        r_u       — ductility ratio σ_uts/σ_y
        N_init    — initiation life
        N_prop    — propagation life
        N_static  — static cap life
        regime    — 'fatigue' or 'static'
    """
    if sigma_y <= 0:
        raise ValueError(f"σ_y must be positive, got {sigma_y}")
    if sigma_uts <= sigma_y:
        raise ValueError(f"σ_uts ({sigma_uts}) must exceed σ_y ({sigma_y})")
    if sigma_a <= 0:
        return {'N_f': np.inf, 'r': 0.0, 'r_u': sigma_uts / sigma_y,
                'N_init': np.inf, 'N_prop': np.inf, 'N_static': np.inf,
                'regime': 'below_threshold'}

    if preset is None:
        preset = AM_PRESETS[structure]

    r = sigma_a / sigma_y
    r_u = sigma_uts / sigma_y

    if r >= r_u:
        return {'N_f': 0.0, 'r': r, 'r_u': r_u,
                'N_init': 0.0, 'N_prop': 0.0, 'N_static': 0.0,
                'regime': 'instant_fracture'}

    N_prop = 0.5 / (preset.A * r ** preset.n)
    N_init = preset.C * r ** (-preset.m)
    N_fatigue = N_init + N_prop

    rr = min(r / r_u, 0.9999)
    N_static = preset.D * (1.0 - rr) ** preset.p

    N_f = min(N_fatigue, N_static)
    regime = 'static' if N_static < N_fatigue else 'fatigue'

    return {
        'N_f': N_f,
        'r': r,
        'r_u': r_u,
        'N_init': N_init,
        'N_prop': N_prop,
        'N_static': N_static,
        'regime': regime,
    }


def am_sn_curve(
    sigma_y: float,
    sigma_uts: float,
    structure: Literal['BCC', 'FCC', 'HCP'],
    sigma_a_range: Optional[np.ndarray] = None,
    n_points: int = 200,
    preset: Optional[AMFatiguePreset] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate full S-N curve for AM alloy.

    Parameters
    ----------
    sigma_y : float
        Yield stress [MPa].
    sigma_uts : float
        Ultimate tensile stress [MPa].
    structure : str
        Crystal structure.
    sigma_a_range : array, optional
        Custom stress amplitude array [MPa].
        Default: 10% to 95% of σ_uts.
    n_points : int
        Number of points (used when sigma_a_range is None).
    preset : AMFatiguePreset, optional
        Custom preset.

    Returns
    -------
    sigma_a : ndarray
        Stress amplitudes [MPa].
    N_f : ndarray
        Predicted fatigue lives [cycles].
    """
    if sigma_a_range is None:
        sigma_a_range = np.linspace(sigma_uts * 0.10, sigma_uts * 0.95, n_points)

    N_f = np.array([
        am_fatigue_life(sa, sigma_y, sigma_uts, structure, preset)['N_f']
        for sa in sigma_a_range
    ])

    return sigma_a_range, N_f


def am_sn_curve_temperature(
    sigma_y_func,
    sigma_uts: float,
    structure: Literal['BCC', 'FCC', 'HCP'],
    temperatures: List[float],
    sigma_a_range: Optional[np.ndarray] = None,
    n_points: int = 200,
    preset: Optional[AMFatiguePreset] = None,
) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate S-N curves at multiple temperatures.

    Parameters
    ----------
    sigma_y_func : callable
        Function T_K -> σ_y [MPa]. Typically:
        lambda T: calc_sigma_y(mat, T_K=T, ...)['sigma_y']
    sigma_uts : float
        Ultimate tensile stress [MPa] at reference T.
        (Assumed constant; for T-dependent σ_uts, pass per-T values.)
    structure : str
        Crystal structure.
    temperatures : list of float
        Temperatures [K].
    sigma_a_range : array, optional
        Fixed stress amplitude array [MPa].
    n_points : int
        Number of points.
    preset : AMFatiguePreset, optional
        Custom preset.

    Returns
    -------
    dict : {T_K: (sigma_a, N_f)} for each temperature.
    """
    results = {}
    for T in temperatures:
        sigma_y_T = sigma_y_func(T)
        if sigma_a_range is None:
            sa = np.linspace(sigma_uts * 0.10, sigma_uts * 0.95, n_points)
        else:
            sa = sigma_a_range
        _, nf = am_sn_curve(sigma_y_T, sigma_uts, structure, sa, preset=preset)
        results[T] = (sa, nf)
    return results


def get_structure(alloy_name: str) -> str:
    """Look up crystal structure for AM alloy name."""
    if alloy_name in ALLOY_STRUCTURE:
        return ALLOY_STRUCTURE[alloy_name]
    # Fuzzy match
    key_lower = alloy_name.lower().replace(' ', '').replace('-', '')
    for name, struct in ALLOY_STRUCTURE.items():
        if name.lower().replace(' ', '').replace('-', '') == key_lower:
            return struct
    raise KeyError(
        f"Unknown alloy '{alloy_name}'. "
        f"Known: {', '.join(sorted(ALLOY_STRUCTURE.keys()))}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='delta-theory am-fatigue',
        description='δ-theory AM alloy fatigue life prediction',
    )
    sub = p.add_subparsers(dest='cmd')

    # --- predict ---
    sp = sub.add_parser('predict', help='Predict fatigue life for single condition')
    sp.add_argument('--alloy', type=str, help='Alloy name (e.g. "Ti-6Al-4V")')
    sp.add_argument('--structure', type=str, choices=['BCC', 'FCC', 'HCP'],
                    help='Crystal structure (auto if --alloy given)')
    sp.add_argument('--sigma-a', type=float, required=True, help='Stress amplitude [MPa]')
    sp.add_argument('--sigma-y', type=float, required=True, help='Yield stress [MPa]')
    sp.add_argument('--sigma-uts', type=float, required=True, help='UTS [MPa]')

    # --- curve ---
    sc = sub.add_parser('curve', help='Generate S-N curve')
    sc.add_argument('--alloy', type=str)
    sc.add_argument('--structure', type=str, choices=['BCC', 'FCC', 'HCP'])
    sc.add_argument('--sigma-y', type=float, required=True)
    sc.add_argument('--sigma-uts', type=float, required=True)
    sc.add_argument('--out', type=str, default='am_sn_curve.png')

    # --- list ---
    sub.add_parser('list', help='List known AM alloys and structures')

    return p


def _resolve_structure(args) -> str:
    if args.alloy:
        return get_structure(args.alloy)
    if args.structure:
        return args.structure
    raise ValueError("Specify --alloy or --structure")


def cmd_predict(args):
    structure = _resolve_structure(args)
    result = am_fatigue_life(args.sigma_a, args.sigma_y, args.sigma_uts, structure)
    alloy_tag = args.alloy or structure

    print(f"\n  δ-theory AM Fatigue Prediction")
    print(f"  {'='*45}")
    print(f"  Alloy/Structure : {alloy_tag} ({structure})")
    print(f"  σ_a             : {args.sigma_a:.1f} MPa")
    print(f"  σ_y             : {args.sigma_y:.1f} MPa")
    print(f"  σ_uts           : {args.sigma_uts:.1f} MPa")
    print(f"  r = σ_a/σ_y     : {result['r']:.4f}")
    print(f"  r_u = σ_uts/σ_y : {result['r_u']:.4f}")
    print(f"  {'-'*45}")
    print(f"  N_init          : {result['N_init']:.3e}")
    print(f"  N_prop          : {result['N_prop']:.3e}")
    print(f"  N_static        : {result['N_static']:.3e}")
    print(f"  {'-'*45}")
    nf = result['N_f']
    if nf == 0:
        print(f"  N_f             : INSTANT FRACTURE (r ≥ r_u)")
    elif np.isinf(nf):
        print(f"  N_f             : ∞ (below threshold)")
    else:
        print(f"  N_f             : {nf:.3e} cycles ({result['regime']})")
    print()


def cmd_curve(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    structure = _resolve_structure(args)
    alloy_tag = args.alloy or structure

    sigma_a, N_f = am_sn_curve(args.sigma_y, args.sigma_uts, structure)

    fig, ax = plt.subplots(figsize=(10, 6))
    valid = N_f > 0
    ax.plot(N_f[valid], sigma_a[valid], 'b-', linewidth=2.5)
    ax.set_xscale('log')
    ax.set_xlabel('N_f [cycles]', fontsize=12)
    ax.set_ylabel('σ_a [MPa]', fontsize=12)
    ax.set_title(
        f'δ-theory AM S-N: {alloy_tag} ({structure})\n'
        f'σ_y={args.sigma_y:.0f} MPa, σ_uts={args.sigma_uts:.0f} MPa',
        fontsize=13, fontweight='bold',
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1e2, 1e10)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {args.out}")


def cmd_list(args):
    print(f"\n  Known AM Alloys ({len(ALLOY_STRUCTURE)})")
    print(f"  {'='*50}")
    for struct in ['BCC', 'FCC', 'HCP']:
        alloys = [k for k, v in ALLOY_STRUCTURE.items() if v == struct]
        print(f"\n  {struct} ({len(alloys)}):")
        for a in sorted(alloys):
            print(f"    {a}")
    print()


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == 'predict':
        cmd_predict(args)
    elif args.cmd == 'curve':
        cmd_curve(args)
    elif args.cmd == 'list':
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
