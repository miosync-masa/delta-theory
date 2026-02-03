#!/usr/bin/env python3
"""Unified Yield + δ-fatigue (v6.9b)

- Yield model: v5.0 FINAL
  σ_y = σ_base(δ) + Δσ_ss(c) + Δσ_ρ(ε) + Δσ_ppt(r,f)

- Fatigue model: v6.8 δ-fatigue damage
  dD/dN = 0 (r<=r_th) else A_eff * (r-r_th)^n
  with r = (amplitude)/(yield in the same loading mode)
  and failure when Λ(D)=D/(1-D) reaches 1 (i.e., D>=0.5) by default.

- Added from v4.1 (was missing in original v6.9 integration)
  τ/σ = (α_s/α_t) * C_class * T_twin * A_texture
  σ_c/σ_t = R_comp  (twinning/asymmetry)

This lets v6.9 operate not only with tensile amplitude σ_a, but also
with shear amplitude τ_a or compression amplitude σ_a^c.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import Dict, Literal, Optional, Tuple
import random

import numpy as np

# ==============================================================================
# Optional import: DBT/DBTT unified model (separate module)
# ------------------------------------------------------------------------------
# Place dbt_unified.py in the same folder as this file.
# Then you can do:
#   from unified_yield_fatigue_v6_9b_tau_classes import DBTUnified
#   model = DBTUnified()
#
# The import is optional so that this module remains usable even if dbt_unified.py
# is not present in your runtime environment.
try:
    from dbt_unified import DBTUnified, DBTCore, Material as DBTMaterial, MATERIAL_FE
except Exception:
    DBTUnified = None
    DBTCore = None
    DBTMaterial = None
    MATERIAL_FE = None

# ==============================================================================
# Physical constants
# ==============================================================================
PI = np.pi
ELECTRON_CHARGE_J = 1.602176634e-19  # 1 eV in Joule
M_TAYLOR = 3.0
ALPHA_TAYLOR = 0.3

# ==============================================================================
# v5.0 δ-yield constants
# ==============================================================================
ALPHA_DELTA = {'BCC': 0.289, 'FCC': 0.250, 'HCP': 0.350}
N_EXPONENT = {'interstitial': 0.90, 'substitutional': 0.95}

# Work hardening presets
K_RHO = {
    'Fe': 1.24e15, 'Cu': 1.07e15, 'Al': 5.98e14, 'Ni': 1.27e15,
    'BCC': 1.2e15, 'FCC': 1.0e15, 'HCP': 0.8e15,
}

# ==============================================================================
# v4.1 τ/σ factors
# ==============================================================================
CU_TAU_SIGMA_TORSION = 0.565  # Cu torsion calibration point
DEFAULT_BCC_W110 = 0.0        # {112} dominant

# Twinning factor (tension) and compression/tension asymmetry
T_TWIN = {
    'Cu': 1.0, 'Al': 1.0, 'Ni': 1.0, 'Au': 1.0, 'Ag': 1.0,
    'Fe': 1.0, 'W':  1.0,
    'Ti': 1.0,
    'Mg': 0.6,
    'Zn': 0.9,
}

R_COMP = {  # σ_c/σ_t
    'Cu': 1.0, 'Al': 1.0, 'Ni': 1.0, 'Au': 1.0, 'Ag': 1.0,
    'Fe': 1.0, 'W':  1.0,
    'Ti': 1.0,
    'Mg': 0.6,
    'Zn': 1.2,
}

T_REF = {
    'BCC': np.array([1, 0, 0], dtype=float),
    'FCC': np.array([1, 1, 0], dtype=float),
    'HCP': np.array([1, 0, 0], dtype=float),
}

# ==============================================================================
# Material database
# ==============================================================================
@dataclass(frozen=True)
class Material:
    name: str
    structure: Literal['BCC', 'FCC', 'HCP']

    # v5.0 base-yield inputs
    a: float      # lattice parameter [m]
    T_m: float    # melting point [K]
    dL: float     # Lindemann parameter
    Eb: float     # bond energy [eV]
    f_d: float    # d-electron directionality factor
    G: float      # shear modulus [Pa]

    # v4.1 τ/σ inputs
    c_a: float = 1.633
    A_texture: float = 1.0
    T_twin: float = 1.0
    R_comp: float = 1.0


MATERIALS: Dict[str, Material] = {
    'Fe': Material('Fe', 'BCC', 2.92e-10, 1811, 0.18, 4.28, 1.5, 82e9, 1.633, 1.0, T_TWIN['Fe'], R_COMP['Fe']),
    'W':  Material('W',  'BCC', 3.16e-10, 3695, 0.16, 8.90, 4.7, 161e9, 1.633, 1.0, T_TWIN['W'],  R_COMP['W']),
    'Cu': Material('Cu', 'FCC', 3.61e-10, 1357, 0.10, 3.49, 2.0, 48e9, 1.633, 1.0, T_TWIN['Cu'], R_COMP['Cu']),
    'Al': Material('Al', 'FCC', 4.05e-10, 933,  0.10, 3.39, 1.6, 26e9, 1.633, 1.0, T_TWIN['Al'], R_COMP['Al']),
    'Ni': Material('Ni', 'FCC', 3.52e-10, 1728, 0.11, 4.44, 2.6, 76e9, 1.633, 1.0, T_TWIN['Ni'], R_COMP['Ni']),
    'Au': Material('Au', 'FCC', 4.08e-10, 1337, 0.10, 3.81, 1.1, 27e9, 1.633, 1.0, T_TWIN['Au'], R_COMP['Au']),
    'Ag': Material('Ag', 'FCC', 4.09e-10, 1235, 0.10, 2.95, 2.0, 30e9, 1.633, 1.0, T_TWIN['Ag'], R_COMP['Ag']),
    'Ti': Material('Ti', 'HCP', 2.95e-10, 1941, 0.10, 4.85, 5.7, 44e9, 1.587, 1.0, T_TWIN['Ti'], R_COMP['Ti']),
    'Mg': Material('Mg', 'HCP', 3.21e-10, 923,  0.117,1.51, 8.2, 17e9, 1.624, 1.0, T_TWIN['Mg'], R_COMP['Mg']),
    'Zn': Material('Zn', 'HCP', 2.66e-10, 693,  0.12, 1.35, 2.0, 43e9, 1.856, 1.0, T_TWIN['Zn'], R_COMP['Zn']),
}

# ==============================================================================
# Helpers
# ==============================================================================

def calc_burgers(a: float, structure: str) -> float:
    if structure == 'BCC':
        return a * np.sqrt(3) / 2
    if structure == 'FCC':
        return a / np.sqrt(2)
    return a

# ==============================================================================
# v5.0 yield components
# ==============================================================================

def sigma_base_delta(mat: Material, T_K: float = 300.0) -> float:
    """Base yield stress from δ-theory [MPa]."""
    alpha = ALPHA_DELTA[mat.structure]
    b = calc_burgers(mat.a, mat.structure)
    V_act = b**3
    HP = max(0.0, 1.0 - T_K / mat.T_m)
    E_eff = mat.Eb * ELECTRON_CHARGE_J * alpha * mat.f_d
    sigma = (E_eff / V_act) * mat.dL * HP / (2 * PI * M_TAYLOR)
    return sigma / 1e6


def delta_sigma_ss(c_wt_percent: float, k: float,
                   solute_type: Optional[Literal['interstitial', 'substitutional']]) -> float:
    if solute_type is None or c_wt_percent <= 0:
        return 0.0
    n = N_EXPONENT[solute_type]
    return k * (c_wt_percent ** n)


def calibrate_k_ss(c_wt_percent: float, sigma_exp_MPa: float,
                   sigma_base_MPa: float,
                   solute_type: Literal['interstitial', 'substitutional']) -> float:
    n = N_EXPONENT[solute_type]
    return (sigma_exp_MPa - sigma_base_MPa) / (c_wt_percent ** n)


def delta_sigma_taylor(eps: float, mat: Material, rho_0: float = 1e12) -> float:
    """Work hardening via Taylor relation [MPa]."""
    if eps <= 0 and rho_0 <= 0:
        return 0.0
    K = K_RHO.get(mat.name, K_RHO.get(mat.structure, 1e15))
    rho = rho_0 + K * max(eps, 0.0)
    b = calc_burgers(mat.a, mat.structure)
    return M_TAYLOR * ALPHA_TAYLOR * mat.G * b * np.sqrt(rho) / 1e6


def delta_sigma_cutting(r_nm: float, f: float, gamma: float, G: float, b: float) -> float:
    r = r_nm * 1e-9
    if r <= 0 or f <= 0 or gamma <= 0:
        return 0.0
    T_line = 0.5 * G * b**2
    return M_TAYLOR * (gamma / (2 * b)) * np.sqrt(3 * PI * f * r / T_line) / 1e6


def delta_sigma_orowan(r_nm: float, f: float, G: float, b: float) -> float:
    r = r_nm * 1e-9
    if r <= 0 or f <= 0:
        return 0.0
    factor = max(0.1, (2 * PI / (3 * f))**0.5 - 2)
    lambda_eff = 2 * r * factor
    log_term = np.log(max(2 * r / b, 2))
    return 0.4 * M_TAYLOR * G * b / lambda_eff * log_term / 1e6


def delta_sigma_ppt(r_nm: float, f: float, gamma: float, mat: Material, A: float = 1.0) -> Tuple[float, str]:
    if r_nm <= 0 or f <= 0:
        return 0.0, 'None'
    b = calc_burgers(mat.a, mat.structure)
    d_cut = A * delta_sigma_cutting(r_nm, f, gamma, mat.G, b)
    d_oro = A * delta_sigma_orowan(r_nm, f, mat.G, b)
    return (d_cut, 'Cutting') if d_cut <= d_oro else (d_oro, 'Orowan')


def calc_sigma_y(
    mat: Material,
    T_K: float = 300.0,
    c_wt_percent: float = 0.0,
    k_ss: float = 0.0,
    solute_type: Optional[Literal['interstitial', 'substitutional']] = None,
    eps: float = 0.0,
    rho_0: float = 0.0,
    r_ppt_nm: float = 0.0,
    f_ppt: float = 0.0,
    gamma_apb: float = 0.0,
    A_ppt: float = 1.0,
) -> Dict[str, float | str]:
    base = sigma_base_delta(mat, T_K)
    ss = delta_sigma_ss(c_wt_percent, k_ss, solute_type)
    wh = delta_sigma_taylor(eps, mat, rho_0) if (eps > 0 or rho_0 > 0) else 0.0
    ppt, mech = delta_sigma_ppt(r_ppt_nm, f_ppt, gamma_apb, mat, A_ppt)
    return {
        'sigma_y': base + ss + wh + ppt,
        'sigma_base': base,
        'delta_ss': ss,
        'delta_wh': wh,
        'delta_ppt': ppt,
        'ppt_mechanism': mech,
    }

# ==============================================================================
# v4.1: α_s/α_t and τ/σ
# ==============================================================================

def get_bond_vectors(structure: str, c_a_ratio: float = 1.633):
    bonds = []
    if structure == 'BCC':
        for i in (-1, 1):
            for j in (-1, 1):
                for k in (-1, 1):
                    v = np.array([i, j, k], dtype=float)
                    bonds.append(v / np.linalg.norm(v))
    elif structure == 'FCC':
        for i in (-1, 1):
            for j in (-1, 1):
                bonds.append(np.array([i, j, 0], dtype=float) / np.sqrt(2))
                bonds.append(np.array([i, 0, j], dtype=float) / np.sqrt(2))
                bonds.append(np.array([0, i, j], dtype=float) / np.sqrt(2))
    elif structure == 'HCP':
        # basal-plane
        for i in range(6):
            angle = i * PI / 3
            bonds.append(np.array([np.cos(angle), np.sin(angle), 0.0]))
        # out-of-plane
        r_xy = 1.0 / np.sqrt(3)
        z = c_a_ratio / 2
        length = np.sqrt(r_xy**2 + z**2)
        r_norm, z_norm = r_xy / length, z / length
        for i in range(3):
            angle = i * 2 * PI / 3 + PI / 6
            bonds.append(np.array([r_norm * np.cos(angle), r_norm * np.sin(angle), z_norm]))
            bonds.append(np.array([r_norm * np.cos(angle + PI), r_norm * np.sin(angle + PI), -z_norm]))
    return bonds


def calc_alpha_tensile(bonds, tensile_dir: np.ndarray) -> float:
    d = tensile_dir / np.linalg.norm(tensile_dir)
    Z = len(bonds)
    return sum(max(float(np.dot(b, d)), 0.0) for b in bonds) / Z


def calc_alpha_shear(bonds, n: np.ndarray, s: np.ndarray) -> float:
    n = n / np.linalg.norm(n)
    s = s / np.linalg.norm(s)
    Z = len(bonds)
    return sum(abs(float(np.dot(b, n))) * abs(float(np.dot(b, s))) for b in bonds) / Z


def get_slip_system(structure: str, variant: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    if structure == 'BCC':
        v = variant or '112'
        if v == '110':
            return (np.array([1, 1, 0], dtype=float) / np.sqrt(2),
                    np.array([1, -1, 1], dtype=float) / np.sqrt(3))
        return (np.array([1, 1, 2], dtype=float) / np.sqrt(6),
                np.array([1, 1, -1], dtype=float) / np.sqrt(3))
    if structure == 'FCC':
        return (np.array([1, 1, 1], dtype=float) / np.sqrt(3),
                np.array([1, -1, 0], dtype=float) / np.sqrt(2))
    # HCP: basal (simplest)
    return (np.array([0, 0, 1], dtype=float), np.array([1, 0, 0], dtype=float))


def calc_alpha_ratio(structure: str, c_a: float = 1.633, bcc_w110: float = DEFAULT_BCC_W110) -> Dict[str, float]:
    bonds = get_bond_vectors(structure, c_a)
    alpha_t = calc_alpha_tensile(bonds, T_REF[structure])

    if structure == 'BCC':
        n110, s110 = get_slip_system('BCC', '110')
        n112, s112 = get_slip_system('BCC', '112')
        a110 = calc_alpha_shear(bonds, n110, s110)
        a112 = calc_alpha_shear(bonds, n112, s112)
        alpha_s = bcc_w110 * a110 + (1.0 - bcc_w110) * a112
        return {
            'alpha_t': alpha_t,
            'alpha_s': alpha_s,
            'ratio': alpha_s / alpha_t,
            'ratio_110': a110 / alpha_t,
            'ratio_112': a112 / alpha_t,
        }

    n, s = get_slip_system(structure)
    alpha_s = calc_alpha_shear(bonds, n, s)
    return {'alpha_t': alpha_t, 'alpha_s': alpha_s, 'ratio': alpha_s / alpha_t}


def calibrate_C_class() -> float:
    fcc = calc_alpha_ratio('FCC')
    return CU_TAU_SIGMA_TORSION / fcc['ratio']


C_CLASS_DEFAULT = calibrate_C_class()


def tau_over_sigma(mat: Material,
                   C_class: float = C_CLASS_DEFAULT,
                   bcc_w110: float = DEFAULT_BCC_W110,
                   apply_C_class_hcp: bool = False) -> float:
    al = calc_alpha_ratio(mat.structure, mat.c_a, bcc_w110)
    c_cls = C_class if (mat.structure != 'HCP' or apply_C_class_hcp) else 1.0
    return al['ratio'] * c_cls * mat.T_twin * mat.A_texture


def sigma_c_over_sigma_t(mat: Material) -> float:
    return mat.R_comp


def yield_by_mode(
    mat: Material,
    sigma_y_tension_MPa: float,
    mode: Literal['tensile', 'compression', 'shear'] = 'tensile',
    C_class: float = C_CLASS_DEFAULT,
    bcc_w110: float = DEFAULT_BCC_W110,
    apply_C_class_hcp: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """Return yield level [MPa] in the requested loading mode + diagnostics."""
    tau_sig = tau_over_sigma(mat, C_class=C_class, bcc_w110=bcc_w110, apply_C_class_hcp=apply_C_class_hcp)
    R = sigma_c_over_sigma_t(mat)
    sigma_y_comp = sigma_y_tension_MPa * R
    tau_y = sigma_y_tension_MPa * tau_sig

    if mode == 'tensile':
        y = sigma_y_tension_MPa
    elif mode == 'compression':
        y = sigma_y_comp
    else:
        y = tau_y

    return y, {
        'tau_over_sigma': tau_sig,
        'sigma_c_over_sigma_t': R,
        'sigma_y_tension': sigma_y_tension_MPa,
        'sigma_y_compression': sigma_y_comp,
        'tau_y': tau_y,
    }

# ==============================================================================
# v6.8 δ-fatigue damage model
# ==============================================================================

FATIGUE_CLASS_PRESET = {
    'BCC': {'r_th': 0.65, 'n': 10.0},
    'FCC': {'r_th': 0.02, 'n': 7.0},
    'HCP': {'r_th': 0.20, 'n': 9.0},
}

# A_int from δ parameters (normalized to Fe)
A_INT_DB = {
    'Fe': 1.00,
    'Cu': 1.41,
    'Al': 0.71,
    'Ni': 1.37,
    'W':  0.85,
    'Ti': 1.10,
    'Mg': 0.60,
    'Zn': 0.75,
    'Au': 1.00,
    'Ag': 1.00,
}


def lambda_from_damage(D: float) -> float:
    """Λ(D)=D/(1-D)."""
    if D <= 0:
        return 0.0
    if D >= 1:
        return float('inf')
    return D / (1.0 - D)


def solve_N_fail(
    r: float,
    r_th: float,
    n: float,
    A_eff: float,
    D0: float = 0.0,
    D_fail: float = 0.5,
) -> float:
    """Closed-form cycles to reach D_fail for constant r."""
    if r <= r_th:
        return float('inf')
    if A_eff <= 0:
        return float('inf')
    rate = A_eff * (r - r_th) ** n
    if rate <= 0:
        return float('inf')
    return (D_fail - D0) / rate


def fatigue_life_const_amp(
    mat: Material,
    sigma_a_MPa: float,
    sigma_y_tension_MPa: float,
    A_ext: float,
    mode: Literal['tensile', 'compression', 'shear'] = 'tensile',
    D0: float = 0.0,
    D_fail: float = 0.5,
    C_class: float = C_CLASS_DEFAULT,
    bcc_w110: float = DEFAULT_BCC_W110,
    apply_C_class_hcp: bool = False,
    r_th_override: float | None = None,
    n_override: float | None = None,
) -> Dict[str, float | str]:
    """Fatigue life under constant amplitude for the chosen loading mode."""

    preset = FATIGUE_CLASS_PRESET[mat.structure]
    r_th = r_th_override if r_th_override is not None else preset['r_th']
    n = n_override if n_override is not None else preset['n']

    A_int = A_INT_DB.get(mat.name, 1.0)
    A_eff = A_int * A_ext

    y_mode, diag = yield_by_mode(
        mat,
        sigma_y_tension_MPa=sigma_y_tension_MPa,
        mode=mode,
        C_class=C_class,
        bcc_w110=bcc_w110,
        apply_C_class_hcp=apply_C_class_hcp,
    )

    # In shear mode, sigma_a_MPa is interpreted as τ_a [MPa]
    r = sigma_a_MPa / y_mode if y_mode > 0 else float('inf')

    N_fail = solve_N_fail(r, r_th, n, A_eff, D0=D0, D_fail=D_fail)

    return {
        'mode': mode,
        'amp_input_MPa': sigma_a_MPa,
        'yield_mode_MPa': y_mode,
        'r': r,
        'r_th': r_th,
        'n': n,
        'A_int': A_int,
        'A_ext': A_ext,
        'A_eff': A_eff,
        'D0': D0,
        'D_fail': D_fail,
        'Lambda_fail': lambda_from_damage(D_fail),
        **diag,
        'N_fail': N_fail,
        'log10_N_fail': (np.log10(N_fail) if np.isfinite(N_fail) and N_fail > 0 else float('inf')),
    }


def generate_sn_curve(
    mat: Material,
    sigma_y_tension_MPa: float,
    A_ext: float,
    sigmas_MPa: np.ndarray,
    mode: Literal['tensile', 'compression', 'shear'] = 'tensile',
    D_fail: float = 0.5,
    C_class: float = C_CLASS_DEFAULT,
    bcc_w110: float = DEFAULT_BCC_W110,
    apply_C_class_hcp: bool = False,
    r_th_override: float | None = None,
    n_override: float | None = None,
) -> np.ndarray:
    Ns = []
    for s in sigmas_MPa:
        out = fatigue_life_const_amp(
            mat,
            sigma_a_MPa=float(s),
            sigma_y_tension_MPa=sigma_y_tension_MPa,
            A_ext=A_ext,
            mode=mode,
            D_fail=D_fail,
            C_class=C_class,
            bcc_w110=bcc_w110,
            apply_C_class_hcp=apply_C_class_hcp,
            r_th_override=r_th_override,
            n_override=n_override,
        )
        Ns.append(out['N_fail'])
    return np.array(Ns, dtype=float)

# ==============================================================================
# Alloy validation helper (for FatigueDB integration)
# ==============================================================================

def get_minimal_material(structure: Literal['BCC', 'FCC', 'HCP']) -> Material:
    """結晶構造から最小限のMaterialを取得（合金検証用ヘルパー）
    
    FatigueDB等の実験データ検証時、σ_yは実測値を使うため
    structure (r_th, n_cl) のみが必要なケース用
    """
    base = {'BCC': MATERIALS['Fe'], 'FCC': MATERIALS['Cu'], 'HCP': MATERIALS['Ti']}
    return base[structure]


# ==============================================================================
# CLI
# ==============================================================================

def cmd_point(args: argparse.Namespace) -> None:
    # material 取得（metal or structure_only）
    if args.metal is None and args.structure_only is None:
        raise SystemExit("Error: --metal or --structure_only required")
    
    if args.structure_only:
        mat = get_minimal_material(args.structure_only)
    else:
        mat0 = MATERIALS[args.metal]
        mat = replace(
            mat0,
            A_texture=float(args.A_texture),
            T_twin=(float(args.T_twin) if args.T_twin is not None else mat0.T_twin),
            R_comp=(float(args.R_comp) if args.R_comp is not None else mat0.R_comp),
            c_a=float(args.c_a) if args.c_a is not None else mat0.c_a,
        )

    # σ_y 計算 or override
    if args.sigma_y_override is not None:
        sigma_y = args.sigma_y_override
        y = {
            'sigma_y': sigma_y,
            'sigma_base': sigma_y,
            'delta_ss': 0.0,
            'delta_wh': 0.0,
            'delta_ppt': 0.0,
            'ppt_mechanism': 'N/A (override)',
        }
    else:
        y = calc_sigma_y(
            mat,
            T_K=args.T_K,
            c_wt_percent=args.c_wt,
            k_ss=args.k_ss,
            solute_type=args.solute_type,
            eps=args.eps,
            rho_0=args.rho_0,
            r_ppt_nm=args.r_ppt_nm,
            f_ppt=args.f_ppt,
            gamma_apb=args.gamma_apb,
            A_ppt=args.A_ppt,
        )
        sigma_y = y['sigma_y']

    # Fatigue (optional)
    if args.sigma_a is not None:
        out = fatigue_life_const_amp(
            mat,
            sigma_a_MPa=float(args.sigma_a),
            sigma_y_tension_MPa=float(sigma_y),
            A_ext=float(args.A_ext),
            mode=args.mode,
            D_fail=args.D_fail,
            C_class=args.C_class,
            bcc_w110=args.bcc_w110,
            apply_C_class_hcp=args.apply_C_class_hcp,
            r_th_override=args.r_th,
            n_override=args.n_exp,
        )
    else:
        out = None

    # 表示
    label = f"structure={mat.structure}" if args.structure_only else f"metal={mat.name} ({mat.structure})"
    print("=" * 88)
    print(f"v6.9b point | {label} | mode={args.mode}")
    print("=" * 88)

    # Yield summary
    y_mode, diag = yield_by_mode(
        mat,
        sigma_y_tension_MPa=float(sigma_y),
        mode=args.mode,
        C_class=args.C_class,
        bcc_w110=args.bcc_w110,
        apply_C_class_hcp=args.apply_C_class_hcp,
    )

    print("[Yield v5.0]")
    if args.sigma_y_override is not None:
        print(f"  σ_y(override) = {sigma_y:.2f} MPa")
    else:
        print(f"  σ_base   = {y['sigma_base']:.2f} MPa")
        print(f"  Δσ_ss    = {y['delta_ss']:.2f} MPa")
        print(f"  Δσ_wh    = {y['delta_wh']:.2f} MPa  (rho_0={args.rho_0:.2e})")
        print(f"  Δσ_ppt   = {y['delta_ppt']:.2f} MPa  ({y['ppt_mechanism']})")
        print(f"  σ_y(t)   = {y['sigma_y']:.2f} MPa")
    
    print("[Class factors v4.1]")
    print(f"  C_class  = {args.C_class:.4f}  (apply to HCP: {args.apply_C_class_hcp})")
    print(f"  bcc_w110 = {args.bcc_w110:.3f}")
    print(f"  A_texture= {mat.A_texture:.3f}")
    print(f"  T_twin   = {mat.T_twin:.3f}")
    print(f"  R_comp   = {mat.R_comp:.3f} (σ_c/σ_t)")
    print(f"  τ/σ_pred = {diag['tau_over_sigma']:.4f}")
    print(f"  τ_y      = {diag['tau_y']:.2f} MPa")
    print(f"  σ_y(c)   = {diag['sigma_y_compression']:.2f} MPa")
    print(f"  Yield(mode) = {y_mode:.2f} MPa")

    if out is None:
        return

    print("\n[Fatigue v6.8]")
    unit = "MPa" if args.mode != 'shear' else "MPa (τ_a)"
    print(f"  amplitude input = {out['amp_input_MPa']:.2f} {unit}")
    print(f"  r = amp / yield(mode) = {out['r']:.5f}")
    print(f"  r_th={out['r_th']:.3f}, n={out['n']:.1f}")
    print(f"  A_int={out['A_int']:.3f}, A_ext={out['A_ext']:.3e} => A_eff={out['A_eff']:.3e}")
    print(f"  D_fail={out['D_fail']:.3f} (Λ_fail={out['Lambda_fail']:.3f})")
    if np.isfinite(out['N_fail']):
        print(f"  N_fail = {out['N_fail']:.3e} cycles  (log10={out['log10_N_fail']:.3f})")
    else:
        print("  N_fail = inf (fatigue limit region)")


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Calibrate A_ext from one (σ_a, N_fail) point."""
    # material 取得
    if args.metal is None and args.structure_only is None:
        raise SystemExit("Error: --metal or --structure_only required")
    
    if args.structure_only:
        mat = get_minimal_material(args.structure_only)
    else:
        mat0 = MATERIALS[args.metal]
        mat = replace(
            mat0,
            A_texture=float(args.A_texture),
            T_twin=(float(args.T_twin) if args.T_twin is not None else mat0.T_twin),
            R_comp=(float(args.R_comp) if args.R_comp is not None else mat0.R_comp),
            c_a=float(args.c_a) if args.c_a is not None else mat0.c_a,
        )

    # σ_y
    if args.sigma_y_override is not None:
        sigma_y = args.sigma_y_override
    else:
        y = calc_sigma_y(
            mat,
            T_K=args.T_K,
            c_wt_percent=args.c_wt,
            k_ss=args.k_ss,
            solute_type=args.solute_type,
            eps=args.eps,
            rho_0=args.rho_0,
            r_ppt_nm=args.r_ppt_nm,
            f_ppt=args.f_ppt,
            gamma_apb=args.gamma_apb,
            A_ppt=args.A_ppt,
        )
        sigma_y = y['sigma_y']

    preset = FATIGUE_CLASS_PRESET[mat.structure]
    r_th = args.r_th if args.r_th is not None else preset['r_th']
    n = args.n_exp if args.n_exp is not None else preset['n']

    y_mode, _ = yield_by_mode(
        mat,
        sigma_y_tension_MPa=float(sigma_y),
        mode=args.mode,
        C_class=args.C_class,
        bcc_w110=args.bcc_w110,
        apply_C_class_hcp=args.apply_C_class_hcp,
    )

    r = args.sigma_a / y_mode

    if r <= r_th:
        raise SystemExit("Calibration point is below r_th (fatigue limit); choose a point with finite life.")

    A_int = A_INT_DB.get(mat.name, 1.0)
    D_fail = args.D_fail
    D0 = 0.0

    rate_needed = (D_fail - D0) / args.N_fail
    A_eff = rate_needed / ((r - r_th) ** n)
    A_ext = A_eff / A_int

    label = f"structure={mat.structure}" if args.structure_only else f"metal={mat.name} ({mat.structure})"
    print("=" * 88)
    print("v6.9b calibrate A_ext")
    print("=" * 88)
    print(f"{label}, mode={args.mode}")
    print(f"σ_y = {sigma_y:.3f} MPa {'(override)' if args.sigma_y_override else '(calc)'}")
    print(f"yield(mode)  = {y_mode:.3f} MPa")
    print(f"amp          = {args.sigma_a:.3f} MPa {'(τ_a)' if args.mode=='shear' else ''}")
    print(f"r={r:.6f}, r_th={r_th:.3f}, n={n:.2f}")
    print(f"D_fail={D_fail:.3f}, N_fail={args.N_fail:.3e}")
    print(f"A_int={A_int:.3f} => A_ext={A_ext:.3e} (A_eff={A_eff:.3e})")


def cmd_sn(args: argparse.Namespace) -> None:
    # material 取得
    if args.metal is None and args.structure_only is None:
        raise SystemExit("Error: --metal or --structure_only required")
    
    if args.structure_only:
        mat = get_minimal_material(args.structure_only)
    else:
        mat0 = MATERIALS[args.metal]
        mat = replace(
            mat0,
            A_texture=float(args.A_texture),
            T_twin=(float(args.T_twin) if args.T_twin is not None else mat0.T_twin),
            R_comp=(float(args.R_comp) if args.R_comp is not None else mat0.R_comp),
            c_a=float(args.c_a) if args.c_a is not None else mat0.c_a,
        )

    # σ_y
    if args.sigma_y_override is not None:
        sigma_y = args.sigma_y_override
    else:
        y = calc_sigma_y(
            mat,
            T_K=args.T_K,
            c_wt_percent=args.c_wt,
            k_ss=args.k_ss,
            solute_type=args.solute_type,
            eps=args.eps,
            rho_0=args.rho_0,
            r_ppt_nm=args.r_ppt_nm,
            f_ppt=args.f_ppt,
            gamma_apb=args.gamma_apb,
            A_ppt=args.A_ppt,
        )
        sigma_y = y['sigma_y']

    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.num)
    Ns = generate_sn_curve(
        mat,
        sigma_y_tension_MPa=float(sigma_y),
        A_ext=args.A_ext,
        sigmas_MPa=sigmas,
        mode=args.mode,
        D_fail=args.D_fail,
        C_class=args.C_class,
        bcc_w110=args.bcc_w110,
        apply_C_class_hcp=args.apply_C_class_hcp,
        r_th_override=args.r_th,
        n_override=args.n_exp,
    )

    label = f"structure={mat.structure}" if args.structure_only else f"metal={mat.name} ({mat.structure})"
    print("=" * 88)
    print(f"v6.9b S-N | {label} | mode={args.mode}")
    print("=" * 88)
    print(f"σ_y={sigma_y:.3f} MPa {'(override)' if args.sigma_y_override else '(calc)'} | A_ext={args.A_ext:.3e} | D_fail={args.D_fail:.3f}")

    header_amp = 'sigma_a_MPa' if args.mode != 'shear' else 'tau_a_MPa'
    print(f"{header_amp:>12} {'N_fail':>14} {'log10N':>10} {'r':>10} {'note':>10}")
    for s, N in zip(sigmas, Ns):
        y_mode, _ = yield_by_mode(
            mat,
            sigma_y_tension_MPa=float(sigma_y),
            mode=args.mode,
            C_class=args.C_class,
            bcc_w110=args.bcc_w110,
            apply_C_class_hcp=args.apply_C_class_hcp,
        )
        r = s / y_mode
        if np.isfinite(N):
            print(f"{s:12.2f} {N:14.3e} {np.log10(N):10.3f} {r:10.4f} {'':>10}")
        else:
            print(f"{s:12.2f} {'inf':>14} {'inf':>10} {r:10.4f} {'limit':>10}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='v6.9b: v5.0 yield × v6.8 fatigue + τ/σ, twins, texture')
    sub = p.add_subparsers(dest='cmd', required=True)

    def add_common(sp: argparse.ArgumentParser):
        # metal OR structure_only
        sp.add_argument('--metal', choices=sorted(MATERIALS.keys()), default=None,
                        help='Material from database')
        sp.add_argument('--structure_only', choices=['BCC', 'FCC', 'HCP'], default=None,
                        help='Use structure preset only (for alloy validation)')
        sp.add_argument('--sigma_y_override', type=float, default=None,
                        help='Override σ_y with experimental value [MPa]')
        
        sp.add_argument('--T_K', type=float, default=300.0)
        sp.add_argument('--c_wt', type=float, default=0.0, help='solute wt%% (e.g., 0.10 for 0.10 wt%%)')
        sp.add_argument('--k_ss', type=float, default=0.0, help='solid-solution k [MPa/(wt%%)^n]')
        sp.add_argument('--solute_type', choices=['interstitial', 'substitutional'], default=None)
        sp.add_argument('--eps', type=float, default=0.0, help='monotonic strain for work hardening')
        sp.add_argument('--rho_0', type=float, default=0.0, help='initial dislocation density [m^-2]')
        sp.add_argument('--r_ppt_nm', type=float, default=0.0)
        sp.add_argument('--f_ppt', type=float, default=0.0)
        sp.add_argument('--gamma_apb', type=float, default=0.0)
        sp.add_argument('--A_ppt', type=float, default=1.0)

        # v4.1 class factors
        sp.add_argument('--A_texture', type=float, default=1.0)
        sp.add_argument('--T_twin', type=float, default=None, help='override T_twin')
        sp.add_argument('--R_comp', type=float, default=None, help='override σ_c/σ_t')
        sp.add_argument('--c_a', type=float, default=None, help='override HCP c/a')
        sp.add_argument('--C_class', type=float, default=C_CLASS_DEFAULT)
        sp.add_argument('--bcc_w110', type=float, default=DEFAULT_BCC_W110)
        sp.add_argument('--apply_C_class_hcp', action='store_true')

    def add_fatigue(sp: argparse.ArgumentParser):
        sp.add_argument('--mode', choices=['tensile', 'compression', 'shear'], default='tensile')
        sp.add_argument('--A_ext', type=float, default=2.46e-4, help='external factor (1-point calibration)')
        sp.add_argument('--D_fail', type=float, default=0.5)
        sp.add_argument('--r_th', type=float, default=None, help='Override r_th (fatigue threshold ratio)')
        sp.add_argument('--n_exp', type=float, default=None, help='Override n (fatigue exponent)')

    sp_point = sub.add_parser('point', help='compute yield (+ optional fatigue life)')
    add_common(sp_point)
    add_fatigue(sp_point)
    sp_point.add_argument('--sigma_a', type=float, default=None, help='amplitude [MPa]. If mode=shear, this is τ_a.')
    sp_point.set_defaults(func=cmd_point)

    sp_cal = sub.add_parser('calibrate', help='calibrate A_ext from one fatigue point')
    add_common(sp_cal)
    add_fatigue(sp_cal)
    sp_cal.add_argument('--sigma_a', type=float, required=True, help='amplitude [MPa]. If mode=shear, τ_a.')
    sp_cal.add_argument('--N_fail', type=float, required=True)
    sp_cal.set_defaults(func=cmd_calibrate)

    sp_sn = sub.add_parser('sn', help='generate S-N table')
    add_common(sp_sn)
    add_fatigue(sp_sn)
    sp_sn.add_argument('--sigma_min', type=float, required=True)
    sp_sn.add_argument('--sigma_max', type=float, required=True)
    sp_sn.add_argument('--num', type=int, default=25)
    sp_sn.set_defaults(func=cmd_sn)

    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

# ==============================================================================
# Banners (Rotation List)
# ==============================================================================
BANNERS = [
    # Banner 1: Original Standard
    """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ███╗    ███╗██║ ██████╗ ███████╗██╗   ██╗███╗   ██╗ ██████╗
  ████╗ ████║██║██╔═══██╗██╔════╝╚██╗ ██╔╝████╗  ██║██╔════╝
  ██╔████╔██║██║██║   ██║███████╗ ╚████╔╝ ██╔██╗ ██║██║     
  ██║╚██╔╝██║██║██║   ██║╚════██║  ╚██╔╝  ██║╚██╗██║██║     
  ██║ ╚═╝ ██║██║╚██████╔╝███████║   ██║   ██║ ╚████║╚██████╗
  ╚═╝     ╚═╝╚═╝ ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═══╝ ╚══════╝
                     δ-theory Unified Model v6.9b
          ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            v5.0 Yield × v6.8 Fatigue × τ/σ Class
          
      BCC: r_th=0.65   FCC: r_th=0.02   HCP: r_th=0.20
               Λ(D) = D/(1-D) → Λ=1 : failure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""",

    # Banner 2: Eureka Mode (Minimal & Mathematical)
    """
    ███████╗██╗   ██╗██████╗ ███████╗██╗  ██╗ █████╗ ██╗
    ██╔════╝██║   ██║██╔══██╗██╔════╝██║ ██╔╝██╔══██╗██║
    █████╗  ██║   ██║██████╔╝█████╗  █████╔╝ ███████║██║
    ██╔══╝  ██║   ██║██╔══██╗██╔══╝  ██╔═██╗ ██╔══██║╚═╝
    ███████╗╚██████╔╝██║  ██║███████╗██║  ██╗██║  ██║██╗
    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝
                   δ-Theory v8.0.0
                 "Nature is Geometry"
""",

    # Banner 3: Particle/Network Style (Technical)
    """
    .·:''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''':·.
     :  δ - U N I F I E D   S I M U L A T I O N   E N G I N E        :
     :  [ System: v6.9b ]  [ Status: Yield Criterion Stable ]         :
     '·...............................................................·'
          __      __                 __      __  ___ 
     |  |/  \ /| /  \  |  | |\ | |  |__  |  |  \  |  
     |/\| \__/  | \__/  |__| | \| |  |___ |/|__/  |  
                                                     
          -- Dislocation Density ρ_0 Tracking Active --
""",

    # Banner 4: Blueprint Style (Material Physics Focus)
    """
   ____________________________________________________________________
  |  ________________________________________________________________  |
  | |                                                                | |
  | |   CONSTITUTIVE EQUATION:                                       | |
  | |   σ_y = σ_base(δ) + Δσ_ss(c) + Δσ_ρ(ε) + Δσ_ppt(r,f)           | |
  | |                                                                | |
  | |   [ Miosyc-SYSTEM ]     [ Unified Yield Theory v5.0 ]          | |
  | |________________________________________________________________| |
  |____________________________________________________________________|
""",  

# Banner 5: Physics Revolution
    """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   F = ma  →  Λ = K/|V|_eff                                       ║
    ║                                                                   ║
    ║   m = Z × f_d  (topology)                                        ║
    ║   |V| = E_bond × α × f_d  (geometry)                             ║
    ║                                                                   ║
    ║              ── Beyond Newton, Into Geometry ──                  ║
    ║                                                                   ║
    ║                     δ-Theory v8.0.0                               ║
    ╚═══════════════════════════════════════════════════════════════════╝
"""
]

def main() -> None:
    # 起動時にランダムで表示
    print(random.choice(BANNERS))
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
