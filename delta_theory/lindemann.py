#!/usr/bin/env python3
"""
δ-theory Lindemann Module (v1.0)
================================

飯泉・リンデマンの式 (Iizumi-Lindemann Law)
-------------------------------------------
Parameter-free prediction of Lindemann ratio from first principles.

    δ_L = √(4/5) × (8/Z) × ξ_struct × √(k_B T_m / E_coh)

Where:
    √(4/5) ≈ 0.8944  : Universal geometric constant (BCC reference)
    8/Z              : Coordination number scaling
    ξ_struct         : Stacking structure factor
                       - BCC: 1.0 (isotropic cubic)
                       - FCC: 1.0 (isotropic cubic)  
                       - HCP: 7/8 (c-axis anisotropy penalty)
    k_B T_m / E_coh  : Thermal-to-cohesive energy ratio

Physical Interpretation:
    The Lindemann criterion describes the thermal amplitude threshold
    at which atomic "cage escape" occurs (melting). The √(4/5) factor
    emerges from BCC cage geometry, and the 8/Z scaling reflects
    how coordination number constrains atomic vibration amplitude.

Validation Results (7 metals, 3 structures):
    - Conventional √48/Z formula:     MAE = 7.4%
    - Iizumi-Lindemann formula:       MAE = 4.9%
    - Improvement: ~34% error reduction

Author: 環 & ご主人さま (飯泉真道) / Miosync Inc.
Date: 2025-02-07
License: MIT
================================================================================
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
import random
import numpy as np
from .banners import show_banner　

# Try to import from delta_theory package, fallback to local
try:
    from delta_theory.material import (
        Material, MATERIALS, get_material, list_materials,
        STRUCTURE_PRESETS, k_B as k_B_J, eV_to_J
    )
    _HAS_MATERIAL_MODULE = True
except ImportError:
    _HAS_MATERIAL_MODULE = False


# ==============================================================================
# Physical Constants
# ==============================================================================
k_B_eV: float = 8.617333262e-5   # Boltzmann constant [eV/K]


# ==============================================================================
# Iizumi-Lindemann Constants
# ==============================================================================
# Universal geometric constant: √(4/5) ≈ 0.8944
# This emerges from BCC cage geometry analysis
C_IIZUMI: float = np.sqrt(4/5)  # = 0.894427...

# Reference coordination number (BCC)
Z_REF: int = 8

# Structure-dependent stacking factor ξ
XI_STRUCT: Dict[str, float] = {
    "BCC": 1.0,      # Isotropic cubic
    "FCC": 1.0,      # Isotropic cubic
    "HCP": 7/8,      # c-axis anisotropy penalty (= 0.875)
}

# Conventional formula constant for comparison
C_CONVENTIONAL: float = np.sqrt(48)  # ≈ 6.928


# ==============================================================================
# Core Functions
# ==============================================================================

def iizumi_lindemann(
    Z: int,
    T_m: float,
    E_coh: float,
    structure: Literal["BCC", "FCC", "HCP"] = "FCC",
) -> float:
    """
    Iizumi-Lindemann Law (parameter-free)
    
    δ_L = √(4/5) × (8/Z) × ξ_struct × √(k_B T_m / E_coh)
    
    Args:
        Z: Coordination number (8 for BCC, 12 for FCC/HCP)
        T_m: Melting temperature [K]
        E_coh: Cohesive energy [eV]
        structure: Crystal structure ("BCC", "FCC", or "HCP")
    
    Returns:
        Lindemann ratio δ_L (dimensionless)
    
    Example:
        >>> iizumi_lindemann(Z=8, T_m=1811, E_coh=4.28, structure="BCC")
        0.171  # Fe prediction
    """
    xi = XI_STRUCT.get(structure, 1.0)
    return C_IIZUMI * (Z_REF / Z) * xi * np.sqrt(k_B_eV * T_m / E_coh)


def conventional_lindemann(Z: int, T_m: float, E_coh: float) -> float:
    """
    Conventional Lindemann formula (for comparison)
    
    δ_L = (√48 / Z) × √(k_B T_m / E_coh)
    
    Args:
        Z: Coordination number
        T_m: Melting temperature [K]
        E_coh: Cohesive energy [eV]
    
    Returns:
        Lindemann ratio δ_L (dimensionless)
    """
    return (C_CONVENTIONAL / Z) * np.sqrt(k_B_eV * T_m / E_coh)


def get_c_geo(structure: Literal["BCC", "FCC", "HCP"]) -> float:
    """
    Get geometric coefficient C_geo for a given structure
    
    C_geo = √(4/5) × (8/Z) × ξ_struct
    
    Args:
        structure: Crystal structure
    
    Returns:
        C_geo coefficient
    """
    Z = 8 if structure == "BCC" else 12
    xi = XI_STRUCT[structure]
    return C_IIZUMI * (Z_REF / Z) * xi


def rms_displacement(delta_L: float, r_nn: float) -> float:
    """
    RMS atomic displacement at melting point
    
    √<u²> = δ_L × r_nn
    
    Args:
        delta_L: Lindemann ratio
        r_nn: Nearest neighbor distance [Å or m]
    
    Returns:
        RMS displacement [same unit as r_nn]
    """
    return delta_L * r_nn


# ==============================================================================
# Material-based Interface
# ==============================================================================

def predict_delta_L(material_name: str) -> Tuple[float, float, float]:
    """
    Predict Lindemann ratio for a material from the database
    
    Args:
        material_name: Material name (e.g., "Fe", "Cu", "Ti")
    
    Returns:
        Tuple of (δ_L_predicted, δ_L_experimental, error_percent)
    
    Raises:
        ImportError: If material module is not available
        ValueError: If material not found
    """
    if not _HAS_MATERIAL_MODULE:
        raise ImportError(
            "material module not available. "
            "Install delta-theory package or provide material data directly."
        )
    
    mat = get_material(material_name)
    
    delta_pred = iizumi_lindemann(
        Z=mat.Z_bulk,
        T_m=mat.T_m,
        E_coh=mat.E_bond_eV,
        structure=mat.structure,
    )
    
    delta_exp = mat.delta_L
    error = (delta_pred - delta_exp) / delta_exp * 100
    
    return delta_pred, delta_exp, error


# ==============================================================================
# Validation Dataset (standalone, no material.py dependency)
# ==============================================================================

@dataclass
class MetalData:
    """Standalone metal data for validation"""
    name: str
    structure: Literal["BCC", "FCC", "HCP"]
    Z: int
    T_m: float        # Melting temperature [K]
    E_coh: float      # Cohesive energy [eV]
    r_nn: float       # Nearest neighbor distance [Å]
    delta_exp: float  # Experimental Lindemann ratio


# Reference dataset (7 metals)
VALIDATION_DATA: List[MetalData] = [
    MetalData("Fe", "BCC", 8,  1811, 4.28, 2.48, 0.180),
    MetalData("W",  "BCC", 8,  3695, 8.90, 2.74, 0.160),
    MetalData("Cu", "FCC", 12, 1357, 3.49, 2.56, 0.100),
    MetalData("Al", "FCC", 12,  933, 3.39, 2.86, 0.100),
    MetalData("Ni", "FCC", 12, 1728, 4.44, 2.49, 0.110),
    MetalData("Ti", "HCP", 12, 1941, 4.85, 2.95, 0.100),
    MetalData("Mg", "HCP", 12,  923, 1.51, 3.21, 0.117),
]


def validate_all(use_material_db: bool = False) -> Dict:
    """
    Validate Iizumi-Lindemann formula against experimental data
    
    Args:
        use_material_db: If True, use material.py database instead of built-in data
    
    Returns:
        Dictionary containing validation results and statistics
    """
    results = []
    
    if use_material_db and _HAS_MATERIAL_MODULE:
        # Use material database
        for name in ["Fe", "W", "Cu", "Al", "Ni", "Ti", "Mg"]:
            mat = get_material(name)
            metal = MetalData(
                name=mat.name,
                structure=mat.structure,
                Z=mat.Z_bulk,
                T_m=mat.T_m,
                E_coh=mat.E_bond_eV,
                r_nn=mat.b * 1e10,  # Convert to Å
                delta_exp=mat.delta_L,
            )
            results.append(_validate_single(metal))
    else:
        # Use built-in validation data
        for metal in VALIDATION_DATA:
            results.append(_validate_single(metal))
    
    # Calculate statistics
    errors_iizumi = [abs(r['error_iizumi']) for r in results]
    errors_conv = [abs(r['error_conv']) for r in results]
    
    mae_iizumi = np.mean(errors_iizumi)
    mae_conv = np.mean(errors_conv)
    
    delta_exp = np.array([r['delta_exp'] for r in results])
    delta_iizumi = np.array([r['delta_iizumi'] for r in results])
    delta_conv = np.array([r['delta_conv'] for r in results])
    
    corr_iizumi = np.corrcoef(delta_exp, delta_iizumi)[0, 1]
    corr_conv = np.corrcoef(delta_exp, delta_conv)[0, 1]
    
    return {
        'results': results,
        'mae_iizumi': mae_iizumi,
        'mae_conv': mae_conv,
        'correlation_iizumi': corr_iizumi,
        'correlation_conv': corr_conv,
        'improvement': (mae_conv - mae_iizumi) / mae_conv * 100,
    }


def _validate_single(metal: MetalData) -> Dict:
    """Validate single metal"""
    delta_iizumi = iizumi_lindemann(
        Z=metal.Z, T_m=metal.T_m, E_coh=metal.E_coh, structure=metal.structure
    )
    delta_conv = conventional_lindemann(
        Z=metal.Z, T_m=metal.T_m, E_coh=metal.E_coh
    )
    
    error_iizumi = (delta_iizumi - metal.delta_exp) / metal.delta_exp * 100
    error_conv = (delta_conv - metal.delta_exp) / metal.delta_exp * 100
    
    return {
        'name': metal.name,
        'structure': metal.structure,
        'Z': metal.Z,
        'T_m': metal.T_m,
        'E_coh': metal.E_coh,
        'delta_exp': metal.delta_exp,
        'delta_iizumi': delta_iizumi,
        'delta_conv': delta_conv,
        'error_iizumi': error_iizumi,
        'error_conv': error_conv,
        'rms_exp': metal.delta_exp * metal.r_nn,
        'rms_iizumi': delta_iizumi * metal.r_nn,
    }


# ==============================================================================
# Display Functions
# ==============================================================================

def print_validation_report(validation: Dict) -> None:
    """Print formatted validation report"""
    
    results = validation['results']
    
    print("=" * 80)
    print("IIZUMI-LINDEMANN LAW VALIDATION")
    print("=" * 80)
    print()
    print("Formula: δ_L = √(4/5) × (8/Z) × ξ_struct × √(k_B T_m / E_coh)")
    print()
    print(f"Constants:")
    print(f"  √(4/5) = {C_IIZUMI:.6f}  (universal geometric constant)")
    print(f"  ξ_BCC  = {XI_STRUCT['BCC']:.3f}")
    print(f"  ξ_FCC  = {XI_STRUCT['FCC']:.3f}")
    print(f"  ξ_HCP  = {XI_STRUCT['HCP']:.3f} (= 7/8, c-axis penalty)")
    print()
    
    # C_geo values
    print("Derived C_geo values:")
    for struct in ["BCC", "FCC", "HCP"]:
        c_geo = get_c_geo(struct)
        print(f"  {struct}: C_geo = {c_geo:.4f}")
    print()
    
    # Results table
    print("-" * 80)
    print(f"{'Metal':>6} {'Struct':>6} {'Z':>3} {'T_m':>7} {'E_coh':>6} "
          f"{'δ_exp':>7} {'δ_Iiz':>7} {'δ_conv':>7} {'Err_I%':>7} {'Err_C%':>7}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:>6} {r['structure']:>6} {r['Z']:>3} {r['T_m']:>7.0f} "
              f"{r['E_coh']:>6.2f} {r['delta_exp']:>7.3f} {r['delta_iizumi']:>7.3f} "
              f"{r['delta_conv']:>7.3f} {r['error_iizumi']:>+7.1f} {r['error_conv']:>+7.1f}")
    
    print("-" * 80)
    
    # Statistics
    print()
    print("STATISTICS")
    print("-" * 40)
    print(f"  Iizumi-Lindemann:  MAE = {validation['mae_iizumi']:.2f}%")
    print(f"  Conventional:      MAE = {validation['mae_conv']:.2f}%")
    print(f"  Improvement:       {validation['improvement']:.1f}%")
    print()
    print(f"  Correlation (Iizumi):       r = {validation['correlation_iizumi']:.4f}")
    print(f"  Correlation (Conventional): r = {validation['correlation_conv']:.4f}")
    
    # Structure breakdown
    print()
    print("STRUCTURE BREAKDOWN")
    print("-" * 40)
    
    for struct in ["BCC", "FCC", "HCP"]:
        struct_results = [r for r in results if r['structure'] == struct]
        if struct_results:
            mae_i = np.mean([abs(r['error_iizumi']) for r in struct_results])
            mae_c = np.mean([abs(r['error_conv']) for r in struct_results])
            metals = ", ".join([r['name'] for r in struct_results])
            print(f"  {struct}: MAE(Iizumi)={mae_i:.1f}%, MAE(Conv)={mae_c:.1f}% [{metals}]")


def print_latex_table(validation: Dict) -> None:
    """Generate LaTeX table"""
    
    results = validation['results']
    
    print()
    print("=" * 80)
    print("LaTeX TABLE")
    print("=" * 80)
    
    print(r"""
\begin{table}[htbp]
\centering
\caption{Validation of the Iizumi-Lindemann Law:
$\delta_L = \sqrt{\frac{4}{5}} \cdot \frac{8}{Z} \cdot \xi_{\text{struct}} \cdot \sqrt{\frac{k_B T_m}{E_{\text{coh}}}}$.
No fitting parameters.}
\label{tab:iizumi_lindemann}
\begin{tabular}{lccccccc}
\toprule
Metal & Structure & $Z$ & $T_m$ [K] & $E_{\text{coh}}$ [eV] & $\delta_L^{\text{exp}}$ & $\delta_L^{\text{pred}}$ & Error [\%] \\
\midrule""")
    
    for r in results:
        err_str = f"+{r['error_iizumi']:.1f}" if r['error_iizumi'] > 0 else f"{r['error_iizumi']:.1f}"
        print(f"{r['name']} & {r['structure']} & {r['Z']} & {r['T_m']:.0f} & "
              f"{r['E_coh']:.2f} & {r['delta_exp']:.3f} & {r['delta_iizumi']:.3f} & {err_str} \\\\")
    
    print(r"""\midrule
\multicolumn{8}{l}{\textbf{Mean Absolute Error: """ + f"{validation['mae_iizumi']:.1f}" + r"""\%}} \\
\bottomrule
\end{tabular}
\end{table}
""")


# ==============================================================================
# CLI Interface
# ==============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser"""
    
    parser = argparse.ArgumentParser(
        prog="python -m delta_theory.lindemann",
        description="Iizumi-Lindemann Law: Parameter-free Lindemann ratio prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate against all reference metals
  python lindemann.py --validate
  
  # Calculate for a specific material from database
  python lindemann.py --material Fe
  
  # Calculate with custom parameters
  python lindemann.py --custom --Z 8 --T_m 1811 --E_coh 4.28 --structure BCC
  
  # Generate LaTeX table
  python lindemann.py --validate --latex
  
  # Compare with conventional formula
  python lindemann.py --validate --compare

Formula:
  δ_L = √(4/5) × (8/Z) × ξ_struct × √(k_B T_m / E_coh)
  
  Where:
    √(4/5) ≈ 0.8944  : Universal constant
    ξ_BCC = 1.0, ξ_FCC = 1.0, ξ_HCP = 7/8
""",
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Run validation against reference metals",
    )
    mode_group.add_argument(
        "--material", "-m",
        type=str,
        metavar="NAME",
        help="Calculate for material from database (e.g., Fe, Cu, Ti)",
    )
    mode_group.add_argument(
        "--custom", "-c",
        action="store_true",
        help="Calculate with custom parameters (requires --Z, --T_m, --E_coh)",
    )
    mode_group.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available materials in database",
    )
    
    # Custom parameters
    custom_group = parser.add_argument_group("Custom parameters (with --custom)")
    custom_group.add_argument("--Z", type=int, help="Coordination number (8 or 12)")
    custom_group.add_argument("--T_m", type=float, help="Melting temperature [K]")
    custom_group.add_argument("--E_coh", type=float, help="Cohesive energy [eV]")
    custom_group.add_argument(
        "--structure", "-s",
        type=str,
        choices=["BCC", "FCC", "HCP"],
        default="FCC",
        help="Crystal structure (default: FCC)",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--latex",
        action="store_true",
        help="Generate LaTeX table (with --validate)",
    )
    output_group.add_argument(
        "--compare",
        action="store_true",
        help="Compare with conventional √48/Z formula",
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point"""
    
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Default to validation if no mode specified
    if not any([args.validate, args.material, args.custom, args.list]):
        args.validate = True
    
    # List materials
    if args.list:
        print("Available materials:")
        if _HAS_MATERIAL_MODULE:
            for name in list_materials():
                mat = get_material(name)
                print(f"  {name:4s} ({mat.structure}): T_m={mat.T_m:4.0f}K, "
                      f"E_coh={mat.E_bond_eV:.2f}eV, δ_L={mat.delta_L:.3f}")
        else:
            print("  (material module not available, using built-in data)")
            for m in VALIDATION_DATA:
                print(f"  {m.name:4s} ({m.structure}): T_m={m.T_m:4.0f}K, "
                      f"E_coh={m.E_coh:.2f}eV, δ_L={m.delta_exp:.3f}")
        return 0
    
    # Validation mode
    if args.validate:
        validation = validate_all(use_material_db=_HAS_MATERIAL_MODULE)
        
        if args.json:
            import json
            # Convert to JSON-serializable format
            output = {
                'mae_iizumi': validation['mae_iizumi'],
                'mae_conventional': validation['mae_conv'],
                'improvement_percent': validation['improvement'],
                'results': validation['results'],
            }
            print(json.dumps(output, indent=2))
        else:
            print_validation_report(validation)
            if args.latex:
                print_latex_table(validation)
        return 0
    
    # Material from database
    if args.material:
        if not _HAS_MATERIAL_MODULE:
            # Fallback to built-in data
            metal_data = {m.name: m for m in VALIDATION_DATA}
            if args.material not in metal_data:
                print(f"Error: Material '{args.material}' not found.")
                print(f"Available: {list(metal_data.keys())}")
                return 1
            
            m = metal_data[args.material]
            delta_pred = iizumi_lindemann(m.Z, m.T_m, m.E_coh, m.structure)
            delta_conv = conventional_lindemann(m.Z, m.T_m, m.E_coh)
            delta_exp = m.delta_exp
        else:
            try:
                mat = get_material(args.material)
                delta_pred = iizumi_lindemann(
                    mat.Z_bulk, mat.T_m, mat.E_bond_eV, mat.structure
                )
                delta_conv = conventional_lindemann(
                    mat.Z_bulk, mat.T_m, mat.E_bond_eV
                )
                delta_exp = mat.delta_L
                m = MetalData(
                    mat.name, mat.structure, mat.Z_bulk,
                    mat.T_m, mat.E_bond_eV, mat.b * 1e10, mat.delta_L
                )
            except ValueError as e:
                print(f"Error: {e}")
                return 1
        
        err_pred = (delta_pred - delta_exp) / delta_exp * 100
        err_conv = (delta_conv - delta_exp) / delta_exp * 100
        
        if args.json:
            import json
            output = {
                'material': m.name,
                'structure': m.structure,
                'Z': m.Z,
                'T_m_K': m.T_m,
                'E_coh_eV': m.E_coh,
                'delta_L_experimental': delta_exp,
                'delta_L_iizumi': delta_pred,
                'delta_L_conventional': delta_conv,
                'error_iizumi_percent': err_pred,
                'error_conventional_percent': err_conv,
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"\n{'='*50}")
            print(f"Material: {m.name} ({m.structure}, Z={m.Z})")
            print(f"{'='*50}")
            print(f"  T_m     = {m.T_m:.0f} K")
            print(f"  E_coh   = {m.E_coh:.2f} eV")
            print(f"  C_geo   = {get_c_geo(m.structure):.4f}")
            print()
            print(f"  δ_L (experimental):  {delta_exp:.4f}")
            print(f"  δ_L (Iizumi):        {delta_pred:.4f}  (error: {err_pred:+.1f}%)")
            if args.compare:
                print(f"  δ_L (conventional):  {delta_conv:.4f}  (error: {err_conv:+.1f}%)")
            print()
        
        return 0
    
    # Custom parameters
    if args.custom:
        if not all([args.Z, args.T_m, args.E_coh]):
            print("Error: --custom requires --Z, --T_m, and --E_coh")
            return 1
        
        delta_pred = iizumi_lindemann(
            args.Z, args.T_m, args.E_coh, args.structure
        )
        delta_conv = conventional_lindemann(args.Z, args.T_m, args.E_coh)
        
        if args.json:
            import json
            output = {
                'Z': args.Z,
                'T_m_K': args.T_m,
                'E_coh_eV': args.E_coh,
                'structure': args.structure,
                'delta_L_iizumi': delta_pred,
                'delta_L_conventional': delta_conv,
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"\n{'='*50}")
            print(f"Custom Calculation ({args.structure}, Z={args.Z})")
            print(f"{'='*50}")
            print(f"  T_m     = {args.T_m:.0f} K")
            print(f"  E_coh   = {args.E_coh:.2f} eV")
            print(f"  C_geo   = {get_c_geo(args.structure):.4f}")
            print()
            print(f"  δ_L (Iizumi):        {delta_pred:.4f}")
            if args.compare:
                print(f"  δ_L (conventional):  {delta_conv:.4f}")
            print()
        
        return 0
    
    return 0


# ==============================================================================
# Module Entry Point
# ==============================================================================

def main() -> None:
    # 起動時にランダムで表示
    show_banner()  # ← これだけ！
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    sys.exit(main())
