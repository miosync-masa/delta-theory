#!/usr/bin/env python3
"""δ-Theory Unified FLC v8.1 - Integrated with v6.9

=============================================================================
v8.0 → v8.1: Integration with unified_yield_fatigue_v6_9
=============================================================================

【新機能】
  - v6.9 から材料パラメータを自動取得
  - FLCMaterial.from_v69() で簡単に材料作成
  - FLC₀ 1点校正で全モード予測
  - v6.9のMaterialクラスから直接変換

【使い方】
  # v6.9 材料から直接 FLC 予測
  flc = FLCPredictor()
  flc.add_from_v69('Fe', flc0=0.225)  # SPCC相当
  eps1 = flc.predict('Fe', 'Plane Strain')
  
  # カスタム材料
  flc.add_from_v69(
      name='MyAlloy',
      base_element='Fe',
      structure='BCC',
      flc0=0.28,
      T_twin=1.0,
  )
  
  # v6.9のMaterialクラスから直接
  # (v6.9がimport可能な場合)
  from unified_yield_fatigue_v6_9 import MATERIALS
  flc.add_from_v69_material(MATERIALS['Fe'], flc0=0.225, name='SPCC')

Author: δ-Theory Team (Masamichi Iizumi & Tamaki)
Version: 8.1.0
Date: 2025-02
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import random
from .banners import show_banner

# ==============================================================================
# Try to import v6.9b
# ==============================================================================

V69_AVAILABLE = False
V69_MATERIALS = {}
V69_Material = None

try:
    from unified_yield_fatigue_v6_9 import (
        tau_over_sigma as v69_tau_over_sigma,
        sigma_c_over_sigma_t as v69_sigma_c_over_sigma_t,
        sigma_base_delta,
        calc_sigma_y,
        Material as V69_Material,
        MATERIALS as V69_MATERIALS,
        T_TWIN as V69_T_TWIN,
        R_COMP as V69_R_COMP,
        C_CLASS_DEFAULT,
        DEFAULT_BCC_W110,
    )
    V69_AVAILABLE = True
except ImportError:
    try:
        # Try relative import (as package)
        from .unified_yield_fatigue_v6_9 import (
            tau_over_sigma as v69_tau_over_sigma,
            sigma_c_over_sigma_t as v69_sigma_c_over_sigma_t,
            sigma_base_delta,
            calc_sigma_y,
            Material as V69_Material,
            MATERIALS as V69_MATERIALS,
            T_TWIN as V69_T_TWIN,
            R_COMP as V69_R_COMP,
            C_CLASS_DEFAULT,
            DEFAULT_BCC_W110,
        )
        V69_AVAILABLE = True
    except ImportError:
        warnings.warn(
            "unified_yield_fatigue_v6_9 not found. "
            "Using built-in parameters. "
            "For full functionality, ensure v6.9 is in the same directory."
        )
        V69_AVAILABLE = False
        # Fallback defaults
        v69_tau_over_sigma = None
        v69_sigma_c_over_sigma_t = None
        V69_MATERIALS = {}
        C_CLASS_DEFAULT = 1.415
        DEFAULT_BCC_W110 = 0.0
        V69_T_TWIN = {}
        V69_R_COMP = {}

# ==============================================================================
# Built-in Parameters (Fallback when v6.9 not available)
# ==============================================================================

# τ/σ ratios from δ-theory
BUILTIN_TAU_SIGMA = {
    # Pure metals
    'Fe': 0.565, 'Cu': 0.565, 'Al': 0.565, 'Ni': 0.565,
    'Ti': 0.546, 'Mg': 0.327, 'Zn': 0.480,
    # Crystal structures (default)
    'BCC': 0.565, 'FCC': 0.565, 'HCP': 0.500,
}

# R_comp (σ_c/σ_t) ratios
BUILTIN_R_COMP = {
    'Fe': 1.00, 'Cu': 1.00, 'Al': 1.00, 'Ni': 1.00,
    'Ti': 1.00, 'Mg': 0.60, 'Zn': 1.20,
    'BCC': 1.00, 'FCC': 1.00, 'HCP': 0.80,
}

# Element to structure mapping
ELEMENT_STRUCTURE = {
    'Fe': 'BCC', 'Cr': 'BCC', 'Mo': 'BCC', 'W': 'BCC', 'V': 'BCC',
    'Cu': 'FCC', 'Al': 'FCC', 'Ni': 'FCC', 'Ag': 'FCC', 'Au': 'FCC',
    'Ti': 'HCP', 'Mg': 'HCP', 'Zn': 'HCP', 'Zr': 'HCP',
}


def get_tau_sigma(element_or_structure: str, T_twin: float = 1.0) -> float:
    """
    Get τ/σ ratio.
    
    Uses v6.9b if available for BCC/FCC, built-in interpolation for HCP.
    
    Args:
        element_or_structure: Element name ('Fe', 'Cu') or structure ('BCC', 'FCC')
        T_twin: Twinning factor for HCP (0.0=twin-dominated, 1.0=slip-dominated)
    
    Returns:
        τ/σ ratio
    
    Note:
        For HCP, we always use built-in interpolation because:
        - v6.9's tau_over_sigma multiplies by T_twin directly (→ 0 when T_twin=0)
        - FLC needs T_twin as interpolation parameter between slip/twin modes
        - T_twin=0.0 → Mg-like twin-dominated (τ/σ ≈ 0.327)
        - T_twin=1.0 → slip-dominated (τ/σ ≈ 0.565)
    """
    structure = ELEMENT_STRUCTURE.get(element_or_structure, element_or_structure)
    
    # HCP: Always use built-in interpolation (v6.9 T_twin semantics differ)
    if structure == 'HCP':
        tau_slip = 0.565  # Normal slip-dominated
        if element_or_structure == 'Mg':
            tau_twin = 0.327  # Mg twin-dominated (calibrated)
        elif element_or_structure == 'Ti':
            tau_twin = 0.546  # Ti (mostly slip even with twinning)
        elif element_or_structure == 'Zn':
            tau_twin = 0.480  # Zn
        else:
            tau_twin = 0.50   # Generic HCP
        
        # T_twin=1.0 → slip-dominated (0.565)
        # T_twin=0.0 → twin-dominated (element-specific)
        return tau_slip * T_twin + tau_twin * (1 - T_twin)
    
    # BCC/FCC: Use v6.9 if available
    if V69_AVAILABLE and element_or_structure in V69_MATERIALS:
        mat = V69_MATERIALS[element_or_structure]
        return v69_tau_over_sigma(mat, C_CLASS_DEFAULT, DEFAULT_BCC_W110)
    
    # Fallback to built-in
    if element_or_structure in BUILTIN_TAU_SIGMA:
        return BUILTIN_TAU_SIGMA[element_or_structure]
    else:
        return BUILTIN_TAU_SIGMA.get(structure, 0.565)


def get_R_comp(element_or_structure: str, T_twin: float = 1.0) -> float:
    """
    Get R_comp (σ_c/σ_t) ratio.
    
    Uses v6.9b if available for BCC/FCC, built-in interpolation for HCP.
    
    Args:
        element_or_structure: Element name or structure
        T_twin: Twinning factor for HCP (0.0=twin-dominated, 1.0=slip-dominated)
    
    Returns:
        R_comp ratio
    
    Note:
        For HCP, T_twin affects tension/compression asymmetry:
        - T_twin=0.0 → strong asymmetry (Mg: R_comp ≈ 0.6)
        - T_twin=1.0 → symmetric (R_comp ≈ 1.0)
    """
    structure = ELEMENT_STRUCTURE.get(element_or_structure, element_or_structure)
    
    # HCP: Always use built-in interpolation
    if structure == 'HCP':
        R_symmetric = 1.0  # Slip-dominated (symmetric)
        if element_or_structure == 'Mg':
            R_twin = 0.60  # Mg twin-dominated (strong asymmetry)
        elif element_or_structure == 'Zn':
            R_twin = 1.20  # Zn (compression stronger)
        elif element_or_structure == 'Ti':
            R_twin = 1.00  # Ti (mostly symmetric)
        else:
            R_twin = 0.80  # Generic HCP
        
        # T_twin=1.0 → symmetric (1.0)
        # T_twin=0.0 → asymmetric (element-specific)
        return R_symmetric * T_twin + R_twin * (1 - T_twin)
    
    # BCC/FCC: Use v6.9 if available
    if V69_AVAILABLE and element_or_structure in V69_MATERIALS:
        mat = V69_MATERIALS[element_or_structure]
        return v69_sigma_c_over_sigma_t(mat)
    
    # Fallback to built-in
    if element_or_structure in BUILTIN_R_COMP:
        return BUILTIN_R_COMP[element_or_structure]
    else:
        return BUILTIN_R_COMP.get(structure, 1.0)


def get_v69_material(element: str) -> Optional[object]:
    """Get v6.9b Material object if available."""
    if V69_AVAILABLE and element in V69_MATERIALS:
        return V69_MATERIALS[element]
    return None


# ==============================================================================
# Constants (Frozen)
# ==============================================================================

K1_LOCALIZATION = 0.75
K2_LOCALIZATION = 0.48

STANDARD_MODES = {
    'Uniaxial':      {'j': 1, 'beta': -0.370, 'w_sigma': 1.246, 'w_tau': 0.319, 'w_c': 0.000},
    'Deep Draw':     {'j': 2, 'beta': -0.306, 'w_sigma': 1.354, 'w_tau': 0.458, 'w_c': 0.000},
    'Draw-Plane':    {'j': 3, 'beta': -0.169, 'w_sigma': 1.552, 'w_tau': 0.723, 'w_c': 0.000},
    'Plane Strain':  {'j': 4, 'beta':  0.000, 'w_sigma': 1.732, 'w_tau': 1.000, 'w_c': 0.000},
    'Plane-Stretch': {'j': 5, 'beta':  0.133, 'w_sigma': 1.829, 'w_tau': 0.813, 'w_c': 0.133},
    'Stretch':       {'j': 6, 'beta':  0.247, 'w_sigma': 1.889, 'w_tau': 0.670, 'w_c': 0.247},
    'Equi-biaxial':  {'j': 7, 'beta':  0.430, 'w_sigma': 1.949, 'w_tau': 0.469, 'w_c': 0.430},
}

MODE_ORDER = ['Uniaxial', 'Deep Draw', 'Draw-Plane', 'Plane Strain', 
              'Plane-Stretch', 'Stretch', 'Equi-biaxial']

R_TH_VIRGIN = {'BCC': 0.65, 'FCC': 0.02, 'HCP': 0.20}

# Pre-calculate C_j
LOCALIZATION_COEFFS = {
    mode: 1 + K1_LOCALIZATION * data['beta'] + K2_LOCALIZATION * data['beta']**2
    for mode, data in STANDARD_MODES.items()
}


# ==============================================================================
# FLCMaterial Class
# ==============================================================================

@dataclass
class FLCMaterial:
    """Material data for δ-FLC prediction."""
    name: str
    tau_sigma: float
    R_comp: float
    V_eff: float
    structure: str = 'FCC'
    sigma_y: float = 0.0
    base_element: str = ''
    
    @property
    def r_th(self) -> float:
        return R_TH_VIRGIN.get(self.structure, 0.20)
    
    @classmethod
    def from_v69(cls,
                 name: str,
                 flc0: float,
                 base_element: Optional[str] = None,
                 structure: Optional[str] = None,
                 T_twin: float = 1.0,
                 sigma_y: float = 0.0) -> 'FLCMaterial':
        """
        Create FLCMaterial from v6.9 parameters.
        
        Args:
            name: Material name
            flc0: Experimental FLC₀ (Plane Strain)
            base_element: Base element ('Fe', 'Cu', 'Al', etc.)
            structure: Crystal structure (auto-detected if base_element given)
            T_twin: Twinning factor for HCP (0.0-1.0)
            sigma_y: Yield stress [MPa]
        
        Returns:
            FLCMaterial with calibrated V_eff
        
        Example:
            >>> mat = FLCMaterial.from_v69('SPCC', flc0=0.225, base_element='Fe')
            >>> mat = FLCMaterial.from_v69('Mg_AZ31', flc0=0.265, base_element='Mg', T_twin=0.0)
        """
        # Determine structure
        if structure is None:
            if base_element:
                structure = ELEMENT_STRUCTURE.get(base_element, 'FCC')
            else:
                structure = 'FCC'
        
        # Get multiaxial parameters
        key = base_element if base_element else structure
        tau_sigma = get_tau_sigma(key, T_twin)
        R_comp = get_R_comp(key, T_twin)
        
        # Calibrate V_eff from FLC₀
        V_eff = calibrate_V_eff(flc0, tau_sigma, R_comp)
        
        return cls(
            name=name,
            tau_sigma=tau_sigma,
            R_comp=R_comp,
            V_eff=V_eff,
            structure=structure,
            sigma_y=sigma_y,
            base_element=base_element or '',
        )


# ==============================================================================
# Core Functions
# ==============================================================================

def calc_R_eff(tau_sigma: float, R_comp: float, mode: str) -> float:
    """Calculate effective resistance R_j."""
    m = STANDARD_MODES[mode]
    R_eff = m['w_sigma'] + m['w_tau'] / tau_sigma
    if m['w_c'] > 0:
        R_eff += m['w_c'] / R_comp
    return R_eff


def calc_K_coeff(tau_sigma: float, R_comp: float, mode: str) -> float:
    """Calculate combined coefficient K_j = C_j / R_j."""
    C_j = LOCALIZATION_COEFFS[mode]
    R_j = calc_R_eff(tau_sigma, R_comp, mode)
    return C_j / R_j


def calibrate_V_eff(flc0: float, tau_sigma: float, R_comp: float = 1.0) -> float:
    """Calibrate V_eff from FLC₀ (Plane Strain)."""
    R_ps = calc_R_eff(tau_sigma, R_comp, 'Plane Strain')
    C_ps = LOCALIZATION_COEFFS['Plane Strain']  # = 1.0
    return flc0 * R_ps / C_ps


def predict_flc_mode(material: Union[str, FLCMaterial], mode: str) -> float:
    """Predict FLC for a specific mode."""
    if isinstance(material, str):
        mat = FLC_MATERIALS.get(material)
        if mat is None:
            raise ValueError(f"Material '{material}' not found.")
    else:
        mat = material
    
    K_j = calc_K_coeff(mat.tau_sigma, mat.R_comp, mode)
    return mat.V_eff * K_j


# ==============================================================================
# Built-in Material Database (Optimized)
# ==============================================================================

FLC_MATERIALS: Dict[str, FLCMaterial] = {
    'Cu': FLCMaterial('Cu', 0.565, 1.00, 1.2235, 'FCC', 69.5, 'Cu'),
    'Ti': FLCMaterial('Ti', 0.546, 1.00, 1.0391, 'HCP', 270.8, 'Ti'),
    'SPCC': FLCMaterial('SPCC', 0.565, 1.00, 0.8024, 'BCC', 237.0, 'Fe'),
    'DP590': FLCMaterial('DP590', 0.565, 1.00, 0.6911, 'BCC', 360.0, 'Fe'),
    'Al5052': FLCMaterial('Al5052', 0.565, 1.00, 0.6189, 'FCC', 85.0, 'Al'),
    'SUS304': FLCMaterial('SUS304', 0.565, 1.00, 1.4234, 'FCC', 275.0, 'Ni'),
    'Mg_AZ31': FLCMaterial('Mg_AZ31', 0.327, 0.60, 1.1798, 'HCP', 136.0, 'Mg'),
}


# ==============================================================================
# FLC Predictor Class
# ==============================================================================

class FLCPredictor:
    """
    δ-Theory FLC Predictor (v8.1) - Integrated with v6.9
    
    Example:
        >>> flc = FLCPredictor()
        
        # Use built-in material
        >>> flc.predict('Cu', 'Plane Strain')
        0.346
        
        # Add from v6.9 parameters
        >>> flc.add_from_v69('MySteel', flc0=0.28, base_element='Fe')
        >>> flc.predict('MySteel', 'Uniaxial')
    """
    
    def __init__(self):
        self._materials = FLC_MATERIALS.copy()
    
    @property
    def v69_available(self) -> bool:
        """Check if v6.9 is available."""
        return V69_AVAILABLE
    
    def add_material(self, mat: FLCMaterial) -> None:
        """Add custom material."""
        self._materials[mat.name] = mat
    
    def add_from_v69(self,
                     name: str,
                     flc0: float,
                     base_element: Optional[str] = None,
                     structure: Optional[str] = None,
                     T_twin: float = 1.0,
                     sigma_y: float = 0.0) -> FLCMaterial:
        """
        Add material using v6.9 parameters.
        
        Args:
            name: Material name
            flc0: Experimental FLC₀
            base_element: Base element ('Fe', 'Cu', 'Al', 'Ti', 'Mg', etc.)
            structure: Crystal structure (auto-detected from base_element)
            T_twin: Twinning factor for HCP (0.0=twin-dominated, 1.0=slip-dominated)
            sigma_y: Yield stress [MPa]
        
        Returns:
            Created FLCMaterial
        
        Examples:
            # BCC steel
            >>> flc.add_from_v69('SPCC', flc0=0.225, base_element='Fe')
            
            # FCC aluminum alloy
            >>> flc.add_from_v69('A5052', flc0=0.165, base_element='Al')
            
            # HCP magnesium (twin-dominated)
            >>> flc.add_from_v69('AZ31', flc0=0.265, base_element='Mg', T_twin=0.0)
            
            # HCP titanium (slip-dominated)
            >>> flc.add_from_v69('Ti64', flc0=0.30, base_element='Ti', T_twin=1.0)
        """
        mat = FLCMaterial.from_v69(
            name=name,
            flc0=flc0,
            base_element=base_element,
            structure=structure,
            T_twin=T_twin,
            sigma_y=sigma_y,
        )
        self._materials[name] = mat
        return mat
    
    def add_from_flc0(self,
                      name: str,
                      flc0: float,
                      tau_sigma: float,
                      R_comp: float = 1.0,
                      structure: str = 'FCC',
                      sigma_y: float = 0.0) -> FLCMaterial:
        """Add material with explicit τ/σ and R_comp."""
        V_eff = calibrate_V_eff(flc0, tau_sigma, R_comp)
        mat = FLCMaterial(
            name=name,
            tau_sigma=tau_sigma,
            R_comp=R_comp,
            V_eff=V_eff,
            structure=structure,
            sigma_y=sigma_y,
        )
        self._materials[name] = mat
        return mat
    
    def add_from_v69_material(self,
                               v69_mat,
                               flc0: float,
                               name: Optional[str] = None,
                               T_twin: float = 1.0) -> FLCMaterial:
        """
        Add material from v6.9 Material object.
        
        Args:
            v69_mat: v6.9 Material object (has .structure, .sigma_y, etc.)
            flc0: Experimental FLC₀
            name: Material name (default: v69_mat.name if available)
            T_twin: Twinning factor for HCP
        
        Returns:
            Created FLCMaterial
        
        Example:
            >>> from unified_yield_fatigue_v6_9 import MATERIALS
            >>> flc.add_from_v69_material(MATERIALS['Fe'], flc0=0.225, name='SPCC')
        """
        # Extract info from v6.9 Material
        mat_name = name or getattr(v69_mat, 'name', 'Unknown')
        structure = getattr(v69_mat, 'structure', 'FCC')
        sigma_y = getattr(v69_mat, 'sigma_y', 0.0)
        
        # Get τ/σ and R_comp from v6.9 functions or built-in
        tau_sigma = get_tau_sigma(structure, T_twin)
        R_comp = get_R_comp(structure, T_twin)
        
        # Calibrate V_eff
        V_eff = calibrate_V_eff(flc0, tau_sigma, R_comp)
        
        mat = FLCMaterial(
            name=mat_name,
            tau_sigma=tau_sigma,
            R_comp=R_comp,
            V_eff=V_eff,
            structure=structure,
            sigma_y=sigma_y,
            base_element=getattr(v69_mat, 'element', ''),
        )
        self._materials[mat_name] = mat
        return mat
    
    def get_material(self, name: str) -> FLCMaterial:
        """Get material by name."""
        if name not in self._materials:
            available = list(self._materials.keys())
            raise ValueError(f"Material '{name}' not found. Available: {available}")
        return self._materials[name]
    
    def list_materials(self) -> List[str]:
        """List available materials."""
        return list(self._materials.keys())
    
    def predict(self,
                material: Union[str, FLCMaterial],
                mode: str,
                include_breakdown: bool = False) -> Union[float, Tuple[float, Dict]]:
        """Predict FLC for a mode."""
        if isinstance(material, str):
            mat = self.get_material(material)
        else:
            mat = material
        
        m = STANDARD_MODES[mode]
        C_j = LOCALIZATION_COEFFS[mode]
        R_j = calc_R_eff(mat.tau_sigma, mat.R_comp, mode)
        K_j = C_j / R_j
        eps1 = mat.V_eff * K_j
        
        if include_breakdown:
            return eps1, {
                'mode': mode, 'j': m['j'], 'beta': m['beta'],
                'C_j': C_j, 'R_j': R_j, 'K_j': K_j,
                'V_eff': mat.V_eff, 'tau_sigma': mat.tau_sigma, 'R_comp': mat.R_comp,
            }
        return eps1
    
    def predict_all_modes(self, material: Union[str, FLCMaterial]) -> Dict[str, float]:
        """Predict FLC for all 7 modes."""
        return {mode: self.predict(material, mode) for mode in MODE_ORDER}
    
    def predict_curve(self, material: Union[str, FLCMaterial]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict FLC curve."""
        predictions = self.predict_all_modes(material)
        betas = np.array([STANDARD_MODES[m]['beta'] for m in MODE_ORDER])
        eps1s = np.array([predictions[m] for m in MODE_ORDER])
        return betas, eps1s
    
    def flc0(self, material: Union[str, FLCMaterial]) -> float:
        """Get FLC₀ (Plane Strain)."""
        return self.predict(material, 'Plane Strain')
    
    def summary(self, material: Union[str, FLCMaterial]) -> str:
        """Generate summary report."""
        if isinstance(material, str):
            mat = self.get_material(material)
        else:
            mat = material
        
        lines = [
            f"δ-FLC Summary: {mat.name}",
            "=" * 55,
            f"Structure: {mat.structure}",
            f"Base element: {mat.base_element}" if mat.base_element else "",
            f"σ_y: {mat.sigma_y:.1f} MPa" if mat.sigma_y > 0 else "",
            f"τ/σ: {mat.tau_sigma:.4f}",
            f"R_comp: {mat.R_comp:.2f}",
            f"|V|_eff: {mat.V_eff:.4f}",
            "",
            "FLC Predictions:",
            "-" * 55,
            f"{'Mode':<15} {'β':>7} {'C_j':>7} {'R_j':>7} {'K_j':>7} {'ε₁':>8}",
            "-" * 55,
        ]
        
        for mode in MODE_ORDER:
            eps1, bd = self.predict(mat, mode, include_breakdown=True)
            lines.append(
                f"{mode:<15} {bd['beta']:>7.3f} {bd['C_j']:>7.4f} "
                f"{bd['R_j']:>7.3f} {bd['K_j']:>7.4f} {eps1:>8.4f}"
            )
        
        lines.extend(["-" * 55, f"FLC₀: {self.flc0(mat):.4f}"])
        return '\n'.join(filter(None, lines))


# ==============================================================================
# Convenience Functions
# ==============================================================================

def predict_flc(material: str, mode: str = 'Plane Strain') -> float:
    """Quick FLC prediction."""
    return FLCPredictor().predict(material, mode)


def predict_flc_curve(material: str) -> Tuple[np.ndarray, np.ndarray]:
    """Quick FLC curve."""
    return FLCPredictor().predict_curve(material)


def get_flc0(material: str) -> float:
    """Get FLC₀."""
    return FLCPredictor().flc0(material)


def create_material_from_v69(name: str, flc0: float, base_element: str,
                              T_twin: float = 1.0) -> FLCMaterial:
    """Create material using v6.9 parameters."""
    return FLCMaterial.from_v69(name, flc0, base_element, T_twin=T_twin)


# ==============================================================================
# Demo
# ==============================================================================
def main() -> None:
    # 起動時にランダムで表示
    show_banner()  # ← これだけ！
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

def demo():
    """Demonstration of v8.1 capabilities."""
    
    print("="*70)
    print("δ-Theory FLC v8.1: Integrated with v6.9")
    print("="*70)
    print(f"\nv6.9 available: {V69_AVAILABLE}")
    
    flc = FLCPredictor()
    
    # 1. Built-in materials
    print("\n[1] Built-in Materials")
    print("-"*70)
    print(f"Available: {flc.list_materials()}")
    
    # 2. Add from v6.9 parameters
    print("\n[2] Add Materials from v6.9 Parameters")
    print("-"*70)
    
    # BCC steel example
    mat1 = flc.add_from_v69('NewSteel', flc0=0.28, base_element='Fe')
    print(f"Added: {mat1.name}")
    print(f"  τ/σ = {mat1.tau_sigma:.4f} (from Fe/BCC)")
    print(f"  R_comp = {mat1.R_comp:.2f}")
    print(f"  V_eff = {mat1.V_eff:.4f} (calibrated from FLC₀=0.28)")
    
    # HCP Mg example (twin-dominated)
    mat2 = flc.add_from_v69('MgAlloy', flc0=0.25, base_element='Mg', T_twin=0.0)
    print(f"\nAdded: {mat2.name}")
    print(f"  τ/σ = {mat2.tau_sigma:.4f} (from Mg/HCP, T_twin=0)")
    print(f"  R_comp = {mat2.R_comp:.2f}")
    print(f"  V_eff = {mat2.V_eff:.4f}")
    
    # 3. Simulate v6.9 Material object integration
    print("\n[3] v6.9 Material Object Integration (Simulated)")
    print("-"*70)
    
    # Create a mock v6.9 Material object
    class MockV69Material:
        def __init__(self, name, structure, sigma_y=0.0, element=''):
            self.name = name
            self.structure = structure
            self.sigma_y = sigma_y
            self.element = element
    
    # Simulate v6.9 Material
    v69_fe = MockV69Material('Fe_BCC', 'BCC', sigma_y=250.0, element='Fe')
    mat3 = flc.add_from_v69_material(v69_fe, flc0=0.225, name='SPCC_v69')
    print(f"Added from v6.9 Material: {mat3.name}")
    print(f"  Structure: {mat3.structure}")
    print(f"  σ_y: {mat3.sigma_y:.1f} MPa")
    print(f"  τ/σ = {mat3.tau_sigma:.4f}")
    print(f"  R_comp = {mat3.R_comp:.2f}")
    print(f"  V_eff = {mat3.V_eff:.4f}")
    
    # 4. Predictions
    print("\n[4] FLC Predictions")
    print("-"*70)
    
    for mat_name in ['NewSteel', 'MgAlloy', 'SPCC_v69']:
        print(f"\n{mat_name}:")
        for mode in ['Uniaxial', 'Plane Strain', 'Equi-biaxial']:
            eps1 = flc.predict(mat_name, mode)
            print(f"  {mode}: ε₁ = {eps1:.4f}")
    
    # 5. Full summary
    print("\n[5] Material Summary: SPCC_v69")
    print("-"*70)
    print(flc.summary('SPCC_v69'))
    
    # 6. τ/σ comparison (pure metal native values)
    print("\n[6] τ/σ Values by Base Element (Pure Metal Native)")
    print("-"*70)
    print(f"{'Element':<10} {'Structure':<10} {'τ/σ':>10} {'R_comp':>10} {'Note':>15}")
    print("-"*60)
    
    # Pure metals: BCC/FCC use T_twin=1.0, HCP uses native characteristic
    elements_info = [
        ('Fe', 'BCC', 1.0, ''),
        ('Cu', 'FCC', 1.0, ''),
        ('Al', 'FCC', 1.0, ''),
        ('Ni', 'FCC', 1.0, ''),
        ('Ti', 'HCP', 0.0, 'HCP, c/a effect'),  # Ti native
        ('Mg', 'HCP', 0.0, 'twin-dominant'),   # Mg native
    ]
    
    for elem, struct, T_twin, note in elements_info:
        tau_s = get_tau_sigma(elem, T_twin)
        r_c = get_R_comp(elem, T_twin)
        print(f"{elem:<10} {struct:<10} {tau_s:>10.4f} {r_c:>10.2f} {note:>15}")
    
    # 7. HCP T_twin interpolation
    print("\n[7] HCP τ/σ vs T_twin (Mg)")
    print("-"*70)
    for T_twin in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tau_s = get_tau_sigma('Mg', T_twin)
        r_c = get_R_comp('Mg', T_twin)
        print(f"  T_twin = {T_twin:.2f} → τ/σ = {tau_s:.4f}, R_comp = {r_c:.2f}")
    
    # 8. Quick usage example
    print("\n[8] Quick Usage Example")
    print("-"*70)
    print("""
# Basic usage:
from unified_flc_v8_1 import FLCPredictor, predict_flc

# Use built-in material
eps1 = predict_flc('Cu', 'Plane Strain')  # 0.346

# Add new material from element
flc = FLCPredictor()
flc.add_from_v69('MySteel', flc0=0.28, base_element='Fe')
curve = flc.predict_curve('MySteel')

# Add from explicit parameters
flc.add_from_flc0('Custom', flc0=0.30, tau_sigma=0.565, R_comp=1.0)

# If v6.9 is available:
# from unified_yield_fatigue_v6_9 import MATERIALS
# flc.add_from_v69_material(MATERIALS['Fe'], flc0=0.225, name='SPCC')
""")


if __name__ == '__main__':
    demo()
