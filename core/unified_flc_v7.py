#!/usr/bin/env python3
"""δ-Theory Unified FLC & Forming-Fatigue Integration (v7.2 + v8.0)

FLC Model (v7.2):
  - Free volume consumption approach
  - FLC(β) = FLC₀_pure × (1 - η_total) × h(β, R, τ/σ)
  - Supports: BCC, FCC, HCP with strengthening mechanisms

Forming-Fatigue Integration (v8.0):
  - r_th_eff = r_th_virgin × (1 - η_forming)
  - Predicts fatigue life reduction due to forming history

Usage:
  from unified_flc_v7 import FLCPredictor, FormingFatigueIntegrator
  
  # FLC prediction
  flc = FLCPredictor()
  Em = flc.predict(beta=0.0, material='SPCC')
  
  # Forming-fatigue integration
  integrator = FormingFatigueIntegrator()
  r_th_eff = integrator.effective_r_th(eta_forming=0.4, structure='BCC')
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np

# Try to import from v6.9b
try:
    from unified_yield_fatigue_v6_9b_tau_classes__2_ import (
        Material, MATERIALS, 
        T_TWIN, R_COMP,
        sigma_base_delta,
    )
    V69_AVAILABLE = True
except ImportError:
    V69_AVAILABLE = False
    warnings.warn("v6.9b not found. Using built-in material database.")


# ==============================================================================
# Constants
# ==============================================================================

# Crystal structure fatigue thresholds (from v6.10)
R_TH_VIRGIN = {
    'BCC': 0.65,
    'FCC': 0.02,
    'HCP': 0.20,
}

# Slip system counts (affects FLC shape)
N_SLIP = {
    'BCC': 48,  # {110}<111>:12 + {112}<111>:12 + {123}<111>:24
    'FCC': 12,  # {111}<110>
    'HCP': 3,   # Basal plane (room temperature)
}

# τ/σ ratios (from v5.0)
TAU_SIGMA = {
    'Fe': 0.565, 'Cu': 0.565, 'Al': 0.565, 'Ni': 0.565,
    'Ti': 0.546, 'Mg': 0.327, 'Zn': 0.480,
    'BCC': 0.565, 'FCC': 0.565, 'HCP': 0.50,
}

# R_comp ratios (from v5.0)
R_COMPRESSION = {
    'Fe': 1.00, 'Cu': 1.00, 'Al': 1.00, 'Ni': 1.00,
    'Ti': 1.00, 'Mg': 0.60, 'Zn': 1.20,
    'BCC': 1.00, 'FCC': 1.00, 'HCP': 0.80,
}


# ==============================================================================
# Material Database for FLC (extended)
# ==============================================================================

@dataclass
class FLCMaterial:
    """Material data for FLC prediction."""
    name: str
    structure: Literal['BCC', 'FCC', 'HCP']
    
    # v5.0 base parameters
    f_d: float           # d-electron directionality factor
    E_bond: float        # Bond energy [eV]
    sigma_y: float       # Yield stress [MPa]
    
    # Multiaxial parameters
    tau_sigma: float = 0.565    # τ/σ ratio
    R_comp: float = 1.0         # Compression/tension ratio
    
    # Strengthening mechanisms (free volume consumption)
    C_ss: float = 0.0           # Solid solution concentration
    f_ppt: float = 0.0          # Precipitate/martensite fraction
    rho_d: float = 1e12         # Dislocation density [m^-2]
    d_grain: float = 50e-6      # Grain size [m]
    
    @property
    def r_th(self) -> float:
        """Virgin material fatigue threshold."""
        return R_TH_VIRGIN[self.structure]
    
    @property
    def n_slip(self) -> int:
        """Number of slip systems."""
        return N_SLIP[self.structure]


# Built-in material database
FLC_MATERIALS: Dict[str, FLCMaterial] = {
    # BCC steels
    'SPCC': FLCMaterial(
        'SPCC', 'BCC', f_d=1.5, E_bond=4.28, sigma_y=200,
        tau_sigma=0.565, R_comp=1.00,
        C_ss=0.02, f_ppt=0.0, rho_d=1e13, d_grain=20e-6
    ),
    'DP590': FLCMaterial(
        'DP590', 'BCC', f_d=1.5, E_bond=4.28, sigma_y=590,
        tau_sigma=0.565, R_comp=1.00,
        C_ss=0.08, f_ppt=0.15, rho_d=1e14, d_grain=5e-6
    ),
    'SECD-E16': FLCMaterial(
        'SECD-E16', 'BCC', f_d=1.5, E_bond=4.28, sigma_y=300,
        tau_sigma=0.565, R_comp=1.00,
        C_ss=0.03, f_ppt=0.0, rho_d=1e13, d_grain=15e-6
    ),
    
    # FCC metals/alloys
    'Al': FLCMaterial(
        'Al', 'FCC', f_d=1.6, E_bond=3.39, sigma_y=35,
        tau_sigma=0.565, R_comp=1.00,
        C_ss=0.01, f_ppt=0.0, rho_d=1e12, d_grain=50e-6
    ),
    'Cu': FLCMaterial(
        'Cu', 'FCC', f_d=2.0, E_bond=3.49, sigma_y=70,
        tau_sigma=0.565, R_comp=1.00,
        C_ss=0.0, f_ppt=0.0, rho_d=1e12, d_grain=50e-6
    ),
    'SUS304': FLCMaterial(
        'SUS304', 'FCC', f_d=2.6, E_bond=4.44, sigma_y=250,
        tau_sigma=0.565, R_comp=1.00,
        C_ss=0.25, f_ppt=0.0, rho_d=1e13, d_grain=30e-6
    ),
    
    # HCP metals/alloys
    'Ti': FLCMaterial(
        'Ti', 'HCP', f_d=5.7, E_bond=4.85, sigma_y=275,
        tau_sigma=0.546, R_comp=1.00,
        C_ss=0.02, f_ppt=0.0, rho_d=1e13, d_grain=30e-6
    ),
    'Mg_AZ31': FLCMaterial(
        'Mg_AZ31', 'HCP', f_d=8.2, E_bond=1.51, sigma_y=160,
        tau_sigma=0.327, R_comp=0.60,
        C_ss=0.05, f_ppt=0.02, rho_d=1e13, d_grain=15e-6
    ),
}


# ==============================================================================
# FLC Predictor (v7.2)
# ==============================================================================

@dataclass
class FLCParams:
    """Optimized FLC model parameters."""
    # Base FLC₀ parameters
    A: float = 0.2383
    alpha: float = 0.150        # (1 - r_th) exponent
    beta_fd: float = -0.166     # f_d exponent
    gamma_E: float = 0.227      # E_bond exponent
    
    # Free volume consumption coefficients
    k_ss: float = -0.098        # Solid solution
    k_ppt: float = 0.345        # Precipitate/martensite
    k_wh: float = 0.067         # Work hardening
    k_HP: float = 0.050         # Hall-Petch (grain refinement)
    
    # V-shape parameters
    k_neg: float = 0.597        # Deep draw bonus
    k_pos: float = 0.235        # Biaxial penalty
    w_neg: float = 0.313        # Deep draw width
    w_pos: float = 0.450        # Biaxial width


class FLCPredictor:
    """
    δ-Theory FLC Predictor (v7.2)
    
    Predicts Forming Limit Curve based on:
    - Crystal structure (r_th, n_slip)
    - d-electron directionality (f_d)
    - Bond energy (E_bond)
    - Free volume consumption (strengthening mechanisms)
    - Multiaxial stress state (τ/σ, R)
    """
    
    def __init__(self, params: Optional[FLCParams] = None):
        self.params = params or FLCParams()
        self._materials = FLC_MATERIALS.copy()
    
    def add_material(self, mat: FLCMaterial) -> None:
        """Add custom material to database."""
        self._materials[mat.name] = mat
    
    def get_material(self, name: str) -> FLCMaterial:
        """Get material by name."""
        if name not in self._materials:
            raise ValueError(f"Material '{name}' not found. "
                           f"Available: {list(self._materials.keys())}")
        return self._materials[name]
    
    def free_volume_consumption(self, mat: FLCMaterial) -> Tuple[float, Dict[str, float]]:
        """
        Calculate free volume consumption from strengthening mechanisms.
        
        Returns:
            (remaining_ratio, breakdown_dict)
        """
        p = self.params
        
        # Solid solution
        eta_ss = p.k_ss * mat.C_ss
        
        # Precipitate/martensite
        eta_ppt = p.k_ppt * mat.f_ppt
        
        # Work hardening (dislocation density)
        rho_ref = 1e12
        eta_wh = p.k_wh * np.log10(mat.rho_d / rho_ref) if mat.rho_d > rho_ref else 0.0
        
        # Hall-Petch (grain refinement)
        d_ref = 50e-6
        eta_HP = p.k_HP * (np.sqrt(d_ref / mat.d_grain) - 1) if mat.d_grain < d_ref else 0.0
        
        eta_total = eta_ss + eta_ppt + eta_wh + eta_HP
        remaining = max(0.1, 1 - eta_total)
        
        breakdown = {
            'eta_ss': eta_ss,
            'eta_ppt': eta_ppt,
            'eta_wh': eta_wh,
            'eta_HP': eta_HP,
            'eta_total': eta_total,
            'remaining': remaining,
        }
        
        return remaining, breakdown
    
    def flc0_pure(self, mat: FLCMaterial) -> float:
        """
        Calculate pure metal FLC₀ (β=0).
        
        FLC₀_pure = A × (1-r_th)^α × f_d^β × E_bond^γ
        """
        p = self.params
        return (p.A 
                * ((1 - mat.r_th) ** p.alpha)
                * (mat.f_d ** p.beta_fd)
                * (mat.E_bond ** p.gamma_E))
    
    def shape_factor(self, beta: float, mat: FLCMaterial) -> float:
        """
        Calculate V-shape factor h(β, R, τ/σ).
        
        β < 0: Deep draw → R-dependent bonus
        β > 0: Biaxial → τ/σ-dependent penalty
        """
        p = self.params
        
        if beta <= 0:
            # Deep draw: compression component
            # R = 1: symmetric → bonus
            # R < 1 (Mg): compression weak → reduced bonus
            bonus = p.k_neg * mat.R_comp * np.exp(-((beta + 0.5) / p.w_neg)**2)
            h = 1 + bonus
        else:
            # Biaxial: shear component (thickness reduction)
            # τ/σ = 0.565: normal → base penalty
            # τ/σ < 0.565 (Mg): shear weak → increased penalty
            penalty = p.k_pos * (0.565 / mat.tau_sigma) * (1 - np.exp(-((beta - 0) / p.w_pos)**2))
            h = 1 - penalty
        
        return h
    
    def predict(self, 
                beta: float,
                material: Union[str, FLCMaterial],
                include_breakdown: bool = False) -> Union[float, Tuple[float, Dict]]:
        """
        Predict FLC (major strain limit) at given β.
        
        Args:
            beta: Strain ratio ε₂/ε₁ (-0.5 to 1.0)
            material: Material name or FLCMaterial object
            include_breakdown: If True, return (Em, breakdown_dict)
        
        Returns:
            Em: Major strain limit at β
            breakdown: (optional) Dict with intermediate values
        """
        if isinstance(material, str):
            mat = self.get_material(material)
        else:
            mat = material
        
        # Base FLC₀ for pure metal
        FLC0_pure = self.flc0_pure(mat)
        
        # Free volume consumption
        fv_ratio, fv_breakdown = self.free_volume_consumption(mat)
        FLC0 = FLC0_pure * fv_ratio
        
        # Shape factor
        h = self.shape_factor(beta, mat)
        
        # Final FLC
        Em = FLC0 * h
        
        if include_breakdown:
            breakdown = {
                'FLC0_pure': FLC0_pure,
                'fv_ratio': fv_ratio,
                'FLC0': FLC0,
                'h': h,
                'Em': Em,
                **fv_breakdown
            }
            return Em, breakdown
        
        return Em
    
    def predict_curve(self, 
                      material: Union[str, FLCMaterial],
                      beta_range: Optional[np.ndarray] = None,
                      n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict full FLC curve.
        
        Returns:
            (beta_array, Em_array)
        """
        if beta_range is None:
            beta_range = np.linspace(-0.5, 1.0, n_points)
        
        Em_values = np.array([self.predict(b, material) for b in beta_range])
        return beta_range, Em_values


# ==============================================================================
# Forming-Fatigue Integrator (v8.0)
# ==============================================================================

@dataclass
class FormingState:
    """State of formed material at a point."""
    eta_forming: float          # Free volume consumption from forming
    r_th_eff: float             # Effective fatigue threshold
    structure: str              # Crystal structure
    
    # Optional: position info
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    
    @property
    def fatigue_limit_reduction(self) -> float:
        """Fatigue limit reduction ratio."""
        r_th_virgin = R_TH_VIRGIN[self.structure]
        return 1 - (self.r_th_eff / r_th_virgin)


class FormingFatigueIntegrator:
    """
    δ-Theory Forming-Fatigue Integration (v8.0)
    
    Predicts how forming history affects fatigue life:
    - η_forming: Free volume consumed during forming
    - r_th_eff: Effective fatigue threshold after forming
    - N/N₀: Fatigue life ratio vs virgin material
    """
    
    def __init__(self, kappa: float = 1.0):
        """
        Args:
            kappa: Forming-fatigue coupling exponent
                   κ > 1: Forming damage strongly affects fatigue
                   κ < 1: Forming damage mildly affects fatigue
        """
        self.kappa = kappa
    
    def effective_r_th(self, 
                       eta_forming: float,
                       structure: Literal['BCC', 'FCC', 'HCP']) -> float:
        """
        Calculate effective fatigue threshold after forming.
        
        r_th_eff = r_th_virgin × (1 - η_forming)^κ
        
        Args:
            eta_forming: Free volume consumption (0 to 1)
            structure: Crystal structure
        
        Returns:
            r_th_eff: Effective fatigue threshold
        """
        r_th_v = R_TH_VIRGIN[structure]
        remaining = max(0.01, 1 - eta_forming)
        return r_th_v * (remaining ** self.kappa)
    
    def critical_eta(self, 
                     r_applied: float,
                     structure: Literal['BCC', 'FCC', 'HCP']) -> float:
        """
        Calculate critical forming consumption.
        
        Beyond this η, virgin-safe stress becomes finite-life.
        
        Args:
            r_applied: Applied stress ratio σ_a/σ_y
            structure: Crystal structure
        
        Returns:
            η_critical: Critical forming consumption
        """
        r_th_v = R_TH_VIRGIN[structure]
        
        if r_applied >= r_th_v:
            return 0.0  # Already finite life
        
        # r_th_eff = r_applied at critical point
        # r_th_v × (1 - η)^κ = r_applied
        # (1 - η)^κ = r_applied / r_th_v
        # 1 - η = (r_applied / r_th_v)^(1/κ)
        # η = 1 - (r_applied / r_th_v)^(1/κ)
        
        return 1 - (r_applied / r_th_v) ** (1 / self.kappa)
    
    def fatigue_life_ratio(self,
                           r_applied: float,
                           eta_forming: float,
                           structure: Literal['BCC', 'FCC', 'HCP'],
                           n_exp: float = 15.0) -> float:
        """
        Calculate fatigue life ratio N/N₀.
        
        Args:
            r_applied: Applied stress ratio σ_a/σ_y
            eta_forming: Free volume consumption from forming
            structure: Crystal structure
            n_exp: Basquin exponent
        
        Returns:
            N/N₀: Fatigue life ratio (∞ if below threshold)
        """
        r_th_v = R_TH_VIRGIN[structure]
        r_th_eff = self.effective_r_th(eta_forming, structure)
        
        # Case 1: Below effective threshold → infinite life
        if r_applied <= r_th_eff:
            return np.inf
        
        # Case 2: Virgin was also finite life
        if r_applied > r_th_v:
            term_virgin = (r_applied - r_th_v) / (1 - r_th_v)
            term_formed = (r_applied - r_th_eff) / (1 - r_th_eff)
            return (term_formed / term_virgin) ** (-n_exp)
        
        # Case 3: Virgin was infinite, now finite
        # Return absolute cycles instead of ratio
        return 0.0  # Indicates transition from infinite to finite
    
    def is_safe(self,
                r_applied: float,
                eta_forming: float,
                structure: Literal['BCC', 'FCC', 'HCP']) -> bool:
        """Check if formed material is safe (below fatigue limit)."""
        r_th_eff = self.effective_r_th(eta_forming, structure)
        return r_applied <= r_th_eff
    
    def analyze_forming_steps(self,
                              eta_steps: List[float],
                              r_applied: float,
                              structure: Literal['BCC', 'FCC', 'HCP']) -> List[FormingState]:
        """
        Analyze forming process step by step.
        
        Args:
            eta_steps: Cumulative η at each step
            r_applied: Applied stress during use
            structure: Crystal structure
        
        Returns:
            List of FormingState for each step
        """
        states = []
        for i, eta in enumerate(eta_steps):
            r_th_eff = self.effective_r_th(eta, structure)
            state = FormingState(
                eta_forming=eta,
                r_th_eff=r_th_eff,
                structure=structure
            )
            states.append(state)
        
        return states


# ==============================================================================
# Combined Analysis
# ==============================================================================

class DeltaFormingAnalyzer:
    """
    Combined FLC + Forming-Fatigue Analysis.
    
    Workflow:
    1. Predict FLC for material
    2. Assess forming risk (Λ = ε_actual / FLC)
    3. Calculate η consumption at each point
    4. Predict post-forming fatigue life
    """
    
    def __init__(self, 
                 flc_params: Optional[FLCParams] = None,
                 kappa: float = 1.0):
        self.flc = FLCPredictor(flc_params)
        self.fatigue = FormingFatigueIntegrator(kappa)
    
    def forming_lambda(self,
                       epsilon_major: float,
                       beta: float,
                       material: Union[str, FLCMaterial]) -> float:
        """
        Calculate forming severity parameter Λ.
        
        Λ = ε_actual / FLC
        Λ < 1: Safe
        Λ = 1: Critical
        Λ > 1: Failure
        """
        flc = self.flc.predict(beta, material)
        return epsilon_major / flc
    
    def eta_from_lambda(self, 
                        lambda_val: float,
                        max_eta: float = 0.95) -> float:
        """
        Estimate η from Λ.
        
        Simple model: η ≈ Λ (up to max_eta)
        """
        return min(lambda_val, max_eta)
    
    def full_analysis(self,
                      material: Union[str, FLCMaterial],
                      epsilon_major: float,
                      beta: float,
                      r_applied: float) -> Dict:
        """
        Complete forming + fatigue analysis.
        
        Returns dict with all relevant parameters.
        """
        if isinstance(material, str):
            mat = self.flc.get_material(material)
        else:
            mat = material
        
        # FLC prediction
        flc, flc_breakdown = self.flc.predict(beta, mat, include_breakdown=True)
        
        # Forming severity
        lambda_val = epsilon_major / flc
        
        # Estimate η
        eta = self.eta_from_lambda(lambda_val)
        
        # Effective fatigue threshold
        r_th_eff = self.fatigue.effective_r_th(eta, mat.structure)
        
        # Fatigue life ratio
        N_ratio = self.fatigue.fatigue_life_ratio(r_applied, eta, mat.structure)
        
        # Safety assessment
        is_forming_safe = lambda_val < 1.0
        is_fatigue_safe = self.fatigue.is_safe(r_applied, eta, mat.structure)
        
        return {
            'material': mat.name,
            'structure': mat.structure,
            # Forming
            'FLC': flc,
            'epsilon_major': epsilon_major,
            'beta': beta,
            'Lambda': lambda_val,
            'is_forming_safe': is_forming_safe,
            # Free volume
            'eta_forming': eta,
            'fv_remaining': 1 - eta,
            **flc_breakdown,
            # Fatigue
            'r_th_virgin': mat.r_th,
            'r_th_eff': r_th_eff,
            'r_applied': r_applied,
            'N_ratio': N_ratio,
            'is_fatigue_safe': is_fatigue_safe,
            # Overall
            'overall_safe': is_forming_safe and is_fatigue_safe,
        }


# ==============================================================================
# Convenience functions
# ==============================================================================

def predict_flc(material: str, beta: float = 0.0) -> float:
    """Quick FLC prediction."""
    return FLCPredictor().predict(beta, material)


def effective_fatigue_threshold(eta: float, structure: str) -> float:
    """Quick effective r_th calculation."""
    return FormingFatigueIntegrator().effective_r_th(eta, structure)


def critical_forming_consumption(r_applied: float, structure: str) -> float:
    """Quick critical η calculation."""
    return FormingFatigueIntegrator().critical_eta(r_applied, structure)


# ==============================================================================
# Demo / Test
# ==============================================================================

def demo():
    """Demonstration of v7.2 + v8.0 capabilities."""
    
    print("="*70)
    print("δ-Theory v7.2 + v8.0: FLC & Forming-Fatigue Integration")
    print("="*70)
    
    # FLC Prediction
    print("\n[1] FLC Prediction (v7.2)")
    print("-"*40)
    
    flc = FLCPredictor()
    
    materials = ['SPCC', 'DP590', 'Al', 'SUS304', 'Ti', 'Mg_AZ31']
    betas = [-0.5, 0.0, 1.0]
    
    print(f"{'Material':10} {'β=-0.5':>8} {'β=0':>8} {'β=1':>8}")
    print("-"*40)
    
    for mat_name in materials:
        Em_values = [flc.predict(b, mat_name) for b in betas]
        print(f"{mat_name:10} {Em_values[0]:>8.3f} {Em_values[1]:>8.3f} {Em_values[2]:>8.3f}")
    
    # Free Volume Breakdown
    print("\n[2] Free Volume Consumption Breakdown")
    print("-"*40)
    
    for mat_name in ['SPCC', 'DP590']:
        mat = flc.get_material(mat_name)
        fv_ratio, breakdown = flc.free_volume_consumption(mat)
        print(f"\n{mat_name}:")
        print(f"  η_ss  = {breakdown['eta_ss']:.3f} (solid solution)")
        print(f"  η_ppt = {breakdown['eta_ppt']:.3f} (precipitate)")
        print(f"  η_wh  = {breakdown['eta_wh']:.3f} (work hardening)")
        print(f"  η_HP  = {breakdown['eta_HP']:.3f} (grain refinement)")
        print(f"  FV remaining = {fv_ratio:.3f}")
    
    # Forming-Fatigue Integration
    print("\n[3] Forming-Fatigue Integration (v8.0)")
    print("-"*40)
    
    integrator = FormingFatigueIntegrator()
    
    # Critical η for various stress levels
    print("\nCritical η for BCC steel (r_th_virgin = 0.65):")
    for r in [0.30, 0.40, 0.50, 0.55, 0.60]:
        eta_crit = integrator.critical_eta(r, 'BCC')
        print(f"  r = {r:.2f} → η_critical = {eta_crit*100:.1f}%")
    
    # Nidec part example
    print("\n[4] Nidec Part Example (SECD-E16)")
    print("-"*40)
    
    analyzer = DeltaFormingAnalyzer()
    
    result = analyzer.full_analysis(
        material='SECD-E16',
        epsilon_major=0.25,  # 25% strain at corner
        beta=0.0,            # Plane strain
        r_applied=0.50       # 50% of yield stress during use
    )
    
    print(f"Material: {result['material']} ({result['structure']})")
    print(f"\nForming:")
    print(f"  ε_major = {result['epsilon_major']:.2f}")
    print(f"  FLC     = {result['FLC']:.3f}")
    print(f"  Λ       = {result['Lambda']:.3f}")
    print(f"  Safe?   = {result['is_forming_safe']}")
    print(f"\nFatigue:")
    print(f"  η_forming = {result['eta_forming']:.3f}")
    print(f"  r_th_virgin = {result['r_th_virgin']:.3f}")
    print(f"  r_th_eff    = {result['r_th_eff']:.3f}")
    print(f"  r_applied   = {result['r_applied']:.3f}")
    print(f"  Safe?       = {result['is_fatigue_safe']}")
    print(f"\nOverall: {'✓ SAFE' if result['overall_safe'] else '✗ UNSAFE'}")


if __name__ == '__main__':
    demo()
