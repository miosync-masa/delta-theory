"""
δ-Theory Core Library
=====================
Unified materials science prediction based on geometric first principles.

"Nature is Geometry" - All material properties emerge from crystal structure.

Modules:
    - unified_yield_fatigue_v6_9: Main yield + fatigue model (v6.9b)
    - unified_flc_v8_1: FLC prediction with v6.9 integration (v8.1)
    - dbt_unified: Ductile-Brittle Transition Temperature prediction
    - materials: Material database
    - fatigue_redis_api: FatigueData-AM2022 Redis API (optional)
    - banners: ASCII Art banners (random selection)

Version History:
    v5.0   - Yield stress from δ-theory (f_d, E_bond, crystal geometry)
    v6.9b  - Unified yield + fatigue with multiaxial (τ/σ, R)
    v6.10  - Universal fatigue validation (2472 points, 5 AM materials)
    v7.2   - FLC from free volume consumption
    v8.0   - Forming-Fatigue integration (η → r_th_eff)
    v8.1   - FLC 7-mode discrete formulation + v6.9 integration
           - "Nature is Geometry" - ε₁ = |V|_eff × C_j / R_j

Example:
    >>> from delta_theory import calc_sigma_y, MATERIALS
    >>> sigma_y = calc_sigma_y(MATERIALS['Fe'])
    
    >>> from delta_theory import FLCPredictor, predict_flc
    >>> flc = FLCPredictor()
    >>> flc.add_from_v69('SPCC', flc0=0.225, base_element='Fe')
    >>> eps1 = flc.predict('SPCC', 'Plane Strain')
    
    >>> from delta_theory import show_banner
    >>> show_banner()  # Random ASCII art!
"""

# ==============================================================================
# Banners (ASCII Art) - Load first for startup display
# ==============================================================================
from .banners import show_banner, get_random_banner, BANNERS

# ==============================================================================
# Core: Yield + Fatigue (v6.9b)
# ==============================================================================
from .unified_yield_fatigue_v6_9 import (
    # Material dataclass
    Material,
    MATERIALS,
    
    # Yield stress
    calc_sigma_y,
    sigma_base_delta,
    delta_sigma_ss,
    delta_sigma_taylor,
    delta_sigma_ppt,
    
    # Fatigue
    fatigue_life_const_amp,
    generate_sn_curve,
    FATIGUE_CLASS_PRESET,
    
    # Multiaxial (τ/σ, R)
    yield_by_mode,
    tau_over_sigma,
    sigma_c_over_sigma_t,
    T_TWIN,
    R_COMP,
    C_CLASS_DEFAULT,
)

# ==============================================================================
# FLC v8.1: 7-Mode Discrete + v6.9 Integration
# ==============================================================================
from .unified_flc_v8_1 import (
    # Main class
    FLCPredictor,
    FLCMaterial,
    FLC_MATERIALS,
    
    # Convenience functions
    predict_flc,
    predict_flc_curve,
    get_flc0,
    calibrate_V_eff,
    create_material_from_v69,
    
    # Core functions
    calc_R_eff,
    calc_K_coeff,
    predict_flc_mode,
    
    # v6.9 parameter helpers
    get_tau_sigma,
    get_R_comp,
    get_v69_material,
    
    # Constants
    STANDARD_MODES,
    MODE_ORDER,
    LOCALIZATION_COEFFS,
    R_TH_VIRGIN,
    ELEMENT_STRUCTURE,
    
    # Flags
    V69_AVAILABLE,
)

# ==============================================================================
# DBT: Ductile-Brittle Transition
# ==============================================================================
try:
    from .dbt_unified import (
        DBTUnified,
        DBTCore,
        GrainSizeView,
        TemperatureView,
        SegregationView,
        MATERIAL_FE,
    )
except ImportError:
    DBTUnified = None
    DBTCore = None
    GrainSizeView = None
    TemperatureView = None
    SegregationView = None
    MATERIAL_FE = None

# ==============================================================================
# Materials Database (GPU-accelerated)
# ==============================================================================
try:
    from .materials import MaterialGPU
except ImportError:
    MaterialGPU = None

# ==============================================================================
# Optional: FatigueDB (requires upstash-redis)
# ==============================================================================
try:
    from .fatigue_redis_api import FatigueDB
except ImportError:
    FatigueDB = None  # upstash-redis not installed

# ==============================================================================
# Package Metadata
# ==============================================================================
__version__ = "8.1.0"
__author__ = "Masamichi Iizumi & Tamaki"

__all__ = [
    # === Banners ===
    "show_banner",
    "get_random_banner",
    "BANNERS",
    
    # === v6.9 Yield + Fatigue ===
    "Material",
    "MATERIALS",
    "calc_sigma_y",
    "sigma_base_delta",
    "delta_sigma_ss",
    "delta_sigma_taylor",
    "delta_sigma_ppt",
    "fatigue_life_const_amp",
    "generate_sn_curve",
    "yield_by_mode",
    "tau_over_sigma",
    "sigma_c_over_sigma_t",
    "FATIGUE_CLASS_PRESET",
    "T_TWIN",
    "R_COMP",
    "C_CLASS_DEFAULT",
    
    # === v8.1 FLC ===
    "FLCPredictor",
    "FLCMaterial",
    "FLC_MATERIALS",
    "predict_flc",
    "predict_flc_curve",
    "get_flc0",
    "calibrate_V_eff",
    "create_material_from_v69",
    "calc_R_eff",
    "calc_K_coeff",
    "predict_flc_mode",
    "get_tau_sigma",
    "get_R_comp",
    "get_v69_material",
    "STANDARD_MODES",
    "MODE_ORDER",
    "LOCALIZATION_COEFFS",
    "R_TH_VIRGIN",
    "ELEMENT_STRUCTURE",
    "V69_AVAILABLE",
    
    # === DBT ===
    "DBTUnified",
    "DBTCore",
    "GrainSizeView",
    "TemperatureView",
    "SegregationView",
    "MATERIAL_FE",
    
    # === Materials ===
    "MaterialGPU",
    
    # === FatigueDB (optional) ===
    "FatigueDB",
    
    # === Info ===
    "info",
]


# ==============================================================================
# Quick Reference
# ==============================================================================
def info():
    """Print δ-Theory library overview with random ASCII banner."""
    show_banner()  # 🎲 Random banner every time!
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  δ-Theory Core Library v{__version__}                                      ║
║  "Nature is Geometry"  ─  ε₁ = |V|_eff × C_j / R_j                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  YIELD STRESS (v6.9b)                                                ║
║    σ_y = f(crystal_structure, f_d, E_bond, T)                        ║
║    Mean error: 2.6% across 10 metals (ZERO fitting parameters)       ║
║    >>> calc_sigma_y(MATERIALS['Fe'])                                 ║
║                                                                      ║
║  FATIGUE LIFE (v6.10)                                                ║
║    N = f(r, r_th, structure)  |  r_th: BCC=0.65, FCC=0.02, HCP=0.20 ║
║    Universal r-N normalization across AM materials                   ║
║    >>> fatigue_life_const_amp(sigma_a, MATERIALS['Fe'])              ║
║                                                                      ║
║  FLC v8.1 - 7-MODE DISCRETE FORMULATION                              ║
║    ε₁,j = |V|_eff × C_j / R_j                                        ║
║    C_j = 1 + 0.75β + 0.48β²  (localization, frozen)                  ║
║    R_j = w_σ + w_τ/(τ/σ) + w_c/R_comp  (mixed resistance)            ║
║    Mean error: 4.7% across 49 points (7 materials × 7 modes)         ║
║                                                                      ║
║    >>> flc = FLCPredictor()                                          ║
║    >>> flc.add_from_v69('SPCC', flc0=0.225, base_element='Fe')       ║
║    >>> eps1 = flc.predict('SPCC', 'Plane Strain')                    ║
║                                                                      ║
║  DBT TEMPERATURE                                                     ║
║    3 views: Grain size / Temperature / Segregation (time)            ║
║    >>> DBTUnified().predict_dbtt(material, grain_size)               ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Core Equations:                                                     ║
║    Λ = K / |V|_eff    (Λ > 1 → yield/fracture)                      ║
║    τ/σ = (α_s/α_t) × C_class × T_twin × A_texture                   ║
║    FLC: 1-point calibration from FLC₀ → all 7 modes                  ║
║                                                                      ║
║  Authors: Masamichi Iizumi & Tamaki                                  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
