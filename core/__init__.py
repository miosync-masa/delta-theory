"""
δ-Theory Core Library
=====================
Unified materials science prediction based on geometric first principles.

"Nature is Geometry" - All material properties emerge from crystal structure.

Modules:
    - unified_yield_fatigue_v6_9: Main yield + fatigue model (v6.9b)
    - unified_flc_v7: Forming Limit Curve + Forming-Fatigue Integration (v7.2/v8.0)
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
           - "Nature is Geometry" - m = Z × f_d (topology redefined)

Example:
    >>> from delta_theory import calc_sigma_y, MATERIALS
    >>> sigma_y = calc_sigma_y(MATERIALS['Fe'])
    
    >>> from delta_theory import FLCPredictor
    >>> flc = FLCPredictor()
    >>> Em = flc.predict(beta=0.0, material='SPCC')
    
    >>> from delta_theory import FormingFatigueIntegrator
    >>> integrator = FormingFatigueIntegrator()
    >>> r_th_eff = integrator.effective_r_th(eta_forming=0.4, structure='BCC')
    
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
    T_TWIN,
    R_COMP,
)

# ==============================================================================
# FLC + Forming-Fatigue (v7.2 / v8.0)
# ==============================================================================
from .unified_flc_v7 import (
    # FLC prediction (v7.2)
    FLCPredictor,
    FLCParams,
    FLCMaterial,
    FLC_MATERIALS,
    
    # Forming-Fatigue integration (v8.0)
    FormingFatigueIntegrator,
    FormingState,
    DeltaFormingAnalyzer,
    
    # Convenience functions
    predict_flc,
    effective_fatigue_threshold,
    critical_forming_consumption,
    
    # Constants
    R_TH_VIRGIN,
    N_SLIP,
)

# ==============================================================================
# DBT: Ductile-Brittle Transition
# ==============================================================================
from .dbt_unified import (
    DBTUnified,
    DBTCore,
    GrainSizeView,
    TemperatureView,
    SegregationView,
    MATERIAL_FE,
)

# ==============================================================================
# Materials Database (GPU-accelerated)
# ==============================================================================
from .materials import MaterialGPU

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
__version__ = "8.1.2"
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
    "FATIGUE_CLASS_PRESET",
    "T_TWIN",
    "R_COMP",
    
    # === v7.2 FLC ===
    "FLCPredictor",
    "FLCParams",
    "FLCMaterial",
    "FLC_MATERIALS",
    "predict_flc",
    "R_TH_VIRGIN",
    "N_SLIP",
    
    # === v8.0 Forming-Fatigue ===
    "FormingFatigueIntegrator",
    "FormingState",
    "DeltaFormingAnalyzer",
    "effective_fatigue_threshold",
    "critical_forming_consumption",
    
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
║  "Nature is Geometry"  ─  m = Z × f_d  ─  Λ = K/|V|_eff              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  YIELD STRESS (v5.0)                                                 ║
║    σ_y = f(crystal_structure, f_d, E_bond, T)                        ║
║    Mean error: 2.6% across 10 metals (ZERO fitting parameters)       ║
║    >>> calc_sigma_y(MATERIALS['Fe'])                                 ║
║                                                                      ║
║  FATIGUE LIFE (v6.10)                                                ║
║    N = f(r, r_th, structure)  |  r_th: BCC=0.65, FCC=0.02, HCP=0.20 ║
║    Universal r-N normalization across AM materials                   ║
║    >>> fatigue_life_const_amp(sigma_a, MATERIALS['Fe'])              ║
║                                                                      ║
║  FLC - FORMING LIMIT (v7.2)                                          ║
║    FLC(β) = FLC₀ × (1-η) × h(β, R, τ/σ)                             ║
║    Mean error: ~3% (geometry-based, minimal calibration)             ║
║    >>> FLCPredictor().predict(beta=0.0, material='SPCC')             ║
║                                                                      ║
║  FORMING-FATIGUE (v8.0)                                              ║
║    r_th_eff = r_th_virgin × (1 - η_forming)                          ║
║    >>> FormingFatigueIntegrator().effective_r_th(0.4, 'BCC')         ║
║                                                                      ║
║  DBT TEMPERATURE                                                     ║
║    3 views: Grain size / Temperature / Segregation (time)            ║
║    >>> DBTUnified().predict_dbtt(material, grain_size)               ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Core Principle:                                                     ║
║    Λ = K / |V|_eff    (Λ > 1 → yield/fracture)                      ║
║    m = Z × f_d        (topology = geometry × electron directionality)║
║                                                                      ║
║  Authors: Masamichi Iizumi & tamaki                                ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
