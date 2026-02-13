"""
δ-Theory Core Library
=====================
Unified materials science prediction based on geometric first principles.

"Nature is Geometry" - All material properties emerge from crystal structure.

Modules:
    - material: Unified material database (v10.0 — SSOC parameters)
    - ssoc: Structure-Selective Orbital Coupling calculation layer (v10.0)
    - unified_yield_fatigue_v10: Main yield + fatigue model (v10.0 SSOC)
    - am_fatigue: AM alloy fatigue life prediction (structure presets)
    - unified_flc_v8_1: FLC prediction with v6.9 integration (v8.1)
    - dbt_unified: Ductile-Brittle Transition Temperature prediction
    - fatigue_redis_api: FatigueData-AM2022 Redis API (optional)
    - banners: ASCII Art banners (random selection)

Version History:
    v5.0   - Yield stress from δ-theory (f_d, E_bond, crystal geometry)
    v6.9b  - Unified yield + fatigue with multiaxial (τ/σ, R)
    v6.10  - Universal fatigue validation (2472 points, 5 AM materials)
    v7.0   - Geometric factorization: f_d → (b/d)² × f_d_elec
             Material database centralized in material.py
    v7.2   - FLC from free volume consumption
    v8.0   - Forming-Fatigue integration (η → r_th_eff)
    v8.1   - FLC 7-mode discrete formulation + v6.9 integration
           - "Nature is Geometry" - ε₁ = |V|_eff × C_j / R_j
    v8.2   - material.py as single source of truth for all modules
    v10.0  - SSOC (Structure-Selective Orbital Coupling)
           - δ_L-free formulation: √(E_coh·k_B·T_m)
           - 3-layer architecture: material / ssoc / application
           - f_de: FCC-PCC, BCC-SCC, HCP-PCC
           - Unified M_SSOC=3.0 (structure差をf_deが吸収)
           - 25 metals validated, MAE=3.2%
    v10.1  - AM alloy fatigue module (am_fatigue)
           - Structure presets (BCC/FCC/HCP) validated on 3040 points
           - RMSE=1.113 (logN) across 30 AM alloys
           - Temperature S-N via σ_y(T) passthrough

Example:
    >>> from delta_theory import calc_sigma_y, MATERIALS
    >>> sigma_y = calc_sigma_y(MATERIALS['Fe'])
    
    >>> from delta_theory import calc_f_de, sigma_base_v10
    >>> f_de = calc_f_de(MATERIALS['Fe'])
    >>> sigma = sigma_base_v10(MATERIALS['Fe'])
    
    >>> from delta_theory import am_fatigue_life, am_sn_curve
    >>> result = am_fatigue_life(200, 900, 1050, 'HCP')
    >>> sigma_a, N_f = am_sn_curve(900, 1050, 'HCP')
    
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
# Material Database (v10.0 — SSOC parameters added)
# ==============================================================================
from .material import (
    # Core
    Material,
    MATERIALS,
    BD_RATIO_SQ,
    COEFF_V10,
    
    # Lookup
    get_material,
    list_materials,
    list_by_structure,
    
    # Structure presets
    StructurePreset,
    STRUCTURE_PRESETS,
)

# ==============================================================================
# SSOC: Structure-Selective Orbital Coupling (v10.0 — NEW)
# ==============================================================================
from .ssoc import (
    # Core calculation
    calc_f_de,
    calc_f_de_detail,
    sigma_base_v10,
    sigma_base_v10_with_fde,
    inverse_f_de,
    
    # FCC PCC
    fcc_f_de,
    fcc_gate,
    fcc_f_mu,
    fcc_f_shell,
    fcc_f_core,
    
    # BCC SCC
    bcc_f_de,
    bcc_f_jt,
    bcc_f_5d,
    bcc_f_lattice,
    
    # HCP PCC
    hcp_f_de,
    hcp_f_aniso,
    hcp_f_ca,
    hcp_f_5d as hcp_f_5d_corr,
    hcp_f_elec,
    
    # Constants
    P_DIM,
    M_SSOC,
    FCC_MU_REF,
    FCC_GAMMA_REF,
)

# ==============================================================================
# Core: Yield + Fatigue (v10.0 SSOC Edition)
# ==============================================================================
from .unified_yield_fatigue_v10 import (
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
    C_CLASS_DEFAULT,
)

# ==============================================================================
# AM Alloy Fatigue (v10.1 — Structure Preset S-N)
# ==============================================================================
from .am_fatigue import (
    # Core prediction
    am_fatigue_life,
    am_sn_curve,
    am_sn_curve_temperature,
    
    # Data
    AMFatiguePreset,
    AM_PRESETS,
    AMAlloy,
    ALLOY_DB,
    ALLOY_STRUCTURE,
    
    # Utility
    get_structure,
    get_alloy,
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
# lindemann.py exports (lazy import)
# ==============================================================================
# Imported lazily via __getattr__ to prevent RuntimeWarning
# when running `python -m delta_theory.lindemann`
# See __getattr__ at bottom of file.

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
__version__ = "10.1.0"
__author__ = "Masamichi Iizumi & Tamaki"

__all__ = [
    # === Banners ===
    "show_banner",
    "get_random_banner",
    "BANNERS",
    
    # === Material Database (v10.0) ===
    "Material",
    "MATERIALS",
    "BD_RATIO_SQ",
    "COEFF_V10",
    "get_material",
    "list_materials",
    "list_by_structure",
    "StructurePreset",
    "STRUCTURE_PRESETS",
    
    # === SSOC (v10.0 — NEW) ===
    "calc_f_de",
    "calc_f_de_detail",
    "sigma_base_v10",
    "sigma_base_v10_with_fde",
    "inverse_f_de",
    "fcc_f_de",
    "fcc_gate",
    "fcc_f_mu",
    "fcc_f_shell",
    "fcc_f_core",
    "bcc_f_de",
    "bcc_f_jt",
    "bcc_f_5d",
    "bcc_f_lattice",
    "hcp_f_de",
    "hcp_f_aniso",
    "hcp_f_ca",
    "hcp_f_5d_corr",
    "hcp_f_elec",
    "P_DIM",
    "M_SSOC",
    "FCC_MU_REF",
    "FCC_GAMMA_REF",
    
    # === v10.0 Yield + Fatigue ===
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
    "C_CLASS_DEFAULT",
    
    # === AM Alloy Fatigue (v10.1) ===
    "am_fatigue_life",
    "am_sn_curve",
    "am_sn_curve_temperature",
    "AMFatiguePreset",
    "AM_PRESETS",
    "AMAlloy",
    "ALLOY_DB",
    "ALLOY_STRUCTURE",
    "get_structure",
    "get_alloy",
    
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

    # === Lindemann ===
    "iizumi_lindemann",
    "conventional_lindemann", 
    "get_c_geo",
    "rms_displacement",
    "predict_delta_L",
    "validate_all",
    "print_validation_report",
    "print_latex_table",
    "C_IIZUMI",
    "XI_STRUCT",
    "Z_REF",
    "C_CONVENTIONAL",
    "MetalData",
    "VALIDATION_DATA",
    
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
║  δ-Theory Core Library v{__version__}                                    ║
║  "Nature is Geometry"                                                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  YIELD STRESS (v10.0 — SSOC δ_L-free)                               ║
║    σ_y = (8√5/5πMZ) × α₀ × (b/d)² × f_de × √(E·k_BT_m) / V × HP  ║
║                                                                      ║
║    Pure geometry:  α₀, (b/d)²=3/2, V_act=b³, HP, 8√5/5π            ║
║    Experimental:   E_coh (cohesive), T_m (melting point)             ║
║    SSOC:           f_de (structure-selective orbital coupling)        ║
║                                                                      ║
║    3-Layer Architecture:                                             ║
║      material.py  → Data layer  (SSOC parameters)                   ║
║      ssoc.py      → Calc layer  (f_de + σ_base)                     ║
║      unified_*    → App layer   (σ_y → S-N → FLC)                   ║
║                                                                      ║
║    SSOC Channels:                                                    ║
║      FCC — PCC: f_de = (μ/μ_ref)^(2/3·g_d) × f_shell × f_core      ║
║      BCC — SCC: f_de = f_JT × f_5d × f_lat  (d⁴ anomaly)           ║
║      HCP — PCC: f_de = f_elec × f_aniso(R) × f_ca × f_5d           ║
║                                                                      ║
║    Mean error: 3.2% across 25 metals (ZERO fitting parameters)       ║
║    >>> calc_sigma_y(MATERIALS['Fe'])                                 ║
║    >>> calc_f_de(MATERIALS['Fe'])                                    ║
║    >>> sigma_base_v10(MATERIALS['Fe'])                               ║
║                                                                      ║
║  FATIGUE LIFE                                                        ║
║    Pure metal (v10.0): per-material A_int, academic precision        ║
║    AM alloy  (v10.1): structure presets, practical prediction        ║
║      N = min( C×r^(-m) + 0.5/(A×r^n),  D×(1-r/r_u)^p )             ║
║      3040 points, 30 alloys, RMSE=1.113 (logN)                      ║
║      BCC: 0.699 | FCC: 1.050 | HCP: 1.376                          ║
║      Temperature S-N via σ_y(T) passthrough                         ║
║    >>> am_fatigue_life(200, 900, 1050, 'HCP')                       ║
║    >>> am_sn_curve(900, 1050, 'HCP')                                ║
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
║  Core Principle:                                                     ║
║    Λ = K / |V|_eff    (Λ > 1 → yield/fracture)                      ║
║    "The same geometry governs materials and particles"               ║
║    Materials: (b/d)² = 3/2    Particles: cos30° = √3/2              ║
║                                                                      ║
║  Authors: Masamichi Iizumi & Tamaki                                  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


# ==============================================================================
# Lazy Import for lindemann (avoids -m execution warning)
# ==============================================================================
_LINDEMANN_EXPORTS = {
    "iizumi_lindemann",
    "conventional_lindemann",
    "get_c_geo",
    "rms_displacement",
    "predict_delta_L",
    "validate_all",
    "print_validation_report",
    "print_latex_table",
    "C_IIZUMI",
    "XI_STRUCT",
    "Z_REF",
    "C_CONVENTIONAL",
    "MetalData",
    "VALIDATION_DATA",
}

def __getattr__(name):
    """Lazy import for lindemann module to avoid -m execution warning."""
    if name in _LINDEMANN_EXPORTS:
        from . import lindemann
        return getattr(lindemann, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
