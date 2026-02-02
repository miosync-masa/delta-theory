"""
δ-Theory Core Library
=====================

Unified yield stress and fatigue life prediction based on geometric first principles.

Modules:
    - unified_yield_fatigue_v6_9: Main yield + fatigue model (v6.9b)
    - dbt_unified: Ductile-Brittle Transition Temperature prediction
    - materials: Material database
"""

from .unified_yield_fatigue_v6_9 import (
    Material,
    MATERIALS,
    calc_sigma_y,
    fatigue_life_const_amp,
    generate_sn_curve,
    yield_by_mode,
    FATIGUE_CLASS_PRESET,
)

from .dbt_unified import (
    DBTUnified,
    DBTCore,
    GrainSizeView,
    TemperatureView,
    SegregationView,
    MATERIAL_FE,
)

from .materials import MaterialGPU

__version__ = "6.9.0"
__author__ = "Masamichi Iizumi & Tamaki"
