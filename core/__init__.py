"""
δ-Theory Core Library
=====================
Unified yield stress and fatigue life prediction based on geometric first principles.

Modules:
    - unified_yield_fatigue_v6_9: Main yield + fatigue model (v6.9b)
    - dbt_unified: Ductile-Brittle Transition Temperature prediction
    - materials: Material database
    - fatigue_redis_api: FatigueData-AM2022 Redis API
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

from .fatigue_redis_api import FatigueDB  # ← 追加！

__version__ = "6.10.1"  # ← バージョンも上げる！
__author__ = "Masamichi Iizumi & Tamaki"

__all__ = [
    # v6.9
    "Material",
    "MATERIALS", 
    "calc_sigma_y",
    "fatigue_life_const_amp",
    "generate_sn_curve",
    "yield_by_mode",
    "FATIGUE_CLASS_PRESET",
    # DBT
    "DBTUnified",
    "DBTCore",
    "GrainSizeView",
    "TemperatureView",
    "SegregationView",
    "MATERIAL_FE",
    # Materials
    "MaterialGPU",
    # FatigueDB
    "FatigueDB",  # ← 追加！
]
