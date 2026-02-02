#!/usr/bin/env python3
"""
δ-Theory Unit Tests
===================
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "..")


class TestUnifiedYieldFatigue:
    """unified_yield_fatigue_v6_9.py のテスト"""

    def test_import(self):
        """インポートテスト"""
        from core.unified_yield_fatigue_v6_9 import (
            FATIGUE_CLASS_PRESET,
            MATERIALS,
        )

        assert "Fe" in MATERIALS
        assert "BCC" in FATIGUE_CLASS_PRESET

    def test_materials_database(self):
        """材料データベースの整合性"""
        from core.unified_yield_fatigue_v6_9 import MATERIALS

        for name, mat in MATERIALS.items():
            assert mat.name == name
            assert mat.structure in ["BCC", "FCC", "HCP"]
            assert mat.T_m > 0
            assert mat.Eb > 0
            assert 0 < mat.dL < 1

    def test_sigma_y_positive(self):
        """降伏応力は正"""
        from core.unified_yield_fatigue_v6_9 import MATERIALS, calc_sigma_y

        for name, mat in MATERIALS.items():
            y = calc_sigma_y(mat, T_K=300)
            assert y["sigma_y"] > 0, f"{name}: σ_y should be positive"
            assert y["sigma_base"] > 0, f"{name}: σ_base should be positive"

    def test_sigma_y_temperature_dependence(self):
        """降伏応力は温度上昇で低下"""
        from core.unified_yield_fatigue_v6_9 import MATERIALS, calc_sigma_y

        mat = MATERIALS["Fe"]
        y_300 = calc_sigma_y(mat, T_K=300)
        y_600 = calc_sigma_y(mat, T_K=600)
        y_900 = calc_sigma_y(mat, T_K=900)

        assert y_300["sigma_y"] > y_600["sigma_y"] > y_900["sigma_y"]

    def test_solid_solution_strengthening(self):
        """固溶強化は正の寄与"""
        from core.unified_yield_fatigue_v6_9 import MATERIALS, calc_sigma_y

        mat = MATERIALS["Fe"]
        y_pure = calc_sigma_y(mat, T_K=300)
        y_ss = calc_sigma_y(mat, T_K=300, c_wt_percent=0.1, k_ss=400, solute_type="interstitial")

        assert y_ss["delta_ss"] > 0
        assert y_ss["sigma_y"] > y_pure["sigma_y"]

    def test_fatigue_preset_bcc(self):
        """BCC材料は明確な疲労限度"""
        from core.unified_yield_fatigue_v6_9 import FATIGUE_CLASS_PRESET

        bcc = FATIGUE_CLASS_PRESET["BCC"]
        assert bcc["r_th"] > 0.5, "BCC should have high r_th"

    def test_fatigue_preset_fcc(self):
        """FCC材料は疲労限度なし"""
        from core.unified_yield_fatigue_v6_9 import FATIGUE_CLASS_PRESET

        fcc = FATIGUE_CLASS_PRESET["FCC"]
        assert fcc["r_th"] < 0.1, "FCC should have low r_th"

    def test_fatigue_life_below_threshold(self):
        """r ≤ r_th で無限寿命"""
        from core.unified_yield_fatigue_v6_9 import MATERIALS, calc_sigma_y, fatigue_life_const_amp

        mat = MATERIALS["Fe"]  # BCC, r_th = 0.65
        y = calc_sigma_y(mat, T_K=300)

        # σ_a = 50 MPa で r < r_th となるはず
        result = fatigue_life_const_amp(
            mat,
            sigma_a_MPa=50,
            sigma_y_tension_MPa=y["sigma_y"],
            A_ext=2.46e-4,
        )

        assert result["r"] < result["r_th"]
        assert result["N_fail"] == float("inf")

    def test_fatigue_life_above_threshold(self):
        """r > r_th で有限寿命"""
        from core.unified_yield_fatigue_v6_9 import MATERIALS, calc_sigma_y, fatigue_life_const_amp

        mat = MATERIALS["Fe"]
        y = calc_sigma_y(mat, T_K=300)

        # 高応力で有限寿命
        result = fatigue_life_const_amp(
            mat,
            sigma_a_MPa=200,
            sigma_y_tension_MPa=y["sigma_y"],
            A_ext=2.46e-4,
        )

        if result["r"] > result["r_th"]:
            assert np.isfinite(result["N_fail"])
            assert result["N_fail"] > 0


class TestDBTUnified:
    """dbt_unified.py のテスト"""

    def test_import(self):
        """インポートテスト"""
        from core.dbt_unified import MATERIAL_FE

        assert MATERIAL_FE.name == "Fe"

    def test_core_sigma_y_positive(self):
        """σ_y は正"""
        from core.dbt_unified import DBTCore

        core = DBTCore()
        for d in [1e-6, 10e-6, 100e-6]:
            for T in [200, 300, 500]:
                sigma_y = core.sigma_y(d, T)
                assert sigma_y > 0

    def test_core_sigma_f_positive(self):
        """σ_f は正"""
        from core.dbt_unified import DBTCore

        core = DBTCore()
        for c in [0.001, 0.005, 0.01]:
            sigma_f = core.sigma_f(d=30e-6, c=c, T=300)
            assert sigma_f > 0

    def test_mclean_isotherm_range(self):
        """McLean等温線は0-1の範囲"""
        from core.dbt_unified import DBTCore

        core = DBTCore()
        for c in [0.001, 0.01, 0.05]:
            for T in [300, 500, 800]:
                theta = core.theta_mclean(c, T)
                assert 0 <= theta <= 1

    def test_dbtt_search(self):
        """DBTT探索が動作"""
        from core.dbt_unified import DBTUnified

        model = DBTUnified()
        result = model.temp_view.find_DBTT(d=30e-6, c=0.005)

        assert "T_star" in result
        assert "status" in result

    def test_grain_size_window(self):
        """延性窓解析が動作"""
        from core.dbt_unified import DBTUnified

        model = DBTUnified()
        result = model.grain_view.classify_mode(T=300, c=0.005)

        assert "mode" in result
        assert result["mode"] in ["brittle", "ductile", "transition", "window"]


class TestMaterials:
    """materials.py のテスト"""

    def test_import(self):
        """インポートテスト"""
        from core.materials import MaterialGPU

        assert hasattr(MaterialGPU, "Fe")

    def test_all_materials_exist(self):
        """全材料が存在"""
        from core.materials import MaterialGPU

        for name in MaterialGPU.list_materials():
            mat = MaterialGPU.get(name)
            assert mat.name == name

    def test_material_properties(self):
        """材料プロパティの整合性"""
        from core.materials import MaterialGPU

        fe = MaterialGPU.Fe()
        assert fe.structure == "BCC"
        assert fe.T_m > 0
        assert fe.E > 0
        assert 0 < fe.nu < 0.5

    def test_aliases(self):
        """エイリアスが動作"""
        from core.materials import MaterialGPU

        fe1 = MaterialGPU.get("Fe")
        fe2 = MaterialGPU.get("Iron")
        fe3 = MaterialGPU.get("SECD")

        assert fe1.name == fe2.name == fe3.name == "Fe"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
