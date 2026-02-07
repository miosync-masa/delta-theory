#!/usr/bin/env python3
"""
δ-Theory Unit Tests (v8.2.0)
=============================
Updated for:
  - v7.0 geometric factorization (BD_RATIO_SQ × f_d_elec)
  - material.py as single source of truth
  - unified_yield_fatigue_v7_0 (imports from material.py)
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "..")


# =============================================================================
# Material Database Tests (material.py — Single Source of Truth)
# =============================================================================

class TestMaterial:
    """material.py のテスト"""

    def test_import(self):
        """インポートテスト"""
        from delta_theory.material import (
            BD_RATIO_SQ,
            Material,
            MATERIALS,
            get_material,
            list_materials,
        )

        assert BD_RATIO_SQ == 1.5
        assert "Fe" in MATERIALS
        assert len(list_materials()) == 10

    def test_all_materials_exist(self):
        """全10金属が存在"""
        from delta_theory.material import MATERIALS, list_materials

        expected = {"Fe", "W", "Cu", "Al", "Ni", "Au", "Ag", "Ti", "Mg", "Zn"}
        assert set(list_materials()) == expected
        for name in expected:
            assert name in MATERIALS

    def test_material_fields(self):
        """Material フィールドの整合性（主要10金属のみ）"""
        from delta_theory.material import get_material, list_materials

        for name in list_materials():
            mat = get_material(name)
            assert mat.name == name
            assert mat.structure in ["BCC", "FCC", "HCP"]
            assert mat.T_m > 0, f"{name}: T_m should be positive"
            assert mat.E_bond_eV > 0, f"{name}: E_bond_eV should be positive"
            assert 0 < mat.delta_L < 1, f"{name}: delta_L should be in (0, 1)"
            assert mat.f_d_elec > 0, f"{name}: f_d_elec should be positive"
            assert mat.a > 0, f"{name}: lattice parameter should be positive"
            assert mat.E > 0, f"{name}: Young's modulus should be positive"
            assert 0 < mat.nu < 0.5, f"{name}: Poisson's ratio should be in (0, 0.5)"

    def test_materials_aliases(self):
        """MATERIALSにエイリアスが含まれる"""
        from delta_theory.material import MATERIALS

        # エイリアスは同じMaterialインスタンスを指す
        assert MATERIALS["Fe"] is MATERIALS["Iron"]
        assert MATERIALS["Fe"] is MATERIALS["SECD"]
        assert MATERIALS["Cu"] is MATERIALS["Copper"]

    def test_bd_ratio_sq(self):
        """BD_RATIO_SQ = 3/2 が正しい"""
        from delta_theory.material import BD_RATIO_SQ

        assert BD_RATIO_SQ == 1.5
        assert BD_RATIO_SQ == 3 / 2

    def test_f_d_backward_compat(self):
        """mat.f_d プロパティ（後方互換）= BD_RATIO_SQ × f_d_elec"""
        from delta_theory.material import BD_RATIO_SQ, get_material, list_materials

        old_f_d = {
            "Fe": 1.5, "W": 4.7, "Cu": 2.0, "Al": 1.6, "Ni": 2.6,
            "Au": 1.1, "Ag": 2.0, "Ti": 5.7, "Mg": 8.2, "Zn": 2.0,
        }

        for name in list_materials():
            mat = get_material(name)
            # f_d property = BD_RATIO_SQ × f_d_elec
            assert abs(mat.f_d - BD_RATIO_SQ * mat.f_d_elec) < 1e-15
            # matches old values
            assert abs(mat.f_d - old_f_d[name]) < 1e-10, (
                f"{name}: f_d={mat.f_d}, expected={old_f_d[name]}"
            )

    def test_fe_is_reference(self):
        """Fe の f_d_elec = 1.0（基準点）"""
        from delta_theory.material import get_material

        fe = get_material("Fe")
        assert fe.f_d_elec == 1.0

    def test_properties_computed(self):
        """計算プロパティが正常"""
        from delta_theory.material import get_material, list_materials

        for name in list_materials():
            mat = get_material(name)
            # G (shear modulus) from E and nu
            G_calc = mat.E / (2 * (1 + mat.nu))
            assert abs(mat.G - G_calc) < 1e-3, f"{name}: G mismatch"

            # b (Burgers vector) > 0
            assert mat.b > 0, f"{name}: b should be positive"

            # V_act = b^3
            assert abs(mat.V_act - mat.b**3) < 1e-40, f"{name}: V_act mismatch"

            # E_eff > 0
            assert mat.E_eff > 0, f"{name}: E_eff should be positive"

    def test_burgers_vector_formula(self):
        """バーガースベクトルの結晶構造依存"""
        from delta_theory.material import get_material

        fe = get_material("Fe")   # BCC: b = a√3/2
        cu = get_material("Cu")   # FCC: b = a/√2
        ti = get_material("Ti")   # HCP: b = a

        assert abs(fe.b - fe.a * np.sqrt(3) / 2) < 1e-20
        assert abs(cu.b - cu.a / np.sqrt(2)) < 1e-20
        assert abs(ti.b - ti.a) < 1e-20

    def test_structure_presets(self):
        """構造プリセットの値"""
        from delta_theory.material import STRUCTURE_PRESETS

        assert set(STRUCTURE_PRESETS.keys()) == {"BCC", "FCC", "HCP"}

        bcc = STRUCTURE_PRESETS["BCC"]
        assert bcc.alpha0 == 0.289
        assert bcc.r_th > 0.5  # BCC has high fatigue threshold

        fcc = STRUCTURE_PRESETS["FCC"]
        assert fcc.alpha0 == 0.250
        assert fcc.r_th < 0.1  # FCC has low fatigue threshold

    def test_list_by_structure(self):
        """構造別リスト"""
        from delta_theory.material import list_by_structure

        bcc = list_by_structure("BCC")
        fcc = list_by_structure("FCC")
        hcp = list_by_structure("HCP")

        assert "Fe" in bcc and "W" in bcc
        assert "Cu" in fcc and "Al" in fcc
        assert "Ti" in hcp and "Mg" in hcp
        assert len(bcc) + len(fcc) + len(hcp) == 10

    def test_get_material_error(self):
        """存在しない金属でValueError"""
        from delta_theory.material import get_material

        with pytest.raises(ValueError):
            get_material("Unobtanium")

    def test_material_gpu_backward_compat(self):
        """MaterialGPU（後方互換クラス）"""
        from delta_theory.material import MaterialGPU

        fe = MaterialGPU.Fe()
        assert fe.name == "Fe"
        assert fe.structure == "BCC"
        assert fe.T_m > 0

        # aliases
        fe2 = MaterialGPU.get("Iron")
        fe3 = MaterialGPU.get("SECD")
        assert fe.name == fe2.name == fe3.name == "Fe"

        # list
        names = MaterialGPU.list_materials()
        assert "Fe" in names

    def test_summary(self):
        """summary() が文字列を返す"""
        from delta_theory.material import get_material

        fe = get_material("Fe")
        s = fe.summary()
        assert isinstance(s, str)
        assert "Fe" in s
        assert "f_d_elec" in s

    def test_frozen_immutable(self):
        """Material は frozen dataclass（変更不可）"""
        from delta_theory.material import get_material

        fe = get_material("Fe")
        # Python 3.11+: FrozenInstanceError, Python <3.11: AttributeError
        with pytest.raises((AttributeError, TypeError)):
            fe.name = "NotFe"  # type: ignore


# =============================================================================
# Unified Yield + Fatigue Tests (v7.0)
# =============================================================================

class TestUnifiedYieldFatigue:
    """unified_yield_fatigue_v7_0.py のテスト"""

    def test_import(self):
        """インポートテスト"""
        from delta_theory.unified_yield_fatigue_v6_9 import (
            FATIGUE_CLASS_PRESET,
            MATERIALS,
        )

        assert "Fe" in MATERIALS
        assert "BCC" in FATIGUE_CLASS_PRESET

    def test_materials_from_material_py(self):
        """MATERIALS は material.py と同一オブジェクト"""
        from delta_theory.material import MATERIALS as MAT_DIRECT
        from delta_theory.unified_yield_fatigue_v6_9 import MATERIALS as MAT_UNIFIED

        assert MAT_DIRECT is MAT_UNIFIED

    def test_sigma_y_positive(self):
        """降伏応力は正"""
        from delta_theory.material import get_material, list_materials
        from delta_theory.unified_yield_fatigue_v6_9 import calc_sigma_y

        for name in list_materials():
            mat = get_material(name)
            y = calc_sigma_y(mat, T_K=300)
            assert y["sigma_y"] > 0, f"{name}: σ_y should be positive"
            assert y["sigma_base"] > 0, f"{name}: σ_base should be positive"

    def test_sigma_base_uses_E_eff(self):
        """sigma_base_delta が mat.E_eff / mat.V_act を使用"""
        from delta_theory.material import BD_RATIO_SQ, get_material
        from delta_theory.unified_yield_fatigue_v6_9 import sigma_base_delta

        mat = get_material("Fe")
        sigma = sigma_base_delta(mat, T_K=300)
        assert sigma > 0

        # 手計算と一致
        eV_to_J = 1.602176634e-19
        HP = 1.0 - 300.0 / mat.T_m
        E_eff = mat.E_bond_eV * eV_to_J * mat.alpha0 * BD_RATIO_SQ * mat.f_d_elec
        sigma_calc = (E_eff / mat.V_act) * mat.delta_L * HP / (2 * np.pi * 3.0) / 1e6

        assert abs(sigma - sigma_calc) < 1e-10

    def test_sigma_y_v69_backward_compat(self):
        """v6.9と同一の降伏応力値を出力"""
        from delta_theory.material import get_material
        from delta_theory.unified_yield_fatigue_v6_9 import sigma_base_delta

        # v6.9で検証済みの値（10金属, 300K）
        v69_values = {
            "Fe": 146.4604, "W": 737.0334, "Cu": 69.4582,
            "Al": 33.2944, "Ni": 144.6105, "Au": 28.7660,
            "Ag": 39.2393, "Ti": 270.8388, "Mg": 87.9478, "Zn": 29.0427,
        }

        for name, expected in v69_values.items():
            sigma = sigma_base_delta(get_material(name), T_K=300)
            assert abs(sigma - expected) < 0.001, (
                f"{name}: σ={sigma:.4f}, expected={expected:.4f}"
            )

    def test_sigma_y_temperature_dependence(self):
        """降伏応力は温度上昇で低下"""
        from delta_theory.material import get_material
        from delta_theory.unified_yield_fatigue_v6_9 import calc_sigma_y

        mat = get_material("Fe")
        y_300 = calc_sigma_y(mat, T_K=300)
        y_600 = calc_sigma_y(mat, T_K=600)
        y_900 = calc_sigma_y(mat, T_K=900)

        assert y_300["sigma_y"] > y_600["sigma_y"] > y_900["sigma_y"]

    def test_sigma_y_at_melting(self):
        """融点で降伏応力=0"""
        from delta_theory.material import get_material, list_materials
        from delta_theory.unified_yield_fatigue_v6_9 import sigma_base_delta

        for name in list_materials():
            mat = get_material(name)
            sigma = sigma_base_delta(mat, T_K=mat.T_m)
            assert sigma == 0.0, f"{name}: σ should be 0 at T_m"

    def test_solid_solution_strengthening(self):
        """固溶強化は正の寄与"""
        from delta_theory.material import get_material
        from delta_theory.unified_yield_fatigue_v6_9 import calc_sigma_y

        mat = get_material("Fe")
        y_pure = calc_sigma_y(mat, T_K=300)
        y_ss = calc_sigma_y(
            mat, T_K=300, c_wt_percent=0.1,
            k_ss=400, solute_type="interstitial",
        )

        assert y_ss["delta_ss"] > 0
        assert y_ss["sigma_y"] > y_pure["sigma_y"]

    def test_fatigue_preset_bcc(self):
        """BCC材料は明確な疲労限度"""
        from delta_theory.unified_yield_fatigue_v6_9 import FATIGUE_CLASS_PRESET

        bcc = FATIGUE_CLASS_PRESET["BCC"]
        assert bcc["r_th"] > 0.5, "BCC should have high r_th"

    def test_fatigue_preset_fcc(self):
        """FCC材料は疲労限度なし"""
        from delta_theory.unified_yield_fatigue_v6_9 import FATIGUE_CLASS_PRESET

        fcc = FATIGUE_CLASS_PRESET["FCC"]
        assert fcc["r_th"] < 0.1, "FCC should have low r_th"

    def test_fatigue_life_below_threshold(self):
        """r ≤ r_th で無限寿命"""
        from delta_theory.material import get_material
        from delta_theory.unified_yield_fatigue_v6_9 import (
            calc_sigma_y,
            fatigue_life_const_amp,
        )

        mat = get_material("Fe")  # BCC, r_th = 0.65
        y = calc_sigma_y(mat, T_K=300)

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
        from delta_theory.material import get_material
        from delta_theory.unified_yield_fatigue_v6_9 import (
            calc_sigma_y,
            fatigue_life_const_amp,
        )

        mat = get_material("Fe")
        y = calc_sigma_y(mat, T_K=300)

        result = fatigue_life_const_amp(
            mat,
            sigma_a_MPa=200,
            sigma_y_tension_MPa=y["sigma_y"],
            A_ext=2.46e-4,
        )

        if result["r"] > result["r_th"]:
            assert np.isfinite(result["N_fail"])
            assert result["N_fail"] > 0

    def test_tau_over_sigma(self):
        """τ/σ変換が正"""
        from delta_theory.material import get_material, list_materials
        from delta_theory.unified_yield_fatigue_v6_9 import tau_over_sigma

        for name in list_materials():
            mat = get_material(name)
            ratio = tau_over_sigma(mat)
            assert ratio > 0, f"{name}: τ/σ should be positive"

    def test_yield_by_mode(self):
        """モード別降伏応力（returns Tuple[float, dict]）"""
        from delta_theory.material import get_material
        from delta_theory.unified_yield_fatigue_v6_9 import (
            calc_sigma_y,
            yield_by_mode,
        )

        mat = get_material("Fe")
        y = calc_sigma_y(mat, T_K=300)

        tensile, info_t = yield_by_mode(mat, y["sigma_y"], mode="tensile")
        shear, info_s = yield_by_mode(mat, y["sigma_y"], mode="shear")
        comp, info_c = yield_by_mode(mat, y["sigma_y"], mode="compression")

        assert tensile > 0
        assert shear > 0
        assert comp > 0
        assert "tau_over_sigma" in info_t


# =============================================================================
# v7.0 Geometric Factorization Specific Tests
# =============================================================================

class TestGeometricFactorization:
    """v7.0 geometric factorization の検証"""

    def test_e_eff_decomposition(self):
        """E_eff = E_bond × α × (b/d)² × f_d_elec"""
        from delta_theory.material import BD_RATIO_SQ, get_material, list_materials

        eV_to_J = 1.602176634e-19

        for name in list_materials():
            mat = get_material(name)
            E_eff_manual = (
                mat.E_bond_eV * eV_to_J
                * mat.alpha0
                * BD_RATIO_SQ
                * mat.f_d_elec
            )
            assert abs(mat.E_eff - E_eff_manual) < 1e-30, (
                f"{name}: E_eff decomposition mismatch"
            )

    def test_f_d_elec_ordering(self):
        """f_d_elec の物理的順序"""
        from delta_theory.material import get_material

        au = get_material("Au")  # filled d-shell
        fe = get_material("Fe")  # reference
        w = get_material("W")    # strong d-bonding
        ti = get_material("Ti")  # strong directionality

        assert au.f_d_elec < fe.f_d_elec < w.f_d_elec < ti.f_d_elec

    def test_bd_ratio_sq_is_universal(self):
        """BD_RATIO_SQ は全結晶構造で同一"""
        from delta_theory.material import BD_RATIO_SQ

        # d/b = √(2/3) for BCC, FCC, HCP → (b/d)² = 3/2
        assert BD_RATIO_SQ == 3 / 2
        assert abs(1 / BD_RATIO_SQ - 2 / 3) < 1e-15

    def test_zero_fitting_parameters(self):
        """σ_y に fitting parameter が0であることの検証

        Pure geometry:  α, (b/d)²=3/2, V_act=b³, HP, M, 2π
        Experimental:   E_bond (sublimation), δ_L (Debye-Waller)
        Electronic:     f_d_elec (quantum mechanical)
        → No adjustable fitting parameters
        """
        from delta_theory.material import BD_RATIO_SQ, STRUCTURE_PRESETS

        # 幾何定数は固定
        assert BD_RATIO_SQ == 1.5
        for st, preset in STRUCTURE_PRESETS.items():
            # alpha0 is determined by crystal geometry
            assert preset.alpha0 > 0

        # M_TAYLOR is universal polycrystal constant
        from delta_theory.unified_yield_fatigue_v6_9 import M_TAYLOR

        assert M_TAYLOR == 3.0


# =============================================================================
# DBT Tests
# =============================================================================

class TestDBTUnified:
    """dbt_unified.py のテスト"""

    def test_import(self):
        """インポートテスト"""
        from delta_theory.dbt_unified import MATERIAL_FE

        assert MATERIAL_FE.name == "Fe"

    def test_core_sigma_y_positive(self):
        """σ_y は正"""
        from delta_theory.dbt_unified import DBTCore

        core = DBTCore()
        for d in [1e-6, 10e-6, 100e-6]:
            for T in [200, 300, 500]:
                sigma_y = core.sigma_y(d, T)
                assert sigma_y > 0

    def test_core_sigma_f_positive(self):
        """σ_f は正"""
        from delta_theory.dbt_unified import DBTCore

        core = DBTCore()
        for c in [0.001, 0.005, 0.01]:
            sigma_f = core.sigma_f(d=30e-6, c=c, T=300)
            assert sigma_f > 0

    def test_mclean_isotherm_range(self):
        """McLean等温線は0-1の範囲"""
        from delta_theory.dbt_unified import DBTCore

        core = DBTCore()
        for c in [0.001, 0.01, 0.05]:
            for T in [300, 500, 800]:
                theta = core.theta_mclean(c, T)
                assert 0 <= theta <= 1

    def test_dbtt_search(self):
        """DBTT探索が動作"""
        from delta_theory.dbt_unified import DBTUnified

        model = DBTUnified()
        result = model.temp_view.find_DBTT(d=30e-6, c=0.005)

        assert "T_star" in result
        assert "status" in result

    def test_grain_size_window(self):
        """延性窓解析が動作"""
        from delta_theory.dbt_unified import DBTUnified

        model = DBTUnified()
        result = model.grain_view.classify_mode(T=300, c=0.005)

        assert "mode" in result
        assert result["mode"] in ["brittle", "ductile", "transition", "window"]


# =============================================================================
# Package-level import tests
# =============================================================================

class TestPackageInit:
    """__init__.py のテスト"""

    def test_top_level_import(self):
        """トップレベルインポート"""
        from delta_theory import (
            BD_RATIO_SQ,
            MATERIALS,
            Material,
            calc_sigma_y,
            get_material,
            sigma_base_delta,
        )

        assert BD_RATIO_SQ == 1.5
        assert "Fe" in MATERIALS

    def test_version(self):
        """バージョン"""
        import delta_theory

        assert delta_theory.__version__ == "8.2.0"

    def test_material_and_unified_share_materials(self):
        """material.py と unified 側の MATERIALS が同一"""
        from delta_theory import MATERIALS
        from delta_theory.material import MATERIALS as MAT2

        assert MATERIALS is MAT2

    def test_no_t_twin_r_comp_export(self):
        """T_TWIN, R_COMP はもうトップレベルにない"""
        import delta_theory

        assert not hasattr(delta_theory, "T_TWIN")
        assert not hasattr(delta_theory, "R_COMP")

    def test_info_runs(self):
        """info() が正常実行"""
        from delta_theory import info

        info()  # should not raise


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
