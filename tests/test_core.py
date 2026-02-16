#!/usr/bin/env python3
"""
δ-Theory CI Tests (v10.2.0) — test_core.py
============================================
Purpose: CI動作テスト（import・型・範囲・関数動作の確認）
Note:    物理検証・精度検証は test_validation.py に分離済み

Author: 飯泉真道 & 環
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "..")


# =============================================================================
# Material Database Tests
# =============================================================================

class TestMaterial:
    """material.py 動作テスト"""

    def test_import(self):
        from delta_theory.material import (
            BD_RATIO_SQ,
            COEFF_V10,
            Material,
            MATERIALS,
            eV_to_J,
            get_material,
            k_B,
            list_materials,
        )

        assert BD_RATIO_SQ == 1.5
        assert "Fe" in MATERIALS
        assert len(list_materials()) == 37
        assert abs(COEFF_V10 - 8 * np.sqrt(5) / (5 * np.pi)) < 1e-15

    def test_all_37_materials_exist(self):
        from delta_theory.material import MATERIALS, list_materials

        expected = {
            "Fe", "W", "V", "Cr", "Nb", "Mo", "Ta",
            "Li", "Na", "Mn", "Sn",
            "Cu", "Al", "Ni", "Au", "Ag", "Pt", "Pd", "Ir", "Rh", "Pb",
            "Ce", "In",
            "Ti", "Mg", "Zn", "Zr", "Hf", "Re", "Cd", "Ru",
            "Co", "Be", "Sc", "Y", "Nd", "Bi",
        }
        assert set(list_materials()) == expected

    def test_material_core_fields(self):
        """全金属: 基本フィールドの型・範囲"""
        from delta_theory.material import get_material, list_materials

        for name in list_materials():
            mat = get_material(name)
            assert mat.name == name
            assert mat.structure in ["BCC", "FCC", "HCP"]
            assert mat.T_m > 0
            assert mat.E_bond_eV > 0
            assert 0 < mat.delta_L < 1
            assert mat.f_d_elec > 0
            assert mat.a > 0
            assert mat.E > 0
            assert 0 < mat.nu < 0.5

    def test_ssoc_common_fields(self):
        """全金属: SSOC共通パラメータの範囲"""
        from delta_theory.material import get_material, list_materials

        for name in list_materials():
            mat = get_material(name)
            assert 0 <= mat.n_d <= 10
            assert mat.period in (2, 3, 4, 5, 6)
            assert 0 <= mat.n_f <= 14
            assert mat.n_atoms_cell >= 1
            assert mat.group >= 0

    def test_ssoc_fcc_fields(self):
        from delta_theory.material import get_material, list_by_structure

        for name in list_by_structure("FCC"):
            mat = get_material(name)
            assert mat.mu_GPa > 0
            assert mat.gamma_isf > 0

    def test_ssoc_bcc_fields(self):
        from delta_theory.material import get_material, list_by_structure

        for name in list_by_structure("BCC"):
            mat = get_material(name)
            assert mat.sel in {0, 1, 2}

    def test_ssoc_hcp_fields(self):
        from delta_theory.material import get_material, list_by_structure

        for name in list_by_structure("HCP"):
            mat = get_material(name)
            assert mat.R_crss > 0
            assert 1.2 < mat.c_a < 1.9

    def test_computed_properties(self):
        """計算プロパティが正常"""
        from delta_theory.material import get_material, list_materials

        for name in list_materials():
            mat = get_material(name)
            G_calc = mat.E / (2 * (1 + mat.nu))
            assert abs(mat.G - G_calc) < 1e-3
            assert mat.b > 0
            assert abs(mat.V_act - mat.b**3) < 1e-40
            assert mat.E_eff > 0
            assert mat.sqrt_EkT > 0

    def test_list_by_structure_counts(self):
        from delta_theory.material import list_by_structure

        assert len(list_by_structure("BCC")) == 11
        assert len(list_by_structure("FCC")) == 12
        assert len(list_by_structure("HCP")) == 14

    def test_aliases(self):
        from delta_theory.material import MATERIALS

        assert MATERIALS["Fe"] is MATERIALS["Iron"]
        assert MATERIALS["Cu"] is MATERIALS["Copper"]
        assert MATERIALS["W"] is MATERIALS["Tungsten"]

    def test_get_material_error(self):
        from delta_theory.material import get_material

        with pytest.raises(ValueError):
            get_material("Unobtanium")

    def test_frozen_immutable(self):
        from delta_theory.material import get_material

        fe = get_material("Fe")
        with pytest.raises((AttributeError, TypeError)):
            fe.name = "NotFe"  # type: ignore

    def test_summary_returns_string(self):
        from delta_theory.material import get_material

        for name in ["Fe", "Cu", "Ti", "Ce"]:
            s = get_material(name).summary()
            assert isinstance(s, str)
            assert name in s


# =============================================================================
# SSOC Tests (動作確認のみ)
# =============================================================================

class TestSSOC:
    """ssoc.py 動作テスト"""

    def test_import(self):
        from delta_theory.ssoc import (
            M_SSOC,
            P_DIM,
            calc_f_de,
            sigma_base_v10,
        )

        assert abs(P_DIM - 2.0 / 3.0) < 1e-15
        assert M_SSOC == 3.0

    # --- ゲート動作 (振る舞いベース) ---

    def test_fcc_gate_sp_zero(self):
        """sp-metals (n_d < 2): g_d = 0"""
        from delta_theory.ssoc import fcc_gate

        assert fcc_gate(0) == 0.0
        assert fcc_gate(1) == 0.0

    def test_fcc_gate_d_metals_positive(self):
        """d-metals (n_d >= 2, non-p-block): g_d > 0"""
        from delta_theory.ssoc import fcc_gate

        for n_d in [2, 4, 7, 8, 10]:
            assert fcc_gate(n_d) > 0.0

    def test_fcc_gate_p_block_d10_zero(self):
        """p-block d10 (In): g_d = 0"""
        from delta_theory.ssoc import fcc_gate

        assert fcc_gate(10, group=13) == 0.0

    def test_fcc_gate_lanthanide_zero(self):
        """ランタノイド: g_d = 0 (別チャンネル)"""
        from delta_theory.ssoc import fcc_gate

        assert fcc_gate(1, n_f=1) == 0.0

    def test_bcc_jt_anomaly(self):
        """d4 JT: f > 1, 他は 1.0"""
        from delta_theory.ssoc import bcc_f_jt

        assert bcc_f_jt(4) > 1.0
        assert bcc_f_jt(6) == 1.0

    def test_bcc_sp_less_than_one(self):
        """sp-metals: f < 1.0"""
        from delta_theory.ssoc import bcc_f_sp

        assert bcc_f_sp(0) < 1.0
        assert bcc_f_sp(10, group=14) < 1.0
        assert bcc_f_sp(6) == 1.0

    def test_bcc_complex_gate(self):
        """通常BCC=1.0, 多原子セル>1.0"""
        from delta_theory.ssoc import bcc_f_complex

        assert bcc_f_complex(2) == 1.0
        assert bcc_f_complex(58) > 1.0

    def test_hcp_aniso_monotonic(self):
        """R増加でf_aniso増加"""
        from delta_theory.ssoc import hcp_f_aniso

        assert hcp_f_aniso(10.0) > hcp_f_aniso(1.0)

    def test_hcp_sp_cov_be(self):
        """Be (period<=2, n_d=0): f > 1.0"""
        from delta_theory.ssoc import hcp_f_sp_cov

        assert hcp_f_sp_cov(0, 2) > 1.0
        assert hcp_f_sp_cov(0, 3) == 1.0

    def test_fcc_f_lanthanide(self):
        """非ランタノイド=1.0, ランタノイド>1.0"""
        from delta_theory.ssoc import fcc_f_lanthanide

        assert fcc_f_lanthanide(0, 10) == 1.0
        assert fcc_f_lanthanide(1, 1) > 1.0

    # --- calc_f_de 統一ディスパッチ ---

    def test_calc_f_de_all_positive(self):
        """全37金属: f_de > 0"""
        from delta_theory.material import get_material, list_materials
        from delta_theory.ssoc import calc_f_de

        for name in list_materials():
            f = calc_f_de(get_material(name))
            assert f > 0, f"{name}: f_de should be positive"

    def test_calc_f_de_dispatch(self):
        """構造別関数と一致"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import bcc_f_de, calc_f_de, fcc_f_de, hcp_f_de

        assert abs(calc_f_de(get_material("Fe")) - bcc_f_de(get_material("Fe"))) < 1e-15
        assert abs(calc_f_de(get_material("Cu")) - fcc_f_de(get_material("Cu"))) < 1e-15
        assert abs(calc_f_de(get_material("Ti")) - hcp_f_de(get_material("Ti"))) < 1e-15

    def test_calc_f_de_detail_has_f_de(self):
        """detail辞書にf_deキーが存在"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import calc_f_de, calc_f_de_detail

        for name in ["Fe", "Cu", "Ti", "Li", "Ce", "Nd"]:
            detail = calc_f_de_detail(get_material(name))
            assert "f_de" in detail

    # --- sigma_base_v10 ---

    def test_sigma_positive_all(self):
        """全37金属: sigma > 0 at 300K"""
        from delta_theory.material import get_material, list_materials
        from delta_theory.ssoc import sigma_base_v10

        for name in list_materials():
            assert sigma_base_v10(get_material(name), T_K=300.0) > 0

    def test_sigma_zero_at_melting(self):
        """全37金属: sigma = 0 at T_m"""
        from delta_theory.material import get_material, list_materials
        from delta_theory.ssoc import sigma_base_v10

        for name in list_materials():
            mat = get_material(name)
            assert sigma_base_v10(mat, T_K=mat.T_m) == 0.0

    def test_sigma_with_external_fde(self):
        """外部f_deと内部f_deで一致"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import calc_f_de, sigma_base_v10, sigma_base_v10_with_fde

        mat = get_material("Cu")
        fde = calc_f_de(mat)
        assert abs(sigma_base_v10_with_fde(mat, fde) - sigma_base_v10(mat)) < 1e-10

    def test_inverse_f_de_roundtrip(self):
        """順逆一致"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import calc_f_de, inverse_f_de, sigma_base_v10

        for name in ["Fe", "Cu", "Ti", "W"]:
            mat = get_material(name)
            fde_fwd = calc_f_de(mat)
            sigma = sigma_base_v10(mat, 300.0)
            fde_inv = inverse_f_de(mat, sigma, 300.0)
            assert abs(fde_fwd - fde_inv) < 1e-6


# =============================================================================
# Unified Yield + Fatigue Tests
# =============================================================================

class TestUnifiedYieldFatigue:
    """unified_yield_fatigue 動作テスト"""

    def test_import(self):
        from delta_theory.unified_yield_fatigue_v6_9 import (
            FATIGUE_CLASS_PRESET,
            MATERIALS,
        )

        assert "Fe" in MATERIALS
        assert "BCC" in FATIGUE_CLASS_PRESET

    def test_sigma_base_delegates_to_ssoc(self):
        from delta_theory.material import get_material, list_materials
        from delta_theory.ssoc import sigma_base_v10
        from delta_theory.unified_yield_fatigue_v6_9 import sigma_base_delta

        for name in list_materials():
            mat = get_material(name)
            assert abs(sigma_base_delta(mat) - sigma_base_v10(mat)) < 1e-10

    def test_sigma_y_positive(self):
        from delta_theory.material import get_material, list_materials
        from delta_theory.unified_yield_fatigue_v6_9 import calc_sigma_y

        for name in list_materials():
            y = calc_sigma_y(get_material(name), T_K=300)
            assert y["sigma_y"] > 0
            assert y["sigma_base"] > 0

    def test_sigma_y_returns_expected_keys(self):
        from delta_theory.material import get_material
        from delta_theory.unified_yield_fatigue_v6_9 import calc_sigma_y

        y = calc_sigma_y(get_material("Fe"), T_K=300)
        assert "f_de" in y
        assert "sigma_base_branch" in y

    def test_sigma_y_temperature_monotonic(self):
        """温度上昇で sigma_y 低下"""
        from delta_theory.material import get_material
        from delta_theory.unified_yield_fatigue_v6_9 import calc_sigma_y

        mat = get_material("Fe")
        y300 = calc_sigma_y(mat, T_K=300)["sigma_y"]
        y600 = calc_sigma_y(mat, T_K=600)["sigma_y"]
        y900 = calc_sigma_y(mat, T_K=900)["sigma_y"]
        assert y300 > y600 > y900

    def test_fatigue_below_threshold_infinite(self):
        from delta_theory.material import get_material
        from delta_theory.unified_yield_fatigue_v6_9 import (
            calc_sigma_y,
            fatigue_life_const_amp,
        )

        mat = get_material("Fe")
        y = calc_sigma_y(mat, T_K=300)
        result = fatigue_life_const_amp(
            mat, sigma_a_MPa=50,
            sigma_y_tension_MPa=y["sigma_y"], A_ext=2.46e-4,
        )
        assert result["r"] < result["r_th"]
        assert result["N_fail"] == float("inf")

    def test_tau_over_sigma_positive(self):
        from delta_theory.material import get_material, list_materials
        from delta_theory.unified_yield_fatigue_v6_9 import tau_over_sigma

        for name in list_materials():
            assert tau_over_sigma(get_material(name)) > 0

    def test_yield_by_mode_runs(self):
        from delta_theory.material import get_material
        from delta_theory.unified_yield_fatigue_v6_9 import calc_sigma_y, yield_by_mode

        mat = get_material("Fe")
        sy = calc_sigma_y(mat, T_K=300)["sigma_y"]
        for mode in ["tensile", "shear", "compression"]:
            val, info = yield_by_mode(mat, sy, mode=mode)
            assert val > 0


# =============================================================================
# DBT Tests
# =============================================================================

class TestDBTUnified:

    def test_import(self):
        from delta_theory.dbt_unified import MATERIAL_FE
        assert MATERIAL_FE.name == "Fe"

    def test_sigma_y_positive(self):
        from delta_theory.dbt_unified import DBTCore
        assert DBTCore().sigma_y(30e-6, 300) > 0

    def test_sigma_f_positive(self):
        from delta_theory.dbt_unified import DBTCore
        assert DBTCore().sigma_f(d=30e-6, c=0.005, T=300) > 0

    def test_mclean_range(self):
        from delta_theory.dbt_unified import DBTCore
        theta = DBTCore().theta_mclean(0.01, 500)
        assert 0 <= theta <= 1

    def test_dbtt_search_runs(self):
        from delta_theory.dbt_unified import DBTUnified
        result = DBTUnified().temp_view.find_DBTT(d=30e-6, c=0.005)
        assert "T_star" in result

    def test_grain_classify_runs(self):
        from delta_theory.dbt_unified import DBTUnified
        result = DBTUnified().grain_view.classify_mode(T=300, c=0.005)
        assert "mode" in result


# =============================================================================
# Package Init Tests
# =============================================================================

class TestPackageInit:

    def test_top_level_import(self):
        from delta_theory import (
            BD_RATIO_SQ, COEFF_V10, MATERIALS, Material,
            calc_f_de, calc_sigma_y, get_material,
            sigma_base_delta, sigma_base_v10,
        )
        assert BD_RATIO_SQ == 1.5
        assert "Fe" in MATERIALS

    def test_ssoc_top_level_import(self):
        from delta_theory import (
            M_SSOC, P_DIM, calc_f_de, calc_f_de_detail,
            fcc_f_de, bcc_f_de, hcp_f_de,
            inverse_f_de, sigma_base_v10, sigma_base_v10_with_fde,
        )
        assert abs(P_DIM - 2 / 3) < 1e-15

    def test_version(self):
        import delta_theory
        assert delta_theory.__version__ == "10.3.1"

    def test_materials_identity(self):
        from delta_theory import MATERIALS
        from delta_theory.material import MATERIALS as MAT2
        assert MATERIALS is MAT2

    def test_info_runs(self):
        from delta_theory import info
        info()

    def test_lindemann_lazy_import(self):
        from delta_theory import iizumi_lindemann, C_IIZUMI
        assert callable(iizumi_lindemann)
        assert C_IIZUMI > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
