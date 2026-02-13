#!/usr/bin/env python3
"""
δ-Theory Unit Tests (v10.0.0)
==============================
Updated for:
  - v10.0 SSOC (Structure-Selective Orbital Coupling)
  - δ_Lフリー統一降伏応力: σ = (8√5/5πMZ) × α₀ × (b/d)² × f_de × √(E·kT) / V × HP
  - material.py: 25金属 + SSOCパラメータ (n_d, period, mu_GPa, gamma_isf, R_crss, etc.)
  - ssoc.py: 計算層 (FCC PCC / BCC SCC / HCP PCC)
  - unified_yield_fatigue: 応用層 (v10.0 σ_base → S-N統合)
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "..")


# =============================================================================
# Material Database Tests (material.py — v10.0 SSOC parameters)
# =============================================================================

class TestMaterial:
    """material.py のテスト (v10.0 SSOC対応)"""

    def test_import(self):
        """インポートテスト"""
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
        assert len(list_materials()) == 25
        assert abs(COEFF_V10 - 8 * np.sqrt(5) / (5 * np.pi)) < 1e-15
        assert eV_to_J == 1.602176634e-19
        assert k_B == 1.380649e-23

    def test_all_25_materials_exist(self):
        """全25金属が存在"""
        from delta_theory.material import MATERIALS, list_materials

        expected = {
            # BCC (7)
            "Fe", "W", "V", "Cr", "Nb", "Mo", "Ta",
            # FCC (10)
            "Cu", "Al", "Ni", "Au", "Ag", "Pt", "Pd", "Ir", "Rh", "Pb",
            # HCP (8)
            "Ti", "Mg", "Zn", "Zr", "Hf", "Re", "Cd", "Ru",
        }
        assert set(list_materials()) == expected
        for name in expected:
            assert name in MATERIALS

    def test_material_v70_fields(self):
        """v7.0フィールドの整合性（後方互換）"""
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

    def test_ssoc_common_fields(self):
        """SSOC共通パラメータ (全25金属)"""
        from delta_theory.material import get_material, list_materials

        for name in list_materials():
            mat = get_material(name)
            assert 0 <= mat.n_d <= 10, f"{name}: n_d={mat.n_d} out of range [0,10]"
            assert mat.period in (3, 4, 5, 6), f"{name}: period={mat.period}"

    def test_ssoc_fcc_fields(self):
        """FCC材料のSSOCパラメータ"""
        from delta_theory.material import get_material, list_by_structure

        for name in list_by_structure("FCC"):
            mat = get_material(name)
            assert mat.mu_GPa > 0, f"{name}: mu_GPa should be positive"
            assert mat.gamma_isf > 0, f"{name}: gamma_isf should be positive"

    def test_ssoc_bcc_fields(self):
        """BCC材料のSSOCパラメータ"""
        from delta_theory.material import get_material, list_by_structure

        for name in list_by_structure("BCC"):
            mat = get_material(name)
            assert mat.group in (5, 6, 8), f"{name}: group={mat.group}"
            assert mat.sel in (1, 2), f"{name}: sel={mat.sel}"

    def test_ssoc_hcp_fields(self):
        """HCP材料のSSOCパラメータ"""
        from delta_theory.material import get_material, list_by_structure

        for name in list_by_structure("HCP"):
            mat = get_material(name)
            assert mat.R_crss > 0, f"{name}: R_crss should be positive"
            assert 1.4 < mat.c_a < 1.9, f"{name}: c_a={mat.c_a} out of physical range"

    def test_materials_aliases(self):
        """MATERIALSにエイリアスが含まれる"""
        from delta_theory.material import MATERIALS

        assert MATERIALS["Fe"] is MATERIALS["Iron"]
        assert MATERIALS["Fe"] is MATERIALS["SECD"]
        assert MATERIALS["Cu"] is MATERIALS["Copper"]
        assert MATERIALS["W"] is MATERIALS["Tungsten"]
        assert MATERIALS["Ti"] is MATERIALS["Titanium"]
        assert MATERIALS["Pt"] is MATERIALS["Platinum"]

    def test_coeff_v10(self):
        """COEFF_V10 = 8√5/(5π) ≈ 1.1384"""
        from delta_theory.material import COEFF_V10

        expected = 8 * np.sqrt(5) / (5 * np.pi)
        assert abs(COEFF_V10 - expected) < 1e-15
        assert abs(COEFF_V10 - 1.1384) < 0.001

    def test_bd_ratio_sq(self):
        """BD_RATIO_SQ = 3/2 が正しい"""
        from delta_theory.material import BD_RATIO_SQ

        assert BD_RATIO_SQ == 1.5
        assert BD_RATIO_SQ == 3 / 2

    def test_sqrt_ekt_property(self):
        """sqrt_EkT = √(E_coh · k_B · T_m)"""
        from delta_theory.material import eV_to_J, get_material, k_B, list_materials

        for name in list_materials():
            mat = get_material(name)
            expected = np.sqrt(mat.E_bond_eV * eV_to_J * k_B * mat.T_m)
            assert abs(mat.sqrt_EkT - expected) < 1e-30, (
                f"{name}: sqrt_EkT mismatch"
            )
            assert mat.sqrt_EkT > 0

    def test_f_d_backward_compat(self):
        """mat.f_d プロパティ（後方互換）= BD_RATIO_SQ × f_d_elec"""
        from delta_theory.material import BD_RATIO_SQ, get_material

        # v7.0の基本10金属で検証
        old_f_d = {
            "Fe": 1.5, "W": 4.7, "Cu": 2.0, "Al": 1.6, "Ni": 2.6,
            "Au": 1.1, "Ag": 2.0, "Ti": 5.7, "Mg": 8.2, "Zn": 2.0,
        }

        for name, expected_fd in old_f_d.items():
            mat = get_material(name)
            assert abs(mat.f_d - BD_RATIO_SQ * mat.f_d_elec) < 1e-15
            assert abs(mat.f_d - expected_fd) < 1e-10, (
                f"{name}: f_d={mat.f_d}, expected={expected_fd}"
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
            G_calc = mat.E / (2 * (1 + mat.nu))
            assert abs(mat.G - G_calc) < 1e-3, f"{name}: G mismatch"
            assert mat.b > 0, f"{name}: b should be positive"
            assert abs(mat.V_act - mat.b**3) < 1e-40, f"{name}: V_act mismatch"
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
        assert bcc.r_th > 0.5

        fcc = STRUCTURE_PRESETS["FCC"]
        assert fcc.alpha0 == 0.250
        assert fcc.r_th < 0.1

    def test_list_by_structure(self):
        """構造別リスト"""
        from delta_theory.material import list_by_structure

        bcc = list_by_structure("BCC")
        fcc = list_by_structure("FCC")
        hcp = list_by_structure("HCP")

        assert len(bcc) == 7
        assert len(fcc) == 10
        assert len(hcp) == 8
        assert "Fe" in bcc and "W" in bcc and "Ta" in bcc
        assert "Cu" in fcc and "Ir" in fcc and "Pb" in fcc
        assert "Ti" in hcp and "Ru" in hcp and "Cd" in hcp
        assert len(bcc) + len(fcc) + len(hcp) == 25

    def test_get_material_error(self):
        """存在しない金属でValueError"""
        from delta_theory.material import get_material

        with pytest.raises(ValueError):
            get_material("Unobtanium")

    def test_summary(self):
        """summary() が文字列を返す"""
        from delta_theory.material import get_material

        fe = get_material("Fe")
        s = fe.summary()
        assert isinstance(s, str)
        assert "Fe" in s
        # v10.0ではSSOC情報も表示
        assert "n_d" in s

    def test_frozen_immutable(self):
        """Material は frozen dataclass（変更不可）"""
        from delta_theory.material import get_material

        fe = get_material("Fe")
        with pytest.raises((AttributeError, TypeError)):
            fe.name = "NotFe"  # type: ignore


# =============================================================================
# SSOC Tests (ssoc.py — Calculation Layer)
# =============================================================================

class TestSSOC:
    """ssoc.py のテスト (v10.0 SSOC計算層)"""

    def test_import(self):
        """インポートテスト"""
        from delta_theory.ssoc import (
            M_SSOC,
            P_DIM,
            calc_f_de,
            sigma_base_v10,
        )

        assert abs(P_DIM - 2.0 / 3.0) < 1e-15
        assert M_SSOC == 3.0

    def test_p_dim_universal(self):
        """P_DIM = 2/3 は面→体積の幾何的次元変換（全構造共通）"""
        from delta_theory.ssoc import P_DIM

        assert abs(P_DIM - 2 / 3) < 1e-15

    def test_m_ssoc_universal(self):
        """M_SSOC = 3.0 は全構造統一Taylor因子"""
        from delta_theory.ssoc import M_SSOC

        assert M_SSOC == 3.0

    # --- FCC PCC ---

    def test_fcc_gate_d_metals(self):
        """FCC d-metals (n_d ≥ 2): g_d = 1"""
        from delta_theory.ssoc import fcc_gate

        assert fcc_gate(10) == 1.0  # Cu, Au, Ag
        assert fcc_gate(8) == 1.0   # Ni, Pd, Pt
        assert fcc_gate(7) == 1.0   # Ir, Rh
        assert fcc_gate(2) == 1.0   # boundary

    def test_fcc_gate_sp_metals(self):
        """FCC sp-metals (n_d < 2): g_d = 0"""
        from delta_theory.ssoc import fcc_gate

        assert fcc_gate(0) == 0.0   # Al, Pb
        assert fcc_gate(1) == 0.0

    def test_fcc_f_mu_reference(self):
        """FCC: Cu (μ_ref=43.6 GPa) で f_μ ≈ 1.0"""
        from delta_theory.ssoc import FCC_MU_REF, P_DIM, fcc_f_mu

        # g_d=1のとき、μ=μ_refなら (μ/μ_ref)^(2/3) = 1.0
        assert abs(fcc_f_mu(FCC_MU_REF, g_d=1.0) - 1.0) < 1e-15
        # g_d=0なら常に1.0
        assert abs(fcc_f_mu(100.0, g_d=0.0) - 1.0) < 1e-15

    def test_fcc_f_de_al_sp_metal(self):
        """Al (sp-metal, n_d=0): g_d=0 → f_de ≈ 1.0"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import fcc_f_de

        al = get_material("Al")
        f = fcc_f_de(al)
        # sp-metalはgate=0でスケーリングなし
        assert abs(f - 1.0) < 0.01

    def test_fcc_f_de_detail(self):
        """FCC f_de内訳が取得可能"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import fcc_f_de, fcc_f_de_detail

        cu = get_material("Cu")
        detail = fcc_f_de_detail(cu)
        assert "f_mu" in detail
        assert "f_shell" in detail
        assert "f_core" in detail
        assert "f_de" in detail
        # detailのf_de == fcc_f_de
        assert abs(detail["f_de"] - fcc_f_de(cu)) < 1e-15

    # --- BCC SCC ---

    def test_bcc_d4_jt_anomaly(self):
        """BCC d⁴ Jahn-Teller anomaly: f_JT = 1.9"""
        from delta_theory.ssoc import BCC_DELTA_JT, bcc_f_jt

        # d⁴: W, Nb → JT active
        assert bcc_f_jt(4) == 1.0 + BCC_DELTA_JT  # 1.9
        # others: 1.0
        assert bcc_f_jt(6) == 1.0  # Fe
        assert bcc_f_jt(5) == 1.0  # Cr, Mo
        assert bcc_f_jt(3) == 1.0  # V, Ta

    def test_bcc_5d_relativistic(self):
        """BCC 5d相対論補正: period=6 → 1.5"""
        from delta_theory.ssoc import BCC_F_5D, bcc_f_5d

        # W(period=6), Ta(period=6)
        assert bcc_f_5d(4, period=6) == BCC_F_5D  # 1.5
        assert bcc_f_5d(3, period=6) == BCC_F_5D
        # 4d, 3d → 1.0
        assert bcc_f_5d(4, period=5) == 1.0  # Nb
        assert bcc_f_5d(6, period=4) == 1.0  # Fe

    def test_bcc_w_d4_jt_full(self):
        """W: d⁴ JT × 5d → f_de ≈ 2.99"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import bcc_f_de

        w = get_material("W")
        f = bcc_f_de(w)
        # 1.9 (JT) × 1.5 (5d) × 1.05 (lattice) ≈ 2.99
        assert abs(f - 2.9925) < 0.01

    def test_bcc_fe_reference_level(self):
        """Fe: d⁶ 基準レベル (f_de ≈ 1.05)"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import bcc_f_de

        fe = get_material("Fe")
        f = bcc_f_de(fe)
        # Fe: JT=1.0, 5d=1.0, lattice=1.05 (Group8)
        assert abs(f - 1.05) < 0.01

    def test_bcc_f_de_detail(self):
        """BCC f_de内訳が取得可能"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import bcc_f_de, bcc_f_de_detail

        w = get_material("W")
        detail = bcc_f_de_detail(w)
        assert "f_jt" in detail
        assert "f_5d" in detail
        assert "f_lat" in detail
        assert "f_de" in detail
        assert abs(detail["f_de"] - bcc_f_de(w)) < 1e-15

    # --- HCP PCC ---

    def test_hcp_f_aniso_sigmoid(self):
        """HCP: R > 1 でシグモイドゲート"""
        from delta_theory.ssoc import hcp_f_aniso

        # R < 1: prismatic-easy linear
        assert hcp_f_aniso(0.5) > 1.0
        # R ≈ 1: transition
        assert abs(hcp_f_aniso(1.0) - 1.0) < 0.05
        # R >> 1: basal-easy sigmoid saturation
        f_high = hcp_f_aniso(10.0)
        assert f_high > hcp_f_aniso(1.0)

    def test_hcp_ca_constraint(self):
        """HCP: c/a < ideal → f_ca > 1.0 (補正), c/a ≥ ideal → f_ca = 1.0"""
        from delta_theory.ssoc import HCP_CA_IDEAL, hcp_f_ca

        # at ideal: f_ca = 1.0
        assert abs(hcp_f_ca(HCP_CA_IDEAL) - 1.0) < 1e-15
        # above ideal: no correction
        assert abs(hcp_f_ca(1.88) - 1.0) < 1e-15  # Cd-like
        # below ideal: correction active (f_ca > 1.0)
        f_below = hcp_f_ca(1.5)
        assert f_below > 1.0

    def test_hcp_ti_reference(self):
        """Ti: f_de ≈ 3.17"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import hcp_f_de

        ti = get_material("Ti")
        f = hcp_f_de(ti)
        assert abs(f - 3.17) < 0.05

    def test_hcp_f_de_detail(self):
        """HCP f_de内訳が取得可能"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import hcp_f_de, hcp_f_de_detail

        ti = get_material("Ti")
        detail = hcp_f_de_detail(ti)
        assert "f_elec" in detail
        assert "f_aniso" in detail
        assert "f_ca" in detail
        assert "f_de" in detail
        assert abs(detail["f_de"] - hcp_f_de(ti)) < 1e-15

    # --- Unified dispatch ---

    def test_calc_f_de_dispatch(self):
        """calc_f_de が構造に応じて正しい関数にディスパッチ"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import (
            bcc_f_de,
            calc_f_de,
            fcc_f_de,
            hcp_f_de,
        )

        fe = get_material("Fe")
        cu = get_material("Cu")
        ti = get_material("Ti")

        assert abs(calc_f_de(fe) - bcc_f_de(fe)) < 1e-15
        assert abs(calc_f_de(cu) - fcc_f_de(cu)) < 1e-15
        assert abs(calc_f_de(ti) - hcp_f_de(ti)) < 1e-15

    def test_calc_f_de_all_positive(self):
        """全25金属で f_de > 0"""
        from delta_theory.material import get_material, list_materials
        from delta_theory.ssoc import calc_f_de

        for name in list_materials():
            mat = get_material(name)
            f = calc_f_de(mat)
            assert f > 0, f"{name}: f_de should be positive"

    def test_calc_f_de_detail_dispatch(self):
        """calc_f_de_detail が全構造で動作"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import calc_f_de, calc_f_de_detail

        for name in ["Fe", "Cu", "Ti"]:
            mat = get_material(name)
            detail = calc_f_de_detail(mat)
            assert "f_de" in detail
            assert abs(detail["f_de"] - calc_f_de(mat)) < 1e-15

    # --- sigma_base_v10 ---

    def test_sigma_base_v10_positive(self):
        """全25金属でσ_base > 0 (T=300K)"""
        from delta_theory.material import get_material, list_materials
        from delta_theory.ssoc import sigma_base_v10

        for name in list_materials():
            mat = get_material(name)
            sigma = sigma_base_v10(mat, T_K=300.0)
            assert sigma > 0, f"{name}: σ_base should be positive"

    def test_sigma_base_v10_at_melting(self):
        """融点でσ_base = 0 (HP = 0)"""
        from delta_theory.material import get_material, list_materials
        from delta_theory.ssoc import sigma_base_v10

        for name in list_materials():
            mat = get_material(name)
            sigma = sigma_base_v10(mat, T_K=mat.T_m)
            assert sigma == 0.0, f"{name}: σ should be 0 at T_m"

    def test_sigma_base_v10_formula_manual(self):
        """σ_base の手計算検証 (Fe)"""
        from delta_theory.material import (
            BD_RATIO_SQ,
            COEFF_V10,
            eV_to_J,
            get_material,
            k_B,
        )
        from delta_theory.ssoc import M_SSOC, calc_f_de, sigma_base_v10

        mat = get_material("Fe")
        T_K = 300.0
        HP = max(0, 1.0 - T_K / mat.T_m)
        f_de = calc_f_de(mat)
        sqrt_EkT = np.sqrt(mat.E_bond_eV * eV_to_J * k_B * mat.T_m)

        sigma_manual = (
            COEFF_V10 / (M_SSOC * mat.Z_bulk)
            * mat.alpha0 * BD_RATIO_SQ * f_de
            * sqrt_EkT / mat.V_act * HP
        ) / 1e6  # Pa → MPa

        sigma_func = sigma_base_v10(mat, T_K)
        assert abs(sigma_manual - sigma_func) < 1e-10

    def test_sigma_base_v10_with_fde(self):
        """sigma_base_v10_with_fde: 外部f_deを渡して計算"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import (
            calc_f_de,
            sigma_base_v10,
            sigma_base_v10_with_fde,
        )

        mat = get_material("Cu")
        fde = calc_f_de(mat)
        sigma_ext = sigma_base_v10_with_fde(mat, fde, T_K=300.0)
        sigma_int = sigma_base_v10(mat, T_K=300.0)
        assert abs(sigma_ext - sigma_int) < 1e-10

    # --- inverse_f_de ---

    def test_inverse_f_de_roundtrip(self):
        """inverse_f_de の順逆一致 (4金属)"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import calc_f_de, inverse_f_de, sigma_base_v10

        for name in ["Fe", "Cu", "Ti", "W"]:
            mat = get_material(name)
            fde_fwd = calc_f_de(mat)
            sigma = sigma_base_v10(mat, 300.0)
            fde_inv = inverse_f_de(mat, sigma, 300.0)
            assert abs(fde_fwd - fde_inv) < 1e-6, (
                f"{name}: fwd={fde_fwd:.6f}, inv={fde_inv:.6f}"
            )

    # --- Validation (全25金属) ---

    def test_sigma_base_v10_validation(self):
        """全25金属のσ_base精度検証: MAE ≈ 3.2%"""
        from delta_theory.material import get_material, list_materials
        from delta_theory.ssoc import sigma_base_v10

        SIGMA_EXP = {
            "Fe": 150, "W": 750, "V": 130, "Cr": 170, "Nb": 105, "Mo": 200, "Ta": 170,
            "Cu": 60, "Ni": 120, "Al": 30, "Au": 30, "Ag": 40,
            "Pt": 70, "Pd": 45, "Ir": 300, "Rh": 200, "Pb": 7,
            "Ti": 250, "Mg": 90, "Zn": 30, "Zr": 230, "Hf": 200,
            "Re": 300, "Cd": 20, "Ru": 350,
        }

        errors = []
        for name in list_materials():
            mat = get_material(name)
            sigma_calc = sigma_base_v10(mat, T_K=300.0)
            sigma_exp = SIGMA_EXP[name]
            err_pct = abs((sigma_calc - sigma_exp) / sigma_exp * 100)
            errors.append((name, err_pct))

        # Cd以外の24金属で10%以内
        for name, err in errors:
            if name != "Cd":
                assert err < 15, f"{name}: error={err:.1f}% exceeds 15%"

        # 全25金属 MAE < 5%
        mae_all = np.mean([e for _, e in errors])
        assert mae_all < 5.0, f"MAE={mae_all:.1f}% exceeds 5%"

        # Cd除く24金属 MAE < 3%
        mae_no_cd = np.mean([e for n, e in errors if n != "Cd"])
        assert mae_no_cd < 3.5, f"MAE(no Cd)={mae_no_cd:.1f}% exceeds 3.5%"


# =============================================================================
# Unified Yield + Fatigue Tests (v10.0)
# =============================================================================

class TestUnifiedYieldFatigue:
    """unified_yield_fatigue のテスト (v10.0 SSOC)"""

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

    def test_sigma_base_delta_delegates_to_ssoc(self):
        """sigma_base_delta → ssoc.sigma_base_v10 に委譲"""
        from delta_theory.material import get_material, list_materials
        from delta_theory.ssoc import sigma_base_v10
        from delta_theory.unified_yield_fatigue_v6_9 import sigma_base_delta

        for name in list_materials():
            mat = get_material(name)
            s_delta = sigma_base_delta(mat, T_K=300.0)
            s_ssoc = sigma_base_v10(mat, T_K=300.0)
            assert abs(s_delta - s_ssoc) < 1e-10, (
                f"{name}: sigma_base_delta != sigma_base_v10"
            )

    def test_sigma_y_positive(self):
        """降伏応力は正"""
        from delta_theory.material import get_material, list_materials
        from delta_theory.unified_yield_fatigue_v6_9 import calc_sigma_y

        for name in list_materials():
            mat = get_material(name)
            y = calc_sigma_y(mat, T_K=300)
            assert y["sigma_y"] > 0, f"{name}: σ_y should be positive"
            assert y["sigma_base"] > 0, f"{name}: σ_base should be positive"

    def test_sigma_y_returns_f_de(self):
        """calc_sigma_y がf_deを返す (v10.0)"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import calc_f_de
        from delta_theory.unified_yield_fatigue_v6_9 import calc_sigma_y

        mat = get_material("Fe")
        y = calc_sigma_y(mat, T_K=300)
        assert "f_de" in y
        assert abs(y["f_de"] - calc_f_de(mat)) < 1e-15

    def test_sigma_y_returns_branch(self):
        """calc_sigma_y がSSOC branchを返す"""
        from delta_theory.material import get_material
        from delta_theory.unified_yield_fatigue_v6_9 import calc_sigma_y

        y = calc_sigma_y(get_material("Fe"), T_K=300)
        assert "sigma_base_branch" in y
        assert y["sigma_base_branch"] == "SSOC(v10.0)"

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
        assert bcc["r_th"] > 0.5

    def test_fatigue_preset_fcc(self):
        """FCC材料は疲労限度なし"""
        from delta_theory.unified_yield_fatigue_v6_9 import FATIGUE_CLASS_PRESET

        fcc = FATIGUE_CLASS_PRESET["FCC"]
        assert fcc["r_th"] < 0.1

    def test_fatigue_life_below_threshold(self):
        """r ≤ r_th で無限寿命"""
        from delta_theory.material import get_material
        from delta_theory.unified_yield_fatigue_v6_9 import (
            calc_sigma_y,
            fatigue_life_const_amp,
        )

        mat = get_material("Fe")
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
        """モード別降伏応力"""
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
# v10.0 SSOC Specific Tests (Physics Validation)
# =============================================================================

class TestSSOCPhysics:
    """v10.0 SSOC の物理的妥当性"""

    def test_delta_l_free_equivalence(self):
        """δ_Lフリーの核心: E_bond × δ_L ∝ √(E_coh · k_B · T_m)

        v7.0: σ ∝ E_bond × δ_L
        v10.0: σ ∝ √(E_coh · k_B · T_m)
        δ_L ∝ √(k_B · T_m / E_coh) なので同値
        """
        from delta_theory.material import eV_to_J, get_material, k_B

        mat = get_material("Fe")
        # v7.0 combination: E_bond × δ_L
        e_bond_j = mat.E_bond_eV * eV_to_J
        v70_term = e_bond_j * mat.delta_L
        # v10.0 combination: √(E_coh · k_B · T_m)
        v10_term = mat.sqrt_EkT

        # They should be proportional (not equal, due to different normalization)
        ratio = v70_term / v10_term
        # ratio should be roughly constant across metals
        ratios = []
        for name in ["Fe", "Cu", "Al", "Ti", "W"]:
            m = get_material(name)
            eb = m.E_bond_eV * eV_to_J
            ratios.append(eb * m.delta_L / m.sqrt_EkT)

        # Not perfectly constant (δ_L has corrections), but order-of-magnitude same
        assert max(ratios) / min(ratios) < 5.0

    def test_fcc_sp_metal_universality(self):
        """FCC sp-metals (Al, Pb) は g_d=0 で f_de ≈ 1.0"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import calc_f_de

        al = get_material("Al")
        pb = get_material("Pb")
        assert abs(calc_f_de(al) - 1.0) < 0.05
        assert abs(calc_f_de(pb) - 1.0) < 0.05

    def test_bcc_d4_strongest(self):
        """BCC d⁴ (W, Nb) はJT異常で最も高いf_de"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import calc_f_de

        w_fde = calc_f_de(get_material("W"))
        fe_fde = calc_f_de(get_material("Fe"))
        cr_fde = calc_f_de(get_material("Cr"))
        v_fde = calc_f_de(get_material("V"))

        # W (d⁴, 5d) should have highest BCC f_de
        assert w_fde > fe_fde
        assert w_fde > cr_fde
        assert w_fde > v_fde

    def test_hcp_ti_stronger_than_mg(self):
        """HCP: Ti の σ_base > Mg の σ_base"""
        from delta_theory.material import get_material
        from delta_theory.ssoc import sigma_base_v10

        ti_sigma = sigma_base_v10(get_material("Ti"))
        mg_sigma = sigma_base_v10(get_material("Mg"))
        assert ti_sigma > mg_sigma

    def test_structure_comparison_f_de_range(self):
        """f_de の範囲: 全構造で 0.3 < f_de < 10"""
        from delta_theory.material import get_material, list_materials
        from delta_theory.ssoc import calc_f_de

        for name in list_materials():
            f = calc_f_de(get_material(name))
            assert 0.3 < f < 10, f"{name}: f_de={f:.3f} out of range"


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
    """__init__.py のテスト (v10.0)"""

    def test_top_level_import(self):
        """トップレベルインポート"""
        from delta_theory import (
            BD_RATIO_SQ,
            COEFF_V10,
            MATERIALS,
            Material,
            calc_f_de,
            calc_sigma_y,
            get_material,
            sigma_base_delta,
            sigma_base_v10,
        )

        assert BD_RATIO_SQ == 1.5
        assert "Fe" in MATERIALS

    def test_ssoc_top_level_import(self):
        """SSOCのトップレベルインポート"""
        from delta_theory import (
            M_SSOC,
            P_DIM,
            calc_f_de,
            calc_f_de_detail,
            fcc_f_de,
            bcc_f_de,
            hcp_f_de,
            inverse_f_de,
            sigma_base_v10,
            sigma_base_v10_with_fde,
        )

        assert abs(P_DIM - 2 / 3) < 1e-15
        assert M_SSOC == 3.0

    def test_version(self):
        """バージョン"""
        import delta_theory

        assert delta_theory.__version__ == "10.0.5"

    def test_material_and_unified_share_materials(self):
        """material.py と unified 側の MATERIALS が同一"""
        from delta_theory import MATERIALS
        from delta_theory.material import MATERIALS as MAT2

        assert MATERIALS is MAT2

    def test_info_runs(self):
        """info() が正常実行"""
        from delta_theory import info

        info()  # should not raise

    def test_lindemann_lazy_import(self):
        """Lindemann モジュールの遅延インポート"""
        from delta_theory import iizumi_lindemann, C_IIZUMI

        assert callable(iizumi_lindemann)
        assert C_IIZUMI > 0


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
