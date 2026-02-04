#!/usr/bin/env python3
"""
================================================================================
dbt_unified.py - 統一DBT/DBTT予測モデル
================================================================================

概要
----
σ_y(d, T) = σ_f(d, c, T) という単一の方程式を、異なる軸から解く統合モジュール。

  【Core】共通の物理モデル
    - σ_y(d, T): δ理論 + Hall-Petch
    - σ_f(d, c, T): 粒界凝集破壊 + McLean偏析
    - θ(c, T): McLean等温線
    - g_seg(θ): 脆化関数（閾値的onset対応）

  【View 1】粒径軸（T固定 → d* を求める）
    - 延性窓 [d_low, d_high]
    - c_crit（延性窓消失の臨界濃度）
    - Re-entrant構造の検出

  【View 2】温度軸（d固定 → T* を求める）
    - DBTT(d, c) [K]
    - DBTT相図（d-c-DBTT）

  【View 3】時間軸（偏析発展）
    - θ_eq(c, T_age): 平衡偏析
    - PHR → θ 変換（AES校正）

統合の経緯
----------
  v4.0（延性窓/Re-entrant）+ v5.2（DBTT threshold）+ Fe-P θ-DBTT
  → 全て同じ物理モデル、見る軸が違うだけ → 1本に統合

Author: 環 & ご主人さま (飯泉真道)
Date: 2026-02-02
================================================================================
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np

# ==============================================================================
# 物理定数
# ==============================================================================
eV_to_J = 1.602176634e-19
k_B = 1.380649e-23
PI = np.pi
M_TAYLOR = 3.0


# ==============================================================================
# 材料データクラス
# ==============================================================================
@dataclass
class Material:
    """材料パラメータ（Fe基準、他材料は拡張可能）"""
    name: str
    # δ理論パラメータ
    E_bond_eV: float      # 結合エネルギー [eV]
    f_d: float            # d電子方向性係数
    a_lat: float          # 格子定数 [m]
    alpha0: float         # 幾何係数
    delta_L: float        # Lindemannパラメータ
    T_m: float            # 融点 [K]
    # 弾性定数
    E_mod: float          # ヤング率 [Pa]
    nu: float             # ポアソン比
    # 粒界パラメータ
    W_sep0: float         # 基準分離仕事 [J/m²]
    delta_sep: float      # 分離変位 [m]
    # 偏析パラメータ（デフォルト）
    E_seg_eV: float       # 偏析エネルギー [eV]
    
    @property
    def b(self) -> float:
        """バーガースベクトル（BCC: a√3/2）"""
        return self.a_lat * np.sqrt(3) / 2
    
    @property
    def V_act(self) -> float:
        """活性化体積"""
        return self.b ** 3
    
    @property
    def G(self) -> float:
        """剛性率"""
        return self.E_mod / (2 * (1 + self.nu))
    
    @property
    def E_eff(self) -> float:
        """有効結合エネルギー"""
        return self.E_bond_eV * eV_to_J * self.alpha0 * self.f_d


# デフォルト材料（Fe）
MATERIAL_FE = Material(
    name="Fe",
    E_bond_eV=4.28,
    f_d=1.5,
    a_lat=2.92e-10,
    alpha0=0.289,
    delta_L=0.18,
    T_m=1811.0,
    E_mod=211e9,
    nu=0.29,
    W_sep0=2.0,
    delta_sep=0.2e-9,
    E_seg_eV=0.45,  # Fe-P系のデフォルト
)


# ==============================================================================
# Core: 共通の物理モデル
# ==============================================================================

class DBTCore:
    """DBT/DBTT計算のコアエンジン"""
    
    def __init__(
        self,
        mat: Material = MATERIAL_FE,
        beta_hp: float = 3.5e-3,
        C_f: float = 5.0,
        K_emb: float = 3.0,
        n_emb: float = 1.0,  # 脆化指数（n>1で閾値的）
    ):
        self.mat = mat
        self.beta_hp = beta_hp
        self.C_f = C_f
        self.K_emb = K_emb
        self.n_emb = n_emb
        
        # 事前計算
        self._A0_noR = (mat.E_eff / mat.V_act) * mat.delta_L / (2 * PI * M_TAYLOR)
    
    # --------------------------------------------------------------------------
    # 基本関数
    # --------------------------------------------------------------------------
    
    def R_block(self, d: float) -> float:
        """Hall-Petch遮断係数 R(d) = 1 + β/√d"""
        return 1.0 + self.beta_hp / np.sqrt(d)
    
    def HP(self, T: float) -> float:
        """熱的余白 HP = 1 - T/T_m"""
        return max(1e-9, 1.0 - T / self.mat.T_m)
    
    def theta_mclean(self, c: float, T: float, E_seg_eV: Optional[float] = None) -> float:
        """McLean等温線による粒界被覆率 θ(c, T)
        
        θ = Kc / (1 - c + Kc),  K = exp(E_seg / kT)
        """
        c = float(np.clip(c, 0.0, 1.0 - 1e-9))
        E_seg = E_seg_eV if E_seg_eV is not None else self.mat.E_seg_eV
        K = np.exp((E_seg * eV_to_J) / (k_B * T))
        return (K * c) / (1.0 - c + K * c + 1e-30)
    
    def g_seg(self, theta: float) -> float:
        """偏析による脆化関数 g_seg(θ) = 1 / (1 + K_emb × θ^n)
        
        n=1: 線形的脆化
        n>1: 閾値的onset（パーコレーション的）
        """
        return 1.0 / (1.0 + self.K_emb * (theta ** self.n_emb))
    
    def g_T_cohesion(self, T: float) -> float:
        """温度による凝集力低下 g_T = 1 - T/T_m"""
        return max(1e-3, 1.0 - T / self.mat.T_m)
    
    # --------------------------------------------------------------------------
    # 応力計算
    # --------------------------------------------------------------------------
    
    def sigma_y(self, d: float, T: float) -> float:
        """降伏応力 σ_y(d, T) [Pa]
        
        σ_y = A0 × HP × R(d)
        """
        return self._A0_noR * self.HP(T) * self.R_block(d)
    
    def sigma_GB(self, c: float, T: float, E_seg_eV: Optional[float] = None) -> float:
        """粒界強度 σ_GB(c, T) [Pa]
        
        σ_GB = W_sep(c, T) / δ_sep
        W_sep = W_sep0 × g_T × g_seg(θ)
        """
        theta = self.theta_mclean(c, T, E_seg_eV)
        W_sep = self.mat.W_sep0 * self.g_T_cohesion(T) * self.g_seg(theta)
        return W_sep / self.mat.delta_sep
    
    def sigma_f(self, d: float, c: float, T: float, E_seg_eV: Optional[float] = None) -> float:
        """破壊応力 σ_f(d, c, T) [Pa]
        
        σ_f = C_f × √(σ_GB × G × b / (R × d))
        """
        sig_gb = self.sigma_GB(c, T, E_seg_eV)
        R = self.R_block(d)
        return self.C_f * np.sqrt(sig_gb * self.mat.G * self.mat.b / (R * d))
    
    def g_diff(self, d: float, c: float, T: float) -> float:
        """g(d, c, T) = σ_y - σ_f
        
        g > 0: 脆性（σ_y > σ_f）
        g < 0: 延性（σ_y < σ_f）
        g = 0: 遷移点
        """
        return self.sigma_y(d, T) - self.sigma_f(d, c, T)


# ==============================================================================
# View 1: 粒径軸（T固定 → d* を求める）
# ==============================================================================

class GrainSizeView:
    """粒径軸からのDBT解析（v4.0相当）"""
    
    def __init__(self, core: DBTCore):
        self.core = core
    
    def find_all_crossings(
        self,
        T: float,
        c: float,
        d_min: float = 1e-8,
        d_max: float = 1e-3,
        n_pts: int = 1000,
    ) -> np.ndarray:
        """σ_y = σ_f の全交差点を検出（Re-entrant対応）"""
        d_arr = np.logspace(np.log10(d_min), np.log10(d_max), n_pts)
        g_arr = np.array([self.core.g_diff(d, c, T) for d in d_arr])
        
        # 符号変化点を検出
        sign_changes = np.where(np.diff(np.sign(g_arr)) != 0)[0]
        
        crossings = []
        for idx in sign_changes:
            # 線形補間
            d0, d1 = d_arr[idx], d_arr[idx + 1]
            g0, g1 = g_arr[idx], g_arr[idx + 1]
            d_cross = d0 + (d1 - d0) * (-g0) / (g1 - g0 + 1e-30)
            crossings.append(d_cross)
        
        return np.array(crossings)
    
    def classify_mode(
        self,
        T: float,
        c: float,
        d_min: float = 1e-8,
        d_max: float = 1e-3,
    ) -> Dict:
        """破壊モードを分類（延性窓検出）
        
        Returns:
            mode: "brittle" / "ductile" / "transition" / "window"
            d_low: 延性窓下限（存在しなければ nan）
            d_high: 延性窓上限（存在しなければ nan）
        """
        crossings = self.find_all_crossings(T, c, d_min, d_max)
        
        # 境界でのg値
        g_min = self.core.g_diff(d_min, c, T)
        g_max = self.core.g_diff(d_max, c, T)
        
        if len(crossings) == 0:
            if g_min > 0:
                return {"mode": "brittle", "d_low": np.nan, "d_high": np.nan,
                        "msg": "全域脆性（σ_y > σ_f everywhere）"}
            else:
                return {"mode": "ductile", "d_low": np.nan, "d_high": np.nan,
                        "msg": "全域延性（σ_y < σ_f everywhere）"}
        
        if len(crossings) == 1:
            d_star = crossings[0]
            if g_min > 0:
                return {"mode": "transition", "d_low": np.nan, "d_high": d_star,
                        "msg": f"単一DBT: d* = {d_star*1e6:.2f} μm（小d側が脆性）"}
            else:
                return {"mode": "transition", "d_low": d_star, "d_high": np.nan,
                        "msg": f"単一DBT: d* = {d_star*1e6:.2f} μm（小d側が延性）"}
        
        # 交差2つ以上: Re-entrant（延性窓）
        d_low = crossings[0]
        d_high = crossings[-1]
        return {"mode": "window", "d_low": d_low, "d_high": d_high,
                "msg": f"延性窓: {d_low*1e6:.3f} ~ {d_high*1e6:.2f} μm"}
    
    def find_c_crit(
        self,
        T: float,
        d_min: float = 1e-8,
        d_max: float = 1e-3,
        c_max: float = 0.1,
        tol: float = 1e-5,
    ) -> Dict:
        """延性窓が消失する臨界濃度 c_crit を二分法で求める"""
        
        def has_ductile_region(c: float) -> bool:
            d_arr = np.logspace(np.log10(d_min), np.log10(d_max), 200)
            g_arr = np.array([self.core.g_diff(d, c, T) for d in d_arr])
            return np.any(g_arr < 0)
        
        # 境界チェック
        if not has_ductile_region(0):
            return {"c_crit": 0.0, "msg": "c=0で既に全域脆性"}
        if has_ductile_region(c_max):
            return {"c_crit": c_max, "msg": f"c={c_max*100:.1f}%でも延性領域あり"}
        
        # 二分法
        c_lo, c_hi = 0.0, c_max
        while (c_hi - c_lo) > tol:
            c_mid = (c_lo + c_hi) / 2
            if has_ductile_region(c_mid):
                c_lo = c_mid
            else:
                c_hi = c_mid
        
        c_crit = (c_lo + c_hi) / 2
        theta_crit = self.core.theta_mclean(c_crit, T)
        return {
            "c_crit": c_crit,
            "theta_crit": theta_crit,
            "msg": f"c_crit = {c_crit*100:.4f} at%, θ_crit = {theta_crit:.4f}"
        }


# ==============================================================================
# View 2: 温度軸（d固定 → T* を求める）
# ==============================================================================

class TemperatureView:
    """温度軸からのDBTT解析（v5.2相当）"""
    
    def __init__(self, core: DBTCore):
        self.core = core
    
    def find_DBTT(
        self,
        d: float,
        c: float,
        T_lo: float = 50.0,
        T_hi: float = 1500.0,
        n_scan: int = 500,
    ) -> Dict:
        """固定粒径 d における DBTT（脆性遷移温度）を求める
        
        g(T) = σ_y(d,T) - σ_f(d,c,T) の符号変化点を探索
        
        Returns:
            T_star: DBTT [K]
            status: "crossing" / "always_brittle" / "always_ductile"
        """
        T_arr = np.linspace(T_lo, T_hi, n_scan)
        g_arr = np.array([self.core.g_diff(d, c, T) for T in T_arr])
        
        # 全域チェック
        if np.all(g_arr > 0):
            return {"T_star": np.nan, "status": "always_brittle",
                    "msg": "全温度域で脆性"}
        if np.all(g_arr < 0):
            return {"T_star": np.nan, "status": "always_ductile",
                    "msg": "全温度域で延性"}
        
        # 符号変化点を探す
        sign_changes = np.where(np.diff(np.sign(g_arr)) != 0)[0]
        if len(sign_changes) == 0:
            return {"T_star": np.nan, "status": "no_crossing",
                    "msg": "交差点なし（数値問題？）"}
        
        # 最初の交差点（低温側のDBTT）
        idx = sign_changes[0]
        a, b = T_arr[idx], T_arr[idx + 1]
        
        # 二分法で高精度化
        for _ in range(60):
            m = 0.5 * (a + b)
            gm = self.core.g_diff(d, c, m)
            ga = self.core.g_diff(d, c, a)
            if np.sign(gm) == np.sign(ga):
                a = m
            else:
                b = m
        
        T_star = 0.5 * (a + b)
        
        # 交差の向き
        g_low = self.core.g_diff(d, c, T_lo)
        if g_low > 0:
            status = "brittle_to_ductile"
            msg = f"DBTT = {T_star:.1f} K（低温側が脆性）"
        else:
            status = "ductile_to_brittle"
            msg = f"DBTT = {T_star:.1f} K（低温側が延性）"
        
        return {"T_star": T_star, "status": status, "msg": msg}
    
    def generate_DBTT_table(
        self,
        d_list: List[float],
        c_list: List[float],
    ) -> np.ndarray:
        """DBTT(d, c) テーブルを生成"""
        table = np.zeros((len(d_list), len(c_list)))
        for i, d in enumerate(d_list):
            for j, c in enumerate(c_list):
                result = self.find_DBTT(d, c)
                table[i, j] = result["T_star"]
        return table


# ==============================================================================
# View 3: 時間軸（偏析発展）
# ==============================================================================

class SegregationView:
    """偏析発展からのDBTT解析（Fe-P相当）"""
    
    def __init__(self, core: DBTCore):
        self.core = core
    
    def theta_from_phr(self, phr: float, S: float, phr_bg: float = 0.0) -> float:
        """AES PHR → θ 変換（プロキシ）
        
        θ = S × max(PHR - PHR_bg, 0) / (1 + S × max(PHR - PHR_bg, 0))
        """
        r = max(0.0, float(phr) - float(phr_bg))
        return (S * r) / (1.0 + S * r)
    
    def fit_S_from_equilibrium(
        self,
        c_bulk: float,
        eq_points: List[Tuple[float, float]],  # [(T_age, phr), ...]
        E_seg_eV: Optional[float] = None,
    ) -> float:
        """McLean平衡点からS（PHR→θ変換係数）をフィット
        
        条件: θ_McLean(c, T_age) = θ_from_phr(phr, S)
        """
        from scipy.optimize import minimize_scalar
        
        def loss(S):
            err = 0.0
            for T_age, phr in eq_points:
                theta_eq = self.core.theta_mclean(c_bulk, T_age, E_seg_eV)
                theta_phr = self.theta_from_phr(phr, S)
                err += (theta_eq - theta_phr) ** 2
            return err
        
        result = minimize_scalar(loss, bounds=(0.1, 100.0), method='bounded')
        return result.x
    
    def DBTT_from_theta(
        self,
        d: float,
        theta: float,
        DBTT_base_K: float,
    ) -> float:
        """θ から DBTT を予測（閾値モデル）
        
        DBTT = max(DBTT_base, T_GB(θ))
        """
        # Q0: 基準状態でのHP
        HP_base = self.core.HP(DBTT_base_K)
        R = self.core.R_block(d)
        
        # σ_f = σ_y の条件から T_GB を導出
        # HP_req = Q0 / g_seg(θ)
        Q0 = HP_base  # 基準状態
        HP_req = Q0 / self.core.g_seg(theta)
        HP_req = np.clip(HP_req, 1e-9, 1.0 - 1e-9)
        
        T_GB = self.core.mat.T_m * (1.0 - HP_req)
        
        return max(DBTT_base_K, T_GB)


# ==============================================================================
# 統合インターフェース
# ==============================================================================

class DBTUnified:
    """統一DBT/DBTTモデル"""
    
    def __init__(
        self,
        mat: Material = MATERIAL_FE,
        beta_hp: float = 3.5e-3,
        C_f: float = 5.0,
        K_emb: float = 3.0,
        n_emb: float = 1.0,
    ):
        self.core = DBTCore(mat, beta_hp, C_f, K_emb, n_emb)
        self.grain_view = GrainSizeView(self.core)
        self.temp_view = TemperatureView(self.core)
        self.seg_view = SegregationView(self.core)
    
    def summary(self, d: float, c: float, T: float) -> Dict:
        """状態サマリー"""
        sigma_y = self.core.sigma_y(d, T)
        sigma_f = self.core.sigma_f(d, c, T)
        theta = self.core.theta_mclean(c, T)
        g = self.core.g_diff(d, c, T)
        
        mode = "brittle" if g > 0 else "ductile"
        
        return {
            "d_um": d * 1e6,
            "c_at_pct": c * 100,
            "T_K": T,
            "sigma_y_MPa": sigma_y / 1e6,
            "sigma_f_MPa": sigma_f / 1e6,
            "theta": theta,
            "g_diff_MPa": g / 1e6,
            "mode": mode,
        }


# ==============================================================================
# CLI
# ==============================================================================

def cmd_point(args):
    """単点計算"""
    model = DBTUnified(
        beta_hp=args.beta_hp,
        C_f=args.C_f,
        K_emb=args.K_emb,
        n_emb=args.n_emb,
    )
    
    result = model.summary(args.d * 1e-6, args.c / 100, args.T)
    
    print("=" * 70)
    print("DBT Unified - 単点計算")
    print("=" * 70)
    print(f"入力: d = {args.d} μm, c = {args.c} at%, T = {args.T} K")
    print("-" * 70)
    print(f"σ_y  = {result['sigma_y_MPa']:.1f} MPa")
    print(f"σ_f  = {result['sigma_f_MPa']:.1f} MPa")
    print(f"θ    = {result['theta']:.4f}")
    print(f"Δσ   = {result['g_diff_MPa']:.1f} MPa")
    print(f"モード: {result['mode'].upper()}")
    print("=" * 70)


def cmd_d_axis(args):
    """粒径軸解析"""
    model = DBTUnified(
        beta_hp=args.beta_hp,
        C_f=args.C_f,
        K_emb=args.K_emb,
        n_emb=args.n_emb,
    )
    
    result = model.grain_view.classify_mode(args.T, args.c / 100)
    
    print("=" * 70)
    print("DBT Unified - 粒径軸解析（View 1）")
    print("=" * 70)
    print(f"条件: T = {args.T} K, c = {args.c} at%")
    print("-" * 70)
    print(f"モード: {result['mode']}")
    print(f"メッセージ: {result['msg']}")
    if np.isfinite(result['d_low']):
        print(f"d_low  = {result['d_low']*1e6:.3f} μm")
    if np.isfinite(result['d_high']):
        print(f"d_high = {result['d_high']*1e6:.2f} μm")
    
    if args.find_c_crit:
        c_result = model.grain_view.find_c_crit(args.T)
        print("-" * 70)
        print(f"臨界濃度: {c_result['msg']}")
    
    print("=" * 70)


def cmd_T_axis(args):
    """温度軸解析"""
    model = DBTUnified(
        beta_hp=args.beta_hp,
        C_f=args.C_f,
        K_emb=args.K_emb,
        n_emb=args.n_emb,
    )
    
    result = model.temp_view.find_DBTT(args.d * 1e-6, args.c / 100)
    
    print("=" * 70)
    print("DBT Unified - 温度軸解析（View 2）")
    print("=" * 70)
    print(f"条件: d = {args.d} μm, c = {args.c} at%")
    print("-" * 70)
    print(f"ステータス: {result['status']}")
    print(f"メッセージ: {result['msg']}")
    if np.isfinite(result['T_star']):
        print(f"DBTT = {result['T_star']:.1f} K = {result['T_star']-273.15:.1f} °C")
    print("=" * 70)


def cmd_table(args):
    """DBTT テーブル生成"""
    model = DBTUnified(
        beta_hp=args.beta_hp,
        C_f=args.C_f,
        K_emb=args.K_emb,
        n_emb=args.n_emb,
    )
    
    d_list = [float(x) * 1e-6 for x in args.d_list.split(",")]
    c_list = [float(x) / 100 for x in args.c_list.split(",")]
    
    print("=" * 70)
    print("DBT Unified - DBTTテーブル")
    print("=" * 70)
    
    # ヘッダー
    header = f"{'d [μm]':<10}" + "".join([f"c={c*100:.2f}%".center(12) for c in c_list])
    print(header)
    print("-" * (10 + 12 * len(c_list)))
    
    for d in d_list:
        row = f"{d*1e6:<10.0f}"
        for c in c_list:
            result = model.temp_view.find_DBTT(d, c)
            if np.isfinite(result['T_star']):
                row += f"{result['T_star']:^12.0f}"
            else:
                row += f"{result['status'][:8]:^12}"
        print(row)
    
    print("=" * 70)


def build_parser():
    ap = argparse.ArgumentParser(description="DBT Unified - 統一DBT/DBTTモデル")
    
    # 共通引数
    ap.add_argument("--beta_hp", type=float, default=3.5e-3, help="Hall-Petch係数")
    ap.add_argument("--C_f", type=float, default=5.0, help="幾何係数")
    ap.add_argument("--K_emb", type=float, default=3.0, help="脆化強度")
    ap.add_argument("--n_emb", type=float, default=1.0, help="脆化指数（>1で閾値的）")
    
    sub = ap.add_subparsers(dest="cmd", required=True)
    
    # point
    p1 = sub.add_parser("point", help="単点計算")
    p1.add_argument("--d", type=float, required=True, help="粒径 [μm]")
    p1.add_argument("--c", type=float, required=True, help="濃度 [at%]")
    p1.add_argument("--T", type=float, required=True, help="温度 [K]")
    
    # d_axis
    p2 = sub.add_parser("d_axis", help="粒径軸解析（延性窓）")
    p2.add_argument("--T", type=float, required=True, help="温度 [K]")
    p2.add_argument("--c", type=float, required=True, help="濃度 [at%]")
    p2.add_argument("--find_c_crit", action="store_true", help="c_critも計算")
    
    # T_axis
    p3 = sub.add_parser("T_axis", help="温度軸解析（DBTT）")
    p3.add_argument("--d", type=float, required=True, help="粒径 [μm]")
    p3.add_argument("--c", type=float, required=True, help="濃度 [at%]")
    
    # table
    p4 = sub.add_parser("table", help="DBTTテーブル生成")
    p4.add_argument("--d_list", type=str, default="5,10,20,50,100", help="粒径リスト [μm]")
    p4.add_argument("--c_list", type=str, default="0,0.2,0.5,0.8,1.0", help="濃度リスト [at%]")
    
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    
    if args.cmd == "point":
        cmd_point(args)
    elif args.cmd == "d_axis":
        cmd_d_axis(args)
    elif args.cmd == "T_axis":
        cmd_T_axis(args)
    elif args.cmd == "table":
        cmd_table(args)


if __name__ == "__main__":
    main()
