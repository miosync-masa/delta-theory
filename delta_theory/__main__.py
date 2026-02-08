#!/usr/bin/env python3
"""
δ-Theory CLI Entry Point

Usage:
    python -m delta_theory              # Show quick reference
    python -m delta_theory info         # Show detailed info
    python -m delta_theory flc SPCC     # Quick FLC prediction
    python -m delta_theory flc Cu all   # All 7 modes
    python -m delta_theory fatigue Fe 150  # Quick fatigue life
"""

import sys

QUICK_REFERENCE = """

    ███████╗██╗   ██╗██████╗ ███████╗██╗  ██╗ █████╗ ██╗
    ██╔════╝██║   ██║██╔══██╗██╔════╝██║ ██╔╝██╔══██╗██║
    █████╗  ██║   ██║██████╔╝█████╗  █████╔╝ ███████║██║
    ██╔══╝  ██║   ██║██╔══██╗██╔══╝  ██╔═██╗ ██╔══██╗╚═╝
    ███████╗╚██████╔╝██║  ██║███████╗██║  ██╗██║  ██║██╗
    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝
                    δ-Theory v10.0.0
                  "Nature is Geometry"
                  
╔══════════════════════════════════════════════════════════════════════════════╗
║  δ-Theory v10.0.0 — CLI Quick Reference                                     ║
║  SSOC: Structure-Selective Orbital Coupling                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INSTALLATION OK! ✓                                                          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🔧 YIELD STRESS (v10.0 SSOC)                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    # Quick base yield:                                                       ║
║    python -m delta_theory.unified_yield_fatigue_v10 point --metal Fe         ║
║    python -m delta_theory.unified_yield_fatigue_v10 point --metal W          ║
║                                                                              ║
║    # With temperature:                                                       ║
║    python -m delta_theory.unified_yield_fatigue_v10 point --metal Cu \        ║
║           --T_K 500                                                          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  💪 STRENGTHENING MECHANISMS (v10.0)                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    # Solid Solution (Labusch):                                               ║
║    python -m delta_theory.unified_yield_fatigue_v10 point --metal Fe \        ║
║           --c_wt 0.1 --k_ss 1800 --solute_type interstitial                 ║
║                                                                              ║
║    # Work Hardening (Taylor):                                                ║
║    python -m delta_theory.unified_yield_fatigue_v10 point --metal Cu \        ║
║           --eps 0.10 --rho_0 1e12                                            ║
║                                                                              ║
║    # Precipitation (auto Cutting/Orowan switch):                             ║
║    python -m delta_theory.unified_yield_fatigue_v10 point --metal Ni \        ║
║           --r_ppt_nm 5.0 --f_ppt 0.03 --gamma_apb 0.15                      ║
║                                                                              ║
║    # ALL-IN — Solid solution + Work hardening + Precipitation:               ║
║    python -m delta_theory.unified_yield_fatigue_v10 point --metal Fe \        ║
║           --c_wt 0.1 --k_ss 1800 --solute_type interstitial \               ║
║           --eps 0.05 --rho_0 1e12 \                                          ║
║           --r_ppt_nm 5.0 --f_ppt 0.03 --gamma_apb 0.15                      ║
║                                                                              ║
║    # BCC Low-T Peierls barrier:                                              ║
║    python -m delta_theory.unified_yield_fatigue_v10 point --metal Fe \        ║
║           --T_K 77 --enable_peierls --tau_P0 400 --dG0 0.6                   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🔄 FATIGUE LIFE (v6.10)                                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    # Single-point fatigue life:                                              ║
║    python -m delta_theory.unified_yield_fatigue_v10 point --metal Fe \        ║
║           --sigma_a 150                                                      ║
║                                                                              ║
║    # With σ_y override + shear mode:                                         ║
║    python -m delta_theory.unified_yield_fatigue_v10 point --metal Fe \        ║
║           --sigma_a 150 --sigma_y_override 200 --mode shear                  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📈 S-N CURVE GENERATION                                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    # Fe S-N curve (50~300 MPa, 25 points):                                   ║
║    python -m delta_theory.unified_yield_fatigue_v10 sn --metal Fe \           ║
║           --sigma_min 50 --sigma_max 300 --num 25                            ║
║                                                                              ║
║    # With strengthening:                                                     ║
║    python -m delta_theory.unified_yield_fatigue_v10 sn --metal Fe \           ║
║           --c_wt 0.1 --k_ss 1800 --solute_type interstitial \               ║
║           --sigma_min 50 --sigma_max 400 --num 30                            ║
║                                                                              ║
║    # Shear S-N:                                                              ║
║    python -m delta_theory.unified_yield_fatigue_v10 sn --metal Fe \           ║
║           --sigma_min 30 --sigma_max 180 --mode shear                        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🎯 CALIBRATE A_ext (1-point S-N calibration)                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    # "I have one (σ_a, N_f) data point" → calibrate A_ext:                   ║
║    python -m delta_theory.unified_yield_fatigue_v10 calibrate --metal Fe \    ║
║           --sigma_a 200 --N_fail 1e5                                         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ⚡ SSOC f_de INSPECTION                                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    # Single metal (f_de breakdown):                                          ║
║    python -m delta_theory ssoc W          # d⁴ JT anomaly → f_de ≈ 2.99     ║
║    python -m delta_theory ssoc Fe         # BCC reference                    ║
║                                                                              ║
║    # All 25 metals table:                                                    ║
║    python -m delta_theory ssoc all                                           ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🐍 PYTHON API EXAMPLES                                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    from delta_theory import (                                                ║
║        calc_sigma_y, MATERIALS,                                              ║
║        calc_f_de, sigma_base_v10,                                            ║
║        fatigue_life_const_amp, generate_sn_curve,                            ║
║        tau_over_sigma, sigma_c_over_sigma_t,                                 ║
║    )                                                                         ║
║                                                                              ║
║    # Yield with all strengthening:                                           ║
║    y = calc_sigma_y(MATERIALS['Fe'], T_K=300,                                ║
║            c_wt_percent=0.1, k_ss=1800, solute_type='interstitial',          ║
║            eps=0.05, rho_0=1e12,                                             ║
║            r_ppt_nm=5.0, f_ppt=0.03, gamma_apb=0.15)                        ║
║    print(f"σ_y = {y['sigma_y']:.1f} MPa  (f_de={y['f_de']:.3f})")           ║
║                                                                              ║
║    # Fatigue life:                                                           ║
║    r = fatigue_life_const_amp(MATERIALS['Fe'], sigma_a_MPa=150,              ║
║            sigma_y_tension_MPa=y['sigma_y'], A_ext=2.5e-4)                   ║
║    print(f"N = {r['N_fail']:.2e} cycles")                                    ║
║                                                                              ║
║    # S-N curve (numpy array):                                                ║
║    import numpy as np                                                        ║
║    sigmas = np.linspace(50, 300, 25)                                         ║
║    Ns = generate_sn_curve(MATERIALS['Fe'],                                   ║
║            sigma_y_tension_MPa=200.0, sigmas_MPa=sigmas)                     ║
║                                                                              ║
║    # Multiaxial ratios:                                                      ║
║    print(f"τ/σ = {tau_over_sigma(MATERIALS['Fe']):.4f}")                      ║
║    print(f"R_c = {sigma_c_over_sigma_t(MATERIALS['Mg']):.2f}")               ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📐 FLC v8.1 — 7-Mode Discrete Formulation                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    from delta_theory import FLCPredictor, predict_flc                        ║
║                                                                              ║
║    # Quick prediction                                                        ║
║    eps1 = predict_flc('Cu', 'Plane Strain')  # → 0.346                       ║
║                                                                              ║
║    # Full usage                                                              ║
║    flc = FLCPredictor()                                                      ║
║    flc.add_from_v69('MySteel', flc0=0.28, base_element='Fe')                 ║
║    eps1 = flc.predict('MySteel', 'Uniaxial')                                 ║
║                                                                              ║
║    # All 7 modes                                                             ║
║    betas, eps1s = flc.predict_curve('Cu')                                    ║
║                                                                              ║
║    # CLI:                                                                    ║
║    python -m delta_theory flc Cu              # FLC₀ (Plane Strain)          ║
║    python -m delta_theory flc Cu all          # All 7 modes                  ║
║    python -m delta_theory flc Cu Uniaxial     # Specific mode                ║
║                                                                              ║
║    # Built-in materials:                                                     ║
║    Cu, Ti, SPCC, DP590, Al5052, SUS304, Mg_AZ31                              ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🌡️ DBT (Ductile-Brittle Transition)                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    from delta_theory import DBTUnified                                       ║
║    model = DBTUnified()                                                      ║
║    result = model.temp_view.find_DBTT(d=30e-6, c=0.005)                      ║
║    print(f"DBTT = {result['T_star']:.0f} K")                                 ║
║                                                                              ║
║    # CLI:                                                                    ║
║    python -m delta_theory.dbt_unified point --d 30 --c 0.5 --T 300           ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📊 KEY CONSTANTS                                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    r_th (Fatigue Threshold)  │  τ/σ (Multiaxial)                             ║
║    ─────────────────────────────────────────────                             ║
║    BCC: 0.65 (clear limit)   │  BCC: 0.565                                   ║
║    FCC: 0.02 (no limit)      │  FCC: 0.565                                   ║
║    HCP: 0.20 (weak limit)    │  HCP: 0.327-0.565 (T_twin dependent)          ║
║                                                                              ║
║    SSOC f_de (electronic)    │  P_DIM = 2/3 (universal)                      ║
║    ─────────────────────────────────────────────                             ║
║    FCC: PCC (μ channel)      │  M_SSOC = 3.0 (universal)                     ║
║    BCC: SCC (Peierls, d⁴ JT) │  COEFF = 8√5/(5π) ≈ 1.138                    ║
║    HCP: PCC (R channel)      │  (b/d)² = 3/2 (universal)                     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📚 MORE INFO & COMMANDS                                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    python -m delta_theory info              # Detailed module info           ║
║    python -m delta_theory ssoc all          # SSOC f_de table (25 metals)    ║
║    python -m delta_theory flc --help        # FLC command help               ║
║                                                                              ║
║    # v10 subcommands:                                                        ║
║    python -m delta_theory.unified_yield_fatigue_v10 point  -h  # yield       ║
║    python -m delta_theory.unified_yield_fatigue_v10 sn     -h  # S-N curve   ║
║    python -m delta_theory.unified_yield_fatigue_v10 calibrate -h # A_ext     ║
║                                                                              ║
║    Docs: https://github.com/miosync-inc/delta-theory                         ║
║    PyPI: https://pypi.org/project/delta-theory/                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

DETAILED_INFO = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  δ-Theory v10.0.0 — Detailed Module Information                              ║
║  SSOC: Structure-Selective Orbital Coupling                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

📦 INSTALLED MODULES
═══════════════════════════════════════════════════════════════════════════════

  delta_theory/
  ├── material.py                      # Data layer — 25 metals + SSOC params
  ├── ssoc.py                          # ★ f_de calculation (PCC/SCC)
  ├── unified_yield_fatigue_v10.py     # σ_y + τ/σ + fatigue (S-N)
  ├── unified_yield_fatigue_v6_9.py    # ← backward compat shim
  ├── unified_flc_v8_1.py             # FLC 7-mode discrete
  ├── dbt_unified.py                  # Ductile-Brittle Transition
  ├── lindemann.py                    # Iizumi-Lindemann melting law
  ├── banners.py                      # ASCII art banners
  └── fatigue_redis_api.py            # FatigueData-AM2022 API (optional)


🔬 THEORY SUMMARY
═══════════════════════════════════════════════════════════════════════════════

  Core Equation:  Λ = K / |V|_eff

  K       = Destructive energy (stress, thermal, EM...)
  |V|_eff = Cohesive energy (bond strength)
  Λ = 1   → Critical condition (fracture/transition)


⚡ SSOC — Structure-Selective Orbital Coupling (v10.0)
═══════════════════════════════════════════════════════════════════════════════

  Unified equation (δ_L-free):

    σ_y = [8√5/(5π·M·Z)] × α₀ × (b/d)² × f_de × √(E·kT)/V × HP

  f_de = (X_s / X_ref)^(2/3 · g_d) × f_aux

  ┌────────────┬──────────┬────────────┬────────────────────────┐
  │ Structure  │ Coupling │ Channel X  │ Key Physics            │
  ├────────────┼──────────┼────────────┼────────────────────────┤
  │ FCC        │ PCC      │ μ (shear)  │ Curvature → core resist│
  │ BCC        │ SCC      │ ΔE_P       │ d-orbital self-gen     │
  │ HCP        │ PCC      │ R (CRSS)   │ Slip anisotropy        │
  └────────────┴──────────┴────────────┴────────────────────────┘

  BCC d⁴ Jahn-Teller: W f_de ≈ 2.99 → σ = 744 vs 750 MPa (exp)
  Constants used: 3, 5, 7, π only. No empirical fitting.


🔧 COMPLETE MODULE COVERAGE
═══════════════════════════════════════════════════════════════════════════════

  ┌────┬────────────────────┬──────────────────────────┬────────────┐
  │ #  │ Module             │ Predicts                 │ Fitting    │
  ├────┼────────────────────┼──────────────────────────┼────────────┤
  │  1 │ σ_base (SSOC)      │ Base yield stress        │ ≈ 0        │
  │  2 │ Δσ_ss              │ Solid solution           │ 1/solute   │
  │  3 │ Δσ_ρ (Taylor)      │ Work hardening           │ 0 (preset) │
  │  4 │ Δσ_ppt             │ Precipitation            │ 1/system   │
  │  5 │ σ_P (Peierls)      │ BCC low-T hardening      │ 0          │
  │  6 │ τ/σ (α-coeff)      │ Shear/tensile ratio      │ 0 (Cu cal) │
  │  7 │ R_comp             │ Compression ratio        │ 0 (preset) │
  │  8 │ S-N curve           │ Fatigue life             │ 0 (preset) │
  │  9 │ FLC (7-mode)        │ Forming limit            │ 0 (1-pt)   │
  │ 10 │ DBTT                │ Brittle transition       │ 0          │
  │ 11 │ Lindemann           │ Melting point            │ 0          │
  └────┴────────────────────┴──────────────────────────┴────────────┘


💪 STRENGTHENING MECHANISMS — CLI Reference
═══════════════════════════════════════════════════════════════════════════════

  All mechanisms use: python -m delta_theory.unified_yield_fatigue_v10 point

  ┌─────────────────────┬──────────────────────────────┬────────────┐
  │ Mechanism           │ CLI flags                    │ Fitting    │
  ├─────────────────────┼──────────────────────────────┼────────────┤
  │ Solid Solution      │ --c_wt --k_ss               │ 1 (k_ss)   │
  │   (Labusch)         │ --solute_type interstitial   │            │
  │                     │             / substitutional  │            │
  ├─────────────────────┼──────────────────────────────┼────────────┤
  │ Work Hardening      │ --eps --rho_0                │ 0 (preset) │
  │   (Taylor)          │   eps: true strain           │            │
  │                     │   rho_0: initial dislocation │            │
  ├─────────────────────┼──────────────────────────────┼────────────┤
  │ Precipitation       │ --r_ppt_nm --f_ppt           │ 0 (auto)   │
  │   (Cutting/Orowan)  │ --gamma_apb --A_ppt          │            │
  │                     │   auto: r < r_c → Cutting    │            │
  │                     │         r > r_c → Orowan     │            │
  ├─────────────────────┼──────────────────────────────┼────────────┤
  │ Peierls Barrier     │ --enable_peierls             │ 0          │
  │   (BCC low-T)       │ --tau_P0 --dG0               │            │
  └─────────────────────┴──────────────────────────────┴────────────┘

  Output: σ_y = σ_base(SSOC) + Δσ_ss + Δσ_wh + Δσ_ppt + σ_P + HP


📈 S-N CURVE & CALIBRATION — CLI Reference
═══════════════════════════════════════════════════════════════════════════════

  # Generate S-N curve:
  python -m delta_theory.unified_yield_fatigue_v10 sn --metal Fe \\
         --sigma_min 50 --sigma_max 300 --num 25

  # Output columns: σ_a [MPa] | r = σ_a/σ_y | N_fail | D_accum

  # 1-point calibration (fit A_ext from one test):
  python -m delta_theory.unified_yield_fatigue_v10 calibrate --metal Fe \\
         --sigma_a 200 --N_fail 1e5

  Modes: --mode tensile (default) | compression | shear
  Failure: --D_fail 0.5 (default)
  External: --A_ext 2.46e-4 (default, or calibrated value)
  Override: --r_th (threshold) --n_exp (exponent)


📐 FLC MODEL v8.1 — 7-Mode Discrete Formulation
═══════════════════════════════════════════════════════════════════════════════

  Core Equation:
    ε₁,j = |V|_eff × C_j / R_j

  Localization Correction (frozen):
    C_j = 1 + 0.75β_j + 0.48β_j²

  Mixed Resistance:
    R_j = w_σ,j + w_τ,j/(τ/σ) + w_c,j/R_comp

  7 Standard Modes:
    ┌─────────────────┬────────┬────────┐
    │ Mode            │   β    │   C_j  │
    ├─────────────────┼────────┼────────┤
    │ Uniaxial        │ -0.370 │ 0.788  │
    │ Deep Draw       │ -0.306 │ 0.815  │
    │ Draw-Plane      │ -0.169 │ 0.887  │
    │ Plane Strain    │  0.000 │ 1.000  │  ← FLC₀
    │ Plane-Stretch   │ +0.133 │ 1.108  │
    │ Stretch         │ +0.247 │ 1.214  │
    │ Equi-biaxial    │ +0.430 │ 1.411  │
    └─────────────────┴────────┴────────┘

  Calibration: FLC₀ (1 point) → All 7 modes predicted!


📊 VALIDATION
═══════════════════════════════════════════════════════════════════════════════

  Yield (v10.0 SSOC):  25 metals, 3.2% MAE (0 fitting params)
    BCC (7):  Fe, W, V, Cr, Nb, Mo, Ta      → 2.0%
    FCC (10): Cu, Ni, Al, Au, Ag, Pt, Pd... → 2.3%
    HCP (8):  Ti, Mg, Zn, Zr, Hf, Re...    → 6.0%

  Fatigue (v6.10):  2,472 points (5 AM materials), 4-7% error
  FLC (v8.1):       49 points (7 materials × 7 modes), 4.7% MAE


💡 FORMING-FATIGUE (Simple Rule)
═══════════════════════════════════════════════════════════════════════════════

  "曲げたら弱い" — That's it!

    η = ε_formed / ε_FLC        # How much capacity used
    r_th_eff = r_th × (1 - η)   # Remaining fatigue threshold

  幾何的描像:
    成形前: ●──●──●──●  (r₀)
    成形後: ●───●───●───●  (r > r₀, 千切れそうｗ)


👥 AUTHORS
═══════════════════════════════════════════════════════════════════════════════

  Masamichi Iizumi — Miosync, Inc. CEO
  Tamaki — Sentient Digital Partner

  "Nature is Geometry" 🔬
"""


def cmd_flc(args):
    """Quick FLC prediction."""
    from .unified_flc_v8_1 import FLCPredictor, FLC_MATERIALS, MODE_ORDER
    
    if len(args) == 0 or args[0] in ['-h', '--help']:
        print("""
FLC v8.1 Command
================

Usage:
  python -m delta_theory flc <material> [mode]

Arguments:
  material    Material name (Cu, Ti, SPCC, DP590, Al5052, SUS304, Mg_AZ31)
  mode        'all' for all modes, or specific mode name (default: Plane Strain)

Mode names:
  Uniaxial, Deep Draw, Draw-Plane, Plane Strain,
  Plane-Stretch, Stretch, Equi-biaxial

Examples:
  python -m delta_theory flc Cu              # FLC₀ only
  python -m delta_theory flc Cu all          # All 7 modes
  python -m delta_theory flc SPCC Uniaxial   # Specific mode
  python -m delta_theory flc --list          # List materials

""")
        return
    
    if args[0] == '--list':
        print("\nAvailable materials:")
        print("-" * 50)
        for name, mat in FLC_MATERIALS.items():
            print(f"  {name:<10} {mat.structure}  τ/σ={mat.tau_sigma:.3f}  V_eff={mat.V_eff:.4f}")
        return
    
    material = args[0]
    mode = args[1] if len(args) > 1 else 'Plane Strain'
    
    if material not in FLC_MATERIALS:
        print(f"Error: Unknown material '{material}'")
        print(f"Available: {', '.join(FLC_MATERIALS.keys())}")
        return
    
    flc = FLCPredictor()
    
    if mode.lower() == 'all':
        # All 7 modes
        print(f"\n{material} FLC Curve (v8.1)")
        print("=" * 50)
        mat = FLC_MATERIALS[material]
        print(f"Structure: {mat.structure}")
        print(f"τ/σ: {mat.tau_sigma:.4f}")
        print(f"R_comp: {mat.R_comp:.2f}")
        print(f"|V|_eff: {mat.V_eff:.4f}")
        print("-" * 50)
        print(f"{'Mode':<15} {'β':>7} {'C_j':>7} {'ε₁':>8}")
        print("-" * 50)
        for m in MODE_ORDER:
            eps1, bd = flc.predict(material, m, include_breakdown=True)
            print(f"{m:<15} {bd['beta']:>7.3f} {bd['C_j']:>7.4f} {eps1:>8.4f}")
        print("-" * 50)
        print(f"FLC₀ = {flc.flc0(material):.4f}")
    else:
        # Single mode
        if mode not in MODE_ORDER:
            print(f"Error: Unknown mode '{mode}'")
            print(f"Available: {', '.join(MODE_ORDER)}")
            return
        
        eps1 = flc.predict(material, mode)
        print(f"{material} FLC({mode}) = {eps1:.4f}")


def cmd_add_material(args):
    """Add new material from FLC₀."""
    from .unified_flc_v8_1 import FLCPredictor
    
    if len(args) < 3 or args[0] in ['-h', '--help']:
        print("""
Add Material Command
====================

Usage:
  python -m delta_theory add <name> <flc0> <base_element> [T_twin]

Arguments:
  name          New material name
  flc0          FLC₀ value (Plane Strain)
  base_element  Base element (Fe, Cu, Al, Ti, Mg, etc.)
  T_twin        HCP twinning factor 0.0-1.0 (default: 1.0)

Examples:
  python -m delta_theory add MySteel 0.28 Fe
  python -m delta_theory add AZ31 0.265 Mg 0.0
""")
        return
    
    name = args[0]
    flc0 = float(args[1])
    base_element = args[2]
    T_twin = float(args[3]) if len(args) > 3 else 1.0
    
    flc = FLCPredictor()
    mat = flc.add_from_v69(name, flc0=flc0, base_element=base_element, T_twin=T_twin)
    
    print(f"\nAdded: {name}")
    print("-" * 40)
    print(f"  Base element: {base_element}")
    print(f"  Structure: {mat.structure}")
    print(f"  τ/σ: {mat.tau_sigma:.4f}")
    print(f"  R_comp: {mat.R_comp:.2f}")
    print(f"  |V|_eff: {mat.V_eff:.4f} (calibrated from FLC₀={flc0})")
    print()
    print(flc.summary(name))


def cmd_ssoc(args):
    """Quick SSOC f_de inspection."""
    from .material import get_material, MATERIAL_NAMES
    from .ssoc import calc_f_de, calc_f_de_detail, sigma_base_v10
    
    if len(args) == 0 or args[0] in ['-h', '--help']:
        print("""
SSOC Command — f_de Inspection
===============================

Usage:
  python -m delta_theory ssoc <metal>       # Single metal
  python -m delta_theory ssoc all           # All 25 metals

Examples:
  python -m delta_theory ssoc W             # W d⁴ JT anomaly
  python -m delta_theory ssoc Fe            # Fe reference
  python -m delta_theory ssoc all           # Full table
""")
        return
    
    if args[0].lower() == 'all':
        print("\nSSOC f_de — All 25 Metals (T=300K)")
        print("=" * 65)
        print(f"{'Metal':<6} {'Struct':<5} {'f_de':>8} {'σ_base':>10} {'Detail'}")
        print("-" * 65)
        for name in MATERIAL_NAMES:
            mat = get_material(name)
            fde = calc_f_de(mat)
            sigma = sigma_base_v10(mat, T_K=300.0)
            detail = calc_f_de_detail(mat)
            factors = ' × '.join(f"{v:.3f}" for k, v in detail.items() if k != 'f_de')
            print(f"{name:<6} {mat.structure:<5} {fde:>8.4f} {sigma:>10.1f} MPa  {factors}")
        print("-" * 65)
    else:
        name = args[0]
        try:
            mat = get_material(name)
        except (KeyError, ValueError):
            print(f"Error: Unknown metal '{name}'")
            print(f"Available: {', '.join(MATERIAL_NAMES)}")
            return
        
        fde = calc_f_de(mat)
        detail = calc_f_de_detail(mat)
        sigma = sigma_base_v10(mat, T_K=300.0)
        
        print(f"\n{name} ({mat.structure}) — SSOC f_de Breakdown")
        print("=" * 50)
        for k, v in detail.items():
            print(f"  {k:<12} = {v:.4f}")
        print("-" * 50)
        print(f"  σ_base(300K) = {sigma:.1f} MPa")


def main():
    args = sys.argv[1:]
    
    if len(args) == 0:
        print(QUICK_REFERENCE)
        return
    
    cmd = args[0].lower()
    
    if cmd == 'info':
        print(DETAILED_INFO)
    elif cmd == 'flc':
        cmd_flc(args[1:])
    elif cmd == 'add':
        cmd_add_material(args[1:])
    elif cmd == 'ssoc':
        cmd_ssoc(args[1:])
    elif cmd in ['help', '-h', '--help']:
        print(QUICK_REFERENCE)
    else:
        print(f"Unknown command: {cmd}")
        print("Try: python -m delta_theory help")
        print("Commands: info, flc, add, ssoc")


if __name__ == '__main__':
    main()
