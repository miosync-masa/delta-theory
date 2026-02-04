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
                    δ-Theory v8.1.0
                  "Nature is Geometry"
                  
╔══════════════════════════════════════════════════════════════════════════════╗
║  δ-Theory v8.1.0 — CLI Quick Reference                                       ║
║  "Nature is Geometry"                                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INSTALLATION OK! ✓                                                          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🔧 YIELD STRESS (v6.9b)                                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    from delta_theory import calc_sigma_y, MATERIALS                          ║
║    result = calc_sigma_y(MATERIALS['Fe'], T_K=300)                           ║
║    print(f"σ_y = {result['sigma_y']:.1f} MPa")                               ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🔄 FATIGUE LIFE (v6.9b)                                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    from delta_theory import fatigue_life_const_amp, MATERIALS                ║
║    result = fatigue_life_const_amp(MATERIALS['Fe'], sigma_a_MPa=150,         ║
║                                    sigma_y_tension_MPa=200, A_ext=2.5e-4)    ║
║    print(f"N = {result['N_fail']:.2e} cycles")                               ║
║                                                                              ║
║    # CLI:                                                                    ║
║    python -m delta_theory.unified_yield_fatigue_v6_9 point --metal Fe \\     ║
║           --sigma_a 150 --sigma_y_override 200                               ║
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
╠══════════════════════════════════════════════════════════════════════════════╣
║  📚 MORE INFO                                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    python -m delta_theory info              # Detailed module info           ║
║    python -m delta_theory flc --help        # FLC command help               ║
║                                                                              ║
║    Docs: https://github.com/miosync/delta-theory                             ║
║    PyPI: https://pypi.org/project/delta-theory/                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

DETAILED_INFO = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  δ-Theory v8.1.0 — Detailed Module Information                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

📦 INSTALLED MODULES
═══════════════════════════════════════════════════════════════════════════════

  delta_theory/
  ├── unified_yield_fatigue_v6_9.py   # Yield + Fatigue (v6.9b)
  ├── unified_flc_v8_1.py             # FLC 7-mode discrete (v8.1)
  ├── dbt_unified.py                  # Ductile-Brittle Transition
  ├── materials.py                    # Material database
  ├── banners.py                      # ASCII art banners
  └── fatigue_redis_api.py            # FatigueData-AM2022 API (optional)


🔬 THEORY SUMMARY
═══════════════════════════════════════════════════════════════════════════════

  Core Equation:  Λ = K / |V|_eff

  K       = Destructive energy (stress, thermal, EM...)
  |V|_eff = Cohesive energy (bond strength)
  Λ = 1   → Critical condition (fracture/transition)


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


🔗 v6.9 INTEGRATION
═══════════════════════════════════════════════════════════════════════════════

  τ/σ and R_comp from δ-theory v6.9:

    # Add new material with v6.9 parameters
    flc = FLCPredictor()
    flc.add_from_v69('MySteel', flc0=0.28, base_element='Fe')
    flc.add_from_v69('MgAlloy', flc0=0.25, base_element='Mg', T_twin=0.0)

  HCP T_twin interpolation:
    T_twin=0.0 → twin-dominated (Mg: τ/σ=0.327, R_comp=0.60)
    T_twin=1.0 → slip-dominated (τ/σ=0.565, R_comp=1.00)


📊 VALIDATION
═══════════════════════════════════════════════════════════════════════════════

  Yield (v6.9b):    10 pure metals, 2.6% mean error
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
    elif cmd in ['help', '-h', '--help']:
        print(QUICK_REFERENCE)
    else:
        print(f"Unknown command: {cmd}")
        print("Try: python -m delta_theory help")


if __name__ == '__main__':
    main()
