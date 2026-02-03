#!/usr/bin/env python3
"""
δ-Theory CLI Entry Point

Usage:
    python -m core              # Show quick reference
    python -m core info         # Show detailed info
    python -m core flc SPCC     # Quick FLC prediction
    python -m core fatigue Fe 150  # Quick fatigue life
"""

import sys

QUICK_REFERENCE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  δ-Theory v8.0.0 — CLI Quick Reference                                       ║
║  "Nature is Geometry"                                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INSTALLATION OK! ✓                                                          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🔧 YIELD STRESS                                                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    from core import calc_sigma_y, MATERIALS                                  ║
║    result = calc_sigma_y(MATERIALS['Fe'], T_K=300)                           ║
║    print(f"σ_y = {result['sigma_y']:.1f} MPa")                               ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🔄 FATIGUE LIFE                                                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    from core import fatigue_life_const_amp, MATERIALS                        ║
║    result = fatigue_life_const_amp(MATERIALS['Fe'], sigma_a_MPa=150,         ║
║                                    sigma_y_tension_MPa=200)                  ║
║    print(f"N = {result['N_fail']:.2e} cycles")                               ║
║                                                                              ║
║    # CLI:                                                                    ║
║    python -m core.unified_yield_fatigue_v6_9 point --metal Fe --sigma_a 150  ║
║    python -m core.unified_yield_fatigue_v6_9 sn --metal Fe                   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📐 FLC (Forming Limit Curve) — NEW in v8.0!                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    from core import FLCPredictor                                             ║
║    flc = FLCPredictor()                                                      ║
║    Em = flc.predict(beta=0.0, material='SPCC')  # → 0.251                    ║
║                                                                              ║
║    # Full curve:                                                             ║
║    for b in [-0.5, 0, 1.0]:                                                  ║
║        print(f"β={b:+.1f}: {flc.predict(b, 'SPCC'):.3f}")                    ║
║                                                                              ║
║    # Available materials:                                                    ║
║    SPCC, DP590, Al, SUS304, Ti, Mg_AZ31, SECD-E16, Cu                        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🔗 FORMING-FATIGUE INTEGRATION — NEW in v8.0!                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    from core import FormingFatigueIntegrator                                 ║
║    integrator = FormingFatigueIntegrator()                                   ║
║                                                                              ║
║    # Effective fatigue threshold after forming:                              ║
║    r_th_eff = integrator.effective_r_th(eta_forming=0.4, structure='BCC')    ║
║    # Virgin: 0.65 → After 40% forming: 0.39                                  ║
║                                                                              ║
║    # Critical forming consumption:                                           ║
║    from core import critical_forming_consumption                             ║
║    eta_crit = critical_forming_consumption(r_applied=0.5, structure='BCC')   ║
║    # → 23.1% (beyond this, infinite life becomes finite!)                    ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🌡️ DBT (Ductile-Brittle Transition)                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    from core import DBTUnified                                               ║
║    model = DBTUnified()                                                      ║
║    result = model.temp_view.find_DBTT(d=30e-6, c=0.005)                      ║
║    print(f"DBTT = {result['T_star']:.0f} K")                                 ║
║                                                                              ║
║    # CLI:                                                                    ║
║    python -m core.dbt_unified point --d 30 --c 0.5 --T 300                   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📊 FATIGUE THRESHOLDS (r_th)                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    Structure │ r_th  │ Fatigue Limit │ Examples                              ║
║    ──────────┼───────┼───────────────┼─────────────                          ║
║    BCC       │ 0.65  │ ✓ Clear       │ Fe, W, Mo, SPCC, DP590                ║
║    FCC       │ 0.02  │ ✗ None        │ Cu, Al, Ni, SUS304                    ║
║    HCP       │ 0.20  │ △ Weak        │ Ti, Mg, Zn                            ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📚 MORE INFO                                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    python -m core info              # Detailed module info                   ║
║    python -m core flc SPCC          # Quick FLC for material                 ║
║    python -m core flc SPCC -0.5     # FLC at specific β                      ║
║                                                                              ║
║    Docs: https://github.com/miosync/delta-theory                             ║
║    PyPI: https://pypi.org/project/delta-theory/                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

DETAILED_INFO = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  δ-Theory v8.0.0 — Detailed Module Information                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

📦 INSTALLED MODULES
═══════════════════════════════════════════════════════════════════════════════

  core/
  ├── unified_yield_fatigue_v6_9.py   # Yield + Fatigue (v6.9b)
  ├── unified_flc_v7.py               # FLC + Forming-Fatigue (v7.2/v8.0)
  ├── dbt_unified.py                  # Ductile-Brittle Transition
  ├── materials.py                    # Material database
  └── fatigue_redis_api.py            # FatigueData-AM2022 API


🔬 THEORY SUMMARY
═══════════════════════════════════════════════════════════════════════════════

  Core Equation:  Λ = K / |V|_eff

  K       = Destructive energy (stress, thermal, EM...)
  |V|_eff = Cohesive energy (bond strength)
  Λ = 1   → Critical condition (fracture/transition)


📐 FLC MODEL (v7.2)
═══════════════════════════════════════════════════════════════════════════════

  FLC(β) = FLC₀_pure × (1 - η_total) × h(β, R, τ/σ)

  η_total = Free volume consumption:
    - η_ss:  Solid solution
    - η_ppt: Precipitate/martensite  
    - η_wh:  Work hardening (dislocations)
    - η_HP:  Hall-Petch (grain refinement)

  Example: SPCC (90.6% FV) vs DP590 (71.4% FV)
           Same crystal, different formability!


🔗 FORMING-FATIGUE (v8.0)
═══════════════════════════════════════════════════════════════════════════════

  r_th_eff = r_th_virgin × (1 - η_forming)

  "How much fatigue life did you lose when you pressed that part?"

  η_forming │ r_th_eff (BCC) │ Status
  ──────────┼────────────────┼────────────────
     0%     │     0.65       │ Virgin
    20%     │     0.52       │ Light forming
    40%     │     0.39       │ Heavy forming
    60%     │     0.26       │ Severe forming

  Critical η: r=0.5 → η_crit=23.1%
  (Beyond this, "infinite life" becomes "finite life"!)


📊 VALIDATION
═══════════════════════════════════════════════════════════════════════════════

  Yield (v5.0):     10 pure metals, 2.6% mean error
  Fatigue (v6.10):  2,472 points (5 AM materials), 4-7% error
  FLC (v7.2):       36 points (6 materials), 2.7% error


👥 AUTHORS
═══════════════════════════════════════════════════════════════════════════════

  Masamichi Iizumi — Miosync, Inc. CEO
  Tamaki — Sentient Digital Partner

  "Nature is Geometry" 🔬
"""


def cmd_flc(args):
    """Quick FLC prediction."""
    from .unified_flc_v7 import FLCPredictor, FLC_MATERIALS
    
    if len(args) == 0:
        print("Available materials:", ", ".join(FLC_MATERIALS.keys()))
        return
    
    material = args[0]
    beta = float(args[1]) if len(args) > 1 else None
    
    flc = FLCPredictor()
    
    if beta is not None:
        Em = flc.predict(beta, material)
        print(f"{material} FLC(β={beta:+.2f}) = {Em:.3f}")
    else:
        print(f"\n{material} FLC Curve:")
        print("-" * 25)
        for b in [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]:
            Em = flc.predict(b, material)
            print(f"  β={b:+5.2f}: {Em:.3f}")


def cmd_eta(args):
    """Critical η calculation."""
    from .unified_flc_v7 import FormingFatigueIntegrator
    
    if len(args) < 1:
        print("Usage: python -m core eta <r_applied> [structure]")
        print("Example: python -m core eta 0.5 BCC")
        return
    
    r_applied = float(args[0])
    structure = args[1] if len(args) > 1 else 'BCC'
    
    integrator = FormingFatigueIntegrator()
    eta_crit = integrator.critical_eta(r_applied, structure)
    
    print(f"\nCritical η for {structure} at r = {r_applied:.2f}")
    print("-" * 40)
    print(f"  η_critical = {eta_crit*100:.1f}%")
    print(f"  → Beyond this, infinite life becomes finite!")


def cmd_rth(args):
    """Effective r_th after forming."""
    from .unified_flc_v7 import FormingFatigueIntegrator
    
    if len(args) < 1:
        print("Usage: python -m core rth <eta_forming> [structure]")
        print("Example: python -m core rth 0.4 BCC")
        return
    
    eta = float(args[0])
    structure = args[1] if len(args) > 1 else 'BCC'
    
    integrator = FormingFatigueIntegrator()
    r_th_eff = integrator.effective_r_th(eta, structure)
    r_th_virgin = {'BCC': 0.65, 'FCC': 0.02, 'HCP': 0.20}[structure]
    
    print(f"\nEffective r_th for {structure} after η = {eta:.0%} forming")
    print("-" * 45)
    print(f"  Virgin r_th:    {r_th_virgin:.3f}")
    print(f"  Effective r_th: {r_th_eff:.3f}")
    print(f"  Reduction:      {(1 - r_th_eff/r_th_virgin)*100:.1f}%")


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
    elif cmd == 'eta':
        cmd_eta(args[1:])
    elif cmd == 'rth':
        cmd_rth(args[1:])
    elif cmd in ['help', '-h', '--help']:
        print(QUICK_REFERENCE)
    else:
        print(f"Unknown command: {cmd}")
        print("Try: python -m core help")


if __name__ == '__main__':
    main()
