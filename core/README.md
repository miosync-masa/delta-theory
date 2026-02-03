```
    ███████╗██╗   ██╗██████╗ ███████╗██╗  ██╗ █████╗ ██╗
    ██╔════╝██║   ██║██╔══██╗██╔════╝██║ ██╔╝██╔══██╗██║
    █████╗  ██║   ██║██████╔╝█████╗  █████╔╝ ███████║██║
    ██╔══╝  ██║   ██║██╔══██╗██╔══╝  ██╔═██╗ ██╔══██║╚═╝
    ███████╗╚██████╔╝██║  ██║███████╗██║  ██╗██║  ██║██╗
    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝
                      δ-Theory v8.0.0
                    "Nature is Geometry"
```

# δ-Theory Quick Reference

---

## 📦 Installation

```bash
pip install delta-theory
```

---

## 🔬 Core Equations

| Domain | Equation | Critical |
|--------|----------|----------|
| **Universal** | Λ = K / \|V\|_eff | Λ = 1 |
| **Yield** | σ_y = σ_base(δ) + Δσ_ss + Δσ_wh + Δσ_ppt | — |
| **Fatigue** | dD/dN = A_eff(r - r_th)^n | r > r_th |
| **FLC** | FLC(β) = FLC₀ × (1-η) × h(β,R,τ/σ) | Λ = 1 |
| **Forming→Fatigue** | r_th_eff = r_th × (1 - η_forming) | — |

---

## 🎯 Structure Presets (No Fitting!)

| Structure | r_th | n | τ/σ | R | Fatigue Limit |
|-----------|------|---|-----|---|---------------|
| **BCC** | 0.65 | 10 | 0.577 | 1.0 | ✅ Clear |
| **FCC** | 0.02 | 7 | 0.577 | 1.0 | ❌ None |
| **HCP** | 0.20 | 9 | 0.327 | 0.6 | △ Weak |

---

## ⌨️ CLI Examples

### Yield Stress

```python
from core import calc_sigma_y, MATERIALS
result = calc_sigma_y(MATERIALS['Fe'], T_K=300)
print(f"σ_y = {result['sigma_y']:.1f} MPa")
```

### Fatigue Life

```python
from core import fatigue_life_const_amp, MATERIALS
result = fatigue_life_const_amp(
    MATERIALS['Fe'],
    sigma_a_MPa=150,
    sigma_y_tension_MPa=200,
)
print(f"N = {result['N_fail']:.2e}")
```

### FLC Prediction

```python
from core import FLCPredictor
flc = FLCPredictor()
for b in [-0.5, 0, 0.5, 1.0]:
    print(f"β={b:+.1f}: {flc.predict(b, 'SPCC'):.3f}")
```

Output:
```
β=-0.5: 0.383
β=+0.0: 0.251
β=+0.5: 0.200
β=+1.0: 0.184
```

### Forming-Fatigue Integration

```python
from core import FormingFatigueIntegrator, critical_forming_consumption

# Effective r_th after forming
integrator = FormingFatigueIntegrator()
r_th_eff = integrator.effective_r_th(eta_forming=0.40, structure='BCC')
print(f"r_th: 0.65 → {r_th_eff:.3f}")  # → 0.390

# Critical η for given load ratio
eta_crit = critical_forming_consumption(r_applied=0.50, structure='BCC')
print(f"η_critical = {eta_crit*100:.1f}%")  # → 23.1%
```

### Full Forming Analysis

```python
from core import DeltaFormingAnalyzer
analyzer = DeltaFormingAnalyzer()
result = analyzer.full_analysis(
    material='SECD-E16',
    epsilon_major=0.25,
    beta=0.0,
    r_applied=0.50
)
print(f"Λ = {result['Lambda']:.3f}")
print(f"r_th_eff = {result['r_th_eff']:.3f}")
print(f"Safe? {result['overall_safe']}")
```

### DBTT Prediction

```python
from core import DBTUnified
model = DBTUnified()
result = model.temp_view.find_DBTT(d=30e-6, c=0.005)
print(f"DBTT = {result['T_star']:.0f} K")
```

---

## 📊 Material Database

### Built-in (FLC)

| Material | Structure | σ_y (MPa) | FLC₀ |
|----------|-----------|-----------|------|
| SPCC | BCC | 180 | 0.25 |
| DP590 | BCC | 400 | 0.20 |
| SECD-E16 | BCC | 300 | 0.22 |
| Al | FCC | 30 | 0.30 |
| SUS304 | FCC | 290 | 0.28 |
| Ti | HCP | 350 | 0.24 |
| Mg_AZ31 | HCP | 200 | 0.16 |

### Built-in (Yield/Fatigue)

```python
from core import MATERIALS
print(list(MATERIALS.keys()))
# ['Fe', 'W', 'Cu', 'Al', 'Ni', 'Au', 'Ag', 'Ti', 'Mg', 'Zn', 'Zr', 'Co', 'Nb', 'Mo', 'Ta']
```

---

## 🔥 Key Insight

> **Free Volume (余白) = Finite Shared Resource**
>
> - Strengthening mechanisms consume it → Higher σ_y, lower ductility
> - Forming consumes it → Lower fatigue threshold
> - Same physics, unified framework!

---

## 📚 Citation

```bibtex
@software{delta_theory_2025,
  author = {Iizumi, Masamichi and Tamaki},
  title = {δ-Theory: Unified Materials Framework},
  version = {8.0.0},
  year = {2025},
  doi = {10.5281/zenodo.18457897}
}
```

---

<div align="center">

**"Nature is Geometry"** 🔬

</div>
