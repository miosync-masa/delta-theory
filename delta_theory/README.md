```
    ███████╗██╗   ██╗██████╗ ███████╗██╗  ██╗ █████╗ ██╗
    ██╔════╝██║   ██║██╔══██╗██╔════╝██║ ██╔╝██╔══██╗██║
    █████╗  ██║   ██║██████╔╝█████╗  █████╔╝ ███████║██║
    ██╔══╝  ██║   ██║██╔══██╗██╔══╝  ██╔═██╗ ██╔══██╗╚═╝
    ███████╗╚██████╔╝██║  ██║███████╗██║  ██╗██║  ██║██╗
    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝
                      δ-Theory v10.0.0
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
| **FLC v8.1** | ε₁,j = \|V\|_eff × C_j / R_j | Λ = 1 |

---

## 🎯 Structure Presets (No Fitting!)

| Structure | r_th | n | τ/σ | R_comp | Fatigue Limit |
|-----------|------|---|-----|--------|---------------|
| **BCC** | 0.65 | 10 | 0.565 | 1.0 | ✅ Clear |
| **FCC** | 0.02 | 7 | 0.565 | 1.0 | ❌ None |
| **HCP** | 0.20 | 9 | 0.327* | 0.6* | △ Weak |

*HCP values depend on T_twin (twinning factor)

---

## 📐 FLC v8.1: 7-Mode Discrete Formulation

### Core Equation
```
ε₁,j = |V|_eff × C_j / R_j
```

- **C_j** = 1 + 0.75β + 0.48β² (localization, frozen)
- **R_j** = w_σ + w_τ/(τ/σ) + w_c/R_comp (mixed resistance)

### 7 Standard Modes

| Mode | β | C_j | Description |
|------|---|-----|-------------|
| Uniaxial | -0.370 | 0.788 | Deep drawing |
| Deep Draw | -0.306 | 0.815 | Drawing dominant |
| Draw-Plane | -0.169 | 0.887 | Transition |
| **Plane Strain** | **0.000** | **1.000** | **FLC₀ (reference)** |
| Plane-Stretch | +0.133 | 1.108 | Transition |
| Stretch | +0.247 | 1.214 | Stretching dominant |
| Equi-biaxial | +0.430 | 1.411 | Balanced biaxial |

### 1-Point Calibration
Measure FLC₀ only → Predict all 7 modes!

---

## ⌨️ CLI Examples

### Yield Stress
```python
from delta_theory import calc_sigma_y, MATERIALS
result = calc_sigma_y(MATERIALS['Fe'], T_K=300)
print(f"σ_y = {result['sigma_y']:.1f} MPa")
```

### Fatigue Life
```python
from delta_theory import fatigue_life_const_amp, MATERIALS
result = fatigue_life_const_amp(
    MATERIALS['Fe'],
    sigma_a_MPa=150,
    sigma_y_tension_MPa=200,
    A_ext=2.5e-4,
)
print(f"N = {result['N_fail']:.2e}")
```

### FLC Prediction (v8.1)
```python
from delta_theory import FLCPredictor, predict_flc

# Quick prediction
eps1 = predict_flc('Cu', 'Plane Strain')  # → 0.346

# Full usage
flc = FLCPredictor()
flc.add_from_v69('MySteel', flc0=0.28, base_element='Fe')
eps1 = flc.predict('MySteel', 'Uniaxial')

# All 7 modes
for mode in ['Uniaxial', 'Plane Strain', 'Equi-biaxial']:
    print(f"{mode}: {flc.predict('Cu', mode):.3f}")
```

Output:
```
Uniaxial: 0.533
Plane Strain: 0.346
Equi-biaxial: 0.538
```

### FLC Curve
```python
flc = FLCPredictor()
betas, eps1s = flc.predict_curve('SPCC')

# Or get all modes as dict
results = flc.predict_all_modes('SPCC')
```

### HCP with T_twin
```python
# Twin-dominated Mg (τ/σ = 0.327)
flc.add_from_v69('AZ31', flc0=0.265, base_element='Mg', T_twin=0.0)

# Slip-dominated Ti (τ/σ = 0.565)
flc.add_from_v69('Ti64', flc0=0.30, base_element='Ti', T_twin=1.0)
```

### DBTT Prediction
```python
from delta_theory import DBTUnified
model = DBTUnified()
result = model.temp_view.find_DBTT(d=30e-6, c=0.005)
print(f"DBTT = {result['T_star']:.0f} K")
```

---

## 🖥️ CLI Commands
```bash
# FLC prediction
python -m delta_theory flc Cu              # FLC₀ only
python -m delta_theory flc SPCC all        # All 7 modes
python -m delta_theory flc SPCC Uniaxial   # Specific mode
python -m delta_theory flc --list          # Available materials

# Info
python -m delta_theory info
```

---

## 📊 Material Database

### Built-in FLC Materials (v8.1)

| Material | Structure | τ/σ | R_comp | \|V\|_eff | FLC₀ |
|----------|-----------|-----|--------|-----------|------|
| Cu | FCC | 0.565 | 1.00 | 1.224 | 0.346 |
| Ti | HCP | 0.546 | 1.00 | 1.039 | 0.293 |
| SPCC | BCC | 0.565 | 1.00 | 0.802 | 0.225 |
| DP590 | BCC | 0.565 | 1.00 | 0.691 | 0.194 |
| Al5052 | FCC | 0.565 | 1.00 | 0.619 | 0.165 |
| SUS304 | FCC | 0.565 | 1.00 | 1.423 | 0.400 |
| Mg_AZ31 | HCP | 0.327 | 0.60 | 1.180 | 0.265 |

### Built-in Yield/Fatigue Materials
```python
from delta_theory import MATERIALS
print(list(MATERIALS.keys()))
# ['Fe', 'W', 'Cu', 'Al', 'Ni', 'Au', 'Ag', 'Ti', 'Mg', 'Zn']
```

---

## 🔥 Key Insights

> **"Forming makes it weak"**
>
> Stretched lattice = Nearly broken bonds
> ```
> Before: ●──●──●──●  (r₀)
> After:  ●───●───●───●  (r > r₀, about to break!)
> ```
> Simple rule: `r_th_eff = r_th × (1 - ε/ε_FLC)`

> **FLC v8.1: Geometry Determines Formability**
>
> - C_j: Localization depends on strain path (β)
> - R_j: Resistance depends on crystal (τ/σ, R_comp)
> - |V|_eff: Material capacity (1-point calibration)

---

## 📊 Validation

| Model | Data Points | Error |
|-------|-------------|-------|
| Yield (v6.9b) | 10 pure metals | 2.6% MAE |
| Fatigue (v6.10) | 2,472 points | 4-7% |
| FLC (v8.1) | 49 points (7×7) | 4.7% MAE |

---

## 📚 Citation
```bibtex
@software{delta_theory_2025,
  author = {Iizumi, Masamichi and Tamaki},
  title = {δ-Theory: Unified Materials Framework},
  version = {8.2.0},
  year = {2025},
  doi = {10.5281/zenodo.18457897}
}
```

---

<div align="center">

**"Nature is Geometry"** 🔬

Masamichi Iizumi & Tamaki

</div>
