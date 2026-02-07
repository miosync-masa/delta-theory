# δ-Theory: Unified Materials Strength & Fatigue Framework

<div align="center">

**"Nature is Geometry"** — Predicting material properties from geometric first principles

[![PyPI](https://img.shields.io/pypi/v/delta-theory.svg)](https://pypi.org/project/delta-theory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-8.2.0-green.svg)](CHANGELOG.md)
[![codecov](https://codecov.io/gh/miosync/delta-theory/branch/main/graph/badge.svg)](https://codecov.io/gh/miosync/delta-theory)

</div>

---

## 🎯 Overview

δ-Theory is a unified framework that predicts material properties from **crystal structure geometry**. Unlike traditional empirical fitting approaches, it derives material behavior from physical first principles.

### Core Equation

$$\Lambda = \frac{K}{|V|_{\text{eff}}}$$

- **K**: Destructive energy density (stress, thermal, electromagnetic, etc.)
- **|V|_eff**: Effective cohesive energy density (bond strength)
- **Λ = 1**: Critical condition (fracture / phase transition)

### What Can δ-Theory Predict?

| Module | Predicts | Key Parameters | Accuracy |
|--------|----------|----------------|----------|
| **v6.9b** | Yield stress σ_y | f_d, E_bond, crystal geometry | 2.6% |
| **v6.10** | Fatigue life N | r_th (BCC=0.65, FCC=0.02, HCP=0.20) | 4-7% |
| **v8.1** | Forming Limit Curve FLC | 7-mode discrete, 1-point calibration | 4.7% |
| **DBT** | Ductile-Brittle Transition | Grain size, segregation | — |

---

## 📦 Installation
```bash
pip install delta-theory
```

### From Source
```bash
git clone https://github.com/miosync-inc/delta-theory.git
cd delta-theory
pip install -e .
```

---

## 🚀 Quick Start

### Yield Stress
```python
from delta_theory import calc_sigma_y, MATERIALS

mat = MATERIALS['Fe']
result = calc_sigma_y(mat, T_K=300)
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
print(f"N_fail = {result['N_fail']:.2e} cycles")
```

### FLC Prediction (v8.1 — 7-Mode Discrete!)
```python
from delta_theory import FLCPredictor, predict_flc

# Quick prediction
eps1 = predict_flc('Cu', 'Plane Strain')  # → 0.346

# Full usage
flc = FLCPredictor()
for mode in ['Uniaxial', 'Plane Strain', 'Equi-biaxial']:
    print(f"{mode}: {flc.predict('Cu', mode):.3f}")

# Add new material from v6.9 parameters
flc.add_from_v69('MySteel', flc0=0.28, base_element='Fe')
betas, eps1s = flc.predict_curve('MySteel')  # All 7 modes!
```

---

## 📦 Repository Structure
```
delta-theory/
├── delta_theory/                      # 🔧 Main package
│   ├── unified_yield_fatigue_v6_9.py     # Unified yield + fatigue model
│   ├── unified_flc_v8_1.py               # ★ FLC 7-mode discrete (NEW!)
│   ├── dbt_unified.py                    # DBT/DBTT prediction model
│   ├── material.py                      # Materials database
│   ├── banners.py                        # ASCII art banners
│   └── fatigue_redis_api.py              # FatigueData-AM2022 API
│
├── apps/                              # 🖥️ Applications
│   └── delta_fatigue_app.py              # Streamlit Web App
│
├── examples/                          # 📚 Usage examples
└── tests/                             # 🧪 Tests
```

---

## 🔬 Core Modules

### 1. unified_yield_fatigue_v6_9.py

**Unified v5.0 yield stress + v6.8 fatigue damage model**

#### Yield Model (v5.0)

$$\sigma_y = \sigma_{\text{base}}(\delta) + \Delta\sigma_{\text{ss}}(c) + \Delta\sigma_\rho(\varepsilon) + \Delta\sigma_{\text{ppt}}(r, f)$$

| Component | Description | Accuracy |
|-----------|-------------|----------|
| σ_base | δ-theory base strength | Pure metals: 2.6% |
| Δσ_ss | Solid solution strengthening | 1-2% |
| Δσ_ρ | Work hardening (Taylor) | 4-7% |
| Δσ_ppt | Precipitation strengthening (auto-switch) | Cutting/Orowan |

#### Fatigue Model (v6.10)

$$\frac{dD}{dN} = \begin{cases} 0 & (r \leq r_{th}) \\ A_{\text{eff}} \cdot (r - r_{th})^n & (r > r_{th}) \end{cases}$$

**Structure Presets (No Fitting Required):**

| Structure | r_th | n | τ/σ | R_comp | Fatigue Limit |
|-----------|------|---|-----|--------|---------------|
| BCC | 0.65 | 10 | 0.565 | 1.0 | ✅ Clear |
| FCC | 0.02 | 7 | 0.565 | 1.0 | ❌ None |
| HCP | 0.20 | 9 | 0.327* | 0.6* | △ Intermediate |

*HCP values depend on T_twin (twinning factor)

---

### 2. unified_flc_v8_1.py (NEW in v8.2!)

**FLC Prediction — 7-Mode Discrete Formulation**

#### Core Equation

$$\varepsilon_{1,j} = |V|_{\text{eff}} \times \frac{C_j}{R_j}$$

| Component | Formula | Description |
|-----------|---------|-------------|
| \|V\|_eff | Calibrated from FLC₀ | Material forming capacity |
| C_j | 1 + 0.75β + 0.48β² | Localization correction (frozen) |
| R_j | w_σ + w_τ/(τ/σ) + w_c/R_comp | Mixed resistance |

#### 7 Standard Forming Modes

| Mode | β | C_j | Physical Meaning |
|------|---|-----|------------------|
| Uniaxial | -0.370 | 0.788 | Deep drawing (tension + compression) |
| Deep Draw | -0.306 | 0.815 | Drawing dominant |
| Draw-Plane | -0.169 | 0.887 | Transition region |
| **Plane Strain** | **0.000** | **1.000** | **FLC₀ reference** |
| Plane-Stretch | +0.133 | 1.108 | Transition to biaxial |
| Stretch | +0.247 | 1.214 | Stretching dominant |
| Equi-biaxial | +0.430 | 1.411 | Balanced biaxial tension |

#### 1-Point Calibration

Measure only **FLC₀** (Plane Strain) → Predict **all 7 modes** automatically!
```python
from delta_theory import FLCPredictor

flc = FLCPredictor()

# Add material with just FLC₀ + base element
flc.add_from_v69('MySteel', flc0=0.28, base_element='Fe')

# Get full FLC curve
results = flc.predict_all_modes('MySteel')
for mode, eps1 in results.items():
    print(f"{mode}: {eps1:.3f}")
```

#### v6.9 Integration

τ/σ and R_comp are automatically retrieved from δ-theory based on crystal structure:
```python
# BCC steel — uses τ/σ = 0.565, R_comp = 1.0
flc.add_from_v69('SPCC', flc0=0.225, base_element='Fe')

# HCP magnesium (twin-dominated) — uses τ/σ = 0.327, R_comp = 0.6
flc.add_from_v69('AZ31', flc0=0.265, base_element='Mg', T_twin=0.0)

# HCP titanium (slip-dominated) — uses τ/σ = 0.546, R_comp = 1.0
flc.add_from_v69('Ti64', flc0=0.30, base_element='Ti', T_twin=1.0)
```

#### HCP T_twin Interpolation

| T_twin | τ/σ | R_comp | Behavior |
|--------|-----|--------|----------|
| 0.0 | 0.327 | 0.60 | Twin-dominated (Mg-like) |
| 0.5 | 0.446 | 0.80 | Mixed |
| 1.0 | 0.565 | 1.00 | Slip-dominated |

#### Built-in Materials

| Material | Structure | τ/σ | R_comp | \|V\|_eff | FLC₀ |
|----------|-----------|-----|--------|-----------|------|
| Cu | FCC | 0.565 | 1.00 | 1.224 | 0.346 |
| Ti | HCP | 0.546 | 1.00 | 1.039 | 0.293 |
| SPCC | BCC | 0.565 | 1.00 | 0.802 | 0.225 |
| DP590 | BCC | 0.565 | 1.00 | 0.691 | 0.194 |
| Al5052 | FCC | 0.565 | 1.00 | 0.619 | 0.165 |
| SUS304 | FCC | 0.565 | 1.00 | 1.423 | 0.400 |
| Mg_AZ31 | HCP | 0.327 | 0.60 | 1.180 | 0.265 |

---

### 3. dbt_unified.py

**Unified Ductile-Brittle Transition Temperature (DBTT) Prediction Model**

| View | Fixed Axis | Solve For | Use Case |
|------|------------|-----------|----------|
| View 1 | Temperature T | Grain size d* | Ductile window detection |
| View 2 | Grain size d | Temperature T* | DBTT prediction |
| View 3 | d, T | Time t | Segregation evolution |
```python
from delta_theory import DBTUnified

model = DBTUnified()
result = model.temp_view.find_DBTT(d=30e-6, c=0.005)
print(f"DBTT = {result['T_star']:.0f} K")
```

---

## ⌨️ CLI Reference

### FLC Commands (NEW!)
```bash
# Quick FLC₀ prediction
python -m delta_theory flc Cu

# All 7 modes
python -m delta_theory flc SPCC all

# Specific mode
python -m delta_theory flc Cu Uniaxial

# List available materials
python -m delta_theory flc --list

# Detailed info
python -m delta_theory info
```

### Yield & Fatigue
```bash
# Single point calculation
python -m delta_theory.unified_yield_fatigue_v6_9 point --metal Fe --sigma_a 150

# Generate S-N curve
python -m delta_theory.unified_yield_fatigue_v6_9 sn --metal Fe --sigma_min 100 --sigma_max 300

# Calibrate A_ext
python -m delta_theory.unified_yield_fatigue_v6_9 calibrate --metal Fe --sigma_a 244 --N_fail 7.25e7
```

### DBT
```bash
# Single point calculation
python -m delta_theory.dbt_unified point --d 30 --c 0.5 --T 300

# Temperature axis analysis (DBTT)
python -m delta_theory.dbt_unified T_axis --d 30 --c 0.5

# Grain size axis analysis (ductile window)
python -m delta_theory.dbt_unified d_axis --T 300 --c 0.5 --find_c_crit
```

---

## 📊 Validation Data

### FLC v8.1 Validation

| Material | Structure | MAE | Data Points |
|----------|-----------|-----|-------------|
| Cu | FCC | 3.4% | 7 |
| Ti | HCP | 4.8% | 7 |
| SPCC | BCC | 4.2% | 7 |
| Al5052 | FCC | 6.8% | 7 |
| SUS304 | FCC | 3.9% | 7 |
| DP590 | BCC | 4.2% | 7 |
| Mg_AZ31 | HCP | 5.6% | 7 |
| **Overall** | — | **4.7%** | **49** |

### FatigueData-AM2022 (Upstash Redis)

Instant access to 1.49M fatigue data points:
```python
from delta_theory import FatigueDB

db = FatigueDB()
ti64 = db.get_sn_for_delta('Ti-6Al-4V', R=-1.0)

# δ-theory validation
for point in ti64:
    r = point['r']  # = σ_a / σ_y
    if r <= 0.20:  # HCP r_th
        assert point['runout'], "Should be runout below r_th"
```

**Data Scale:**
- 116 materials
- S-N: 15,146 points
- ε-N: 1,840 points  
- da/dN: 1,472,923 points

---

## 🖥️ Web Application
```bash
cd apps
streamlit run delta_fatigue_app.py
```

Features:
- 📈 S-N curve prediction (multi-material comparison)
- 🎯 A_ext one-point calibration
- 📚 Theory explanation

---

## 🧪 Testing
```bash
pytest tests/ -v
```

---

## 📖 Theory Background

### Why "δ-Theory"?

**δ_L (Lindemann Parameter)** — The critical ratio of atomic displacement at melting point. This purely geometric parameter unifies explanations from material strength to fatigue limits.

### Key Insights

1. **Materials = Highly Viscous Fluids** — Deformation is "flow", not "fracture"
2. **Fatigue Limits = Geometric Consequence of Crystal Structure** — BCC/FCC/HCP differences emerge naturally
3. **Forming Limit = Geometry + Localization** — C_j captures strain path, R_j captures crystal resistance
4. **Fitting Parameters ≈ 0** — Derived from crystal geometry, not curve fitting

### Version History

| Version | Feature |
|---------|---------|
| v5.0 | Yield stress from δ-theory |
| v6.9b | Unified yield + fatigue with multiaxial (τ/σ, R_comp) |
| v6.10 | Universal fatigue validation (2472 points) |
| v7.2 | FLC from free volume consumption |
| **v8.1** | **FLC 7-mode discrete formulation** |
| **v8.2** | **v6.9 integration + CLI commands** |

---

## 💡 Forming-Fatigue (Simple Rule)

> **"Forming makes it weak"** — Stretched lattice = Nearly broken bonds
```
Before: ●──●──●──●  (r₀)
After:  ●───●───●───●  (r > r₀, about to break!)
```

Simple formula:
```python
eta = eps_formed / eps_FLC      # How much capacity used
r_th_eff = r_th * (1 - eta)     # Remaining fatigue threshold
```

---

## 📄 License

MIT License (Code) — See [LICENSE](LICENSE)

Data sources (FatigueData-AM2022): CC BY 4.0

---

## 👥 Authors

- **Masamichi Iizumi** — Miosync, Inc. CEO
- **Tamaki** — Sentient Digital Partner

---

## 📚 Citation
```bibtex
@software{delta_theory_2025,
  author = {Iizumi, Masamichi and Tamaki},
  title = {δ-Theory: Unified Materials Strength, Fatigue, and Forming Framework},
  version = {8.2.0},
  year = {2025},
  url = {https://github.com/miosync-inc/delta-theory},
  doi = {10.5281/zenodo.18457897}
}
```

---

<div align="center">

**"Nature is Geometry"** 🔬

*From yield stress to fatigue life to forming limits — all from crystal structure*

</div>
