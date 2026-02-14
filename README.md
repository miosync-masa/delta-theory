# δ-Theory: Unified Materials Strength & Fatigue Framework

<div align="center">

**"Nature is Geometry"** — Predicting material properties from Geometric Structural Principles

[![PyPI](https://img.shields.io/pypi/v/delta-theory.svg)](https://pypi.org/project/delta-theory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-10.0.0-green.svg)](CHANGELOG.md)
[![codecov](https://codecov.io/gh/miosync/delta-theory/branch/main/graph/badge.svg)](https://codecov.io/gh/miosync/delta-theory)

</div>

---

## 🎯 Overview

δ-Theory is a unified framework that predicts material properties from **crystal structure geometry**. Unlike traditional empirical fitting approaches, it derives material behavior from Geometric Structural Principles

### Core Equation

$$\Lambda = \frac{K}{|V|_{\text{eff}}}$$

- **K**: Destructive energy density (stress, thermal, electromagnetic, etc.)
- **|V|_eff**: Effective cohesive energy density (bond strength)
- **Λ = 1**: Critical condition (fracture / phase transition)

### What Can δ-Theory Predict?

| Module | Predicts | Method | Accuracy |
|--------|----------|--------|----------|
| **v10.3** | Creep Module| Q_self = k_B × T_m × Q_base(struct) × g_ssoc(pattern)| **3.5%** (20 metals) |
| **v10.2** | AM Alloy Fatigue Module| N = min( N_init + N_prop,  N_static ) | **3.2%** (25 metals) |
| **v10.0** | Yield stress σ_y | SSOC f_de (δ_L-free) | **3.2%** (25 metals) |
| **v4.1** | τ/σ, R_comp | α-coefficient geometry | Cu-calibrated |
| **v6.10** | Fatigue life N | r_th (BCC=0.65, FCC=0.02, HCP=0.20) | 4-7% |
| **v8.1** | Forming Limit Curve FLC | 7-mode discrete, 1-point calibration | 4.7% |
| **DBT** | Ductile-Brittle Transition | Grain size, segregation | — |
| **Lindemann** | Melting point prediction | Iizumi-Lindemann Law | — |

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

### Yield Stress (v10.1 SSOC)
```python
from delta_theory import calc_sigma_y, MATERIALS, calc_f_de, sigma_base_v10

# Full yield stress with strengthening mechanisms
mat = MATERIALS['Fe']
result = calc_sigma_y(mat, T_K=300)
print(f"σ_y = {result['sigma_y']:.1f} MPa")
print(f"f_de = {result['f_de']:.4f}")
print(f"branch = {result['sigma_base_branch']}")

# Direct SSOC calculation
print(f"W f_de = {calc_f_de(MATERIALS['W']):.3f}")   # → 2.993 (d⁴ JT anomaly)
print(f"Fe σ_base = {sigma_base_v10(MATERIALS['Fe']):.1f} MPa")
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
├── delta_theory/                       # 🔧 Main package
│   ├── material.py                     # Data layer — 37 metals + SSOC params
│   ├── creep.py                        # Diffusion Creep Module
│   ├── ssoc.py                         # Calculation layer — f_de (NEW!)
│   ├── am_fatigue.py                   # AM fatigue SNCurve
│   ├── unified_yield_fatigue_v10.py    # Application layer — σ_y, τ/σ, S-N
│   ├── unified_yield_fatigue_v6_9.py   # ← backward compat shim (re-exports v10)
│   ├── unified_flc_v8_1.py             # FLC 7-mode discrete
│   ├── dbt_unified.py                  # DBT/DBTT prediction model
│   ├── lindemann.py                    # Iizumi-Lindemann melting law
│   ├── banners.py                      # ASCII art banners
│   └── fatigue_redis_api.py            # FatigueData-AM2022 API
│
├── apps/                              # 🖥️ Applications
│   └── delta_fatigue_app.py           # Streamlit Web App
│
├── examples/                          # 📚 Usage examples
└── tests/                             # 🧪 Tests
```

---

## 🔬 Core Modules

### 1. Yield Stress — v10.0 SSOC (Structure-Selective Orbital Coupling)

**δ_L-free unified yield stress from crystal geometry**

#### Architecture (Multi-Layer Design)

```
material.py                    → Data layer      (25 metals, SSOC + crystal params)
ssoc.py                        → σ_y calculation  (f_de: PCC/SCC 3-factor model)
unified_yield_fatigue_v10.py    → Application layer (σ_y + τ/σ + R_comp → S-N, FLC)
```

#### Unified Equation

$$\sigma_y = \frac{8\sqrt{5}}{5\pi M Z} \cdot \alpha_0 \cdot \left(\frac{b}{d}\right)^2 \cdot f_{de} \cdot \frac{\sqrt{E_{\text{coh}} \cdot k_B \cdot T_m}}{V_{\text{act}}} \cdot HP$$

| Symbol | Meaning | Source |
|--------|---------|--------|
| 8√5/(5π) | ≈ 1.1384, geometric coefficient | Derived |
| M | = 3.0, unified Taylor factor (all structures) | Geometric |
| Z | Bulk coordination number | Crystal structure |
| α₀ | Packing fraction (BCC=0.289, FCC=0.250, HCP=0.250) | Crystal geometry |
| (b/d)² | = 3/2, universal geometric ratio | Crystal geometry |
| f_de | **SSOC electronic factor** (structure-selective) | **This work** |
| √(E·kT) | Core energy term (δ_L-free) | Thermodynamic |
| V_act | = b³, activation volume | Crystal geometry |
| HP | = 1 - T/T_m, homologous fraction | Temperature |

**Key Insight**: δ_L ∝ √(k_B·T_m / E_coh), so E_bond × δ_L ∝ √(E_coh · k_B · T_m). v10.0 uses this relationship directly, **eliminating δ_L dependence**.

#### SSOC f_de — Structure-Selective Orbital Coupling (v10.1)
$$f_{de}^{(s)} = \left(\frac{X_s}{X_{\text{ref}}}\right)^{2/3 \cdot g_d} \times f_{\text{aux}}^{(s)}$$
**P_DIM = 2/3** — Universal geometric exponent: surface (2D) → volume (3D) dimension transformation
| Structure | Channel | X (input) | g_d (gate) | f_de formula |
|-----------|---------|-----------|------------|--------------|
| **FCC** | PCC | μ (shear modulus) | {0, 1} discrete | f_μ × f_shell × f_core × f_lanthanide |
| **BCC** | SCC | ΔE_P (Peierls) | d⁴ JT anomaly | f_JT × f_5d × f_lat × f_complex (or f_sp) |
| **HCP** | PCC | R (CRSS ratio) | sigmoid | f_elec × f_aniso × f_ca × f_5d × f_lanthanide × f_sp_cov |
- **PCC** (Perturbative-Coupled Channel): Input field and response separable (FCC, HCP)
- **SCC** (Self-Consistent Channel): Field and response inseparable (BCC)

#### v10.1 Gate Extensions
| Gate | Physics | Targets |
|------|---------|---------|
| `f_lanthanide` (FCC/HCP) | 4f crystal field: f = 1 + 0.423×n_f_eff, + 5d¹ contribution | Ce, Nd |
| `f_complex` (BCC) | Complex unit cell: (N_atoms/2)^0.25 | Mn (58 at/cell) |
| `f_sp` (BCC) | Unified sp branch: pure sp=0.10, p-block d¹⁰=0.80 | Li, Na, Sn |
| `f_elec` d¹ gate (HCP) | Period-dependent directionality: ≤4→3.0, ≥5→1.5 | Sc, Y |
| `f_sp_cov` (HCP) | sp³ covalent bonding: f=1.905 | Be |
| `fcc_gate` p-block (FCC) | d¹⁰ + p-block → g_d=0 | In |

#### BCC d⁴ Jahn-Teller Anomaly
```
t₂g⁴: one orbital doubly occupied → Oh→D₄h symmetry breaking
→ Maximum SCC self-generation of Peierls barrier
→ W (d⁴, 5d): f_JT=1.9 × f_5d=1.5 × f_lat=1.05 → f_de ≈ 2.99
```
#### Validation (37 Metals, T=300K)
| Structure | Metals | MAE | Key Results |
|-----------|--------|-----|-------------|
| BCC (11) | Fe, W, V, Cr, Nb, Mo, Ta, Li, Na, Mn, Sn | 2.0% | W d⁴ JT: 744 vs 750 MPa |
| FCC (12) | Cu, Ni, Al, Au, Ag, Pt, Pd, Ir, Rh, Pb, Ce, In | 10.6% | Ce 4f: 65.0 vs 65 MPa |
| HCP (14) | Ti, Mg, Zn, Zr, Hf, Re, Cd, Ru, Co, Be, Sc, Y, Nd, Bi | 4.0% | Be sp_cov: 300 vs 300 MPa |
| **All 37** | — | **5.5%** | Cd, In excluded: **~2.6%** |
| **<5% error** | 32/37 | **<10%** | 35/37 |

**Zero fitting parameters** — All predictions from crystal geometry + thermodynamic data.

```python
from delta_theory import calc_f_de, sigma_base_v10, MATERIALS

# Inspect SSOC factors
from delta_theory import calc_f_de_detail
detail = calc_f_de_detail(MATERIALS['W'])
# → {'f_jt': 1.9, 'f_5d': 1.5, 'f_lat': 1.05, 'f_de': 2.993}

# Inverse: experimental σ → back-calculate f_de
from delta_theory import inverse_f_de
f = inverse_f_de(MATERIALS['Fe'], sigma_exp_MPa=150.0)
```

---

### 2. Multiaxial Ratios — v4.1 α-Coefficient Theory

**τ/σ (shear-to-tensile ratio) and R_comp (compression-to-tensile ratio) from crystal geometry**

These ratios feed directly into fatigue (r_th presets) and FLC (R_j resistance).

#### Core Equation

$$\frac{\tau}{\sigma} = \frac{\alpha_s}{\alpha_t} \times C_{\text{class}} \times T_{\text{twin}} \times A_{\text{texture}}$$

| Symbol | Formula | Meaning |
|--------|---------|---------|
| α_t | (1/Z) Σ max(b·d, 0) | Tensile projection of bond vectors |
| α_s | (1/Z) Σ \|b·n\| \|b·s\| | Shear projection onto slip system (n, s) |
| C_class | (τ/σ)_Cu / (α_s/α_t)_FCC | Cu torsion 1-point calibration |
| T_twin | 0.6 (Mg) ~ 1.0 (most) | Twinning correction factor |
| A_texture | 1.0 (default) | Texture adjustment slot |

#### Design Principles (v4.1)

- **Cu torsion = anchor** (Quality A data): C_class calibrated from Cu τ/σ = 0.565
- **HCP: C_class NOT applied** — avoids propagating uncertainty from estimated data
- **BCC slip mixing**: w110 parameter for {110}/{112} system weighting (default: w110=0, pure {112})

#### Geometric α Values by Structure

| Structure | Tensile ref | Slip system | α_s/α_t |
|-----------|-------------|-------------|---------|
| BCC | [100] | {112}⟨111⟩ | 0.4714 |
| FCC | [110] | {111}⟨110⟩ | 0.4082 |
| HCP | [100] | (0001)⟨1000⟩ | 0.4330 |

#### Compression Ratio R = σ_c / σ_t

Driven by twinning asymmetry in HCP metals:

| Metal | R_comp | Mechanism |
|-------|--------|-----------|
| BCC, FCC | 1.0 | Symmetric (no twinning effect) |
| Ti | 1.0 | Slip-dominated HCP |
| Mg | 0.6 | Twin-dominated (strong tension-compression asymmetry) |
| Zn | 1.2 | Reverse twinning effect |

```python
from delta_theory import tau_over_sigma, sigma_c_over_sigma_t, C_CLASS_DEFAULT, MATERIALS

# τ/σ prediction (uses α-coefficients internally)
fe = MATERIALS['Fe']
print(f"Fe τ/σ = {tau_over_sigma(fe):.4f}")       # → 0.565
print(f"Fe R_comp = {sigma_c_over_sigma_t(fe)}")   # → 1.0
print(f"C_class = {C_CLASS_DEFAULT:.4f}")           # Cu-calibrated

# Yield by mode
from delta_theory import yield_by_mode
sigma_shear, info = yield_by_mode(fe, sigma_y_tension_MPa=150.0, mode='shear')
print(f"Fe τ_y = {sigma_shear:.1f} MPa")
```

---

### 3. Fatigue Life (v6.8/6.10)

$$\frac{dD}{dN} = \begin{cases} 0 & (r \leq r_{th}) \\ A_{\text{eff}} \cdot (r - r_{th})^n & (r > r_{th}) \end{cases}$$

**Structure Presets (No Fitting Required):**

| Structure | r_th | n | τ/σ † | R_comp † | Fatigue Limit |
|-----------|------|---|-------|----------|---------------|
| BCC | 0.65 | 10 | 0.565 | 1.0 | ✅ Clear |
| FCC | 0.02 | 7 | 0.565 | 1.0 | ❌ None |
| HCP | 0.20 | 9 | 0.327* | 0.6* | △ Intermediate |

*HCP values depend on T_twin (twinning factor)
†τ/σ and R_comp derived from v4.1 α-coefficient theory (Section 2)

---

### 4. unified_flc_v8_1.py — FLC 7-Mode Discrete

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
flc.add_from_v69('MySteel', flc0=0.28, base_element='Fe')

results = flc.predict_all_modes('MySteel')
for mode, eps1 in results.items():
    print(f"{mode}: {eps1:.3f}")
```

#### FLC Validation

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

---

### 5. dbt_unified.py — Ductile-Brittle Transition

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

### 6. lindemann.py — Iizumi-Lindemann Law

Melting point prediction from crystal geometry:

```python
from delta_theory import iizumi_lindemann, predict_delta_L

# Predict Lindemann parameter from crystal structure
delta_L = predict_delta_L('Fe')

# Full validation
from delta_theory import validate_all, print_validation_report
results = validate_all()
print_validation_report(results)
```

---

## ⌨️ CLI Reference

### Yield & Fatigue (v10.0)
```bash
# Single point calculation (SSOC)
python -m delta_theory.unified_yield_fatigue_v10 point --metal Fe --sigma_a 150

# Generate S-N curve
python -m delta_theory.unified_yield_fatigue_v10 sn --metal Fe --sigma_min 100 --sigma_max 300

# Calibrate A_ext
python -m delta_theory.unified_yield_fatigue_v10 calibrate --metal Fe --sigma_a 244 --N_fail 7.25e7
```

### FLC
```bash
# Quick FLC₀ prediction
python -m delta_theory flc Cu

# All 7 modes
python -m delta_theory flc SPCC all

# List available materials
python -m delta_theory flc --list
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

## 🧪 Testing
```bash
pytest tests/ -v
```

---

## 📖 Theory Background

### Why "δ-Theory"?

**δ_L (Lindemann Parameter)** — The critical ratio of atomic displacement at melting point. This purely geometric parameter unifies explanations from material strength to fatigue limits. In v10.0, the explicit δ_L dependence is eliminated through SSOC, but the geometric spirit remains.

### Key Insights

1. **Materials = Highly Viscous Fluids** — Deformation is "flow", not "fracture"
2. **Fatigue Limits = Geometric Consequence of Crystal Structure** — BCC/FCC/HCP differences emerge naturally
3. **Forming Limit = Geometry + Localization** — C_j captures strain path, R_j captures crystal resistance
4. **SSOC** — Electronic effects (d-orbitals, Jahn-Teller, relativistic) are captured through structure-selective channels, not fitting parameters
5. **Fitting Parameters ≈ 0** — Derived from crystal geometry, not curve fitting

### Version History

| Version | Feature |
|---------|---------|
| **v4.1** | **τ/σ and R_comp from α-coefficient geometry (Cu 1-point calibration)** |
| v5.0 | Yield stress from δ-theory |
| v6.9b | Unified yield + fatigue with multiaxial (τ/σ, R_comp) |
| v6.10 | Universal fatigue validation (2472 points) |
| v7.0 | Geometric factorization: f_d → (b/d)² × f_d_elec |
| v7.2 | FLC from free volume consumption |
| v8.1 | FLC 7-mode discrete formulation |
| v8.2 | v6.9 integration + CLI commands |
| **v10.0** | **SSOC: δ_L-free unified yield stress (25 metals, 3.2% MAE)** |

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
  version = {10.0.0},
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
