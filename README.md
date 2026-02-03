# δ-Theory: Unified Materials Strength & Fatigue Framework

<div align="center">

**"Nature is Geometry"** — Predicting material properties from geometric first principles

[![Tests](https://github.com/miosync/delta-theory/actions/workflows/tests.yml/badge.svg)](https://github.com/miosync-masa/delta-theory/blob/main/.github/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/delta-theory.svg)](https://pypi.org/project/delta-theory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-8.0.0-green.svg)](CHANGELOG.md)
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
| **v5.0** | Yield stress σ_y | f_d, E_bond, crystal geometry | 2.6% |
| **v6.10** | Fatigue life N | r_th (BCC=0.65, FCC=0.02, HCP=0.20) | 4-7% |
| **v7.2** | Forming Limit Curve FLC(β) | Free volume consumption | 2.7% |
| **v8.0** | Post-forming fatigue life | η_forming → r_th_eff | — |
| **DBT** | Ductile-Brittle Transition | Grain size, segregation | — |

---

## 📦 Installation

```bash
pip install delta-theory
```

### From Source

```bash
git clone https://github.com/miosync/delta-theory.git
cd delta-theory
pip install -e .
```

---

## 🚀 Quick Start

### Yield Stress

```python
from core import calc_sigma_y, MATERIALS

mat = MATERIALS['Fe']
result = calc_sigma_y(mat, T_K=300)
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
print(f"N_fail = {result['N_fail']:.2e} cycles")
```

### FLC Prediction (NEW in v8.0!)

```python
from core import FLCPredictor

flc = FLCPredictor()
for beta in [-0.5, 0.0, 1.0]:
    Em = flc.predict(beta, 'SPCC')
    print(f"β={beta:+.1f}: FLC = {Em:.3f}")
```

### Forming-Fatigue Integration (NEW in v8.0!)

```python
from core import FormingFatigueIntegrator

integrator = FormingFatigueIntegrator()

# After 40% free volume consumption from forming:
r_th_eff = integrator.effective_r_th(eta_forming=0.40, structure='BCC')
print(f"r_th: 0.65 → {r_th_eff:.3f}")  # Fatigue threshold drops!
```

---

## 📦 Repository Structure

```
delta-theory/
├── core/                              # 🔧 Main modules
│   ├── unified_yield_fatigue_v6_9.py     # Unified yield + fatigue model
│   ├── unified_flc_v7.py                 # ★ FLC + Forming-Fatigue (NEW!)
│   ├── dbt_unified.py                    # DBT/DBTT prediction model
│   ├── materials.py                      # Materials database
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

| Structure | r_th | n | Fatigue Limit | Representative Materials |
|-----------|------|---|---------------|--------------------------|
| BCC | 0.65 | 10 | ✅ Clear | Fe, W, Mo |
| FCC | 0.02 | 7 | ❌ None | Cu, Al, Ni |
| HCP | 0.20 | 9 | △ Intermediate | Ti, Mg, Zn |

---

### 2. unified_flc_v7.py (NEW in v8.0!)

**FLC Prediction + Forming-Fatigue Integration**

#### FLC Model (v7.2)

$$\text{FLC}(\beta) = \text{FLC}_0^{\text{pure}} \times (1 - \eta_{\text{total}}) \times h(\beta, R, \tau/\sigma)$$

| Parameter | Description |
|-----------|-------------|
| FLC₀_pure | Pure metal formability from δ-theory |
| η_total | Free volume consumed by strengthening mechanisms |
| h(β) | V-shape factor from multiaxial stress state |
| R | Compression/tension ratio (twin effect for HCP) |
| τ/σ | Shear/tension ratio |

**Free Volume Consumption:**

```
η_total = η_ss + η_ppt + η_wh + η_HP
        = k_ss×C_ss + k_ppt×f_ppt + k_wh×log(ρ/ρ_ref) + k_HP×(√(d_ref/d)-1)
```

**Why SPCC vs DP590 have different FLC:**

| Material | Free Volume Remaining | FLC₀ |
|----------|----------------------|------|
| SPCC | 90.6% | 0.25 |
| DP590 | 71.4% | 0.20 |

#### Forming-Fatigue Integration (v8.0)

**Revolutionary insight:** Forming consumes free volume → Less available for fatigue!

$$r_{th}^{\text{eff}} = r_{th}^{\text{virgin}} \times (1 - \eta_{\text{forming}})$$

| η_forming | r_th_eff (BCC) | Implication |
|-----------|----------------|-------------|
| 0% | 0.65 | Virgin material |
| 20% | 0.52 | Some forming |
| 40% | 0.39 | Heavy forming |
| 60% | 0.26 | Severe forming |

**Critical η:** The forming level where "infinite life" becomes "finite life"

```python
from core import critical_forming_consumption

eta_crit = critical_forming_consumption(r_applied=0.50, structure='BCC')
print(f"η_critical = {eta_crit*100:.1f}%")  # → 23.1%
```

---

### 3. dbt_unified.py

**Unified Ductile-Brittle Transition Temperature (DBTT) Prediction Model**

| View | Fixed Axis | Solve For | Use Case |
|------|------------|-----------|----------|
| View 1 | Temperature T | Grain size d* | Ductile window detection |
| View 2 | Grain size d | Temperature T* | DBTT prediction |
| View 3 | d, T | Time t | Segregation evolution |

```python
from core import DBTUnified

model = DBTUnified()
result = model.temp_view.find_DBTT(d=30e-6, c=0.005)
print(f"DBTT = {result['T_star']:.0f} K")
```

---

## ⌨️ CLI Reference

### Yield & Fatigue

```bash
# Single point calculation
python -m core.unified_yield_fatigue_v6_9 point --metal Fe --sigma_a 150

# Generate S-N curve
python -m core.unified_yield_fatigue_v6_9 sn --metal Fe --sigma_min 100 --sigma_max 300

# Calibrate A_ext
python -m core.unified_yield_fatigue_v6_9 calibrate --metal Fe --sigma_a 244 --N_fail 7.25e7
```

### FLC (NEW!)

```bash
# Single material FLC
python3 -c "
from core import FLCPredictor
flc = FLCPredictor()
for b in [-0.5, -0.25, 0, 0.25, 0.5, 1.0]:
    print(f'β={b:+.2f}: {flc.predict(b, \"SPCC\"):.3f}')
"

# Quick FLC value
python3 -c "from core import predict_flc; print(predict_flc('SPCC', 0.0))"

# Forming-fatigue analysis
python3 -c "
from core import FormingFatigueIntegrator
integrator = FormingFatigueIntegrator()
for eta in [0.0, 0.2, 0.4, 0.6]:
    r_th = integrator.effective_r_th(eta, 'BCC')
    print(f'η={eta:.0%}: r_th_eff = {r_th:.3f}')
"

# Critical η calculation
python3 -c "
from core import critical_forming_consumption
for r in [0.3, 0.4, 0.5, 0.6]:
    eta = critical_forming_consumption(r, 'BCC')
    print(f'r={r:.1f}: η_critical = {eta*100:.1f}%')
"
```

### DBT

```bash
# Single point calculation
python -m core.dbt_unified point --d 30 --c 0.5 --T 300

# Temperature axis analysis (DBTT)
python -m core.dbt_unified T_axis --d 30 --c 0.5

# Grain size axis analysis (ductile window)
python -m core.dbt_unified d_axis --T 300 --c 0.5 --find_c_crit
```

---

## 📊 Validation Data

### FatigueData-AM2022 (Upstash Redis)

Instant access to 1.49M fatigue data points:

```python
from core import FatigueDB

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
3. **Free Volume = Shared Resource** — Strength, ductility, and fatigue compete for the same "余白"
4. **Fitting Parameters ≈ 0** — Derived from crystal geometry, not curve fitting

### Version History

| Version | Feature |
|---------|---------|
| v5.0 | Yield stress from δ-theory |
| v6.9b | Unified yield + fatigue with multiaxial |
| v6.10 | Universal fatigue validation (2472 points) |
| v7.2 | FLC from free volume consumption |
| **v8.0** | **Forming-Fatigue integration** |

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
  version = {8.0.0},
  year = {2025},
  url = {https://github.com/miosync/delta-theory},
  doi = {10.5281/zenodo.18457897}
}
```

---

<div align="center">

**"Nature is Geometry"** 🔬

*From yield stress to fatigue life to forming limits — all from crystal structure*

</div>
