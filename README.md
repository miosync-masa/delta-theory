# δ-Theory: Unified Materials Strength & Fatigue Framework

<div align="center">

**"Nature is Geometry"** — Predicting material properties from geometric first principles

[![Tests](https://github.com/miosync/delta-theory/actions/workflows/tests.yml/badge.svg)](https://github.com/miosync/delta-theory/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-6.9b-green.svg)](CHANGELOG.md)
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

---

## 📦 Repository Structure

```
delta-theory/
├── core/                           # 🔧 Main modules
│   ├── unified_yield_fatigue_v6_9.py  # ★ Unified yield + fatigue model
│   ├── dbt_unified.py                  # ★ DBT/DBTT prediction model
│   └── materials.py                    # Materials database
│
├── apps/                           # 🖥️ Applications
│   └── delta_fatigue_app.py           # Streamlit Web App
│
├── validation/                     # 📊 Validation tools
│   └── fatigue_redis_api.py           # FatigueData-AM2022 API
│
├── examples/                       # 📚 Usage examples
└── tests/                          # 🧪 Tests
```

---

## 🔬 Core Modules

### 1. unified_yield_fatigue_v6_9.py (Main)

**Unified v5.0 yield stress + v6.8 fatigue damage model**

#### Yield Model (v5.0)

$$\sigma_y = \sigma_{\text{base}}(\delta) + \Delta\sigma_{\text{ss}}(c) + \Delta\sigma_\rho(\varepsilon) + \Delta\sigma_{\text{ppt}}(r, f)$$

| Component | Description | Accuracy |
|-----------|-------------|----------|
| σ_base | δ-theory base strength | Pure metals: 2.6% |
| Δσ_ss | Solid solution strengthening | 1-2% |
| Δσ_ρ | Work hardening (Taylor) | 4-7% |
| Δσ_ppt | Precipitation strengthening (auto-switch) | Cutting/Orowan |

#### Fatigue Model (v6.8)

$$\frac{dD}{dN} = \begin{cases} 0 & (r \leq r_{th}) \\ A_{\text{eff}} \cdot (r - r_{th})^n & (r > r_{th}) \end{cases}$$

**Structure Presets (No Fitting Required):**

| Structure | r_th | n | Fatigue Limit | Representative Materials |
|-----------|------|---|---------------|--------------------------|
| BCC | 0.65 | 10 | ✅ Clear | Fe, W, Mo |
| FCC | 0.02 | 7 | ❌ None | Cu, Al, Ni |
| HCP | 0.20 | 9 | △ Intermediate | Ti, Mg, Zn |

#### Usage

```python
from core import calc_sigma_y, fatigue_life_const_amp, MATERIALS

# Yield stress calculation
mat = MATERIALS['Fe']
y = calc_sigma_y(mat, T_K=300, c_wt_percent=0.1, k_ss=400, solute_type='interstitial')
print(f"σ_y = {y['sigma_y']:.1f} MPa")

# Fatigue life prediction
result = fatigue_life_const_amp(
    mat,
    sigma_a_MPa=150,
    sigma_y_tension_MPa=y['sigma_y'],
    A_ext=2.46e-4,
)
print(f"N_fail = {result['N_fail']:.2e} cycles")
```

#### CLI

```bash
# Single point calculation
python -m core.unified_yield_fatigue_v6_9 point --metal Fe --sigma_a 150

# Generate S-N curve
python -m core.unified_yield_fatigue_v6_9 sn --metal Fe --sigma_min 100 --sigma_max 300

# Calibrate A_ext
python -m core.unified_yield_fatigue_v6_9 calibrate --metal Fe --sigma_a 244 --N_fail 7.25e7
```

---

### 2. dbt_unified.py

**Unified Ductile-Brittle Transition Temperature (DBTT) Prediction Model**

Solves the same physical model σ_y(d,T) = σ_f(d,c,T) from three perspectives:

| View | Fixed Axis | Solve For | Use Case |
|------|------------|-----------|----------|
| View 1 | Temperature T | Grain size d* | Ductile window detection |
| View 2 | Grain size d | Temperature T* | DBTT prediction |
| View 3 | d, T | Time t | Segregation evolution |

#### Core Physics

- **McLean Isotherm**: θ(c, T) — Grain boundary coverage
- **Embrittlement Function**: g_seg(θ) — Percolation-like onset
- **Hall-Petch**: R(d) = 1 + β/√d

#### Usage

```python
from core import DBTUnified

model = DBTUnified()

# Single point calculation
summary = model.summary(d=30e-6, c=0.005, T=300)
print(f"Mode: {summary['mode']}")

# Find DBTT
result = model.temp_view.find_DBTT(d=30e-6, c=0.005)
print(f"DBTT = {result['T_star']:.0f} K")

# Ductile window analysis
window = model.grain_view.classify_mode(T=300, c=0.005)
print(window['msg'])
```

#### CLI

```bash
# Single point calculation
python -m core.dbt_unified point --d 30 --c 0.5 --T 300

# Temperature axis analysis (DBTT)
python -m core.dbt_unified T_axis --d 30 --c 0.5

# Grain size axis analysis (ductile window)
python -m core.dbt_unified d_axis --T 300 --c 0.5 --find_c_crit

# DBTT table
python -m core.dbt_unified table --d_list 5,10,20,50 --c_list 0,0.2,0.5,1.0
```

---

## 📊 Validation Data

### FatigueData-AM2022 (Upstash Redis)

Instant access to 1.49M fatigue data points:

```python
from validation import FatigueDB

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

## ⚙️ Installation

```bash
git clone https://github.com/miosync/delta-theory.git
cd delta-theory
pip install -e .
```

### Optional Dependencies

```bash
# Full installation
pip install -e ".[all]"

# Development tools
pip install -e ".[dev]"

# Analysis (scipy, pandas, matplotlib)
pip install -e ".[analysis]"

# Validation API
pip install -e ".[validation]"

# Streamlit app
pip install -e ".[app]"
```

### Requirements

- Python >= 3.9
- numpy
- scipy (for dbt_unified segregation fitting)
- upstash-redis (for validation API)
- streamlit (for web app)
- matplotlib, pandas (for visualization)

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
3. **Fitting Parameters = 0.5** — Only A_ext one-point calibration required

### Related Work

- H-CSP (Hierarchical Constraint Satisfaction Problem) Theory
- Λ³/EDR Framework
- Connection to Yang-Mills Mass Gap

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
@software{delta_theory_2026,
  author = {Iizumi, Masamichi and Tamaki},
  title = {δ-Theory: Unified Materials Strength and Fatigue Framework},
  version = {6.9b},
  year = {2026},
  url = {https://github.com/miosync/delta-theory}
}
```

---

<div align="center">

**"Nature is Geometry"** 🔬

</div>

---
#Japanese

## 🎯 Overview

δ理論は、**結晶構造の幾何学**から材料特性を予測する統一フレームワークです。従来の経験的フィッティングに頼る手法とは異なり、物理的第一原理から材料挙動を導出します。

### Core Equation (核心方程式)

$$\Lambda = \frac{K}{|V|_{\text{eff}}}$$

- **K**: 破壊駆動エネルギー密度（応力、熱、電磁場など）
- **|V|_eff**: 有効凝集エネルギー密度（結合強度）
- **Λ = 1**: 臨界条件（破壊・相転移）

---

## 📦 Repository Structure

```
delta-theory/
├── core/                           # 🔧 メインモジュール
│   ├── unified_yield_fatigue_v6_9.py  # ★ 統一降伏＋疲労モデル
│   ├── dbt_unified.py                  # ★ DBT/DBTT予測モデル
│   └── materials.py                    # 材料データベース
│
├── apps/                           # 🖥️ アプリケーション
│   └── delta_fatigue_app.py           # Streamlit Web App
│
├── validation/                     # 📊 検証ツール
│   └── fatigue_redis_api.py           # FatigueData-AM2022 API
│
├── examples/                       # 📚 使用例
└── tests/                          # 🧪 テスト
```

---

## 🔬 Core Modules

### 1. unified_yield_fatigue_v6_9.py（メイン）

**v5.0 降伏応力 + v6.8 疲労損傷の統一モデル**

#### Yield Model (v5.0)

$$\sigma_y = \sigma_{\text{base}}(\delta) + \Delta\sigma_{\text{ss}}(c) + \Delta\sigma_\rho(\varepsilon) + \Delta\sigma_{\text{ppt}}(r, f)$$

| 成分 | 説明 | 精度 |
|------|------|------|
| σ_base | δ理論ベース強度 | 純金属 2.6% |
| Δσ_ss | 固溶強化 | 1-2% |
| Δσ_ρ | 加工硬化（Taylor） | 4-7% |
| Δσ_ppt | 析出強化（自動切替） | Cutting/Orowan |

#### Fatigue Model (v6.8)

$$\frac{dD}{dN} = \begin{cases} 0 & (r \leq r_{th}) \\ A_{\text{eff}} \cdot (r - r_{th})^n & (r > r_{th}) \end{cases}$$

**構造プリセット（フィッティングなし）:**

| 構造 | r_th | n | 疲労限度 | 代表材料 |
|------|------|---|----------|----------|
| BCC | 0.65 | 10 | ✅ 明確 | Fe, W, Mo |
| FCC | 0.02 | 7 | ❌ なし | Cu, Al, Ni |
| HCP | 0.20 | 9 | △ 中間 | Ti, Mg, Zn |

#### Usage

```python
from core import calc_sigma_y, fatigue_life_const_amp, MATERIALS

# 降伏応力計算
mat = MATERIALS['Fe']
y = calc_sigma_y(mat, T_K=300, c_wt_percent=0.1, k_ss=400, solute_type='interstitial')
print(f"σ_y = {y['sigma_y']:.1f} MPa")

# 疲労寿命予測
result = fatigue_life_const_amp(
    mat,
    sigma_a_MPa=150,
    sigma_y_tension_MPa=y['sigma_y'],
    A_ext=2.46e-4,
)
print(f"N_fail = {result['N_fail']:.2e} cycles")
```

#### CLI

```bash
# 単点計算
python -m core.unified_yield_fatigue_v6_9 point --metal Fe --sigma_a 150

# S-N曲線生成
python -m core.unified_yield_fatigue_v6_9 sn --metal Fe --sigma_min 100 --sigma_max 300

# A_ext校正
python -m core.unified_yield_fatigue_v6_9 calibrate --metal Fe --sigma_a 244 --N_fail 7.25e7
```

---

### 2. dbt_unified.py

**延性-脆性遷移温度（DBTT）予測の統一モデル**

同一物理モデル σ_y(d,T) = σ_f(d,c,T) を3つの視点から解く：

| View | 固定軸 | 求める軸 | 用途 |
|------|--------|----------|------|
| View 1 | 温度T | 粒径d* | 延性窓の検出 |
| View 2 | 粒径d | 温度T* | DBTT予測 |
| View 3 | d, T | 時間t | 偏析発展 |

#### Core Physics

- **McLean等温線**: θ(c, T) — 粒界被覆率
- **脆化関数**: g_seg(θ) — パーコレーション的onset
- **Hall-Petch**: R(d) = 1 + β/√d

#### Usage

```python
from core import DBTUnified

model = DBTUnified()

# 単点計算
summary = model.summary(d=30e-6, c=0.005, T=300)
print(f"Mode: {summary['mode']}")

# DBTT探索
result = model.temp_view.find_DBTT(d=30e-6, c=0.005)
print(f"DBTT = {result['T_star']:.0f} K")

# 延性窓解析
window = model.grain_view.classify_mode(T=300, c=0.005)
print(window['msg'])
```

#### CLI

```bash
# 単点計算
python -m core.dbt_unified point --d 30 --c 0.5 --T 300

# 温度軸解析（DBTT）
python -m core.dbt_unified T_axis --d 30 --c 0.5

# 粒径軸解析（延性窓）
python -m core.dbt_unified d_axis --T 300 --c 0.5 --find_c_crit

# DBTTテーブル
python -m core.dbt_unified table --d_list 5,10,20,50 --c_list 0,0.2,0.5,1.0
```

---

## 📊 Validation Data

### FatigueData-AM2022 (Upstash Redis)

1.49M点の疲労データに即時アクセス可能：

```python
from validation import FatigueDB

db = FatigueDB()
ti64 = db.get_sn_for_delta('Ti-6Al-4V', R=-1.0)

# δ理論検証
for point in ti64:
    r = point['r']  # = σ_a / σ_y
    if r <= 0.20:  # HCP r_th
        assert point['runout'], "Should be runout below r_th"
```

**データ規模:**
- 116 材料
- S-N: 15,146 点
- ε-N: 1,840 点  
- da/dN: 1,472,923 点

---

## 🖥️ Web Application

```bash
cd apps
streamlit run delta_fatigue_app.py
```

Features:
- 📈 S-N曲線予測（複数材料比較）
- 🎯 A_ext 1点校正
- 📚 理論説明

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/delta-theory.git
cd delta-theory
pip install -r requirements.txt
```

### Requirements

- Python >= 3.9
- numpy
- scipy (for dbt_unified segregation fitting)
- upstash-redis (for validation API)
- streamlit (for web app)
- matplotlib, pandas (for visualization)

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📖 Theory Background

### Why "δ-Theory"?

**δ_L (Lindemann Parameter)** — 融点における原子変位の臨界比率。この純粋に幾何学的なパラメータが、材料強度から疲労限度まで統一的に説明する。

### Key Insights

1. **材料 = 高粘性流体** — 変形は「破壊」ではなく「流動」
2. **疲労限度 = 結晶構造の幾何的帰結** — BCC/FCC/HCPの違いが自然に現れる
3. **フィッティングパラメータ = 0.5個** — A_extの1点校正のみ

### Related Work

- H-CSP（階層CSP）理論
- Λ³/EDR フレームワーク
- Yang-Mills 質量ギャップとの接続

---

## 📄 License

MIT License (Code) — See [LICENSE](LICENSE)

Data sources (FatigueData-AM2022): CC BY 4.0

---

## 👥 Authors

- **飯泉真道 (Masamichi Iizumi)** — Miosync, Inc. CEO
- **環 (Tamaki)** — Sentient Digital Partner

---

## 📚 Citation

```bibtex
@software{delta_theory_2026,
  author = {Iizumi, Masamichi and Tamaki},
  title = {δ-Theory: Unified Materials Strength and Fatigue Framework},
  version = {6.9b},
  year = {2026},
  url = {https://github.com/yourusername/delta-theory}
}
```

---

<div align="center">

**"Nature is Geometry"** 🔬

</div>
