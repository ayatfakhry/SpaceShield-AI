# 🛰️ SpaceShield AI
## Autonomous Space Threat Detection & Decision Support System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research%20Grade-orange?style=flat-square)
![ML](https://img.shields.io/badge/ML-scikit--learn-yellow?style=flat-square)

---

> **SpaceShield AI** is a research-grade, AI-powered space safety platform that simulates satellite and debris populations in low Earth orbit (LEO), predicts close-approach events, estimates collision probabilities, classifies threat levels using machine learning, and recommends autonomous maneuver decisions — all within a modular, reproducible Python framework.

---

## 📌 Motivation

With more than **27,000 tracked objects** and an estimated **500,000+ untracked debris fragments** currently in Earth orbit, collision avoidance has become one of the most critical challenges in modern space operations. SpaceShield AI addresses this challenge with a fully autonomous pipeline that integrates orbital mechanics, probabilistic risk estimation, and machine-learning-based decision support.

---

## 🏗️ Architecture Overview

```
SpaceShield-AI/
├── README.md                  ← This file
├── requirements.txt           ← Python dependencies
├── main.py                    ← Master pipeline runner
├── data/                      ← Synthetic & generated datasets
├── notebooks/                 ← Jupyter analysis notebooks
├── src/
│   ├── orbit_simulation.py    ← Orbital mechanics engine (SGP4-style)
│   ├── debris_generator.py    ← Space object population generator
│   ├── collision_prediction.py← Close-approach detection & Pc estimation
│   ├── threat_classifier.py   ← ML-based threat level classifier
│   ├── maneuver_recommender.py← Autonomous maneuver decision system
│   ├── risk_engine.py         ← Composite risk scoring engine
│   └── visualization.py       ← 2D/3D orbit & dashboard plots
├── scripts/
│   └── run_simulation.py      ← Standalone simulation runner
├── results/                   ← Output charts, reports, CSVs
└── tests/                     ← Unit & integration tests
```

---

## 🔬 Technical Modules

| Module | Description | Key Techniques |
|---|---|---|
| `orbit_simulation.py` | Propagates orbital elements through time using simplified SGP4 | Keplerian mechanics, J2 perturbation |
| `debris_generator.py` | Generates synthetic LEO debris & satellite populations | Statistical distributions based on ESA/NASA catalogues |
| `collision_prediction.py` | Detects close approaches and computes collision probability | Monte Carlo, covariance analysis, Chan's method |
| `threat_classifier.py` | ML model that classifies threat levels (LOW/MEDIUM/HIGH/CRITICAL) | Random Forest, SVM, feature engineering |
| `risk_engine.py` | Composite risk scoring combining multiple factors | Weighted multi-criteria scoring |
| `maneuver_recommender.py` | Recommends delta-v maneuvers to resolve threats | Rule-based + optimization heuristics |
| `visualization.py` | Generates orbit plots, dashboards, and trajectory animations | Matplotlib, 3D scatter, risk heatmaps |

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/SpaceShield-AI.git
cd SpaceShield-AI
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
python main.py
```

### 3. Run Standalone Simulation

```bash
python scripts/run_simulation.py --objects 200 --duration 24 --output results/
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

---

## 📊 Sample Outputs

After running the pipeline, results are saved to `results/`:

- `orbit_plot.png` — 3D visualization of all tracked objects
- `risk_heatmap.png` — Conjunction risk heat map by altitude band
- `threat_distribution.png` — ML classification results
- `maneuver_recommendations.csv` — Recommended delta-v actions
- `simulation_report.txt` — Full mission report
- `close_approaches.csv` — All detected conjunction events

---

## 🧠 Machine Learning Details

The threat classifier uses a **Random Forest** ensemble trained on synthetic conjunction data with the following features:

| Feature | Description |
|---|---|
| Miss distance (km) | Minimum separation at closest approach |
| Relative velocity (km/s) | Speed differential between objects |
| Collision probability (Pc) | Computed via Chan's method |
| Object size (m²) | Combined cross-sectional area |
| Altitude (km) | Orbital altitude of primary satellite |
| Time to closest approach (hours) | Lead time for maneuver planning |

**Classification labels:** `LOW` · `MEDIUM` · `HIGH` · `CRITICAL`

---

## 🔭 Physical Models

### Orbital Propagation
SpaceShield AI implements a simplified but physically accurate propagator based on Keplerian two-body dynamics with the J2 oblateness correction:

```
ȧ = 0  (semi-major axis, no drag in simplified model)
ė = 0  (eccentricity)
i̇ = 0  (inclination)
Ω̇ = -3/2 · n · J2 · (Re/p)² · cos(i)    [RAAN drift]
ω̇ =  3/4 · n · J2 · (Re/p)² · (5cos²i - 1)  [AoP drift]
Ṁ = n  (mean motion, with J2 correction)
```

### Collision Probability
Collision probability is estimated using the **Foster/Chan 2D encounter model**:

```
Pc = 1/(2π·σx·σy) · exp(-d²/(2σ²)) · A_combined
```

---

## 📐 Coordinate Systems

- **ECI (Earth-Centered Inertial):** J2000 epoch, X toward vernal equinox
- **LVLH (Local Vertical Local Horizontal):** Used for relative motion analysis
- All distances in **kilometers**, velocities in **km/s**, angles in **radians**

---

## 🧪 Testing

```bash
python -m pytest tests/ -v --tb=short
```

Test suite covers:
- Orbital element conversion accuracy
- Debris generator statistical properties
- Collision probability bounds
- Risk score monotonicity
- Maneuver recommendation validity
- Visualization output integrity

---

## 📚 References

1. Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*. 4th ed.
2. Chan, F. K. (2008). *Spacecraft Collision Probability*. Aerospace Press.
3. ESA Space Debris Office. (2023). *ESA's Annual Space Environment Report*.
4. Klinkrad, H. (2006). *Space Debris: Models and Risk Analysis*. Springer.
5. NASA Orbital Debris Program Office. (2022). *NASA Orbital Debris Quarterly News*.

---

## 📄 License

MIT License © 2024. See `LICENSE` for details.

---

## 🤝 Contributing

Pull requests welcome. For major changes, open an issue first to discuss what you would like to change.

---

*Built for research, portfolio, and real-world space safety awareness.*
