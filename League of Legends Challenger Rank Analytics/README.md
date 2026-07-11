# 🎮 League of Legends: Challenger Match Analytics & Win Prediction

An end-to-end data science project on **300 Challenger-tier League of Legends matches** (~311K in-game events), streamed from a 10-million-event public dataset. The project covers the full lifecycle: **streaming ingestion → feature engineering → win prediction → model interpretation → unsupervised player profiling → anomaly detection → champion-synergy mining**, with a strong emphasis on *methodological rigor* (data-leakage auditing, distribution-shift/patch robustness, ablation studies, and honest documentation of data-quality issues).

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange" />
  <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green" />
  <img src="https://img.shields.io/badge/SHAP-Interpretability-red" />
  <img src="https://img.shields.io/badge/Plotly%20%7C%20NetworkX-Viz-lightgrey" />
</p>

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Key Results](#-key-results)
- [Pipeline Architecture](#-pipeline-architecture)
- [Methodology Highlights](#-methodology-highlights)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Limitations](#️-limitations)
- [Future Work](#-future-work)
- [What This Project Demonstrates](#-what-this-project-demonstrates)

---

## 🔍 Overview

The core question: **can we predict which team wins a Challenger-tier LoL match from its early- and mid-game state, and which factors matter most?**

To answer it, the notebook is split into two parts:

- **Part 1: Preprocessing.** Streams event-level match data from Hugging Face, reconstructs complete matches, attributes events to teams, snapshots team economy at the 10- and 20-minute marks, and engineers two feature tables (`match_features`, `player_features`) saved as Parquet.
- **Part 2: Analysis, Modelling & Visualisation.** Exploratory analysis, supervised win prediction (Logistic Regression / Random Forest / XGBoost), SHAP interpretation, a **data-leakage audit**, **patch-robustness** checks, K-Means player archetypes, Isolation-Forest anomaly detection, and champion-pair synergy mining, capped by an automated text report and an interactive win-probability dashboard.

---

## 📚 Dataset

| Property | Value |
|---|---|
| Source | `gptilt/lol-ultimate-events-challenger-10m` (Hugging Face) |
| Region / split | Americas (`train_region_americas`) |
| Complete matches processed | **300** |
| Player-match rows | **3,000** |
| Raw event rows | **311,266** |
| Event types used | `CHAMPION_KILL` (15.3K), `ELITE_MONSTER_KILL` (3.6K), `BUILDING_KILL` (3.7K) |
| Champions mapped | 169 (ID → name lookup) |
| Patches present | 15.6 (9%), **15.7 (71%)**, 15.8 (20%) |

Data is pulled via **streaming** (never fully materialised in memory), buffered per `matchId`, and finalised only when a `GAME_END` event is seen, which keeps the working set small while guaranteeing complete matches.

---

## 📈 Key Results

### Objective control → win rate (Team 100 perspective)

| Objective secured | Win rate WITH | Win rate WITHOUT |
|---|---|---|
| First Tower | **69.7%** | 35.9% |
| First Herald | **68.3%** | 36.0% |
| First Dragon | **65.3%** | 45.6% |
| First Blood | 57.1% | 48.9% |

> First Tower and First Herald are the strongest early-signal objectives. Securing either roughly doubles the effective win rate versus conceding it.

### Model performance (hold-out test set, stratified 80/20)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **XGBoost** | 0.800 | 0.813 | **0.901** |
| Random Forest | 0.800 | 0.818 | 0.897 |
| Logistic Regression | 0.800 | 0.829 | 0.868 |
| Baseline XGBoost (3 features) | 0.583 | 0.627 | 0.650 |

### The leakage audit (the most important finding)

Splitting features by *when they become known* reveals that the headline AUC is driven almost entirely by mid-game information:

| Feature horizon | ROC-AUC |
|---|---|
| Early game (≤ 10 min only) | **0.68** |
| Mid game (≤ 20 min) | **0.90** |
| Full model | 0.90 |

> **Conclusion:** the strong model is a *mid-game* predictor, not an early-game one. Predicting a Challenger match from the first 10 minutes alone is genuinely hard (AUC ≈ 0.68), a nuance that's easy to miss and easy to overclaim.

### Top drivers (SHAP, mean |value|)
`xp_diff_20` (1.63) ≫ `dragon_diff_20` (0.60) > `cs_diff_20` (0.46) > `total_dragons` (0.34) > `kill_diff_20` (0.33).
XP/gold advantage at 20 minutes dominates; per-team champion-pick flags were shown by ablation to *add noise*, not signal, at this sample size.

### Player archetypes (K-Means, k = 5)
`Scaling Farmer` (28%), `Supportive Engager` (25%), `Low-Farm Support` (21%), `Farming Jungler` (19%), `High-Efficiency Carry` (6%), all separated cleanly on a PCA projection (PC1+PC2 ≈ 65% variance).

---

## 🏗 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  PART 1 - PREPROCESSING                                     │
├─────────────────────────────────────────────────────────────┤
│  HF streaming  ──►  buffer by matchId  ──►  keep on GAME_END│
│        │                                          │         │
│        ▼                                          ▼         │
│  select 75 cols        dtype downcast (float32)  +  gc      │
│        │                                                    │
│        ▼                                                    │
│  event attribution (vectorised killer-team, teamId fallback)│
│        │                                                    │
│        ▼                                                    │
│  10-min & 20-min stat snapshots                             │
│        │                                                    │
│        ▼                                                    │
│  match_features.parquet  +  player_features.parquet         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  PART 2 - ANALYSIS & MODELLING                              │
├─────────────────────────────────────────────────────────────┤
│  EDA  ─►  Supervised ML  ─►  SHAP  ─►  Leakage audit        │
│                              │                              │
│                              ├─►  Permutation importance    │
│                              ├─►  Feature ablation          │
│                              └─►  Patch-robustness AUC      │
│  K-Means archetypes  ─►  PCA / UMAP  ─►  Isolation Forest   │
│  Champion synergy (co-pick pairs → network graph)           │
│  Automated report  +  interactive ipywidgets dashboard      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧪 Methodology Highlights

What makes this more than a standard "train a classifier" notebook:

- **Data-leakage auditing.** Features are explicitly partitioned into ≤10-min vs ≤20-min horizons, and three models are trained side-by-side to quantify exactly how much predictive power comes from future information. This reframes the headline result honestly.
- **Data-quality diagnosis.** The notebook detects that `killerId` is unpopulated for `CHAMPION_KILL` events (so per-player *kills* can't be derived and are correctly flagged as constant-zero, while *deaths* remain valid via `victimId`), and recovers missing `killerTeamId` on `BUILDING_KILL` events via a `teamId` fallback. These are the kinds of silent traps that break naive analyses.
- **Distribution-shift / patch robustness.** Because the meta changes every patch, the model is retrained on the dominant patch (15.7) and cross-evaluated on the others; win-rate spread across patches (~10%) is measured and reported as a caveat.
- **Model interpretation triangulated three ways.** SHAP, permutation importance, and leave-one-group-out ablation are cross-checked rather than trusting a single importance metric.
- **Feature-redundancy analysis.** Highly correlated pairs (e.g. `xp_lead_pct_10 ↔ xp_diff_10`, r ≈ 0.997) are surfaced with drop recommendations.
- **Leakage-free baseline.** A separate early-game-only model is trained to serve as the honest "real-time prediction" artifact for deployment framing.

---

## 🛠 Tech Stack

| Area | Tools |
|---|---|
| Data handling | `pandas`, `numpy`, `pyarrow` (Parquet) |
| Ingestion | Hugging Face `datasets` (streaming) |
| Modelling | `scikit-learn` (LogisticRegression, RandomForest, KMeans, IsolationForest, PCA, StandardScaler), `xgboost` |
| Interpretation | `shap`, `sklearn.inspection.permutation_importance` |
| Dim. reduction | `PCA`, `umap-learn` (optional fallback to PCA) |
| Visualisation | `matplotlib`, `seaborn`, `plotly`, `networkx` |
| Interactivity | `ipywidgets` |
| Utilities | `tqdm` |

---

## ⚠️ Limitations

The notebook documents these openly and treats them as *strengths* (scientific honesty), not weaknesses:

- **Challenger-only:** patterns here don't generalise to lower ranks, where objective priorities and pacing differ.
- **Single region + patch drift:** Americas only, spanning three patches; the meta is not constant across the sample.
- **Schema gaps:** assist data is absent, and per-player kills can't be reconstructed (`killerId` unpopulated). Economy and deaths remain valid.
- **Small-sample synergy:** only 50 of 4,085 champion pairs reach ≥5 co-picks (and just 2 reach ≥10), so synergy scores are exploratory, not conclusive, at n = 300 matches.
- **Modest N for ML:** 300 matches is enough to demonstrate methodology, not to ship a production model.

---

## 🚀 Future Work

- Scale ingestion to thousands of matches (and multiple regions) to firm up synergy and champion-level estimates.
- Add **patch as a covariate** or train patch-specific models.
- Build a true **time-series / real-time** predictor that updates win probability minute-by-minute.
- Package the early-game (leakage-free) model behind a small API + live dashboard.
- Add role/lane inference to enrich the player-archetype clustering.

---

## 🎯 What This Project Demonstrates

- **End-to-end ownership:** streaming ETL, feature engineering, modelling, interpretation, and reporting in one coherent pipeline.
- **Scientific rigor over vanity metrics:** the leakage audit, ablation, and patch-robustness work show a habit of *interrogating* results rather than presenting the best-looking number.
- **Practical data-quality instincts:** catching unpopulated ID fields and silent attribution gaps before they corrupt downstream analysis.
- **Communication:** automated summary report, an interactive dashboard, and a full suite of publication-quality visualisations.

---

*Built with Python. Data © Riot Games / dataset authors; used here for non-commercial analysis and education.*
