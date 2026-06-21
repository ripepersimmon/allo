# Δw Redesign Plan — Crisis Weight-Shift Drivers (OLS → Gradient Boosting)

**Date:** 2026-06-21
**Status:** Plan (scaffold: `delta_weight_study.py`)
**Supersedes the flawed designs in:** `crisis_weight_test.py` (autocorrelated daily t-tests), `variance_decomp.py` / `variable_mining.py` / `intervention_analysis.py` (Σ-tautological / overlapping-window panel OLS).

---

## 1. Motivation — what was wrong

A code+methodology audit (2026-06-21) found three CRITICAL problems that any OLS→XGBoost report on the *current* design would amplify:

| # | Problem | Where | Why a stronger model makes it worse |
|---|---------|-------|-------------------------------------|
| C1 | **Tautology** — GMV weight `w ∝ Σ⁻¹1 ≈ 1/σᵢ²` regressed on variance features from the *same* Σ | `variance_decomp.py`, `variable_mining.py`, `*_decomp.py` | XGBoost reconstructs the algebraic identity → R²≈0.95 that *means nothing* |
| C2 | **Overlapping-window leakage** — 252d rolling ⇒ adjacent daily rows share 251/252 days (ρ≈0.996) treated as iid | `crisis_weight_test.py`, `intervention_analysis.py` | Random k-fold on the daily panel puts near-identical rows in train+test → fake high score |
| C3 | **Survivorship bias** — 2024 S&P 100 list applied back to 2007 GFC | `src/data_loader.py:TICKERS` | Distressed GFC financials (Lehman, Bear, WaMu) absent ⇒ GFC results unreliable |

## 2. Design fix — cross-sectional DiD on Δw

**Core idea:** collapse the time axis into two snapshots per crisis and model the *change* in weight, using only **pre-crisis (predetermined)** features. This structurally removes C1 and C2.

### 2.1 Target
For each crisis `c` and estimator `e`:
```
w_pre_i     = GMV_longonly( Σ_e over 252 td ending at T0(c) − 1 )
w_crisis_i  = GMV_longonly( Σ_e over 252 td ending at peak(c) )
Δw_i        = w_crisis_i − w_pre_i          # one value per asset
```
Unit of analysis = **asset (cross-section)**, not daily panel ⇒ no overlapping-window leakage (C2 solved).
Pooled sample ≈ 100 assets × 3 crises ≈ 300 rows.

### 2.2 Features — all measured on the PRE window (predetermined)
| Group | Variables | Tautology-safe? |
|-------|-----------|-----------------|
| **Control (required)** | `w_pre` | Absorbs mechanical mean-reversion in Δw (the −w_pre term) |
| Σ-derived | `pre_total_var`, `pre_syst_share`, `pre_avg_corr`, `pre_beta` | Interpretable **only after** controlling `w_pre` |
| Non-Σ (cleanest) | `pre_amihud`, `pre_momentum` (12-1), `pre_log_dolvol`, `pre_downside_vol` | Yes — not algebraically in Σ |
| Optional | GICS sector dummies | — |

Tautology fix (C1) = predetermined pre-features **AND** `w_pre` as a control. Coefficient on any other feature then answers *"among assets with the same starting weight, who gained weight during the crisis?"* — a genuine, non-circular question because `w_crisis` comes from different (crisis-window) data.

### 2.3 Evaluation — leakage-safe CV
- **Leave-One-Crisis-Out (LOCO):** train on 2 crises, test on the held-out crisis. Honest test of cross-crisis generalization; the natural split once time is collapsed to snapshots.
- **No random k-fold** (same asset would leak across folds). If a finer split is needed, group by asset.
- Report **out-of-sample LOCO R² / MAE**, not in-sample fit.

### 2.4 Survivorship (C3)
Course/internal scope ⇒ keep the 2024 universe but **state the limitation explicitly**, and treat GFC results as indicative only. Optional robustness: rerun on COVID+Rates only (post-2015 universe is far more stable).

## 3. Model ladder (all on identical target / features / LOCO CV)

| Model | Spec | Question answered |
|-------|------|-------------------|
| **M0** | `Δw ~ w_pre` | Mechanical baseline (mean-reversion only) |
| **M1** | M0 + Σ-derived, OLS (HC3 robust SE) | Do risk-structure features add linear signal beyond starting weight? |
| **M2** | M1 + non-Σ (liquidity, momentum, size), OLS (HC3) | Genuine economic drivers |
| **M3** | Gradient boosting on M2 feature set | Does **nonlinearity / interactions** add explanatory power over M2? |

- M3 uses `HistGradientBoostingRegressor` (sklearn, no extra install); auto-swaps to `xgboost.XGBRegressor` if installed.
- Interpretability: `permutation_importance` (always) + SHAP if `shap` is installed. → M3 is not a black box vs OLS.

## 4. Small-data discipline (~300 rows)

- M3 framed as **"does nonlinearity help?"**, NOT precise prediction.
- Shallow trees (`max_depth=2–3` / `max_leaf_nodes≤8`), strong regularization, early stopping via LOCO.
- A low / negative LOCO R² is a **finding** ("weight-shift mechanism differs across crises"), not a failure.
- Always report M0 baseline alongside so added value is visible.

## 5. Deliverables

- `delta_weight_study.py` — single script, run from repo root.
- Figures → `results/figures/delta_weight/`: Δw distributions, OLS coefficient (M1/M2) with HC3 CIs, LOCO R² ladder bar (M0→M3), permutation/SHAP importance, partial-dependence of top non-Σ driver.
- Report → `reports/delta_weight_study_report.md`: per-crisis + pooled coefficient tables, LOCO scores, interpretation, **explicit limitations** (survivorship, static sectors, snapshot overlap).

## 6. What is reused vs dropped

**Reused:** `rolling_gmv`/`gmv_long_only` (weights); `PERIODS` pre/crisis windows from `crisis_weight_test.py`; feature math from `variance_decomp.decompose_variance`, `variable_mining` (amihud/downside), `intervention_analysis` (avg_corr/momentum) — but evaluated **on the pre window only**.

**Dropped:** daily overlapping panel OLS; Σ-derived features explaining weight *level*; autocorrelation-blind t/permutation/MWU tests; SSR/BIC δ grid (irrelevant under the snapshot design).

## 7. Open items to confirm

- [ ] `peak(c)` definition: max-drawdown date of the EW universe inside each crisis window, vs. crisis-window end. **Default: max-drawdown date.**
- [ ] Pooled vs per-crisis as the headline (default: report both; pooled with crisis fixed effects as headline).
- [ ] Whether to add the COVID+Rates-only survivorship robustness cut (default: yes, as appendix).
