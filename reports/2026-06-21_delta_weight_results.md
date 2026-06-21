# Δw Study — Results Report

**Date:** 2026-06-21
**Experiment:** `delta_weight_study.py`
**Plan:** `reports/2026-06-21_delta_weight_redesign_plan.md`
**Environment:** conda `allo` (Python 3.13, xgboost backend) — see §6 for reproduction.

---

## 1. What this experiment does

Redesigned study of **which pre-crisis asset characteristics drive the change in GMV
weight during a crisis**, fixing the three CRITICAL flaws found in the 2026-06-21 audit
(tautology, overlapping-window leakage, survivorship).

- **Target:** `Δw_i = w_crisis_i − w_pre_i` (per-asset weight change; cross-sectional DiD).
- **Features:** predetermined (pre-crisis window only) + `w_pre` as control.
- **Evaluation:** Leave-One-Crisis-Out (LOCO) **out-of-sample** R²/MAE — no random k-fold.
- **Sample:** 3 crises × ~96 assets = 288 rows. Crisis peaks (EW max-drawdown date):
  GFC 2009-03-09, COVID 2020-03-23, Rates 2022-09-30.
- **Estimators:** Sample, Ledoit-Wolf (LW), Gerber. Market proxy: equal-weighted.

---

## 2. Model ladder — LOCO out-of-sample R²

| Model | Spec | Gerber | LW | Sample |
| --- | --- | --- | --- | --- |
| **M0** | `Δw ~ w_pre` (mechanical baseline) | **0.381** | **0.173** | **0.209** |
| **M1** | M0 + Σ-derived (OLS, HC3) | 0.365 | 0.143 | 0.181 |
| **M2** | M1 + non-Σ (OLS, HC3) | 0.192 | −0.178 | −0.042 |
| **M3** | M2 features, XGBoost (depth-2) | 0.268 | 0.106 | 0.093 |

**Reading:**
- **The starting weight `w_pre` alone (M0) is the dominant out-of-sample predictor.** Crisis weight-shift is largely **mean-reversion**: assets that enter a crisis with high GMV weight shed it.
- **Adding features linearly (M1→M2) does not help and even hurts OOS** (LW/Sample go negative). With ~96 assets/crisis, a 9-regressor linear model overfits and fails to generalize across regimes.
- **Nonlinearity recovers part of that loss (M2→M3):** XGBoost on the *same* feature set beats linear M2 everywhere (Gerber +0.08, LW +0.28, Sample +0.13 vs M2), i.e. the usable signal beyond `w_pre` is interaction/nonlinear, not linear.
- **But no model beats the M0 baseline OOS.** Honest conclusion: **beyond mean-reversion of the starting weight, pre-crisis characteristics carry only weak, regime-specific signal.** This is a finding, not a failure — and it is exactly what the old (leaky/tautological) design hid.

---

## 3. OLS coefficients (M2, HC3 robust) — what is actually significant

Consolidated across estimators (coef [t-stat]; **bold** = p<0.05):

| Term | Sample | LW | Gerber | Robust? |
| --- | --- | --- | --- | --- |
| `w_pre` | **−0.81** [−5.9] | **−0.74** [−5.2] | **−0.93** [−9.5] | ✅ all 3, p≈0 |
| `pre_downside_vol` | **−2.09** [−2.5] | **−1.70** [−2.3] | **−1.98** [−2.5] | ✅ all 3 |
| `pre_momentum` | **−0.021** [−2.4] | **−0.018** [−2.2] | −0.011 [−1.4] | ✅ Sample/LW |
| `pre_total_var` | +16.6 [1.4] | +12.3 [1.1] | +22.5 [1.8†] | ✗ (Gerber p=0.07) |
| `pre_syst_share` | −0.06 [−1.1] | −0.06 [−1.4] | −0.07 [−1.6] | ✗ |
| `pre_avg_corr` | +0.07 [0.9] | +0.09 [1.2] | +0.11 [1.4] | ✗ |
| `pre_beta` | −0.003 [−0.4] | −0.001 [−0.2] | −0.010 [−1.5] | ✗ |
| `pre_amihud` | +10.9 [0.7] | +10.7 [0.7] | +8.4 [0.4] | ✗ |
| `pre_log_dolvol` | +0.003 [0.8] | +0.002 [0.8] | +0.000 [0.1] | ✗ |

† marginal (p=0.072).

**Three takeaways:**
1. **`w_pre` (mean-reversion)** is the strongest, most robust effect — consistent and highly significant across all estimators.
2. **`pre_downside_vol` is the one genuinely robust *characteristic* driver:** higher pre-crisis downside volatility → larger weight **loss** in the crisis (negative, significant in all three). GMV de-allocates from assets already showing downside risk. **`pre_momentum`** adds a secondary robust effect (pre-crisis winners lose weight).
3. **The Σ-derived variance-decomposition features (`total_var`, `syst_share`, `avg_corr`, `beta`) are NOT significant once `w_pre` is controlled.** This directly validates the audit's tautology finding: their apparent explanatory power in the old design was mechanical (they *are* `w`), and it vanishes under a correct, predetermined-feature + control specification.

---

## 4. Connection to the audit

| Audit CRITICAL | How this design resolves it | Evidence in results |
| --- | --- | --- |
| C1 Tautology (`w ∝ Σ⁻¹1` regressed on same-Σ features) | Δw target + predetermined features + `w_pre` control | Σ-features lose significance (§3); only non-Σ `downside_vol`/`momentum` survive |
| C2 Overlapping-window leakage (ρ≈0.996 daily panel) | Time collapsed to 2 snapshots; CV = LOCO | OOS R² < in-sample; M2 goes negative — leakage no longer inflates fit |
| C3 Survivorship (2024 universe → 2007) | Stated limitation; `--skip-gfc` robustness cut | GFC n=89 vs 99–100 later (missing distressed financials visible) |

---

## 5. Figures (`results/figures/delta_weight/`)

| File | Content |
| --- | --- |
| `ladder_loco_{Sample,LW,Gerber}.png` | M0→M3 LOCO R² bars |
| `importance_{Sample,LW,Gerber}.png` | XGBoost permutation importance over the M2 feature set |

Per-estimator panels: `reports/delta_panel_{Sample,LW,Gerber}.csv`.
Machine-readable scores: `reports/delta_weight_loco_summary.csv`.

---

## 6. Environment & reproduction

Managed via conda env **`allo`** (created 2026-06-21). Spec: `environment.yml`;
exact pin: `requirements-allo.lock.txt`.

```bash
# create once
conda env create -f environment.yml      # or: conda create -n allo python=3.13
conda activate allo
pip install -r requirements-allo.lock.txt # exact reproduction

# run
python delta_weight_study.py                 # ew proxy, all 3 crises (this report)
python delta_weight_study.py --skip-gfc      # survivorship robustness cut
python delta_weight_study.py --proxy spy     # SPY market proxy (run fetch_spy.py first)
```

Key versions: python 3.13.14 · numpy 2.4.6 · pandas 3.0.3 · scikit-learn 1.9.0 ·
statsmodels 0.14.6 · xgboost 3.3.0 · shap 0.52.0 · cvxpy 1.9.1.

> With xgboost installed, M3 uses `XGBRegressor`; without it the script falls back to
> sklearn `HistGradientBoostingRegressor` (M3 LOCO is then lower, e.g. Gerber 0.097).

---

## 7. Limitations

- **Survivorship bias** — 2024 S&P 100 applied to all crises; GFC indicative only (use `--skip-gfc`).
- **Static 2024 GICS sectors** (relevant only if sector dummies are added).
- **Snapshot overlap** — `w_pre` and `w_crisis` windows overlap partially; Δw is a two-point difference, not a clean event study.
- **Small sample** (~96 assets × 3 crises) — XGBoost detects nonlinearity, it does not predict precisely; LOCO R² is the honest metric.
- **EW market proxy** is endogenous to the universe; `--proxy spy` is the robustness alternative.
