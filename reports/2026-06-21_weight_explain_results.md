# Weight-Explanation Study — Results Report

**Date:** 2026-06-21
**Experiment:** `weight_explain_study.py`
**Environment:** conda `allo` (Python 3.13, xgboost 3.3.0, shap 0.52.0)
**References:** `references_weight_explain.md` ([CST2011], [FP2014], [BBW2011], [LW2004], [JM2003], [DGU2009])
**Auto-dump:** `weight_explain_report.md` · **Panel:** `weight_explain_panel.csv`

---

## 1. Question & design

**What interpretable asset characteristics explain the cross-section of Ledoit-Wolf
long-only GMV portfolio weights?**

- **Covariance:** Ledoit-Wolf only ([LW2004]).
- **Target:** weight level `w_i` (long-only GMV).
- **Headline features (non-Σ):** `beta`, `log_dolvol` (size), `amihud` (illiquidity),
  `momentum`, GICS sector dummies.
- **Benchmark (Σ-derived, reported separately):** `total_var`, `syst_share` — kept apart
  because for *unconstrained* GMV `w ∝ Σ⁻¹1` makes them tautological.
- **Time structure:** 20 annual non-overlapping year-end snapshots (2005–2024), 1,921
  asset-rows. Annual spacing ≈ window length ⇒ no daily-overlap leakage.
- **Evaluation:** Leave-One-Year-Out (LOYO) **out-of-sample** R²/MAE.

---

## 2. Headline results

### LOYO out-of-sample R²
| Model | Features | R² | MAE |
|---|---|---|---|
| **B0** Σ-benchmark | total_var, syst_share | **0.059** | 0.0152 |
| **M1** non-Σ | beta, size, liquidity, momentum | 0.172 | 0.0154 |
| **M2** + sectors | M1 + GICS dummies | 0.176 | 0.0152 |
| **M3** GBR (xgboost) | M2 features, nonlinear | **0.356** | 0.0101 |
| *(ref)* Σ + non-Σ | all | 0.187 | — |

- **non-Σ adds +0.128 R² over the Σ-benchmark; Σ adds only +0.011 over non-Σ.** The
  interpretable characteristics carry essentially all the explainable signal.
- **M3 doubles the linear M2** (0.176 → 0.356) → the characteristic→weight mapping is
  strongly **nonlinear**.

### OLS — non-Σ + sectors (HC3 robust), selected terms
| Term | coef | t | p |
|---|---|---|---|
| **beta** | **−0.0340** | **−11.1** | ~0 |
| log_dolvol (size) | +0.0016 | +2.1 | 0.039 |
| amihud | +6.47 | +1.4 | 0.165 |
| momentum | −0.0022 | −1.5 | 0.146 |
| sec_Financials | +0.0073 | +4.5 | ~0 |
| sec_ConsStap | +0.0063 | +1.8 | 0.066 |

(Σ-benchmark OLS: `total_var` −3.93 [t=−4.3], `syst_share` −0.038 [t=−10.9] — both
negative as mechanically expected, but jointly only R²≈0.06 out-of-sample.)

---

## 3. SHAP interpretation (TreeExplainer on the GBR)

Mean |SHAP value| (contribution magnitude to predicted weight):

| Feature | mean &#124;SHAP&#124; | Rank |
|---|---|---|
| **beta** | **0.01263** | 1 (≈3× the next) |
| amihud | 0.00424 | 2 |
| log_dolvol | 0.00316 | 3 |
| momentum | 0.00161 | 4 |
| sec_ConsStap | 0.00082 | 5 |
| (other sectors) | < 0.0007 | — |

Figures: `results/figures/weight_explain/shap_summary.png`,
`shap_dependence_{beta,log_dolvol,amihud}.png`.

### 3.1 The beta effect is a **threshold/cliff**, not linear — the key SHAP finding
`shap_dependence_beta.png` shows a sharp hockey-stick:
- **β ≲ 0.6:** strong *positive* contribution to weight, up to **+0.12**, rising steeply as beta falls.
- **β ≈ 0.7–0.9:** SHAP crosses zero — an empirical **threshold beta**.
- **β ≳ 1.0:** flat *negative floor* (~−0.01) for the entire high-beta range (β up to 3.0)
  — high-beta names are uniformly pushed toward zero weight.

This nonlinear threshold is **why the gradient-boosted M3 doubles the linear M2**: a single
linear beta slope cannot represent a cliff.

### 3.2 Beeswarm summary
`shap_summary.png`: `beta` has by far the widest impact (low beta = large positive weight
push); `amihud` (high illiquidity → negative push) and `log_dolvol` add secondary, partly
nonlinear effects; sector effects are small (`ConsStap`/`Financials` slightly positive).
SHAP ranks `amihud` above `log_dolvol`, the reverse of permutation importance — consistent
with liquidity acting through interactions rather than a clean main effect.

---

## 4. Connection to prior research

| Our result | Prior work | Relationship |
|---|---|---|
| Beta dominates; hard threshold above which weight → 0 | **[CST2011]** Minimum-Variance Portfolio Composition | **Direct theoretical match.** CST derive analytically that long-only MVP weight is a function of beta with a **threshold beta** above which securities leave the solution. Our SHAP cliff at β≈0.7–0.9 is the empirical analog. |
| GMV = low-beta tilt | **[FP2014]** Betting Against Beta; **[BBW2011]** Low-Volatility Anomaly | The tilt loads on the documented defensive/low-beta premium, not just variance minimization. |
| Σ-benchmark explains only R²≈0.06 (tautology is mild) | **[JM2003]** Wrong Constraints Help | The long-only constraint regularizes Σ and **breaks the `w ∝ Σ⁻¹1` identity**, explaining why variance summaries barely predict constrained weights. |
| Out-of-sample (LOYO) evaluation | **[DGU2009]** Optimal vs Naive | Estimation error can offset optimization gains; OOS is the honest metric. |
| LW shrinkage covariance | **[LW2004]** | The estimator used (`src/estimators.lw_cov`). |

**Contribution of this study relative to [CST2011]:** CST derive the threshold *analytically*
under a single-factor model; here we **recover the same threshold structure empirically and
model-free** from realized LW-GMV weights (SHAP on a gradient-boosted fit), and quantify how
much *additional* explanatory power non-variance characteristics (size, liquidity, sector)
contribute beyond the mechanical variance terms.

---

## 5. Take-away

> **The LW long-only GMV portfolio is, first and foremost, a low-beta portfolio with a sharp
> beta threshold (≈0.7–0.9) above which stocks receive ~zero weight — matching the analytic
> result of Clarke, de Silva & Thorley (2011). Size and liquidity add secondary, nonlinear
> tilts; sectors add little. Because the relationship is a threshold rather than linear, a
> gradient-boosted model doubles the out-of-sample R² of linear OLS (0.18 → 0.36), and
> interpretable non-Σ characteristics explain far more than the mechanical variance terms.**

---

## 6. Reproduction
```bash
conda activate allo
python weight_explain_study.py                 # ew proxy, 2005–2024 (this report)
python weight_explain_study.py --proxy spy     # SPY proxy robustness (run fetch_spy.py first)
python weight_explain_study.py --years 2010 2024
```

## 7. Limitations
- **Descriptive, not causal** — `w` is a deterministic function of Σ; characteristics are
  *associated* with weight.
- **Survivorship bias** — 2024 S&P 100 universe applied to all snapshot years.
- **Static 2024 GICS sectors** applied to all years.
- **Concentrated target** — long-only `w` is highly right-skewed; OLS is descriptive only.
- **In-sample SHAP** — SHAP is computed on the pooled GBR fit (interpretation of learned
  structure), while R² metrics are the LOYO out-of-sample numbers.
