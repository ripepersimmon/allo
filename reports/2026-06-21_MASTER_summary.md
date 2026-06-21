# Master Summary — Covariance Estimators & GMV Weight Allocation (2026-06-21)

A single-file overview of the full redesign cycle: from a methodology audit, through
three rebuilt studies, to a crisis case study — with environment, robustness, and
references. Detailed reports are linked from each section.

**Environment:** conda `allo` (Python 3.13 · numpy 2.4.6 · scikit-learn 1.9.0 · xgboost
3.3.0 · shap 0.52.0 · statsmodels 0.14.6 · cvxpy 1.9.1). Spec: `environment.yml`; pin:
`requirements-allo.lock.txt`.

---

## 0. The arc

The original scripts asked *"what drives GMV portfolio weights in crises"* but did so in a
way that a methodology audit found **fatally circular and leaky**. We redesigned the
question three times, each fixing the prior flaw, ending with a clean, prior-research-
anchored result and a crisis case study.

```
Audit (3 CRITICAL)
   │
   ├─ Study 1  Δw crisis weight-change (DiD)         → mean-reversion dominates; signal weak
   │
   ├─ Study 2  Weight-level explanation (LW only)    → GMV = low-beta threshold portfolio  ★
   │
   └─ Study 3  Crisis case study (VIX-defined)       → relative de-risking, absolute β compresses
```

---

## 1. Audit — what was wrong (`see git history of old scripts`)

| # | CRITICAL flaw | Where | Fixed by |
|---|---|---|---|
| C1 | **Tautology** — `w ∝ Σ⁻¹1` regressed on same-Σ variance features | `variance_decomp.py`, `variable_mining.py` | Predetermined / non-Σ features; benchmark-only Σ block |
| C2 | **Overlapping-window leakage** — 252-d rolling ⇒ ρ≈0.996 daily panel treated as iid | `crisis_weight_test.py`, `intervention_analysis.py` | Non-overlapping snapshots; leakage-safe CV (LOCO/LOYO) |
| C3 | **Survivorship bias** — 2024 S&P 100 applied to 2007 | `src/data_loader.py` | Stated limitation; VIX widens crisis set |

Two claimed bugs were **verified as non-issues** (corrected the audit): the BBC back-group
denominator off-by-one (argmax-invariant) and the SSR-vs-BIC δ selection (equivalent at
fixed model dimension).

---

## 2. Study 1 — Δw crisis weight-change (DiD)
*Scripts:* `delta_weight_study.py` · *Reports:* `2026-06-21_delta_weight_redesign_plan.md`,
`2026-06-21_delta_weight_results.md`

Target `Δw = w_crisis − w_pre`; predetermined features + `w_pre` control; Leave-One-
Crisis-Out OOS. **LOCO R²:**

| Model | Gerber | LW | Sample |
|---|---|---|---|
| M0 `w_pre` | **0.381** | 0.173 | 0.209 |
| M1 +Σ | 0.365 | 0.143 | 0.181 |
| M2 +non-Σ | 0.192 | −0.178 | −0.042 |
| M3 GBR | 0.268 | 0.106 | 0.093 |

**Finding:** crisis weight change is dominated by **mean-reversion of the starting weight**
(`w_pre` coef −0.74…−0.93, p≈0). Only `downside_vol` and `momentum` survive as robust
characteristic drivers; **Σ-derived variance features go insignificant once controlled —
validating C1**. No feature beats the M0 baseline OOS (honest weak signal).

---

## 3. Study 2 — Weight-level explanation ★ (headline)
*Scripts:* `weight_explain_study.py` · *Report:* `2026-06-21_weight_explain_results.md`

LW only. Target = GMV weight level `w`; 20 annual non-overlapping snapshots (1,921 rows);
non-Σ headline features, Σ as a separate benchmark; Leave-One-Year-Out OOS. **LOYO R²:**

| Model | Features | R² |
|---|---|---|
| B0 Σ-bench | total_var, syst_share | 0.059 |
| M1 non-Σ | beta, size, liquidity, momentum | 0.172 |
| M2 +sector | + GICS | 0.176 |
| **M3 GBR** | M2, nonlinear | **0.356** |

**Findings:**
- **GMV is a low-beta portfolio.** `beta` dominates (OLS −0.034, t=−11; SHAP mean&#124;val&#124;
  3× the next feature).
- **The beta effect is a threshold/cliff, not linear** (SHAP `shap_dependence_beta.png`):
  strong positive weight for β≲0.6, hard drop, flat ~zero floor for β≳1. This nonlinearity
  is **why M3 doubles linear M2 (0.18→0.36).**
- **Matches [CST2011]** "Minimum-Variance Portfolio Composition" almost exactly (analytic
  threshold beta) — recovered here empirically and model-free.
- The tautology (C1) is **mild for long-only weights**: the Σ-benchmark explains only
  R²≈0.06 OOS, because the long-only constraint breaks `w ∝ Σ⁻¹1` ([JM2003]).

---

## 4. Study 3 — Crisis case study (VIX-defined)
*Scripts:* `crisis_case_study.py`, `src/crises.py`, `fetch_vix.py` · *Report:*
`2026-06-21_crisis_case_results.md`

**Crisis definition:** VIX two-threshold hysteresis (enter>30 / exit<20, min 10 d, merge
<42 td) → **8 named episodes** (`vix_crisis_periods.csv`): GFC×2, Flash Crash/EU 2010,
EU debt 2011, China 2015, Volmageddon 2018, COVID 2020, Rates 2022.

**Findings:**
- **F1 — Concentration in severe crises only:** Effective-N COVID 16.5→8.5, GFC-Bear
  13.3→5.6; mild crises barely move; 2022 even diversifies.
- **F2 — Two-layer beta effect (headline nuance):** the cross-sectional **β-weight slope
  steepens** into the peak (6/8 crises; COVID −0.040→−0.072) — GMV pulls *harder* toward
  low-beta ([CST2011] threshold tightening). **Yet absolute portfolio β rises** in severe
  crises (COVID 0.52→0.60) because correlations spike and betas **compress toward 1**
  ([LS2001]). → *De-risking is relative, not absolute.*
- **F3 — Flight to defensives:** gainers are low-β names (VZ, JNJ, SO, WMT, LMT).
- **F4 — Liquidity null:** `amihud` coefficient does not shift consistently — no clean
  flight-to-liquidity (honest null).

**Robustness:** threshold scheme (fixed 30/20 vs percentile 90/60) → 8 vs 9 episodes, 8
identical. Market proxy (EW vs SPY) → portfolio-β direction identical in all 8; Effective-N
unchanged. Conclusions hold.

---

## 5. Headline conclusion

> **The Ledoit-Wolf long-only GMV portfolio is, structurally, a low-beta portfolio with a
> sharp beta threshold (Clarke–de Silva–Thorley 2011), recovered here empirically via SHAP.
> In a crisis the low-beta tilt *intensifies* (the threshold tightens), but the realized
> portfolio beta still *rises* because market-wide correlations compress all betas toward 1
> (Longin–Solnik 2001). Severe shocks (COVID, GFC) sharply concentrate the portfolio into a
> handful of defensive names. Variance-decomposition characteristics — the focus of the old,
> circular design — add little once the tautology and leakage are removed.**

---

## 6. Artifacts

**Scripts:** `delta_weight_study.py` · `weight_explain_study.py` · `crisis_case_study.py`
· `fetch_vix.py` · `src/crises.py`

**Reports (`reports/`):**
- Plans/results: `2026-06-21_delta_weight_redesign_plan.md`, `…_delta_weight_results.md`,
  `…_weight_explain_results.md`, `…_crisis_case_results.md`, **this file**
- References: `references_weight_explain.md` (8 verified citations + BibTeX)
- Data tables: `vix_crisis_periods.csv`, `delta_weight_loco_summary.csv`,
  `weight_explain_panel.csv`, `crisis_case_summary.csv`, `crisis_case_summary_spy.csv`
- Auto-dumps: `delta_weight_study_report.md`, `weight_explain_report.md`,
  `crisis_case_report.md`

**Figures (`results/figures/`):** `delta_weight/` · `weight_explain/` (incl. SHAP
beeswarm + dependence) · `crisis_case/` (8 per-crisis panels + pre-vs-peak summary)

**Environment / data:** `environment.yml` · `requirements-allo.lock.txt` ·
`sp500/VIX.parquet`

---

## 7. References (verified — see `references_weight_explain.md`)

| Key | Citation | Role |
|---|---|---|
| CST2011 | Clarke, de Silva & Thorley (2011), *JPM* 37(2) | Threshold-beta MVP — **primary anchor** |
| LS2001 | Longin & Solnik (2001), *J. Finance* 56(2) | Correlations rise in bear markets — beta compression |
| FP2014 | Frazzini & Pedersen (2014), *JFE* 111(1) | Betting Against Beta / defensive premium |
| BBW2011 | Baker, Bradley & Wurgler (2011), *FAJ* 67(1) | Low-volatility anomaly |
| JM2003 | Jagannathan & Ma (2003), *J. Finance* 58(4) | Long-only constraint as regularization |
| LW2004 | Ledoit & Wolf (2004), *JPM* 30(4) | Shrinkage covariance estimator |
| DGU2009 | DeMiguel, Garlappi & Uppal (2009), *RFS* 22(5) | 1/N — motivates OOS evaluation |
| KIM2025 | Kim et al. (2025), arXiv:2508.10776 | Repo's BBC / decision-focused GMV ref |

---

## 8. Honest limitations (carried across all studies)
- **Survivorship bias** — 2024 S&P 100 universe; GFC results partial (missing Lehman/Bear/WaMu).
- **Descriptive, not causal** — `w` is a deterministic function of Σ; characteristics are *associated*.
- **Beta endogeneity** — crisis-window betas spike/compress; F2.2 is itself a manifestation.
- **Static 2024 GICS sectors**; small samples (case study is per-episode, not inferential).

## 9. Possible next steps
- LaTeX paper draft (plan → write) from Studies 2–3.
- Point-in-time universe to remove survivorship (cleanest for GFC).
- Per-crisis SHAP (replace pre/peak OLS coefficients) for sharper structure shifts.
