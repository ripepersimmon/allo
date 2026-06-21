# Variance Decomposition Analysis of GMV Weights

**Date**: 2026-05-24
**Estimators**: Sample covariance, Ledoit-Wolf (LW), Gerber (threshold=0.3)
**Method**: Unconstrained GMV (w ∝ Σ⁻¹1), equal-weighted market proxy
**Window**: 252 trading days

---

## 1. Methodology

### Variance Decomposition

For each asset *i*, we run a time-series OLS against the equal-weighted market return *r_m*:

$$r_{it} = \alpha_i + \beta_i r_{mt} + \varepsilon_{it}$$

yielding:

$$\sigma^2_i = \underbrace{\beta^2_i \sigma^2_m}_{	ext{systematic}} + \underbrace{\sigma^2_{\varepsilon,i}}_{	ext{idiosyncratic}}$$

### Cross-Sectional OLS

At each date, we run four models across all active assets:

| Model | Specification | Purpose |
|-------|--------------|---------|
| (A) β-only | w_i = α + γ · β_i | baseline |
| (B) total-σ² | w_i = α + γ · σ²_i | variance level only |
| (C) decomposed | w_i = α + γ₁ · syst_var_i + γ₂ · idio_var_i | raw decomposition |
| **(D) orthogonal** | **w_i = α + γ₁ · total_var_i + γ₂ · syst_share_i** | **primary spec** |

**Model (D) is the primary specification.** Since syst_var + idio_var = total_var exactly, Model (C) suffers from structural near-collinearity (VIF inflates as cross-sectional β dispersion shrinks in crises). Model (D) orthogonalizes by using total_var (the level) and syst_share = syst_var/total_var (the composition), which are genuinely distinct: VIF is typically much lower. The key test is whether γ₂ in (D) is non-zero — i.e., does the *fraction* of systematic variance matter for weights, beyond total variance alone?

Model (C) is still reported for context; its R² vs. (B) should be interpreted with caution because (C) has an extra free parameter and will mechanically fit at least as well.

### Theoretical Expectation

From the Woodbury identity applied to a single-factor covariance:

$$\Sigma^{-1}\mathbf{1} = D^{-1}\mathbf{1} - D^{-1}\beta\left(\sigma^{-2}_m + \beta' D^{-1}\beta\right)^{-1}\beta' D^{-1}\mathbf{1}$$

- High-**idiosyncratic** variance → lower GMV weight (D⁻¹ term penalizes it directly)
- High-**systematic** variance → extra penalty via the β correction term
- So both γ₁ < 0 and γ₂ < 0 are expected; **|γ₁| > |γ₂|** if systematic variance is more penalizing

**Gerber hypothesis**: the Gerber statistic filters sub-threshold moves, discarding idiosyncratic small-noise. This makes its estimated covariance more "systematic-like", predicting stronger γ₁ relative to γ₂.

---

## 2. Snapshot Analysis at Crisis Peaks

**Peak dates**: GFC = 2009-03-31, COVID = 2020-04-30, Rates = 2023-01-31

### OLS Results Table

| Crisis | Estimator | γ₁ (syst) | γ₂ (idio) | R²(A) β | R²(B) σ² | R²(C)† | R²(D) | VIF(C) | N |
|--------|-----------|-----------|-----------|---------|----------|--------|-------|--------|---|
| GFC | Sample | -30.0819** | 11.9858 | 0.088 | 0.028 | 0.074 | 0.073 | 2.5 | 93 |
| GFC | LW | -29.4093*** | 11.6705* | 0.147 | 0.049 | 0.127 | 0.145 | 2.5 | 93 |
| GFC | Gerber | -27.2310* | 7.3595 | 0.053 | 0.025 | 0.042 | 0.048 | 2.5 | 93 |
| COVID | Sample | -168.7591*** | 80.9289* | 0.085 | 0.026 | 0.104 | 0.072 | 1.4 | 100 |
| COVID | LW | -141.7126*** | 58.3554** | 0.261 | 0.092 | 0.283 | 0.225 | 1.4 | 100 |
| COVID | Gerber | -148.6045*** | 82.7990** | 0.087 | 0.018 | 0.110 | 0.084 | 1.4 | 100 |
| Rates | Sample | -96.3647 | -6.3096 | 0.048 | 0.025 | 0.032 | 0.040 | 1.3 | 100 |
| Rates | LW | -93.2523** | -3.1526 | 0.084 | 0.043 | 0.056 | 0.079 | 1.3 | 100 |
| Rates | Gerber | -86.3636 | -45.7342 | 0.015 | 0.013 | 0.014 | 0.013 | 1.3 | 100 |

*Significance: * p<.10, ** p<.05, *** p<.01 (two-sided)*
*† R²(C) inflated by extra df vs (B); R²(D) is the fair comparison for composition effect.*
*VIF(C) = VIF of syst_var regressed on idio_var in Model (C); values > 5 indicate problematic collinearity.*

### Key Findings by Crisis

#### GFC (2009-03-31)
  - **Sample**: γ₁=-30.0819 (t=-2.50), γ₂=11.9858 (t=1.32), R²(decomp)=0.074 (+-0.014 vs β-only); dominant component = **systematic variance**
  - **LW**: γ₁=-29.4093 (t=-3.36), γ₂=11.6705 (t=1.76), R²(decomp)=0.127 (+-0.021 vs β-only); dominant component = **systematic variance**
  - **Gerber**: γ₁=-27.2310 (t=-1.66), γ₂=7.3595 (t=0.59), R²(decomp)=0.042 (+-0.011 vs β-only); dominant component = **systematic variance**

#### COVID (2020-04-30)
  - **Sample**: γ₁=-168.7591 (t=-3.36), γ₂=80.9289 (t=1.73), R²(decomp)=0.104 (++0.019 vs β-only); dominant component = **systematic variance**
  - **LW**: γ₁=-141.7126 (t=-6.14), γ₂=58.3554 (t=2.72), R²(decomp)=0.283 (++0.023 vs β-only); dominant component = **systematic variance**
  - **Gerber**: γ₁=-148.6045 (t=-3.45), γ₂=82.7990 (t=2.07), R²(decomp)=0.110 (++0.023 vs β-only); dominant component = **systematic variance**

#### Rates (2023-01-31)
  - **Sample**: γ₁=-96.3647 (t=-1.49), γ₂=-6.3096 (t=-0.10), R²(decomp)=0.032 (+-0.016 vs β-only); dominant component = **systematic variance**
  - **LW**: γ₁=-93.2523 (t=-2.04), γ₂=-3.1526 (t=-0.07), R²(decomp)=0.056 (+-0.028 vs β-only); dominant component = **systematic variance**
  - **Gerber**: γ₁=-86.3636 (t=-0.73), γ₂=-45.7342 (t=-0.41), R²(decomp)=0.014 (+-0.002 vs β-only); dominant component = **systematic variance**

---

## 3. Rolling Analysis: Within-Crisis Dynamics

### Average Coefficient Summary

| Crisis | Estimator | E[γ₁] | E[γ₂] | E[|γ₁|/|γ₂|] | E[R²(C)] |
|--------|-----------|--------|--------|--------------|---------|
| GFC | Sample | -297.4583 | 31.0551 | 9.71 | 0.145 |
| GFC | LW | -285.3303 | 29.7236 | 9.80 | 0.234 |
| GFC | Gerber | -292.2285 | 19.6842 | 9.88 | 0.048 |
| COVID | Sample | -212.3734 | 53.7070 | 6.17 | 0.077 |
| COVID | LW | -192.2188 | 46.5078 | 5.44 | 0.199 |
| COVID | Gerber | -127.5394 | 1.5379 | 1.50 | 0.065 |
| Rates | Sample | -193.5997 | 23.7063 | 9.58 | 0.043 |
| Rates | LW | -176.1727 | 21.4016 | 9.62 | 0.085 |
| Rates | Gerber | -164.4603 | -25.4357 | 8.64 | 0.017 |

**Interpretation**:
- **|γ₁|/|γ₂| > 1** → systematic variance is the more penalizing component on average
- **|γ₁|/|γ₂| < 1** → idiosyncratic variance dominates the weight allocation signal
- Ratio rising over a crisis window → correlation regime shift is increasing the role of systematic risk

---

## 4. Size Proxy Analysis (Model E)

> **Note**: Actual market capitalisation (shares outstanding) is not available in the OHLCV source
> files. **Dollar volume (Close × Volume)** is used as a size proxy.  Within S&P 100, dollar volume
> correlates strongly with market cap and institutional coverage, but results should be interpreted
> with this data limitation in mind.

### Size Proxy OLS Results (Model E = D + log dollar volume)

| Crisis | Estimator | γ₃ (size) | γ₂ (syst_share, E) | R²(D) | R²(E) | ΔR² |
|--------|-----------|-----------|-------------------|-------|-------|-----|
| GFC | Sample | 0.0086 | -0.1716** | 0.073 | 0.081 | 0.008 |
| GFC | LW | 0.0091 | -0.1883*** | 0.145 | 0.162 | 0.017 |
| GFC | Gerber | 0.0093 | -0.1665 | 0.048 | 0.053 | 0.005 |
| COVID | Sample | 0.0036 | -0.1911** | 0.072 | 0.073 | 0.001 |
| COVID | LW | 0.0042 | -0.1661*** | 0.225 | 0.228 | 0.003 |
| COVID | Gerber | -0.0001 | -0.1940** | 0.084 | 0.084 | 0.000 |
| Rates | Sample | 0.0189 | -0.0778 | 0.040 | 0.060 | 0.020 |
| Rates | LW | 0.0167* | -0.0844** | 0.079 | 0.110 | 0.031 |
| Rates | Gerber | 0.0236 | -0.0314 | 0.013 | 0.023 | 0.010 |

*Significance: * p<.10, ** p<.05, *** p<.01 (two-sided)*
*γ₃ = coefficient on log(avg daily dollar volume) = size proxy*
*ΔR² = R²(E) − R²(D): marginal contribution of size proxy beyond variance decomposition*

**Size proxy findings**: γ₃ < 0 in 1/9 cells (negative = larger-cap stocks get lower GMV weight).
R²(E) > R²(D) in 9/9 cells; size is significant (p<.10) in 1/9 cells.

---

## 5. Figures

| Figure | File | Description |
|--------|------|-------------|
| Fig 1 | `vardec_coef_snapshot.png` | Model C: γ₁ and γ₂ bar chart at crisis peaks per estimator |
| Fig 2 | `vardec_scatter_gfc.png` | Scatter of syst/idio var vs weight at GFC peak |
| Fig 3a | `vardec_rolling_coef.png` | Model C: rolling γ₁ and γ₂ time-series through crisis periods |
| Fig 3b | `vardec_rolling_coef_D.png` | **Model D (primary)**: rolling total_var and syst_share coefficients |
| Fig 4 | `vardec_r2_comparison.png` | R² comparison across all five models (A–E) |
| Fig 5 | `vardec_ratio_rolling.png` | Rolling |γ₁|/|γ₂| ratio (Model C) |
| Fig 6 | `vardec_size_coef.png` | Model E: γ₃ (size proxy) at crisis peaks + syst_share D vs E comparison |
| Fig 7 | `vardec_rolling_size.png` | Rolling γ₃ (size proxy coefficient) through crisis periods |

---

## 6. Estimator-Level Interpretation

### Sample Covariance
Uses the full empirical covariance without regularization. In a high-correlation crisis regime, the sample estimator may produce extreme precision matrix entries, amplifying the systematic component's influence on weights. Idiosyncratic estimates are noisy.

### Ledoit-Wolf (LW)
Shrinkage pulls the sample covariance toward a structured target (scaled identity), which **compresses the eigenvalue spread**. This reduces the penalty on large-eigenvalue (systematic) directions, potentially weakening γ₁ relative to Sample. R² improvement from decomposition should be smaller if LW already implicitly handles the decomposition via its shrinkage structure.

### Gerber (threshold=0.3)
The Gerber statistic only counts co-movements exceeding 0.3σ. Small idiosyncratic fluctuations are discarded; only large synchronized moves (systematic in nature) contribute to the correlation estimate.

**Observed result**: Gerber showed similar or weaker |γ₁| relative to Sample in 0/3 crises, partially supporting or not supporting the hypothesis that threshold filtering amplifies the systematic-variance signal.

---

## 7. Conclusions

1. **Sign of γ₁ and γ₂**: γ₁ < 0 in 100% of crisis×estimator cells; γ₂ < 0 in 33% — partially consistent with the Woodbury prediction that both variance components negatively predict GMV weight.

2. **Decomposition adds explanatory power**: R²(C) > R²(A) in 3/9 cells. Critically, R²(D) > R²(B) in 9/9 cells — meaning the *composition* effect (syst_share) explains weight variation beyond total variance level even in the collinearity-corrected specification.

3. **Collinearity in Model (C)**: Average VIF(C) = 1.8; 0/9 cells have VIF > 5. Collinearity is moderate — Model (C) and (D) results are broadly consistent.

4. **Estimator heterogeneity**: |γ₁|/|γ₂| differs across estimators, especially during active crisis windows. Gerber showed similar or weaker |γ₁| relative to Sample in 0/3 crises, not strongly supporting the hypothesis that threshold filtering amplifies the systematic-variance signal.

5. **Crisis dynamics**: The rolling |γ₁|/|γ₂| ratio tends to shift during crisis windows. This is consistent with the view that during market stress, the covariance structure becomes increasingly driven by common factors, changing how estimators translate variance decomposition into weight allocation.

6. **Implication**: The variance decomposition reveals estimator-level heterogeneity that beta-only regressions miss. Choosing an estimator that is "systematic-aware" (Gerber) versus one that treats all variance symmetrically (Sample) leads to materially different weight allocations precisely when crises make systematic risk dominant.

---

*Analysis code: `variance_decomp.py` | Figures: `results/figures/vardec_*.png`*


---

## 8. Sector Fixed-Effects Robustness

**Purpose**: Test whether γ₂(D) — the coefficient on `syst_share` in Model D — survives
GICS sector controls. If sector membership (not idiosyncratic risk) drove the syst_share
signal, adding sector dummies should absorb it. Conversely, if γ₂(D) survives, the
variance-decomposition narrative strengthens.

**Specification**: Model D + sector dummies (11 GICS sectors; InfoTech = reference, 10 dummies).
All-zero sector columns (sectors absent from each 252-day window) are dropped before OLS.

### γ₂(D) Comparison: No FE vs Sector FE

| Crisis | Estimator | γ₂ no-FE (t) | R²(D) | γ₂ with-FE (t) | R²(D+FE) | Sign stable? |
|--------|-----------|-------------|-------|----------------|----------|--------------|
| GFC | Sample | -0.1549** (-2.08) | 0.073 | -0.1835** (-2.04) | 0.151 | ✓ sig |
| GFC | LW | -0.1706*** (-3.19) | 0.145 | -0.1915*** (-3.02) | 0.247 | ✓ sig |
| GFC | Gerber | -0.1484 (-1.46) | 0.048 | -0.2070* (-1.70) | 0.135 | ✓ sig |
| COVID | Sample | -0.1899** (-2.21) | 0.072 | -0.3689** (-2.98) | 0.187 | ✓ sig |
| COVID | LW | -0.1647*** (-4.09) | 0.225 | -0.2084*** (-3.81) | 0.397 | ✓ sig |
| COVID | Gerber | -0.1941** (-2.65) | 0.084 | -0.3076** (-2.83) | 0.153 | ✓ sig |
| Rates | Sample | -0.0681 (-1.23) | 0.040 | -0.1437 (-1.63) | 0.085 | ✓  |
| Rates | LW | -0.0758* (-1.94) | 0.079 | -0.1410** (-2.28) | 0.132 | ✓ sig |
| Rates | Gerber | -0.0192 (-0.19) | 0.013 | -0.0549 (-0.33) | 0.037 | ✓  |

*Significance: * p<.10, ** p<.05, *** p<.01 (two-sided)*

### Summary

- **Sign flips after FE**: 0/9 cells. No sign flips — syst_share sign is robust to sector controls.
- **γ₂ significant after FE** (p<.10): 7/9 cells.
- **Average R² gain from sector FE**: +0.083 (R²(D+FE) − R²(D)); FE adds explanatory power beyond the variance decomposition.
- **Rates crisis**: γ₂(D) is broadly stable: 1/3 estimators significant after sector FE (same as without FE); sector controls neither absorb nor rescue the syst_share signal in the Rates period.

*Full FE comparison table: `reports/variance_decomp_sector_fe_table.csv`*
*Figure: `results/figures/vardec_sector_fe_gamma2.png`*
