# Variable Mining: GMV Weight Predictor Search

**Date**: 2026-05-23
**Universe**: S&P 100, three crisis peaks (GFC / COVID / Rates)
**Estimators**: Sample, Ledoit-Wolf (LW), Gerber (threshold=0.3)
**Method**: Unconstrained GMV (w ∝ Σ⁻¹1), 252-day estimation window

---

## 1. Candidate Variables (24 total)

| Group | Variables | Theoretical motivation |
|-------|-----------|----------------------|
| VarDecomp | β, β², total σ², syst σ², idio σ², syst_share | Baseline decomposition |
| **Woodbury** | **1/σ²_ε, β/σ²_ε, β²/σ²_ε** | **Exact GMV formula terms** |
| Vol levels | σ, σ_ε, β·σ_m | Level vs squared risk |
| Moments | skewness, ex_kurtosis, downside vol, VaR 5%, CVaR 5%, max drawdown | Tail / asymmetry risk |
| CorrLiq | avg pairwise corr, autocorr lag-1, Amihud, log(dolvol)† | Correlation structure / liquidity |

† SIZE PROXY: log average daily dollar volume (Close×Volume). True market cap unavailable (OHLCV-only data).

---

## 2. Univariate OLS Results (z-scored variables)

Selection threshold: **|t| > 1.65** (p < .10, two-sided) in **≥ 4 of 9** crisis×estimator cells.

**Selected variables** (13 of 22): β (market beta), β², Systematic σ², Syst. share (R²_mkt), 1/σ²_ε  [Woodbury], β²/σ²_ε, Total vol σ, Syst. vol β·σ_m, Downside vol, VaR 5% (loss), CVaR 5% (loss), Max drawdown, Avg pairwise corr

### Full Univariate t-stat Table (sorted by significance count)

| Variable | Group | Exp. sign | GFC/Sample | GFC/LW | GFC/Gerber | COVID/Sample | COVID/LW | COVID/Gerber | Rates/Sample | Rates/LW | Rates/Gerber | Sig cells |
|---|---|---||---||---||---||---||---||---||---||---||---|---|
| Downside vol | Moments | − | -1.98* | -2.58** | -1.72* | -1.97* | -3.67*** | -2.12** | -1.82* | -2.21** | -1.83* | **9/9** |
| β (market beta) | VarDecomp | −? | -2.96** | -3.96*** | -2.26** | -3.03*** | -5.88*** | -3.05*** | -2.23** | -2.99** | -1.22 | **8/9** |
| β² | VarDecomp | − | -2.33** | -3.12*** | -1.90* | -2.85** | -5.39*** | -2.72** | -1.79* | -2.41** | -1.08 | **8/9** |
| Systematic σ² | VarDecomp | − | -2.33** | -3.12*** | -1.90* | -2.85** | -5.39*** | -2.72** | -1.79* | -2.41** | -1.08 | **8/9** |
| Total vol σ | Vol | − | -2.38** | -3.11*** | -1.94* | -2.11** | -4.11*** | -1.99* | -1.96* | -2.53** | -1.37 | **8/9** |
| Syst. vol β·σ_m | Vol | − | -2.96** | -3.96*** | -2.26** | -3.03*** | -5.88*** | -3.05*** | -2.23** | -2.99** | -1.22 | **8/9** |
| VaR 5% (loss) | Moments | − | -2.36** | -3.05*** | -1.98* | -1.77* | -3.89*** | -1.94* | -1.88* | -2.38** | -0.97 | **8/9** |
| CVaR 5% (loss) | Moments | − | -2.14** | -2.78** | -1.82* | -1.98* | -3.77*** | -2.00** | -1.96* | -2.42** | -1.63 | **8/9** |
| 1/σ²_ε  [Woodbury] | Woodbury | + | +3.58*** | +4.02*** | +1.98* | +1.57 | +1.85* | +1.32 | +2.16** | +2.09** | +2.09** | **7/9** |
| Syst. share (R²_mkt) | VarDecomp | − | -1.72* | -2.63** | -1.15 | -2.11** | -3.70*** | -2.56** | -1.46 | -2.24** | -0.37 | **6/9** |
| β²/σ²_ε | Woodbury | − | -1.88* | -2.75** | -0.81 | -1.74* | -3.35*** | -2.05** | -1.17 | -1.98* | -0.33 | **6/9** |
| Max drawdown | Moments | − | -2.59** | -3.43*** | -1.22 | -1.46 | -2.73** | -1.80* | -1.67* | -2.10** | -0.93 | **6/9** |
| Avg pairwise corr | CorrLiq | − | -1.43 | -2.14** | -1.00 | -2.19** | -3.65*** | -2.42** | -1.56 | -2.31** | -0.39 | **5/9** |
| Total σ² | VarDecomp | − | -1.63 | -2.16** | -1.52 | -1.60 | -3.14*** | -1.34 | -1.59 | -2.09** | -1.14 | **3/9** |
| Amihud illiquidity | CorrLiq | ? | -0.89 | -1.17 | -0.72 | -0.87 | -1.96* | -0.68 | -1.60 | -2.02** | -1.27 | **2/9** |
| Idio. vol σ_ε | Vol | − | -1.63 | -2.07** | -1.52 | -0.37 | -1.01 | -0.09 | -1.22 | -1.44 | -1.12 | **1/9** |
| Autocorr lag-1 | CorrLiq | ? | +0.25 | +0.43 | +0.33 | +0.85 | +1.27 | -0.50 | +1.57 | +1.83* | +0.95 | **1/9** |
| Idiosyncratic σ² | VarDecomp | − | -0.96 | -1.27 | -1.09 | -0.14 | -0.69 | +0.18 | -0.98 | -1.25 | -0.90 | **0/9** |
| β/σ²_ε  [Woodbury corr] | Woodbury | − | +1.25 | +1.08 | +0.93 | -0.27 | -1.09 | -0.70 | +0.10 | -0.51 | +0.68 | **0/9** |
| Skewness | Moments | ? | -0.30 | -0.42 | -0.73 | +0.84 | +1.28 | +0.77 | -0.28 | -0.57 | +0.17 | **0/9** |
| Excess kurtosis | Moments | − | +0.49 | +0.46 | +0.58 | -0.43 | -1.04 | -1.00 | -0.56 | -0.83 | -1.20 | **0/9** |
| log(dollar vol) [SIZE†] | CorrLiq | ? | -0.02 | -0.01 | -0.02 | +0.27 | +0.62 | -0.02 | +0.30 | +0.27 | +0.24 | **0/9** |

*Values = standardised t-statistic. * p<.10  ** p<.05  *** p<.01 (two-sided)*

---

## 3. Multivariate OLS (significant variables, VIF ≤ 5 enforced)

| Crisis | Estimator | Variable | β* | t-stat | VIF | R² | N |
|--------|-----------|----------|----|--------|-----|----|---|
| GFC | Sample | Systematic σ² | -0.0010 | -0.08 | 2.8 | 0.251 | 93 |
| GFC | Sample | 1/σ²_ε  [Woodbury] | +0.0565*** | +4.18 | 3.6 | 0.251 | 93 |
| GFC | Sample | β²/σ²_ε | -0.0324*** | -3.86 | 1.4 | 0.251 | 93 |
| GFC | Sample | Max drawdown | +0.0265* | +1.71 | 4.7 | 0.251 | 93 |
| GFC | LW | Systematic σ² | -0.0025 | -0.30 | 2.8 | 0.339 | 93 |
| GFC | LW | 1/σ²_ε  [Woodbury] | +0.0429*** | +4.51 | 3.6 | 0.339 | 93 |
| GFC | LW | β²/σ²_ε | -0.0291*** | -4.92 | 1.4 | 0.339 | 93 |
| GFC | LW | Max drawdown | +0.0176 | +1.61 | 4.7 | 0.339 | 93 |
| GFC | Gerber | Systematic σ² | -0.0233 | -1.33 | 2.8 | 0.100 | 93 |
| GFC | Gerber | 1/σ²_ε  [Woodbury] | +0.0456** | +2.29 | 3.6 | 0.100 | 93 |
| GFC | Gerber | β²/σ²_ε | -0.0227* | -1.84 | 1.4 | 0.100 | 93 |
| GFC | Gerber | Max drawdown | +0.0424* | +1.86 | 4.7 | 0.100 | 93 |
| COVID | Sample | 1/σ²_ε  [Woodbury] | +0.0544*** | +3.29 | 2.4 | 0.182 | 100 |
| COVID | Sample | Downside vol | -0.0170 | -0.98 | 2.6 | 0.182 | 100 |
| COVID | Sample | VaR 5% (loss) | +0.0210 | +1.15 | 2.9 | 0.182 | 100 |
| COVID | Sample | Avg pairwise corr | -0.0599*** | -3.99 | 1.9 | 0.182 | 100 |
| COVID | LW | 1/σ²_ε  [Woodbury] | +0.0304*** | +4.07 | 2.4 | 0.368 | 100 |
| COVID | LW | Downside vol | -0.0112 | -1.44 | 2.6 | 0.368 | 100 |
| COVID | LW | VaR 5% (loss) | +0.0032 | +0.39 | 2.9 | 0.368 | 100 |
| COVID | LW | Avg pairwise corr | -0.0392*** | -5.79 | 1.9 | 0.368 | 100 |
| COVID | Gerber | 1/σ²_ε  [Woodbury] | +0.0423** | +2.98 | 2.4 | 0.181 | 100 |
| COVID | Gerber | Downside vol | -0.0162 | -1.09 | 2.6 | 0.181 | 100 |
| COVID | Gerber | VaR 5% (loss) | +0.0157 | +1.00 | 2.9 | 0.181 | 100 |
| COVID | Gerber | Avg pairwise corr | -0.0508*** | -3.93 | 1.9 | 0.181 | 100 |
| Rates | Sample | Systematic σ² | +0.0263 | +1.39 | 4.6 | 0.151 | 100 |
| Rates | Sample | 1/σ²_ε  [Woodbury] | +0.0484*** | +3.05 | 3.2 | 0.151 | 100 |
| Rates | Sample | Downside vol | -0.0164 | -0.86 | 4.6 | 0.151 | 100 |
| Rates | Sample | Max drawdown | +0.0149 | +0.82 | 4.2 | 0.151 | 100 |
| Rates | Sample | Avg pairwise corr | -0.0509*** | -3.26 | 3.1 | 0.151 | 100 |
| Rates | LW | Systematic σ² | +0.0190 | +1.43 | 4.6 | 0.197 | 100 |
| Rates | LW | 1/σ²_ε  [Woodbury] | +0.0340*** | +3.06 | 3.2 | 0.197 | 100 |
| Rates | LW | Downside vol | -0.0173 | -1.31 | 4.6 | 0.197 | 100 |
| Rates | LW | Max drawdown | +0.0132 | +1.05 | 4.2 | 0.197 | 100 |
| Rates | LW | Avg pairwise corr | -0.0420*** | -3.86 | 3.1 | 0.197 | 100 |
| Rates | Gerber | Systematic σ² | +0.0390 | +1.10 | 4.6 | 0.104 | 100 |
| Rates | Gerber | 1/σ²_ε  [Woodbury] | +0.0674** | +2.27 | 3.2 | 0.104 | 100 |
| Rates | Gerber | Downside vol | -0.0535 | -1.51 | 4.6 | 0.104 | 100 |
| Rates | Gerber | Max drawdown | +0.0483 | +1.43 | 4.2 | 0.104 | 100 |
| Rates | Gerber | Avg pairwise corr | -0.0671** | -2.30 | 3.1 | 0.104 | 100 |

*β* = standardised coefficient (z-scored variable). VIF ≤ 5 enforced by iterative drop.*

---

## 4. Figures

| Figure | File | Description |
|--------|------|-------------|
| Fig A | `varmine_tstat_heatmap.png` | t-stat heatmap for all 24 variables × 9 crisis×estimator cells |
| Fig B | `varmine_multivariate.png` | Multivariate β* for selected variables, per crisis |

---

## 5. Conclusions

1. **Woodbury terms dominate**: `1/σ²_ε` and `β/σ²_ε` are expected to be the most significant — they directly appear in the analytical GMV weight formula.
2. **Higher moments**: Tail risk variables (CVaR, max drawdown) may carry additional signal beyond variance.
3. **Average pairwise correlation**: High avg_corr → less diversification benefit → potentially lower weight.
4. **Size proxy**: log dollar volume showed no significance in prior analysis (0/9 cells at p<.10); serves as a null-result benchmark.

---

*Analysis code: `variable_mining.py` | Figures: `results/figures/varmine_*.png`*
