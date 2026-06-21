# Multi-Factor Decomposition: FF3 and FF5 vs. Single-Factor

**Date**: 2026-05-24
**Estimators**: Sample, Ledoit-Wolf (LW), Gerber
**Window**: 252 trading days

---

## 1. Specification

For each asset *i* in the 252-day window, we compute:

| Symbol | Definition |
|--------|-----------|
| `mkt_syst_share` | adj-R² from regressing r_i on EW market return (k=2) |
| `ff3_syst_share` | adj-R² from regressing r_i on Mkt-RF, SMB, HML (k=4) |
| `ff5_syst_share` | adj-R² from regressing r_i on Mkt-RF, SMB, HML, RMW, CMA (k=6) |
| `avg_corr` | mean off-diagonal pairwise correlation in the window |

All syst_share measures use adj-R² so they are on a comparable scale. adj-R² penalty is small at n=252 (≤ 2 pp difference from raw R² for k≤6).

Cross-sectional models:
- **(D)**  w = α + γ₁·total_var + γ₂·mkt_syst_share
- **(G3)** w = α + γ₁·total_var + γ₂·ff3_syst_share
- **(G5)** w = α + γ₁·total_var + γ₂·ff5_syst_share
- **(H3)** w = α + γ₁·total_var + γ₂·ff3_syst_share + γ₃·avg_corr
- **(H5)** w = α + γ₁·total_var + γ₂·ff5_syst_share + γ₃·avg_corr

---

## 2. adj-R² Results

| Crisis | Est | adj-R²(D) | adj-R²(G3) Δ | adj-R²(G5) Δ | adj-R²(H3) Δ | adj-R²(H5) Δ | N |
|--------|-----|-----------|-------------|-------------|-------------|-------------|---|
| GFC | Sample | 0.052 | 0.056 (+0.004) | 0.043 (-0.009) | 0.054 (+0.002) | 0.055 (+0.003) | 93 |
| GFC | LW | 0.126 | 0.126 (-0.000) | 0.105 (-0.021) | 0.136 (+0.009) | 0.137 (+0.010) | 93 |
| GFC | Gerber | 0.026 | 0.028 (+0.001) | 0.020 (-0.006) | 0.028 (+0.002) | 0.031 (+0.005) | 93 |
| COVID | Sample | 0.053 | 0.048 (-0.005) | 0.052 (-0.001) | 0.069 (+0.015) | 0.061 (+0.008) | 100 |
| COVID | LW | 0.209 | 0.192 (-0.017) | 0.196 (-0.013) | 0.252 (+0.043) | 0.239 (+0.030) | 100 |
| COVID | Gerber | 0.065 | 0.071 (+0.005) | 0.072 (+0.007) | 0.063 (-0.003) | 0.064 (-0.001) | 100 |
| Rates | Sample | 0.020 | 0.014 (-0.006) | 0.015 (-0.005) | 0.030 (+0.010) | 0.022 (+0.002) | 100 |
| Rates | LW | 0.060 | 0.046 (-0.014) | 0.045 (-0.015) | 0.081 (+0.022) | 0.072 (+0.013) | 100 |
| Rates | Gerber | 0.000 | 0.000 (+0.000) | 0.000 (+0.000) | 0.006 (+0.006) | 0.000 (+0.000) | 100 |

Average adj-R² gain vs Model D: G3=-0.004, G5=-0.007, H3=+0.012, H5=+0.008
**Best model on average**: H3

---

## 3. γ₂ Coefficient Table (syst_share)

| Crisis | Est | γ₂(D) | γ₂(G3) | γ₂(G5) | γ₂(H3) | γ₂(H5) |
|--------|-----|-------|--------|--------|--------|--------|
| GFC | Sample | -0.1542** | -0.1433** | -0.1230* | -0.0282 | 0.0462 |
| GFC | LW | -0.1699*** | -0.1513*** | -0.1345*** | -0.0221 | 0.0361 |
| GFC | Gerber | -0.1478 | -0.1353 | -0.1138 | 0.0407 | 0.1088 |
| COVID | Sample | -0.1891** | -0.1803** | -0.2000** | 0.2645 | 0.1068 |
| COVID | LW | -0.1641*** | -0.1547*** | -0.1671*** | 0.1878 | 0.0939 |
| COVID | Gerber | -0.1933*** | -0.2028*** | -0.2168*** | -0.1163 | -0.1323 |
| Rates | Sample | -0.0678 | -0.0533 | -0.0589 | 0.1389 | 0.0653 |
| Rates | LW | -0.0755* | -0.0600 | -0.0614 | 0.1217 | 0.0694 |
| Rates | Gerber | -0.0191 | 0.0290 | 0.0340 | 0.3582 | 0.2515 |

*Significance: * p<.10  ** p<.05  *** p<.01 (two-sided, asymptotic z-thresholds: 1.645, 1.960, 2.576)*

---

## 4. γ₃ Coefficient Table (avg_corr in H models)

| Crisis | Est | γ₃(H3) | γ₃(H5) |
|--------|-----|--------|--------|
| GFC | Sample | -0.2920 | -0.4385 |
| GFC | LW | -0.3278 | -0.4423** |
| GFC | Gerber | -0.4467 | -0.5771 |
| COVID | Sample | -0.9439* | -0.6282 |
| COVID | LW | -0.7268*** | -0.5343** |
| COVID | Gerber | -0.1836 | -0.1730 |
| Rates | Sample | -0.4299 | -0.2827 |
| Rates | LW | -0.4063** | -0.2976* |
| Rates | Gerber | -0.7360 | -0.4949 |

*avg_corr is the mean off-diagonal pairwise correlation for that asset's 252-day window.*

---

## 5. Figures

| Figure | File |
|--------|------|
| adj-R² bars | `multifactor_decomp_r2.png` |
| γ₂ comparison | `multifactor_decomp_gamma.png` |

---

*Analysis code: `multifactor_decomp.py`*
