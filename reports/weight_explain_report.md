# Weight-Explanation Study — LW-GMV portfolio weight

Covariance: **Ledoit-Wolf only**  |  Target: weight level `w`  |  Market proxy: `ew`  |  Snapshots: 2005–2024 year-ends  |  N=1921 rows  |  GBR: xgboost

Design: annual non-overlapping snapshots (no daily-panel leakage); Σ-derived features reported only as a mechanical benchmark; Leave-One-Year-Out (LOYO) out-of-sample evaluation.

## LOYO out-of-sample R²

| model | LOYO R² | MAE |
| --- | --- | --- |
| B0 Σ-bench | 0.0589 | 0.0152 |
| M1 nonΣ | 0.1722 | 0.0154 |
| M2 +sector | 0.1755 | 0.0152 |
| M3 GBR | 0.3563 | 0.0101 |

- **B0 (Σ-bench)** `w ~ total_var + syst_share` (R²=0.059): a crude 2-variable proxy for the Σ-structure that mechanically sets `w`. For *unconstrained* GMV (`w ∝ Σ⁻¹1`) this would be near-tautological, but the long-only constraint and the correlation/precision off-diagonals break the clean `1/var` identity — so out-of-sample these two summary stats explain little, and the tautology risk is mild here.

- **Σ + non-Σ combined** R²=0.187 → non-Σ adds **+0.128** beyond the mechanical part.

- M1–M3 use non-Σ characteristics only (+ sectors). M3−M2 = nonlinearity gain.

## OLS — non-Σ + sectors (HC3 robust)

| term | coef | se | t | p |
| --- | --- | --- | --- | --- |
| const | 0.0115 | 0.0163 | 0.7013 | 0.4831 |
| log_dolvol | 0.0016 | 0.0008 | 2.0666 | 0.0388 |
| amihud | 6.469 | 4.6611 | 1.3879 | 0.1652 |
| momentum | -0.0022 | 0.0015 | -1.4528 | 0.1463 |
| beta | -0.034 | 0.0031 | -11.0692 | 0.0 |
| sec_CommSvcs | 0.0002 | 0.0028 | 0.0578 | 0.9539 |
| sec_ConsDis | 0.0032 | 0.0017 | 1.8311 | 0.0671 |
| sec_ConsStap | 0.0063 | 0.0034 | 1.8362 | 0.0663 |
| sec_Energy | 0.0006 | 0.0017 | 0.3759 | 0.707 |
| sec_Financials | 0.0073 | 0.0016 | 4.4966 | 0.0 |
| sec_HealthCare | -0.0011 | 0.0022 | -0.4969 | 0.6193 |
| sec_Industrials | -0.0021 | 0.0014 | -1.4576 | 0.1449 |
| sec_Materials | -0.0001 | 0.0038 | -0.0263 | 0.979 |
| sec_RealEstate | -0.0012 | 0.0032 | -0.3696 | 0.7117 |
| sec_Utilities | 0.0063 | 0.0057 | 1.1034 | 0.2698 |

## OLS — Σ benchmark (HC3 robust)

| term | coef | se | t | p |
| --- | --- | --- | --- | --- |
| const | 0.0265 | 0.0017 | 15.5336 | 0.0 |
| total_var | -3.9332 | 0.9194 | -4.2779 | 0.0 |
| syst_share | -0.0384 | 0.0035 | -10.9278 | 0.0 |

## GBR permutation importance (non-Σ + sectors)

| feature | importance |
| --- | --- |
| beta | 1.079078 |
| log_dolvol | 0.328504 |
| amihud | 0.283793 |
| momentum | 0.097757 |
| sec_ConsStap | 0.019778 |
| sec_Financials | 0.015858 |
| sec_Utilities | 0.012558 |
| sec_ConsDis | 0.011538 |
| sec_HealthCare | 0.005829 |
| sec_CommSvcs | 0.003577 |
| sec_RealEstate | 0.002676 |
| sec_Energy | 0.000417 |
| sec_Materials | 0.000323 |
| sec_Industrials | 1e-05 |

## SHAP interpretation (TreeExplainer, GBR)

Mean |SHAP value| = average magnitude of each feature's contribution to the predicted weight. Figures: `shap_summary.png` (beeswarm), `shap_dependence_{beta,log_dolvol,amihud}.png`.

| feature | mean_abs_shap |
| --- | --- |
| beta | 0.012634 |
| amihud | 0.00424 |
| log_dolvol | 0.003159 |
| momentum | 0.001611 |
| sec_ConsStap | 0.000824 |
| sec_Utilities | 0.00068 |
| sec_HealthCare | 0.000551 |
| sec_ConsDis | 0.00045 |
| sec_Financials | 0.000432 |
| sec_RealEstate | 0.000109 |
| sec_Energy | 9.2e-05 |
| sec_CommSvcs | 4.3e-05 |
| sec_Materials | 1.7e-05 |
| sec_Industrials | 5e-06 |

## Limitations

- **Descriptive, not causal**: `w` is a deterministic function of Σ; these characteristics are *associated* with weight, not causes of it.

- **Survivorship bias**: 2024 S&P 100 universe applied to all snapshot years.

- **Static 2024 GICS sectors** applied to all years.

- **Concentrated target**: long-only `w` is highly right-skewed — most assets carry near-zero weight and a few dominate; OLS on a skewed target is descriptive only.
