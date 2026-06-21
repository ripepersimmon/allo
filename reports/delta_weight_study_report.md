# Δw Study — Crisis Weight-Shift Drivers

Market proxy: `ew`  |  Crises: ['GFC', 'COVID', 'Rates']  |  GBR backend: xgboost

Design: cross-sectional DiD on Δw = w_crisis − w_pre, predetermined pre-features, leave-one-crisis-out (LOCO) out-of-sample evaluation. See `2026-06-21_delta_weight_redesign_plan.md`.

## LOCO out-of-sample R² ladder

| model | Gerber | LW | Sample |
| --- | --- | --- | --- |
| M0 w_pre | 0.381 | 0.173 | 0.209 |
| M1 +Σ | 0.365 | 0.143 | 0.181 |
| M2 +nonΣ | 0.192 | -0.178 | -0.042 |
| M3 GBR | 0.268 | 0.106 | 0.093 |

_M3 over M2 = value of nonlinearity/interactions. Negative R² ⇒ no cross-crisis generalization (a finding, not a bug, given ~100 assets/crisis)._

## OLS (M2, HC3 robust) — Sample

| term | coef | se | t | p |
| --- | --- | --- | --- | --- |
| const | -0.0197 | 0.0614 | -0.3217 | 0.7477 |
| w_pre | -0.8054 | 0.1354 | -5.9498 | 0.0 |
| pre_total_var | 16.5645 | 12.053 | 1.3743 | 0.1693 |
| pre_syst_share | -0.0586 | 0.0521 | -1.1248 | 0.2607 |
| pre_avg_corr | 0.069 | 0.0809 | 0.8518 | 0.3943 |
| pre_beta | -0.0025 | 0.0056 | -0.4435 | 0.6574 |
| pre_amihud | 10.9092 | 16.56 | 0.6588 | 0.51 |
| pre_momentum | -0.0209 | 0.0087 | -2.404 | 0.0162 |
| pre_log_dolvol | 0.0026 | 0.0031 | 0.8172 | 0.4138 |
| pre_downside_vol | -2.0921 | 0.8386 | -2.4947 | 0.0126 |

## OLS (M2, HC3 robust) — LW

| term | coef | se | t | p |
| --- | --- | --- | --- | --- |
| const | -0.0197 | 0.0528 | -0.373 | 0.7091 |
| w_pre | -0.7413 | 0.1423 | -5.208 | 0.0 |
| pre_total_var | 12.2956 | 11.7097 | 1.05 | 0.2937 |
| pre_syst_share | -0.062 | 0.0451 | -1.3725 | 0.1699 |
| pre_avg_corr | 0.0867 | 0.0731 | 1.1867 | 0.2354 |
| pre_beta | -0.0013 | 0.0055 | -0.2316 | 0.8168 |
| pre_amihud | 10.6612 | 14.8764 | 0.7167 | 0.4736 |
| pre_momentum | -0.0176 | 0.0078 | -2.2454 | 0.0247 |
| pre_log_dolvol | 0.0021 | 0.0027 | 0.7749 | 0.4384 |
| pre_downside_vol | -1.7045 | 0.7403 | -2.3025 | 0.0213 |

## OLS (M2, HC3 robust) — Gerber

| term | coef | se | t | p |
| --- | --- | --- | --- | --- |
| const | 0.0239 | 0.0664 | 0.3601 | 0.7188 |
| w_pre | -0.9344 | 0.0984 | -9.492 | 0.0 |
| pre_total_var | 22.5162 | 12.4992 | 1.8014 | 0.0716 |
| pre_syst_share | -0.0749 | 0.0454 | -1.6489 | 0.0992 |
| pre_avg_corr | 0.1093 | 0.0758 | 1.4415 | 0.1494 |
| pre_beta | -0.0098 | 0.0066 | -1.4849 | 0.1376 |
| pre_amihud | 8.3751 | 19.7947 | 0.4231 | 0.6722 |
| pre_momentum | -0.011 | 0.0077 | -1.422 | 0.155 |
| pre_log_dolvol | 0.0002 | 0.0033 | 0.0658 | 0.9476 |
| pre_downside_vol | -1.9767 | 0.7865 | -2.5131 | 0.012 |

## Limitations

- **Survivorship bias**: 2024 S&P 100 universe applied to all crises; GFC results indicative only. Rerun with `--skip-gfc` for the post-2015 robustness cut.

- **Static GICS sectors** (2024) if sector dummies are added later.

- **Snapshot windows** for w_pre and w_crisis overlap partially; Δw is a two-point difference, not a clean event study.

- **Small sample** (~100 assets × N crises): GBR is for detecting nonlinearity, not precise prediction.
