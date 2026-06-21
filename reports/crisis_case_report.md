# Crisis Case Study — LW-GMV weight shifts through VIX-defined crises

Covariance: Ledoit-Wolf  |  Market proxy: `ew`  |  Crises: VIX hysteresis (enter>30 / exit<20), 8 episodes  |  Pre = 252-td window ending ~63 td before onset (calm baseline); Peak = 252-td window at VIX peak.

Theoretical lens: Clarke, de Silva & Thorley (2011) — long-only MVP weight is a function of beta with a volatility-dependent threshold (see `references_weight_explain.md`). Descriptive per-episode case study.

## Cross-crisis summary — pre vs peak

| crisis | peak_date | peak_vix | pre_port_beta | peak_port_beta | pre_eff_n | peak_eff_n | pre_amihud_coef | peak_amihud_coef |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GFC | 2008-03-17 | 32.2 | 0.485 | 0.411 | 13.284 | 5.564 | -30.958 | 47.579 |
| GFC | 2008-11-20 | 80.9 | 0.391 | 0.521 | 5.098 | 7.27 | 29.015 | -23.577 |
| Flash Crash / EU | 2010-05-20 | 45.8 | 0.36 | 0.459 | 8.616 | 9.37 | -7.253 | 76.542 |
| EU debt / US downgrade | 2011-08-08 | 48.0 | 0.472 | 0.46 | 9.581 | 9.471 | 2.209 | -0.629 |
| China devaluation | 2015-08-24 | 40.7 | 0.691 | 0.708 | 14.289 | 12.094 | 29.372 | 23.67 |
| Volmageddon | 2018-12-24 | 36.1 | 0.565 | 0.545 | 17.628 | 15.125 | 144.968 | 638.063 |
| COVID-19 | 2020-03-16 | 82.7 | 0.517 | 0.596 | 16.548 | 8.488 | 333.905 | 201.837 |
| Rate hikes / Ukraine | 2022-03-07 | 36.5 | 0.563 | 0.553 | 14.32 | 19.246 | -79.946 | 109.538 |

- **Portfolio β pre→peak**: does the GMV portfolio de-risk into the crisis peak?
- **Effective N pre→peak**: does it concentrate?
- **Amihud coef pre→peak**: more negative/positive at peak ⇒ liquidity matters more (flight to liquidity).


## GFC — peak 2008-03-17 (VIX 32.2)

- Portfolio β: 0.485 → 0.411  |  Effective N: 13.3 → 5.6

- β-weight slope: -0.0307 → -0.0597  |  amihud coef: -30.958 → 47.579

- Top weight **gainers**: JNJ (+17.4%, β0.41), PG (+13.4%, β0.62), BRK-B (+10.6%, β0.33)

- Top weight **losers**: USB (-11.2%, β0.53), PEP (-5.9%, β0.44), D (-5.9%, β0.49)

- Figure: `results/figures/crisis_case/`


## GFC — peak 2008-11-20 (VIX 80.9)

- Portfolio β: 0.391 → 0.521  |  Effective N: 5.1 → 7.3

- β-weight slope: -0.0537 → -0.0488  |  amihud coef: 29.015 → -23.577

- Top weight **gainers**: SO (+14.4%, β0.58), ABT (+11.0%, β0.55), PEP (+10.3%, β0.49)

- Top weight **losers**: JNJ (-21.0%, β0.36), PG (-8.0%, β0.44), MO (-7.9%, β0.51)

- Figure: `results/figures/crisis_case/`


## Flash Crash / EU — peak 2010-05-20 (VIX 45.8)

- Portfolio β: 0.360 → 0.459  |  Effective N: 8.6 → 9.4

- β-weight slope: -0.0186 → -0.0450  |  amihud coef: -7.253 → 76.542

- Top weight **gainers**: WMT (+7.4%, β0.31), BRK-B (+5.9%, β0.85), MDLZ (+5.9%, β0.49)

- Top weight **losers**: MO (-9.6%, β0.30), GILD (-6.0%, β0.36), MCD (-4.8%, β0.43)

- Figure: `results/figures/crisis_case/`


## EU debt / US downgrade — peak 2011-08-08 (VIX 48.0)

- Portfolio β: 0.472 → 0.460  |  Effective N: 9.6 → 9.5

- β-weight slope: -0.0470 → -0.0557  |  amihud coef: 2.209 → -0.629

- Top weight **gainers**: PEP (+8.2%, β0.55), MCD (+5.9%, β0.59), DUK (+5.5%, β0.53)

- Top weight **losers**: CHTR (-11.3%, β0.27), ABT (-6.9%, β0.53), WMT (-6.1%, β0.44)

- Figure: `results/figures/crisis_case/`


## China devaluation — peak 2015-08-24 (VIX 40.7)

- Portfolio β: 0.691 → 0.708  |  Effective N: 14.3 → 12.1

- β-weight slope: -0.0756 → -0.0833  |  amihud coef: 29.372 → 23.670

- Top weight **gainers**: T (+4.4%, β0.66), SPG (+4.2%, β0.67), KO (+2.7%, β0.53)

- Top weight **losers**: COST (-7.2%, β0.70), BA (-4.0%, β0.92), PEP (-3.1%, β0.68)

- Figure: `results/figures/crisis_case/`


## Volmageddon — peak 2018-12-24 (VIX 36.1)

- Portfolio β: 0.565 → 0.545  |  Effective N: 17.6 → 15.1

- β-weight slope: -0.0485 → -0.0465  |  amihud coef: 144.968 → 638.063

- Top weight **gainers**: D (+6.3%, β0.22), BK (+4.4%, β1.12), DUK (+4.0%, β0.15)

- Top weight **losers**: OXY (-6.4%, β0.76), NEE (-5.1%, β0.24), C (-4.6%, β1.13)

- Figure: `results/figures/crisis_case/`


## COVID-19 — peak 2020-03-16 (VIX 82.7)

- Portfolio β: 0.517 → 0.596  |  Effective N: 16.5 → 8.5

- β-weight slope: -0.0400 → -0.0723  |  amihud coef: 333.905 → 201.837

- Top weight **gainers**: VZ (+16.7%, β0.41), GILD (+12.8%, β0.89), JNJ (+10.9%, β0.65)

- Top weight **losers**: MCD (-11.3%, β0.43), AMT (-10.4%, β0.22), DUK (-9.2%, β0.26)

- Figure: `results/figures/crisis_case/`


## Rate hikes / Ukraine — peak 2022-03-07 (VIX 36.5)

- Portfolio β: 0.563 → 0.553  |  Effective N: 14.3 → 19.2

- β-weight slope: -0.0178 → -0.0279  |  amihud coef: -79.946 → 109.538

- Top weight **gainers**: LMT (+8.7%, β0.72), TMO (+5.3%, β0.40), JNJ (+4.7%, β0.60)

- Top weight **losers**: T (-8.3%, β0.58), MMM (-6.9%, β0.66), VZ (-4.9%, β0.45)

- Figure: `results/figures/crisis_case/`


## Limitations

- **Descriptive, not inferential** — per-episode characterization, not a test across crises.

- **Survivorship bias** — 2024 S&P 100 universe; GFC episodes miss failed financials (Lehman, Bear, WaMu), so GFC weight dynamics are partial.

- **Beta endogeneity** — betas are estimated on crisis-contaminated windows (betas spike/compress in crises); interpret the β-weight relationship descriptively.

- **Static 2024 GICS**; window-overlap between pre and peak snapshots.
