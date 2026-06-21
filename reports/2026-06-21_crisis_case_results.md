# Crisis Case Study — Results Report

**Date:** 2026-06-21
**Experiment:** `crisis_case_study.py` (crisis definition: `src/crises.py`)
**Environment:** conda `allo` (Python 3.13)
**References:** `references_weight_explain.md` — [CST2011], [LS2001], [FP2014], [JM2003]
**Auto-dump:** `crisis_case_report.md` · **Summary:** `crisis_case_summary.csv`

---

## 1. Question & design

**How does the Ledoit-Wolf long-only GMV portfolio's weight structure change through a
crisis?** A descriptive, per-episode case study (not cross-crisis inference).

- **Crisis definition (objective):** VIX two-threshold hysteresis — enter when VIX>30,
  exit when VIX<20; min 10 trading days; merge episodes <42 td apart; label by peak-year
  event. → **8 episodes, 2005–2024** (`reports/vix_crisis_periods.csv`). Recovers the 3
  original `CRISIS_PERIODS` plus 5 (Flash Crash/EU 2010, EU debt 2011, China 2015,
  Volmageddon 2018, Rates 2022).
- **Per crisis:** PRE window (252 td ending ~63 td *before* onset = calm baseline) vs
  PEAK window (252 td ending at the VIX peak); plus a weekly-sampled timeline across
  pre→peak→recovery.
- **Metrics:** portfolio weighted-average beta, Effective-N, top-5 / low-β-decile weight
  share, and the cross-sectional `w ~ beta + amihud + size + momentum` (HC3) coefficients.
- **Theoretical lens:** [CST2011] — long-only MVP weight is a function of beta with a
  **volatility-dependent threshold**; it should move in a crisis.

---

## 2. Cross-crisis summary (EW proxy)

| Crisis | peak | VIX | Port β pre→peak | Eff-N pre→peak | β-slope pre→peak |
|---|---|---|---|---|---|
| GFC (Bear) | 2008-03 | 32 | 0.49 → 0.41 | 13.3 → 5.6 | −0.031 → −0.060 |
| GFC (Lehman) | 2008-11 | 81 | 0.39 → **0.52** | 5.1 → 7.3 | −0.054 → −0.049 |
| Flash Crash/EU | 2010-05 | 46 | 0.36 → 0.46 | 8.6 → 9.4 | −0.019 → −0.045 |
| EU debt/downgrade | 2011-08 | 48 | 0.47 → 0.46 | 9.6 → 9.5 | −0.047 → −0.056 |
| China deval. | 2015-08 | 41 | 0.69 → 0.71 | 14.3 → 12.1 | −0.076 → −0.083 |
| Volmageddon (Q4'18) | 2018-12 | 36 | 0.57 → 0.55 | 17.6 → 15.1 | −0.049 → −0.047 |
| COVID-19 | 2020-03 | 83 | 0.52 → **0.60** | 16.5 → **8.5** | −0.040 → **−0.072** |
| Rate hikes/Ukraine | 2022-03 | 37 | 0.56 → 0.55 | 14.3 → 19.2 | −0.018 → −0.028 |

---

## 3. Key findings

### F1 — Severe crises concentrate the portfolio
Effective-N collapses into the peak in the two most severe episodes: **COVID 16.5 → 8.5**,
**GFC-Bear 13.3 → 5.6** (and top-5 / low-β-decile weight share spike — see
`case_202003_COVID-19.png`). Milder episodes (EU, China, Volmageddon) barely move; the
slow 2022 rate-hike regime even *diversifies* (14.3 → 19.2). **Concentration is a
severe-shock phenomenon, not a generic crisis feature.**

### F2 — A two-layer beta effect (the headline nuance)
1. **The cross-sectional low-β preference *intensifies*** — the `β`-weight slope steepens
   (more negative) into the peak in 6 of 8 crises (COVID −0.040 → −0.072, Flash −0.019 →
   −0.045, GFC-Bear −0.031 → −0.060). This is the empirical analog of **[CST2011]'s
   threshold beta tightening**: GMV pulls harder toward low-beta names.
2. **Yet the *absolute* portfolio beta rises** in the most severe crises (COVID 0.52 →
   0.60, GFC-Lehman 0.39 → 0.52). Not a contradiction: as correlations spike, the whole
   cross-section of betas **compresses toward 1** ([LS2001] — correlations rise in bear
   markets), so even an intensified low-beta tilt lands on a higher absolute beta.

> **The GMV portfolio de-risks in *relative* (cross-sectional) terms but cannot escape
> the market-wide beta compression — so "GMV gets more defensive in a crisis" is true of
> the tilt, false of the realized portfolio beta.** This corrects the naive de-risking
> hypothesis.

### F3 — Flight to defensives (named cases)
Weight gainers are consistently low-beta defensives: **COVID** → VZ (β0.41), JNJ (β0.65);
**GFC-Lehman** → SO (utility, β0.58), ABT, PEP; **Flash** → WMT (β0.31), MDLZ; **2022** →
LMT, TMO, JNJ. Losers are often previously-overweighted defensives mean-reverting (JNJ/PG
in GFC-Lehman) or higher-beta names (BA β0.92 in China, C β1.13 in Volmageddon).

### F4 — Liquidity signal is inconclusive (honest null)
The `amihud` (illiquidity) coefficient does **not** shift consistently pre→peak (COVID
334 → 202 *down*; GFC-Bear −31 → +48; EU ~0). **No clean flight-to-liquidity signal** in
GMV weights at this resolution — reported as a null, not forced into a narrative.

---

## 4. Robustness

| Axis | Variant | Result |
|---|---|---|
| **VIX threshold** | percentile 90/60 (≈28.7/18.5) vs fixed 30/20 | **8 vs 9 episodes; 8 identical.** Percentile adds only the borderline Feb-2018 Volmageddon. Crisis set is stable. |
| **Market proxy** | SPY vs equal-weighted | **Portfolio-β pre→peak direction identical in all 8** (COVID rises under both: ew 0.52→0.60, spy 0.49→0.63; GFC-Bear falls under both). Effective-N is *identical* (GMV weights don't depend on the proxy). |

Robustness artifact: `reports/crisis_case_summary_spy.csv`. Conclusions F1–F3 hold under
both variants.

> Naming caveat surfaced by the threshold check: the fixed-30/20 "Volmageddon 2018" label
> sits on the **Q4-2018 selloff** (peak Dec-24); the *actual* Feb-2018 Volmageddon (XIV
> blow-up, VIX 37) only survives under the percentile scheme. Year-based labeling is
> approximate.

---

## 5. Connection to prior research
- **[CST2011] Minimum-Variance Portfolio Composition** — threshold-beta structure; F2.1
  (slope steepening) is its empirical crisis analog.
- **[LS2001] Extreme Correlation of International Equity Markets** — correlations rise in
  bear markets; explains F2.2 (beta compression → absolute portfolio β rises).
- **[FP2014] Betting Against Beta / [BBW2011] Low-Vol Anomaly** — frame the persistent
  low-beta tilt as the defensive-equity premium.
- **[JM2003] Wrong Constraints Help** — the long-only constraint shapes which low-beta
  names actually receive weight.

---

## 6. Reproduction
```bash
conda activate allo
python fetch_vix.py                       # once
python -m src.crises                      # regenerate vix_crisis_periods.csv
python crisis_case_study.py               # EW proxy (this report)
python crisis_case_study.py --proxy spy   # proxy robustness
```

## 7. Limitations
- **Descriptive, not inferential** — per-episode characterization; 8 crises do not support
  statistical generalization.
- **Survivorship bias** — 2024 S&P 100 universe; GFC episodes miss failed financials
  (Lehman, Bear, WaMu), so GFC weight dynamics are partial.
- **Beta endogeneity** — betas estimated on crisis-contaminated windows; the β-weight
  relationship is descriptive, and F2.2 is itself a manifestation of this.
- **Static 2024 GICS sectors; approximate year-based crisis labels; pre/peak window
  overlap** for the long episodes.
