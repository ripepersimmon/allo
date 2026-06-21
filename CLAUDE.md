# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.




## Project

Research project studying how covariance estimators allocate GMV portfolio weights during financial crises. Compares Sample, Ledoit-Wolf (LW), and Gerber estimators across three crisis periods.

## Setup

```bash
source .venv/bin/activate
```

All scripts must be run from the repo root with the venv active. No build or install step needed.

## Running Analysis Scripts

```bash
python bbc_analysis.py          # BBC precision-matrix permutation, figures saved to results/figures/
python beta_weight.py           # Market beta vs GMV weight scatter + rolling correlation
python crisis_weight_test.py    # Statistical tests: pre-crisis vs crisis weight allocation
python longonly_gmv_analysis.py # Long-only GMV: rolling weights, Effective-N, turnover, realized variance
python intervention_analysis.py # Decay-form OLS: per-asset weight panel regressed on characteristics + crisis term
python variance_decomp.py       # Cross-sectional OLS: systematic vs idiosyncratic variance vs GMV weight
python variable_mining.py       # 24-variable cross-sectional predictor search (univariate + multivariate)
jupyter notebook notebooks/crisis_study.ipynb  # Main exploratory notebook
```

## Architecture

Data flows in one direction: `sp500/` parquet files → `src/data_loader.py` → `src/estimators.py` → `src/portfolio.py` → `src/analysis.py` → figures/reports.

**`src/data_loader.py`**  
Loads per-ticker parquet files from `sp500/` (format: `sp500/AAPL.parquet`, column `Close`). `TICKERS` is the canonical S&P 100 list used everywhere. Returns a wide price DataFrame; `compute_returns()` produces log returns. `load_dollar_volume()` builds a Close×Volume proxy (used as a size proxy; actual shares-outstanding unavailable). `load_field_from_parquet()` is the generic single-column loader underlying both price and volume helpers.

**`src/estimators.py`**  
Three covariance estimators, all with signature `(returns: pd.DataFrame) -> np.ndarray`:
- `sample_cov` — plain `.cov()`
- `lw_cov` — sklearn `LedoitWolf`
- `gerber_cov(threshold=0.3)` — threshold-based co-movement statistic; projects to nearest PSD correlation then scales to covariance
- `bbc_permutation(A)` — Algorithm 1 from Kim et al. (2025): permutes a precision matrix to expose high/low row-sum blocks. Returns a permutation index array.

**`src/portfolio.py`**  
- `gmv_unconstrained(cov)` — analytical `Σ⁻¹1 / 1'Σ⁻¹1`; falls back to `pinv` if singular
- `gmv_long_only(cov)` — cvxpy QP with CLARABEL solver; falls back to equal weight on solver failure
- `effective_n(weights)` — `1 / Σwᵢ²` (Herfindahl inverse)
- `turnover(w_prev, w_curr)` — `0.5 * Σ|Δwᵢ|`

**`src/analysis.py`**  
- `rolling_gmv(returns, estimator_fn, window, constrained)` — rolls a window, excludes NaN tickers per window, returns a weight DataFrame aligned to `TICKERS`
- `CRISIS_PERIODS` dict is the canonical definition of the three crisis windows
- Plotting helpers: `plot_weights`, `plot_concentration`, `plot_cov_heatmap`, `compare_turnover`, `summary_table`

## Additional Analysis Scripts

**`longonly_gmv_analysis.py`**  
Comprehensive long-only experiment: runs `rolling_gmv(constrained=True)` and `rolling_gmv(constrained=False)` for all three estimators, then produces ~11 figure types (area charts, Effective-N overlays, realized variance, turnover bars, heatmaps, boxplots, volcano plots). Figures saved to `results/figures/longonly/`.

**`intervention_analysis.py`**  
Decay-form (Box-Tiao transfer function) panel OLS: `weight_it = controls + ω·δ^(t−T₀)·crisis_indicator`. Regressors: market beta, average correlation, Amihud illiquidity, momentum. Selects δ via BIC over `HALFLIFE_GRID = [5, 10, 21, 42, 63]` trading days. Outputs OLS summaries per estimator and coefficient comparison figure. Uses `LEAD_IN=60` trading days before each crisis onset.

**`variance_decomp.py`**  
Five cross-sectional OLS models (A–E) at crisis-peak snapshots and rolling windows. Market proxy = equal-weighted portfolio (consistent with `beta_weight.py`). Model D (total_var + syst_share) is the primary specification; Model E adds log dollar-volume as size proxy.

**`variable_mining.py`**  
24-variable predictor search across 9 cells (3 estimators × 3 crisis peaks). Pipeline: (A) univariate OLS t-stat heatmap; (B) multivariate with variables significant in ≥4/9 cells; (C) VIF check (threshold 5). Variables span five groups: variance decomposition, Woodbury-direct terms, volatility levels, higher moments/risk, correlation/liquidity.

## Crisis Periods

Defined in `src/analysis.py::CRISIS_PERIODS` and used consistently across all scripts:
- GFC: 2007-01-01 → 2009-06-30
- COVID: 2019-10-01 → 2020-09-30
- Rates: 2021-07-01 → 2023-01-31

Pre-crisis windows used in `crisis_weight_test.py` (not in `CRISIS_PERIODS`): GFC pre = 2005-2006, COVID pre = 2018–2019-09, Rates pre = 2019-07–2021-06.

## Key Conventions

- All scripts import with `sys.path.insert(0, '.')` and must be run from repo root.
- Market proxy for beta calculations: equal-weighted return of all assets in the estimation window (no external index).
- GMV analytical solution: `w ∝ Σ⁻¹1` (row sums of precision matrix). The BBC algorithm exploits this — assets with large precision row sums get high weights.
- Figures always saved to `results/figures/`; reports to `reports/`.
- Rolling window default: 252 trading days throughout all scripts.
- Long-only rolling GMV (`constrained=True`) is the default for the notebook; unconstrained is used in `beta_weight.py` to expose the full beta–weight relationship.
- Statistical tests in `crisis_weight_test.py`: Welch t-test per asset + Benjamini-Hochberg FDR 5% correction; permutation test (n=2000) on squared L2 distance of mean weight vectors; Mann-Whitney U on Effective N series.
