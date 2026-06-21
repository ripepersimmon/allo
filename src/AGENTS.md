<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-21 | Updated: 2026-06-21 -->

# src

## Purpose
The reusable library underpinning every root analysis script. Provides the
one-directional pipeline data loading → covariance estimation → portfolio
optimization → analysis/plotting, plus shared helpers for the OLS scripts
(market proxy, GICS sectors, Fama-French factors). All root scripts import from
here; this package holds no executable entry point of its own.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Empty package marker |
| `data_loader.py` | `TICKERS` (canonical S&P 100 list); loads per-ticker parquet → wide price/field/dollar-volume DataFrames; `compute_returns()` for log/simple returns |
| `estimators.py` | Three covariance estimators `(returns) -> np.ndarray`: `sample_cov`, `lw_cov`, `gerber_cov`; plus `bbc_permutation(A)` (precision-matrix block permutation) |
| `portfolio.py` | GMV solvers `gmv_unconstrained` (analytical `Σ⁻¹1`) and `gmv_long_only` (cvxpy QP); concentration metric `effective_n`; `turnover` |
| `analysis.py` | `CRISIS_PERIODS`, `ESTIMATOR_COLORS`, `rolling_gmv`, `compute_metrics`, and plotting helpers (`plot_weights`, `plot_concentration`, `plot_cov_heatmap`, `compare_turnover`, `summary_table`) |
| `market.py` | Market-proxy utilities: `fetch_spy`, `load_spy_returns`, `get_market_proxy` (equal-weighted vs SPY) |
| `sectors.py` | Hardcoded `GICS_SECTORS` mapping for all tickers + `get_sector_dummies` (one-hot, reference = InfoTech) |
| `ff49.py` | Loaders for Fama-French data: `load_ff49_returns` (49 industry VW) and `load_ff_factors` (3- or 5-factor) |

## For AI Agents

### Working In This Directory
- All estimators share the signature `(returns: pd.DataFrame) -> np.ndarray`. Preserve this when adding a new one so it drops into `rolling_gmv` and the scripts unchanged.
- `TICKERS` and `GICS_SECTORS` are the single source of truth for the cross-section. Keep them in sync — `sectors.py` already covers every ticker in `TICKERS`; a new ticker must be added to both.
- `CRISIS_PERIODS` in `analysis.py` is canonical; do not redefine crisis windows elsewhere.
- `gerber_cov` and any new estimator must return a PSD covariance. `gerber_cov` projects to the nearest PSD correlation via `_nearest_psd_corr` before rescaling — reuse this helper.
- Solvers must degrade gracefully: `gmv_unconstrained` falls back to `pinv` on singular Σ; `gmv_long_only` falls back to equal weight on solver failure. Keep these fallbacks.

### Testing Requirements
- No unit tests. Validate by importing into a script (or notebook) and confirming weights sum to ~1, Σ is symmetric PSD, and `effective_n ∈ [1, n]`.
- `rolling_gmv` excludes per-window NaN tickers (not-yet-listed / data gaps) and zero-fills them — verify new estimators don't break on the reduced sub-matrix.

### Common Patterns
- Loaders forward-fill prices (`ffill(limit=5)`) but keep raw volume so missing-volume days don't create phantom liquidity.
- GMV analytical solution is `w ∝ Σ⁻¹1` (precision row sums); the BBC algorithm exploits this — high precision row-sum ⇒ high weight.
- Plotting helpers create parent dirs and save at dpi=150 when `save_path` is given, then `plt.show()`.

## Dependencies

### Internal
- `analysis.py` imports `portfolio.py` (`gmv_long_only`, `gmv_unconstrained`, `effective_n`, `turnover`).
- Consumed by every root script and by `notebooks/crisis_study.ipynb`.

### External
- numpy, pandas — core numerics
- scikit-learn (`LedoitWolf`) — `lw_cov`
- cvxpy (CLARABEL) — `gmv_long_only`
- matplotlib, seaborn — plotting helpers in `analysis.py`
- yfinance — `market.fetch_spy` only

<!-- MANUAL: -->
