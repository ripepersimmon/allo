<!-- Generated: 2026-06-21 | Updated: 2026-06-21 -->

# allo — Covariance Estimators & GMV Weight Allocation in Crises

## Purpose
Research project studying how covariance estimators (Sample, Ledoit-Wolf, Gerber)
allocate Global Minimum Variance (GMV) portfolio weights during financial crises
(GFC, COVID, Rates). The reusable library lives in `src/`; the repo root holds
one self-contained analysis script per experiment. Each script imports `src/`,
runs from the repo root, and writes figures to `results/figures/` and written
reports to `reports/`.

## Data Flow
`sp500/*.parquet` → `src/data_loader.py` → `src/estimators.py` → `src/portfolio.py`
→ `src/analysis.py` → figures/reports. Cross-sectional OLS scripts additionally
pull `src/market.py` (market proxy), `src/sectors.py` (GICS dummies), and
`src/ff49.py` (Fama-French factors).

## Key Files

### Configuration & Setup
| File | Description |
|------|-------------|
| `CLAUDE.md` | Canonical project guide: architecture, script index, crisis-period definitions, conventions |
| `requirements.txt` | Pinned venv dependencies (numpy, pandas, scikit-learn, cvxpy, statsmodels, matplotlib, seaborn, jupyter) |
| `fetch_spy.py` | One-time SPY OHLCV download → `sp500/SPY.parquet` (wraps `src/market.fetch_spy`) |
| `.gitignore` | Excludes venv, caches, large data |

### Core Experiment Scripts
| File | Description |
|------|-------------|
| `bbc_analysis.py` | BBC precision-matrix permutation (Kim et al. 2025, Alg. 1) per estimator; block-structure figures |
| `beta_weight.py` | Market beta vs unconstrained GMV weight scatter + rolling correlation |
| `crisis_weight_test.py` | Pre-crisis vs crisis weight tests: Welch t + BH-FDR, permutation, Mann-Whitney U |
| `longonly_gmv_analysis.py` | Comprehensive long-only GMV experiment (~11 figure types) → `results/figures/longonly/` |
| `intervention_analysis.py` | Decay-form (Box-Tiao) panel OLS: weight ~ characteristics + crisis term, δ via BIC |
| `variance_decomp.py` | Cross-sectional OLS (Models A–E): systematic vs idiosyncratic variance vs GMV weight |
| `variable_mining.py` | 24-variable cross-sectional predictor search (univariate t-stat → multivariate → VIF) |

### Secondary / Follow-up Scripts
| File | Description |
|------|-------------|
| `aapl_decomp.py` | AAPL-specific Model K weight decomposition at its 2024-09-30 peak weight |
| `advanced_decomp.py` | PC1 variance share, correlation dispersion, sector-correlation split models |
| `asset_syst_share_track.py` | Per-asset systematic-share tracking: top-5 beneficiaries/losers pre vs crisis |
| `intervention_new_analysis.py` | LW-only decay-form panel OLS for COVID & Rates (2015–2024) |
| `k1_analysis.py` | Model K-1: Model K + `cross_corr` (inter-sector mean correlation) |
| `lw10_analysis.py` | LW-only analysis on most recent 10 years (2015–2024), crisis-segmented |
| `lw10_full_analysis.py` | LW-only full-period (2015–2024) rolling analysis, no crisis segmentation |
| `multifactor_decomp.py` | Replaces EW market factor in Model D with Fama-French multi-factor decomposition |
| `ols_assumption_check.py` | OLS assumption diagnostics (LW, 2015–2024) |
| `spy_robustness.py` | Re-runs Model D with SPY market proxy instead of equal-weighted, for robustness |
| `timeseries_analysis.py` | Temporal structure of LW GMV weights & Effective-N (2015–2024) |
| `ppt_figures.py` | PPT-ready (16:9) versions of key figures → `results/figures/ppt/` |

### Raw External Data (CSV)
| File | Description |
|------|-------------|
| `12_Industry_Portfolios_Daily.csv`, `49_Industry_Portfolios_Daily.csv` | Fama-French industry portfolio daily returns (loaded by `src/ff49.py`) |
| `F-F_Research_Data_Factors_daily.csv` | Fama-French 3-factor daily returns |
| `F-F_Research_Data_5_Factors_2x3_daily.csv` | Fama-French 5-factor daily returns |
| `2508.10776v1.pdf` | Reference paper (Kim et al. 2025, BBC algorithm) |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `src/` | Reusable library: data loading, estimators, portfolio solvers, analysis (see `src/AGENTS.md`) |
| `sp500/` | Per-ticker OHLCV parquet files; data source (see `sp500/AGENTS.md`) |
| `notebooks/` | Exploratory Jupyter notebooks (see `notebooks/AGENTS.md`) |
| `reports/` | Generated written reports (Markdown) + result CSVs (see `reports/AGENTS.md`) |
| `results/` | Generated figures, organized by experiment (see `results/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Activate the venv first: `source .venv/bin/activate`. There is no build/install step.
- Run every script from the repo root. Scripts use `sys.path.insert(0, '.')` then `from src... import`.
- One script == one experiment. Prefer adding a new root script over expanding an existing one when the experiment is distinct.
- Shared logic (loaders, estimators, solvers, market proxy, sectors, factors) belongs in `src/`, not duplicated across scripts.
- Figures save to `results/figures/` (or an experiment subfolder); reports to `reports/`. Keep this split.

### Testing Requirements
- No formal test suite. Verify a change by running the affected script end-to-end and checking the regenerated figures/reports are sensible.
- For numerical changes, sanity-check GMV weights sum to ~1 and Effective-N stays in `[1, n]`.

### Common Patterns
- Crisis windows come from `src.analysis.CRISIS_PERIODS` — never hardcode dates in new scripts.
- Default rolling window = 252 trading days throughout.
- Market proxy default = equal-weighted return of the estimation window; SPY is the robustness alternative via `src.market`.
- Estimator color scheme is fixed in `src.analysis.ESTIMATOR_COLORS` (Sample=red, LW=blue, Gerber=green).

## Dependencies

### External
- numpy / pandas — numerics and data frames
- scikit-learn — `LedoitWolf` covariance
- cvxpy (CLARABEL solver) — long-only GMV QP
- statsmodels — OLS / panel regressions
- matplotlib / seaborn — figures
- yfinance — one-time SPY fetch only

<!-- MANUAL: Custom project notes can be added below -->
