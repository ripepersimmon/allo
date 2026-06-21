# allo — Covariance Estimators & GMV Portfolio Weight Allocation

Research code studying how covariance estimators (Sample, Ledoit-Wolf, Gerber) allocate
Global Minimum Variance (GMV) portfolio weights, and how that allocation behaves in
financial crises. S&P 100 universe, 2005–2024.

## Headline findings

- **GMV is a low-beta portfolio with a sharp beta threshold** — recovered empirically via
  SHAP, matching the analytic result of Clarke, de Silva & Thorley (2011).
- **In a crisis the low-beta tilt intensifies, yet the realized portfolio beta rises**
  because market-wide correlations compress all betas toward 1 (Longin & Solnik 2001).
- Variance-decomposition characteristics — the focus of the original (circular) design —
  add little once tautology and look-ahead leakage are removed.

See **[`report/`](report/)** for the full Korean write-up (`report/00_종합요약.md`),
figures, and result tables. Verified references in `report/참고문헌.md`.

## Layout

| Path | Contents |
|------|----------|
| `src/` | Library: data loading, estimators, GMV solvers, analysis, VIX crisis detection |
| `weight_explain_study.py` | ★ Headline: what explains LW-GMV weight (OLS → XGBoost + SHAP) |
| `crisis_case_study.py` | VIX-defined crisis case study of weight shifts |
| `delta_weight_study.py` | DiD study of crisis weight change (Δw) |
| `fetch_spy.py`, `fetch_vix.py` | One-time market-data fetch (SPY, VIX) |
| `report/` | Curated Korean reports + figures + tables (the deliverable) |
| `reports/` | Auto-generated reports and result CSVs |

## Setup

```bash
conda env create -f environment.yml        # or: conda create -n allo python=3.13
conda activate allo
pip install -r requirements-allo.lock.txt   # exact pin
```

## Data (not committed — see `.gitignore`)

Raw data is excluded from the repo (size / licensing). To reproduce, place per-ticker
OHLCV parquet files in `sp500/` (columns `Open,High,Low,Close,Volume`, indexed by `Date`),
then:

```bash
python fetch_spy.py     # → sp500/SPY.parquet
python fetch_vix.py     # → sp500/VIX.parquet
```

Fama-French daily factor / industry CSVs (Ken French data library) go in the repo root.

## Run

```bash
python weight_explain_study.py        # headline study
python -m src.crises                  # regenerate VIX crisis table
python crisis_case_study.py           # crisis case study
python delta_weight_study.py          # Δw study
```

## Environment

Python 3.13 · numpy · pandas · scikit-learn · statsmodels · cvxpy (CLARABEL) ·
xgboost · shap · matplotlib · seaborn. Exact versions in `requirements-allo.lock.txt`.
