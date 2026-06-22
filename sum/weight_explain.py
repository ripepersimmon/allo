"""Cross-sectional explanation of LW-GMV weight levels (Tables 3-4, Figure 1).

For each year-end 2005-2024 the long-only GMV portfolio is built from the prior
252-day Ledoit-Wolf covariance, then its weight w_i is regressed on interpretable
non-covariance characteristics (beta, size, liquidity, momentum, GICS sector).
Annual snapshots are ~non-overlapping, avoiding daily-panel leakage. Models are
scored Leave-One-Year-Out (LOYO); the fitted gradient-boosted model is read with
SHAP to expose the beta threshold.

    python weight_explain.py                 # equal-weighted market proxy
    python weight_explain.py --proxy spy     # SPY proxy (run fetch_data.py first)
"""
import sys, argparse, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from src.data_loader import load_prices_from_parquet, compute_returns, load_dollar_volume, TICKERS
from src.estimators import lw_cov
from src.portfolio import gmv_long_only
from src.market import get_market_proxy, load_spy_returns
from src.sectors import get_sector_dummies

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

np.random.seed(42)

WINDOW = 252
FIGURES = Path('figures/weight_explain')
TABLES = Path('tables')
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

COLOR = '#377eb8'
NONSIGMA = ['log_dolvol', 'amihud', 'momentum', 'beta']   # headline characteristics
SIGMA_BENCH = ['total_var', 'syst_share']                 # covariance-derived benchmark

returns = dvol = SPY_RETURNS = None
PROXY = 'ew'
SECTOR_COLS = []


def year_end_window(year: int) -> pd.DataFrame:
    """252-day window ending on the last trading day <= Dec-31 of `year`."""
    idx = returns.index
    pos = idx.searchsorted(pd.Timestamp(f'{year}-12-31'), side='right')
    lo = max(pos - WINDOW, 0)
    if pos - lo < WINDOW:
        return pd.DataFrame()
    return returns.iloc[lo:pos].dropna(axis=1)


def gmv_weights(win: pd.DataFrame) -> pd.Series:
    if win.shape[1] < 5:
        return pd.Series(dtype=float)
    try:
        w = gmv_long_only(lw_cov(win))
    except Exception:
        w = np.full(win.shape[1], 1.0 / win.shape[1])
    return pd.Series(w, index=win.columns)


def snapshot_features(win: pd.DataFrame) -> pd.DataFrame:
    """Per-asset characteristics on one window (index = ticker)."""
    cols = win.columns
    mkt = get_market_proxy(win, PROXY, SPY_RETURNS)
    valid = mkt.dropna().index.intersection(win.index)
    w, m = win.loc[valid], mkt.loc[valid]
    mkt_var = m.var()
    dv_win = dvol.reindex(index=valid, columns=cols)

    rows = []
    for col in cols:
        r = w[col]
        total_var = r.var()
        if mkt_var > 0 and total_var > 0:
            beta = r.cov(m) / mkt_var
            syst_share = min((beta ** 2 * mkt_var) / total_var, 1.0)
        else:
            beta = syst_share = 0.0
        dv = dv_win[col].replace(0, np.nan) if col in dv_win else pd.Series(dtype=float)
        ratio = (r.abs() / dv.reindex(r.index)).dropna()
        rows.append({
            'ticker': col, 'total_var': total_var, 'syst_share': syst_share,
            'beta': beta, 'momentum': float(r.sum()),
            'amihud': float(ratio.mean() * 1e6) if len(ratio) > 10 else np.nan,
            'log_dolvol': float(np.log(max(dv.mean(), 1.0))) if dv.notna().any() else np.nan,
        })
    out = pd.DataFrame(rows).set_index('ticker')
    return out.fillna(out.median())


def build_panel(years) -> pd.DataFrame:
    """One row per (year, asset): GMV weight + characteristics + sector dummies."""
    global SECTOR_COLS
    frames = []
    for year in years:
        win = year_end_window(year)
        if win.empty:
            continue
        w = gmv_weights(win)
        feats = snapshot_features(win)
        common = w.index.intersection(feats.index)
        df = feats.loc[common].copy()
        df['w'] = w.loc[common]
        df['year'] = year
        df.index.name = 'ticker'
        frames.append(df.reset_index())
        print(f'    {year}: n_assets={len(common)}', flush=True)
    panel = pd.concat(frames, ignore_index=True)

    sec = get_sector_dummies(panel['ticker'].unique().tolist(), drop_first=True)
    SECTOR_COLS = list(sec.columns)
    panel = panel.merge(sec, left_on='ticker', right_index=True, how='left')
    panel[SECTOR_COLS] = panel[SECTOR_COLS].fillna(0.0)
    return panel


def make_gbr():
    if HAS_XGB:
        return XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                            random_state=42, n_jobs=2)
    return HistGradientBoostingRegressor(max_depth=3, max_leaf_nodes=15,
                                         learning_rate=0.03, max_iter=500,
                                         l2_regularization=2.0, random_state=42)


def loyo(panel: pd.DataFrame, feats: list, kind: str) -> dict:
    """Leave-One-Year-Out out-of-sample R² / MAE. kind in {'ols', 'gbr'}."""
    d = panel.dropna(subset=feats + ['w'])
    yt, yp = [], []
    for held in sorted(d['year'].unique()):
        tr, te = d[d['year'] != held], d[d['year'] == held]
        if kind == 'ols':
            res = sm.OLS(tr['w'], sm.add_constant(tr[feats])).fit()
            pred = res.predict(sm.add_constant(te[feats], has_constant='add'))
        else:
            m = make_gbr()
            m.fit(tr[feats].values, tr['w'].values)
            pred = m.predict(te[feats].values)
        yt.append(te['w'].values); yp.append(np.asarray(pred))
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    ss_res, ss_tot = np.sum((yt - yp) ** 2), np.sum((yt - yt.mean()) ** 2)
    return {'loyo_r2': float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            'loyo_mae': float(np.mean(np.abs(yt - yp)))}


def fit_ols(panel: pd.DataFrame, feats: list):
    d = panel.dropna(subset=feats + ['w'])
    return sm.OLS(d['w'], sm.add_constant(d[feats])).fit(cov_type='HC3')


def fig_ladder(ladder: dict):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    names, vals = list(ladder), [ladder[k]['loyo_r2'] for k in ladder]
    bars = ax.bar(names, vals, color=COLOR)
    bars[0].set_color('#999999')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('LOYO out-of-sample R²')
    ax.set_title('LW-GMV weight explained — covariance benchmark vs characteristics')
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    plt.tight_layout(); fig.savefig(FIGURES / 'ladder_loyo.png', dpi=150); plt.close(fig)


def fig_importance(imp: pd.Series):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    imp.iloc[::-1].plot.barh(ax=ax, color=COLOR)
    ax.set_xlabel('Permutation importance (Δ MSE)')
    ax.set_title('GBR feature importance — LW-GMV weight')
    plt.tight_layout(); fig.savefig(FIGURES / 'importance.png', dpi=150); plt.close(fig)


def shap_analysis(model, X: pd.DataFrame) -> pd.Series:
    """TreeExplainer SHAP: beeswarm + beta/size/liquidity dependence; mean|SHAP|."""
    sv = shap.TreeExplainer(model).shap_values(X)

    plt.figure()
    shap.summary_plot(sv, X, show=False, plot_size=(8, 5))
    plt.title('SHAP summary — LW-GMV weight', fontsize=11)
    plt.tight_layout(); plt.savefig(FIGURES / 'shap_summary.png', dpi=150); plt.close()

    for feat in ('beta', 'log_dolvol', 'amihud'):
        if feat not in X.columns:
            continue
        plt.figure()
        shap.dependence_plot(feat, sv, X, show=False, interaction_index='auto')
        plt.tight_layout(); plt.savefig(FIGURES / f'shap_dependence_{feat}.png', dpi=150); plt.close()

    return pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)


def main():
    global returns, dvol, SPY_RETURNS, PROXY
    ap = argparse.ArgumentParser()
    ap.add_argument('--proxy', choices=['ew', 'spy'], default='ew')
    ap.add_argument('--years', type=int, nargs=2, default=[2005, 2024], metavar=('START', 'END'))
    args = ap.parse_args()
    PROXY = args.proxy
    years = range(args.years[0], args.years[1] + 1)

    print('Loading data...')
    prices = load_prices_from_parquet(tickers=TICKERS, start='2000-01-01', end='2024-12-31')
    returns = compute_returns(prices, method='log')
    dvol = load_dollar_volume(tickers=TICKERS, start='2000-01-01', end='2024-12-31')
    SPY_RETURNS = load_spy_returns() if PROXY == 'spy' else None
    print(f'Returns: {returns.shape}  proxy={PROXY}  '
          f'GBR={"xgboost" if HAS_XGB else "sklearn-HistGBR"}  SHAP={HAS_SHAP}\n')

    print('Building year-end snapshot panel (LW-GMV)...')
    panel = build_panel(years)
    panel.to_csv(TABLES / 'weight_panel.csv', index=False)
    print(f'  Panel: {len(panel)} rows  (zero-weight share: {(panel["w"]==0).mean():.1%})\n')

    nonsigma_sec = NONSIGMA + SECTOR_COLS
    ladder = {
        'B0 Σ-bench': loyo(panel, SIGMA_BENCH, 'ols'),
        'M1 nonΣ':    loyo(panel, NONSIGMA, 'ols'),
        'M2 +sector': loyo(panel, nonsigma_sec, 'ols'),
        'M3 GBR':     loyo(panel, nonsigma_sec, 'gbr'),
    }
    pd.DataFrame(ladder).T.to_csv(TABLES / 'model_scores.csv')
    print('LOYO out-of-sample R²:')
    for k, v in ladder.items():
        print(f'  {k:12s} R²={v["loyo_r2"]:+.3f}  MAE={v["loyo_mae"]:.2e}')

    ols = fit_ols(panel, nonsigma_sec)
    pd.DataFrame({'coef': ols.params, 'se': ols.bse, 't': ols.tvalues, 'p': ols.pvalues}) \
        .round(6).to_csv(TABLES / 'ols_coef.csv')

    imp_model = make_gbr()
    dd = panel.dropna(subset=nonsigma_sec + ['w'])
    imp_model.fit(dd[nonsigma_sec].values, dd['w'].values)
    imp = pd.Series(
        permutation_importance(imp_model, dd[nonsigma_sec].values, dd['w'].values,
                               n_repeats=30, random_state=42).importances_mean,
        index=nonsigma_sec).sort_values(ascending=False)
    fig_ladder(ladder); fig_importance(imp)

    if HAS_SHAP and HAS_XGB:
        print('\nComputing SHAP (TreeExplainer)...')
        shap_imp = shap_analysis(imp_model, dd[nonsigma_sec])
        shap_imp.round(6).to_frame('mean_abs_shap').to_csv(TABLES / 'shap_importance.csv')
        print('  SHAP mean|val| top: ' + ', '.join(f'{k}={v:.2e}' for k, v in shap_imp.head(4).items()))
    else:
        print('\nSHAP skipped (needs shap + xgboost).')

    print(f'\nDone. Figures → {FIGURES}/   Tables → {TABLES}/')


if __name__ == '__main__':
    main()
