"""
Δw Study — Crisis Weight-Shift Drivers (OLS → Gradient Boosting)
================================================================
Cross-sectional DiD redesign. See reports/2026-06-21_delta_weight_redesign_plan.md.

Target  : Δw_i = w_crisis_i − w_pre_i   (per-asset weight change, snapshot DiD)
Features: PRE-crisis (predetermined) characteristics + w_pre control
Eval    : Leave-One-Crisis-Out (LOCO) out-of-sample — NO random k-fold

This structurally avoids the three CRITICAL flaws of the old design:
  C1 tautology      → predetermined pre-features + w_pre control (Δw uses crisis-window w)
  C2 window leakage → time collapsed to two snapshots; CV = leave-one-crisis-out
  C3 survivorship   → stated as a limitation (course scope); optional COVID+Rates cut

Run from repo root with the venv active:
    python delta_weight_study.py
    python delta_weight_study.py --proxy spy      # market proxy = SPY (default: ew)
    python delta_weight_study.py --skip-gfc       # survivorship-robustness cut
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
from src.estimators import sample_cov, lw_cov, gerber_cov
from src.portfolio import gmv_long_only
from src.market import get_market_proxy, load_spy_returns

# ── optional drop-in upgrades (used only if installed) ────────────────────────
try:
    from xgboost import XGBRegressor          # noqa: F401
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    import shap                                # noqa: F401
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

np.random.seed(42)

# ── config ────────────────────────────────────────────────────────────────────
WINDOW   = 252
FIGURES  = Path('results/figures/delta_weight')
REPORTS  = Path('reports')
FIGURES.mkdir(parents=True, exist_ok=True)

ESTIMATORS = {'Sample': sample_cov, 'LW': lw_cov, 'Gerber': gerber_cov}
EST_COLORS = {'Sample': '#e41a1c', 'LW': '#377eb8', 'Gerber': '#4daf4a'}

# pre/crisis windows (mirrors crisis_weight_test.py PERIODS; pre windows are
# the project-canonical pre-crisis windows from CLAUDE.md)
PERIODS = {
    'GFC':   {'pre': ('2005-01-01', '2006-12-31'), 'crisis': ('2007-01-01', '2009-06-30')},
    'COVID': {'pre': ('2018-01-01', '2019-09-30'), 'crisis': ('2019-10-01', '2020-09-30')},
    'Rates': {'pre': ('2019-07-01', '2021-06-30'), 'crisis': ('2021-07-01', '2023-01-31')},
}

# Σ-derived features (interpret ONLY after controlling w_pre) and non-Σ features
SIGMA_FEATS = ['pre_total_var', 'pre_syst_share', 'pre_avg_corr', 'pre_beta']
NONSIGMA_FEATS = ['pre_amihud', 'pre_momentum', 'pre_log_dolvol', 'pre_downside_vol']
CONTROL = 'w_pre'
ALL_FEATS = [CONTROL] + SIGMA_FEATS + NONSIGMA_FEATS

# globals populated in main()
returns = None
dvol = None
SPY_RETURNS = None
PROXY = 'ew'


# ── snapshot helpers ──────────────────────────────────────────────────────────

def trailing_window(end_date: str) -> pd.DataFrame:
    """The 252-td window ending at (and excluding) the first index date > end_date.

    Mirrors rolling_gmv's `iloc[i-WINDOW:i]` look-ahead-safe slicing: we take the
    WINDOW rows up to and including the last trading day <= end_date.
    """
    idx = returns.index
    pos = idx.searchsorted(pd.Timestamp(end_date), side='right')  # first row strictly after
    lo = max(pos - WINDOW, 0)
    return returns.iloc[lo:pos].dropna(axis=1)


def crisis_peak_date(crisis: str) -> pd.Timestamp:
    """Max-drawdown date of the equal-weighted universe inside the crisis window."""
    t0, t1 = PERIODS[crisis]['crisis']
    win = returns.loc[t0:t1].dropna(axis=1, how='all')
    ew = win.mean(axis=1)                       # EW daily log return
    cum = ew.cumsum()                           # log cumulative
    drawdown = cum - cum.cummax()
    return drawdown.idxmin()


def gmv_weights(win: pd.DataFrame, est_fn) -> pd.Series:
    """Long-only GMV weights for the assets present in `win` (Series indexed by ticker)."""
    if win.shape[1] < 5:
        return pd.Series(dtype=float)
    try:
        w = gmv_long_only(est_fn(win))
    except Exception:
        w = np.full(win.shape[1], 1.0 / win.shape[1])
    return pd.Series(w, index=win.columns)


# ── predetermined features (PRE window only) ──────────────────────────────────

def pre_features(win: pd.DataFrame) -> pd.DataFrame:
    """All characteristics computed on the PRE-crisis window (index=ticker).

    Σ-derived: total_var, syst_share, avg_corr, beta (market = PROXY).
    Non-Σ    : amihud illiquidity, momentum (Σ log-ret), log dollar volume,
               downside volatility.
    """
    cols = win.columns
    mkt = get_market_proxy(win, PROXY, SPY_RETURNS)
    valid = mkt.dropna().index.intersection(win.index)
    w = win.loc[valid]
    m = mkt.loc[valid]
    mkt_var = m.var()
    corr = w.corr()
    dv_win = dvol.reindex(index=valid, columns=cols)

    rows = []
    for col in cols:
        r = w[col]
        total_var = r.var()
        if mkt_var > 0 and total_var > 0:
            beta = r.cov(m) / mkt_var
            syst_var = beta ** 2 * mkt_var
            syst_share = min(syst_var / total_var, 1.0)
        else:
            beta = 0.0
            syst_share = 0.0

        avg_corr = float(corr[col].drop(col, errors='ignore').mean()) if col in corr else np.nan

        dv = dv_win[col].replace(0, np.nan) if col in dv_win else pd.Series(dtype=float)
        ratio = (r.abs() / dv.reindex(r.index)).dropna()
        amihud = float(ratio.mean() * 1e6) if len(ratio) > 10 else np.nan
        log_dolvol = float(np.log(max(dv.mean(), 1.0))) if dv.notna().any() else np.nan

        downside_vol = float(r[r < 0].std()) if (r < 0).sum() > 5 else float(r.std())
        momentum = float(r.sum())

        rows.append({
            'ticker': col,
            'pre_total_var': total_var, 'pre_syst_share': syst_share,
            'pre_avg_corr': avg_corr, 'pre_beta': beta,
            'pre_amihud': amihud, 'pre_momentum': momentum,
            'pre_log_dolvol': log_dolvol, 'pre_downside_vol': downside_vol,
        })
    out = pd.DataFrame(rows).set_index('ticker')
    # impute remaining NaNs with cross-sectional median (avoids dropping assets)
    return out.fillna(out.median())


# ── Δw panel builder ──────────────────────────────────────────────────────────

def build_delta_panel(est_fn) -> pd.DataFrame:
    """One row per (crisis, asset): Δw target + w_pre control + pre-features."""
    frames = []
    for crisis, spec in PERIODS.items():
        pre_end = spec['pre'][1]
        peak = crisis_peak_date(crisis)

        pre_win = trailing_window(pre_end)
        crisis_win = trailing_window(str(peak.date()))

        w_pre = gmv_weights(pre_win, est_fn)
        w_crisis = gmv_weights(crisis_win, est_fn)
        feats = pre_features(pre_win)

        common = w_pre.index.intersection(w_crisis.index).intersection(feats.index)
        df = feats.loc[common].copy()
        df['w_pre'] = w_pre.loc[common]
        df['delta_w'] = (w_crisis.loc[common] - w_pre.loc[common]).values
        df['crisis'] = crisis
        df['peak'] = peak
        df = df.reset_index().rename(columns={'index': 'ticker'})
        frames.append(df)
        print(f'    {crisis}: peak={peak.date()}  n_assets={len(df)}', flush=True)
    return pd.concat(frames, ignore_index=True)


# ── models ────────────────────────────────────────────────────────────────────

def fit_ols(panel: pd.DataFrame, feats: list[str]):
    """OLS with HC3 heteroscedasticity-robust SE. Returns fitted results."""
    d = panel.dropna(subset=feats + ['delta_w'])
    X = sm.add_constant(d[feats])
    return sm.OLS(d['delta_w'], X).fit(cov_type='HC3')


def make_gbr():
    """Gradient-boosting regressor: XGBoost if available, else sklearn Hist-GBR."""
    if HAS_XGB:
        return XGBRegressor(
            n_estimators=300, max_depth=2, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            random_state=42, n_jobs=2,
        )
    return HistGradientBoostingRegressor(
        max_depth=2, max_leaf_nodes=8, learning_rate=0.03,
        max_iter=400, l2_regularization=2.0, early_stopping=False,
        random_state=42,
    )


def loco_score(panel: pd.DataFrame, feats: list[str], model_kind: str) -> dict:
    """Leave-One-Crisis-Out out-of-sample R²/MAE.

    model_kind ∈ {'ols', 'gbr'}. Trains on all crises but the held-out one,
    predicts the held-out crisis, aggregates OOS predictions.
    """
    d = panel.dropna(subset=feats + ['delta_w']).copy()
    crises = list(d['crisis'].unique())
    y_true, y_pred = [], []
    for held in crises:
        tr = d[d['crisis'] != held]
        te = d[d['crisis'] == held]
        if model_kind == 'ols':
            Xtr = sm.add_constant(tr[feats])
            res = sm.OLS(tr['delta_w'], Xtr).fit()
            Xte = sm.add_constant(te[feats], has_constant='add')
            pred = res.predict(Xte)
        else:
            m = make_gbr()
            m.fit(tr[feats].values, tr['delta_w'].values)
            pred = m.predict(te[feats].values)
        y_true.append(te['delta_w'].values)
        y_pred.append(np.asarray(pred))
    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(yt - yp)))
    return {'loco_r2': float(r2), 'loco_mae': mae, 'n': len(yt)}


def gbr_importance(panel: pd.DataFrame, feats: list[str]) -> pd.Series:
    """Permutation importance of the GBR on the full pooled sample (descriptive)."""
    d = panel.dropna(subset=feats + ['delta_w'])
    m = make_gbr()
    m.fit(d[feats].values, d['delta_w'].values)
    imp = permutation_importance(m, d[feats].values, d['delta_w'].values,
                                 n_repeats=30, random_state=42)
    return pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False)


# ── figures ───────────────────────────────────────────────────────────────────

def fig_ladder(ladder: dict, est_name: str):
    """LOCO R² bar across the M0→M3 model ladder for one estimator."""
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(ladder.keys())
    vals = [ladder[k]['loco_r2'] for k in names]
    ax.bar(names, vals, color=EST_COLORS.get(est_name, 'gray'))
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('LOCO out-of-sample R²')
    ax.set_title(f'Model ladder — Δw drivers ({est_name})')
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES / f'ladder_loco_{est_name}.png', dpi=150)
    plt.close(fig)


def fig_importance(imp: pd.Series, est_name: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    imp.iloc[::-1].plot.barh(ax=ax, color=EST_COLORS.get(est_name, 'gray'))
    ax.set_xlabel('Permutation importance (Δ MSE)')
    ax.set_title(f'GBR feature importance — Δw ({est_name})')
    plt.tight_layout()
    fig.savefig(FIGURES / f'importance_{est_name}.png', dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global returns, dvol, SPY_RETURNS, PROXY
    ap = argparse.ArgumentParser()
    ap.add_argument('--proxy', choices=['ew', 'spy'], default='ew')
    ap.add_argument('--skip-gfc', action='store_true',
                    help='survivorship-robustness cut: drop GFC (post-2015 universe)')
    args = ap.parse_args()
    PROXY = args.proxy

    if args.skip_gfc:
        PERIODS.pop('GFC', None)

    print('Loading data...')
    prices = load_prices_from_parquet('sp500', tickers=TICKERS,
                                      start='2000-01-01', end='2024-12-31')
    returns = compute_returns(prices, method='log')
    dvol = load_dollar_volume('sp500', tickers=TICKERS, start='2000-01-01', end='2024-12-31')
    SPY_RETURNS = load_spy_returns() if PROXY == 'spy' else None
    print(f'Returns: {returns.shape}   proxy={PROXY}   '
          f'GBR={"xgboost" if HAS_XGB else "sklearn-HistGBR"}   SHAP={HAS_SHAP}\n')

    summary_rows = []
    ols_tables = {}

    for est_name, est_fn in ESTIMATORS.items():
        print(f'[{est_name}] building Δw panel...')
        panel = build_delta_panel(est_fn)
        panel.to_csv(REPORTS / f'delta_panel_{est_name}.csv', index=False)

        # ── model ladder ──
        ladder = {
            'M0 w_pre':     loco_score(panel, [CONTROL], 'ols'),
            'M1 +Σ':        loco_score(panel, [CONTROL] + SIGMA_FEATS, 'ols'),
            'M2 +nonΣ':     loco_score(panel, ALL_FEATS, 'ols'),
            'M3 GBR':       loco_score(panel, ALL_FEATS, 'gbr'),
        }
        for k, v in ladder.items():
            print(f'    {k:12s} LOCO R²={v["loco_r2"]:+.3f}  MAE={v["loco_mae"]:.2e}  n={v["n"]}')
            summary_rows.append({'estimator': est_name, 'model': k, **v})

        # ── in-sample OLS coefficients (M2, HC3) for interpretation ──
        ols_tables[est_name] = fit_ols(panel, ALL_FEATS)

        # ── GBR importance + figures ──
        imp = gbr_importance(panel, ALL_FEATS)
        fig_ladder(ladder, est_name)
        fig_importance(imp, est_name)
        print()

    pd.DataFrame(summary_rows).to_csv(REPORTS / 'delta_weight_loco_summary.csv', index=False)

    # ── write report skeleton ──
    write_report(summary_rows, ols_tables, args)
    print(f'Done. Figures → {FIGURES}/   Report → {REPORTS}/delta_weight_study_report.md')


def _md_table(df: pd.DataFrame, index_label: str = '') -> str:
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    cols = [index_label] + [str(c) for c in df.columns]
    head = '| ' + ' | '.join(cols) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    body = [
        '| ' + ' | '.join([str(idx)] + [str(v) for v in row]) + ' |'
        for idx, row in zip(df.index, df.values)
    ]
    return '\n'.join([head, sep] + body)


def write_report(summary_rows, ols_tables, args):
    lines = []
    A = lines.append
    A('# Δw Study — Crisis Weight-Shift Drivers\n')
    A(f'Market proxy: `{args.proxy}`  |  Crises: {list(PERIODS)}  |  '
      f'GBR backend: {"xgboost" if HAS_XGB else "sklearn HistGBR"}\n')
    A('Design: cross-sectional DiD on Δw = w_crisis − w_pre, predetermined pre-features, '
      'leave-one-crisis-out (LOCO) out-of-sample evaluation. '
      'See `2026-06-21_delta_weight_redesign_plan.md`.\n')

    A('## LOCO out-of-sample R² ladder\n')
    df = pd.DataFrame(summary_rows)
    pivot = df.pivot(index='model', columns='estimator', values='loco_r2')
    A(_md_table(pivot.round(3), 'model'))
    A('\n_M3 over M2 = value of nonlinearity/interactions. Negative R² ⇒ no cross-crisis '
      'generalization (a finding, not a bug, given ~100 assets/crisis)._\n')

    for est_name, res in ols_tables.items():
        A(f'## OLS (M2, HC3 robust) — {est_name}\n')
        tbl = pd.DataFrame({
            'coef': res.params, 'se': res.bse, 't': res.tvalues, 'p': res.pvalues,
        }).round(4)
        A(_md_table(tbl, 'term'))
        A('')

    A('## Limitations\n')
    A('- **Survivorship bias**: 2024 S&P 100 universe applied to all crises; GFC results '
      'indicative only. Rerun with `--skip-gfc` for the post-2015 robustness cut.\n')
    A('- **Static GICS sectors** (2024) if sector dummies are added later.\n')
    A('- **Snapshot windows** for w_pre and w_crisis overlap partially; Δw is a two-point '
      'difference, not a clean event study.\n')
    A('- **Small sample** (~100 assets × N crises): GBR is for detecting nonlinearity, '
      'not precise prediction.\n')

    (REPORTS / 'delta_weight_study_report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
