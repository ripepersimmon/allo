"""
Weight-Explanation Study — what characteristics attract LW-GMV portfolio weight
===============================================================================
Cross-sectional study of the LW-covariance long-only GMV weight LEVEL w_i.

Design choices (locked 2026-06-21):
  - Covariance        : Ledoit-Wolf ONLY.
  - Target            : w_i  (long-only GMV weight level, NOT Δw).
  - Σ-derived features: kept ONLY as a separately-reported "mechanical benchmark"
                        (total_var, syst_share) — never mixed into the headline
                        ladder, because w ∝ Σ⁻¹1 makes them tautological.
  - Headline features : non-Σ characteristics (size, liquidity, momentum, beta,
                        sector) — a legitimate descriptive question.
  - Time structure    : annual year-end snapshots. 252-td windows one year apart
                        are ~non-overlapping → avoids the 99.6%-overlap leakage
                        that invalidated the old daily-panel OLS.
  - Evaluation        : Leave-One-Year-Out (LOYO) out-of-sample R²/MAE.

Run from repo root with the `allo` conda env active:
    conda activate allo
    python weight_explain_study.py
    python weight_explain_study.py --proxy spy     # market proxy = SPY (run fetch_spy.py first)
    python weight_explain_study.py --years 2010 2024
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
except Exception:
    HAS_XGB = False
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

np.random.seed(42)

# ── config ────────────────────────────────────────────────────────────────────
WINDOW   = 252
FIGURES  = Path('results/figures/weight_explain')
REPORTS  = Path('reports')
FIGURES.mkdir(parents=True, exist_ok=True)

LW_COLOR = '#377eb8'

NONSIGMA = ['log_dolvol', 'amihud', 'momentum', 'beta']   # headline (non-Σ)
SIGMA_BENCH = ['total_var', 'syst_share']                 # mechanical benchmark
SECTOR_COLS = []                                          # filled at panel build

# globals
returns = None
dvol = None
SPY_RETURNS = None
PROXY = 'ew'


# ── snapshot helpers ──────────────────────────────────────────────────────────

def year_end_window(year: int) -> pd.DataFrame:
    """252-td window ending at the last trading day <= Dec-31 of `year`."""
    idx = returns.index
    cutoff = pd.Timestamp(f'{year}-12-31')
    pos = idx.searchsorted(cutoff, side='right')       # first row strictly after Dec-31
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
    """Per-asset characteristics on one snapshot window (index=ticker).

    Headline non-Σ : log_dolvol (size), amihud (illiquidity), momentum, beta.
    Benchmark Σ    : total_var, syst_share.
    """
    cols = win.columns
    mkt = get_market_proxy(win, PROXY, SPY_RETURNS)
    valid = mkt.dropna().index.intersection(win.index)
    w = win.loc[valid]
    m = mkt.loc[valid]
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
            beta = 0.0
            syst_share = 0.0

        dv = dv_win[col].replace(0, np.nan) if col in dv_win else pd.Series(dtype=float)
        ratio = (r.abs() / dv.reindex(r.index)).dropna()
        amihud = float(ratio.mean() * 1e6) if len(ratio) > 10 else np.nan
        log_dolvol = float(np.log(max(dv.mean(), 1.0))) if dv.notna().any() else np.nan

        rows.append({
            'ticker': col,
            'total_var': total_var, 'syst_share': syst_share,
            'beta': beta, 'momentum': float(r.sum()),
            'amihud': amihud, 'log_dolvol': log_dolvol,
        })
    out = pd.DataFrame(rows).set_index('ticker')
    return out.fillna(out.median())


# ── panel builder (one row per year × asset) ──────────────────────────────────

def build_panel(years) -> pd.DataFrame:
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
        frames.append(df.reset_index().rename(columns={'index': 'ticker'}))
        print(f'    {year}: n_assets={len(common)}', flush=True)
    panel = pd.concat(frames, ignore_index=True)

    # static GICS sector dummies (reference = InfoTech)
    sec = get_sector_dummies(panel['ticker'].unique().tolist(), drop_first=True)
    SECTOR_COLS = list(sec.columns)
    panel = panel.merge(sec, left_on='ticker', right_index=True, how='left')
    panel[SECTOR_COLS] = panel[SECTOR_COLS].fillna(0.0)
    return panel


# ── models & LOYO evaluation ──────────────────────────────────────────────────

def make_gbr():
    if HAS_XGB:
        return XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                            random_state=42, n_jobs=2)
    return HistGradientBoostingRegressor(max_depth=3, max_leaf_nodes=15,
                                         learning_rate=0.03, max_iter=500,
                                         l2_regularization=2.0, random_state=42)


def loyo(panel: pd.DataFrame, feats: list[str], kind: str) -> dict:
    """Leave-One-Year-Out out-of-sample R²/MAE. kind ∈ {'ols','gbr'}."""
    d = panel.dropna(subset=feats + ['w']).copy()
    yrs = sorted(d['year'].unique())
    yt, yp = [], []
    for held in yrs:
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
    ss_res = np.sum((yt - yp) ** 2); ss_tot = np.sum((yt - yt.mean()) ** 2)
    return {'loyo_r2': float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            'loyo_mae': float(np.mean(np.abs(yt - yp))), 'n': len(yt)}


def fit_ols(panel: pd.DataFrame, feats: list[str]):
    d = panel.dropna(subset=feats + ['w'])
    return sm.OLS(d['w'], sm.add_constant(d[feats])).fit(cov_type='HC3')


# ── figures ───────────────────────────────────────────────────────────────────

def fig_ladder(ladder: dict):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    names = list(ladder); vals = [ladder[k]['loyo_r2'] for k in names]
    bars = ax.bar(names, vals, color=LW_COLOR)
    bars[0].set_color('#999999')                       # benchmark greyed
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('LOYO out-of-sample R²')
    ax.set_title('LW-GMV weight explained — Σ-benchmark vs non-Σ ladder')
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    plt.tight_layout(); fig.savefig(FIGURES / 'ladder_loyo.png', dpi=150); plt.close(fig)


def fig_importance(imp: pd.Series):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    imp.iloc[::-1].plot.barh(ax=ax, color=LW_COLOR)
    ax.set_xlabel('Permutation importance (Δ MSE)')
    ax.set_title('GBR feature importance — LW-GMV weight (non-Σ + sectors)')
    plt.tight_layout(); fig.savefig(FIGURES / 'importance.png', dpi=150); plt.close(fig)


def shap_analysis(model, X: pd.DataFrame, top_dep=('beta', 'log_dolvol', 'amihud')) -> pd.Series:
    """TreeExplainer SHAP on the fitted GBR. Saves a beeswarm summary + dependence
    plots and returns mean(|SHAP|) importance. Requires shap + an xgboost model.
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)                       # (n, p)

    # beeswarm summary
    plt.figure()
    shap.summary_plot(sv, X, show=False, plot_size=(8, 5))
    plt.title('SHAP summary — LW-GMV weight', fontsize=11)
    plt.tight_layout(); plt.savefig(FIGURES / 'shap_summary.png', dpi=150); plt.close()

    # dependence plots for the top continuous drivers
    for feat in top_dep:
        if feat not in X.columns:
            continue
        plt.figure()
        shap.dependence_plot(feat, sv, X, show=False, interaction_index='auto')
        plt.tight_layout()
        plt.savefig(FIGURES / f'shap_dependence_{feat}.png', dpi=150); plt.close()

    return pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global returns, dvol, SPY_RETURNS, PROXY
    ap = argparse.ArgumentParser()
    ap.add_argument('--proxy', choices=['ew', 'spy'], default='ew')
    ap.add_argument('--years', type=int, nargs=2, default=[2005, 2024],
                    metavar=('START', 'END'))
    args = ap.parse_args()
    PROXY = args.proxy
    years = range(args.years[0], args.years[1] + 1)

    print('Loading data...')
    prices = load_prices_from_parquet('sp500', tickers=TICKERS, start='2000-01-01', end='2024-12-31')
    returns = compute_returns(prices, method='log')
    dvol = load_dollar_volume('sp500', tickers=TICKERS, start='2000-01-01', end='2024-12-31')
    SPY_RETURNS = load_spy_returns() if PROXY == 'spy' else None
    print(f'Returns: {returns.shape}  proxy={PROXY}  '
          f'GBR={"xgboost" if HAS_XGB else "sklearn-HistGBR"}  SHAP={HAS_SHAP}\n')

    print('Building year-end snapshot panel (LW-GMV)...')
    panel = build_panel(years)
    panel.to_csv(REPORTS / 'weight_explain_panel.csv', index=False)
    print(f'  Panel: {len(panel)} rows  (zero-weight share: {(panel["w"]==0).mean():.1%})\n')

    nonsigma_sec = NONSIGMA + SECTOR_COLS
    combined = SIGMA_BENCH + nonsigma_sec

    # ── ladder (all LOYO out-of-sample) ──
    ladder = {
        'B0 Σ-bench':   loyo(panel, SIGMA_BENCH, 'ols'),     # mechanical ceiling
        'M1 nonΣ':      loyo(panel, NONSIGMA, 'ols'),
        'M2 +sector':   loyo(panel, nonsigma_sec, 'ols'),
        'M3 GBR':       loyo(panel, nonsigma_sec, 'gbr'),
    }
    # incremental: does non-Σ add over the mechanical benchmark?
    r2_bench   = ladder['B0 Σ-bench']['loyo_r2']
    r2_combined = loyo(panel, combined, 'ols')['loyo_r2']
    print('LOYO out-of-sample R²:')
    for k, v in ladder.items():
        print(f'  {k:12s} R²={v["loyo_r2"]:+.3f}  MAE={v["loyo_mae"]:.2e}  n={v["n"]}')
    print(f'  {"Σ+nonΣ":12s} R²={r2_combined:+.3f}  (incremental over Σ-bench: '
          f'{r2_combined - r2_bench:+.3f})\n')

    ols_main = fit_ols(panel, nonsigma_sec)
    ols_bench = fit_ols(panel, SIGMA_BENCH)

    imp_model = make_gbr()
    dd = panel.dropna(subset=nonsigma_sec + ['w'])
    imp_model.fit(dd[nonsigma_sec].values, dd['w'].values)
    imp = pd.Series(
        permutation_importance(imp_model, dd[nonsigma_sec].values, dd['w'].values,
                               n_repeats=30, random_state=42).importances_mean,
        index=nonsigma_sec).sort_values(ascending=False)

    fig_ladder(ladder); fig_importance(imp)

    # ── SHAP interpretation of the GBR (descriptive, pooled fit) ──
    shap_imp = None
    if HAS_SHAP and HAS_XGB:
        print('Computing SHAP (TreeExplainer)...')
        shap_imp = shap_analysis(imp_model, dd[nonsigma_sec])
        print('  SHAP mean|val| top: '
              + ', '.join(f'{k}={v:.2e}' for k, v in shap_imp.head(4).items()))
    else:
        print('SHAP skipped (needs shap + xgboost).')

    write_report(ladder, r2_bench, r2_combined, ols_main, ols_bench, imp, shap_imp, args, len(panel))
    print(f'Done. Figures → {FIGURES}/   Report → {REPORTS}/weight_explain_report.md')


def _md(df: pd.DataFrame, idx_label: str) -> str:
    cols = [idx_label] + [str(c) for c in df.columns]
    out = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for i, row in zip(df.index, df.values):
        out.append('| ' + ' | '.join([str(i)] + [str(v) for v in row]) + ' |')
    return '\n'.join(out)


def write_report(ladder, r2_bench, r2_combined, ols_main, ols_bench, imp, shap_imp, args, n):
    L = []
    A = L.append
    A('# Weight-Explanation Study — LW-GMV portfolio weight\n')
    A(f'Covariance: **Ledoit-Wolf only**  |  Target: weight level `w`  |  '
      f'Market proxy: `{args.proxy}`  |  Snapshots: {args.years[0]}–{args.years[1]} year-ends  '
      f'|  N={n} rows  |  GBR: {"xgboost" if HAS_XGB else "sklearn HistGBR"}\n')
    A('Design: annual non-overlapping snapshots (no daily-panel leakage); Σ-derived '
      'features reported only as a mechanical benchmark; Leave-One-Year-Out (LOYO) '
      'out-of-sample evaluation.\n')

    A('## LOYO out-of-sample R²\n')
    df = pd.DataFrame({k: [v['loyo_r2'], v['loyo_mae']] for k, v in ladder.items()},
                      index=['LOYO R²', 'MAE']).T.round(4)
    A(_md(df, 'model'))
    A(f'\n- **B0 (Σ-bench)** `w ~ total_var + syst_share` (R²={r2_bench:.3f}): a crude '
      f'2-variable proxy for the Σ-structure that mechanically sets `w`. For *unconstrained* '
      f'GMV (`w ∝ Σ⁻¹1`) this would be near-tautological, but the long-only constraint and the '
      f'correlation/precision off-diagonals break the clean `1/var` identity — so out-of-sample '
      f'these two summary stats explain little, and the tautology risk is mild here.\n')
    A(f'- **Σ + non-Σ combined** R²={r2_combined:.3f} → non-Σ adds '
      f'**{r2_combined - r2_bench:+.3f}** beyond the mechanical part.\n')
    A('- M1–M3 use non-Σ characteristics only (+ sectors). M3−M2 = nonlinearity gain.\n')

    A('## OLS — non-Σ + sectors (HC3 robust)\n')
    t = pd.DataFrame({'coef': ols_main.params, 'se': ols_main.bse,
                      't': ols_main.tvalues, 'p': ols_main.pvalues}).round(4)
    A(_md(t, 'term'))

    A('\n## OLS — Σ benchmark (HC3 robust)\n')
    tb = pd.DataFrame({'coef': ols_bench.params, 'se': ols_bench.bse,
                       't': ols_bench.tvalues, 'p': ols_bench.pvalues}).round(4)
    A(_md(tb, 'term'))

    A('\n## GBR permutation importance (non-Σ + sectors)\n')
    A(_md(imp.round(6).to_frame('importance'), 'feature'))

    if shap_imp is not None:
        A('\n## SHAP interpretation (TreeExplainer, GBR)\n')
        A('Mean |SHAP value| = average magnitude of each feature\'s contribution to the '
          'predicted weight. Figures: `shap_summary.png` (beeswarm), '
          '`shap_dependence_{beta,log_dolvol,amihud}.png`.\n')
        A(_md(shap_imp.round(6).to_frame('mean_abs_shap'), 'feature'))

    A('\n## Limitations\n')
    A('- **Descriptive, not causal**: `w` is a deterministic function of Σ; these '
      'characteristics are *associated* with weight, not causes of it.\n')
    A('- **Survivorship bias**: 2024 S&P 100 universe applied to all snapshot years.\n')
    A('- **Static 2024 GICS sectors** applied to all years.\n')
    A('- **Concentrated target**: long-only `w` is highly right-skewed — most assets carry '
      'near-zero weight and a few dominate; OLS on a skewed target is descriptive only.\n')
    (REPORTS / 'weight_explain_report.md').write_text('\n'.join(L))


if __name__ == '__main__':
    main()
