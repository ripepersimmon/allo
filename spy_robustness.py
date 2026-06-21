"""
SPY Robustness Check for Market Proxy
======================================
Runs Model D (w_i = α + γ₁·total_var_i + γ₂·syst_share_i) at the three
crisis peaks for all three estimators, twice: once with the equal-weighted
(EW) market proxy and once with SPY.

Compares γ₁, γ₂, R²(D) across both proxies to test whether the variance-
decomposition narrative survives replacing the in-sample EW mean with an
exogenous index.

Outputs:
    reports/spy_robustness_table.csv
    results/figures/spy_robustness_bars.png
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from src.data_loader import load_prices_from_parquet, compute_returns, TICKERS
from src.estimators import sample_cov, lw_cov, gerber_cov
from src.market import get_market_proxy, load_spy_returns

WINDOW  = 252
FIGURES = Path('results/figures')
REPORTS = Path('reports')
FIGURES.mkdir(parents=True, exist_ok=True)

ESTIMATORS = {'Sample': sample_cov, 'LW': lw_cov, 'Gerber': gerber_cov}
EST_COLORS = {'Sample': '#e41a1c', 'LW': '#377eb8', 'Gerber': '#4daf4a'}

CRISIS_PEAKS = {
    'GFC':   '2009-03-31',
    'COVID': '2020-04-30',
    'Rates': '2023-01-31',
}

# ── data ──────────────────────────────────────────────────────────────────────
print('Loading data...')
prices  = load_prices_from_parquet('sp500', tickers=TICKERS,
                                   start='2000-01-01', end='2024-12-31')
returns = compute_returns(prices, method='log')
spy_ret = load_spy_returns(start='2000-01-01', end='2024-12-31')
print(f'Returns: {returns.shape[0]} days × {returns.shape[1]} assets')
print(f'SPY returns: {len(spy_ret)} observations\n')


# ── core helpers (self-contained; mirrors variance_decomp.py logic) ───────────

def decompose_and_ols(win: pd.DataFrame, cov: np.ndarray,
                      proxy: str, spy_returns) -> dict | None:
    """Compute variance decomp + Model D OLS for one (window, cov) pair."""
    mkt     = get_market_proxy(win, proxy, spy_returns)
    # drop dates where SPY is NaN (data gaps)
    valid_idx = mkt.dropna().index.intersection(win.index)
    if len(valid_idx) < WINDOW // 2:
        return None
    win  = win.loc[valid_idx]
    mkt  = mkt.loc[valid_idx]
    mkt_var = mkt.var()
    if mkt_var < 1e-14:
        return None

    rows = {}
    for col in win.columns:
        r         = win[col]
        total_var = r.var()
        if total_var < 1e-14:
            continue
        beta      = r.cov(mkt) / mkt_var
        syst_var  = beta ** 2 * mkt_var
        syst_share = syst_var / total_var
        rows[col] = dict(total_var=total_var, syst_share=syst_share)

    decomp = pd.DataFrame(rows).T

    # GMV weights from pre-computed cov (same as variance_decomp.py gmv_weights)
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)
    raw_w = prec @ np.ones(cov.shape[0])
    if abs(raw_w.sum()) < 1e-10:
        return None
    w = pd.Series(raw_w / raw_w.sum(), index=win.columns)

    common = decomp.index.intersection(w.index)
    if len(common) < 6:
        return None
    d  = decomp.loc[common]
    wt = w[common].values
    tv = d['total_var'].values
    ss = d['syst_share'].values

    n     = len(wt)
    ones  = np.ones(n)
    X     = np.column_stack([ones, tv, ss])
    try:
        Q, R  = np.linalg.qr(X)
        beta_v = np.linalg.solve(R, Q.T @ wt)
        XtX_inv = np.linalg.inv(R) @ np.linalg.inv(R).T
    except np.linalg.LinAlgError:
        beta_v, _, _, _ = np.linalg.lstsq(X, wt, rcond=None)
        XtX_inv = np.linalg.pinv(X.T @ X)

    y_hat  = X @ beta_v
    ss_res = np.sum((wt - y_hat) ** 2)
    ss_tot = np.sum((wt - wt.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 1e-14 else 0.0
    dof    = n - 3
    if dof > 0 and ss_res > 1e-14:
        sigma2 = ss_res / dof
        se     = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0))
        tstat  = beta_v / np.where(se > 1e-14, se, np.nan)
        pval   = 2 * (1 - stats.t.cdf(np.abs(tstat), df=dof))
    else:
        tstat = pval = np.full(3, np.nan)

    return dict(
        g1=beta_v[1], t1=tstat[1], p1=pval[1],
        g2=beta_v[2], t2=tstat[2], p2=pval[2],
        r2=r2, n=n,
    )


# ── Run both proxies at all 9 cells ──────────────────────────────────────────
print('Running Model D with EW proxy...')
results_ew  = {}
print('Running Model D with SPY proxy...')
results_spy = {}

for crisis, peak_date in CRISIS_PEAKS.items():
    end   = pd.Timestamp(peak_date)
    start = end - pd.offsets.BDay(WINDOW)
    win   = returns.loc[start:end].dropna(axis=1)
    print(f'  {crisis} ({peak_date}): {win.shape[1]} assets')

    for est_name, est_fn in ESTIMATORS.items():
        try:
            cov = est_fn(win)
        except Exception:
            continue
        key = (crisis, est_name)
        results_ew[key]  = decompose_and_ols(win, cov, 'ew',  None)
        results_spy[key] = decompose_and_ols(win, cov, 'spy', spy_ret)

print()


# ── Build comparison table ────────────────────────────────────────────────────
def _star(t):
    if np.isnan(t):
        return ''
    a = abs(t)
    return '***' if a > 3 else ('**' if a > 2 else ('*' if a > 1.65 else ''))


rows = []
for crisis in CRISIS_PEAKS:
    for est in ESTIMATORS:
        key = (crisis, est)
        ew  = results_ew.get(key)
        spy = results_spy.get(key)
        if ew is None and spy is None:
            continue
        rows.append({
            'crisis': crisis, 'estimator': est,
            'g1_ew':  ew['g1']  if ew  else np.nan,
            't1_ew':  ew['t1']  if ew  else np.nan,
            'g2_ew':  ew['g2']  if ew  else np.nan,
            't2_ew':  ew['t2']  if ew  else np.nan,
            'r2_ew':  ew['r2']  if ew  else np.nan,
            'g1_spy': spy['g1'] if spy else np.nan,
            't1_spy': spy['t1'] if spy else np.nan,
            'g2_spy': spy['g2'] if spy else np.nan,
            't2_spy': spy['t2'] if spy else np.nan,
            'r2_spy': spy['r2'] if spy else np.nan,
        })

table = pd.DataFrame(rows)
csv_path = REPORTS / 'spy_robustness_table.csv'
table.to_csv(csv_path, index=False, float_format='%.6f')
print(f'Saved → {csv_path}')

# Console summary
print('\n=== SPY Robustness: Model D (total_var + syst_share) ===')
print(f'{"Crisis":<8} {"Est":<8} {"g2(EW)":>10} {"t2(EW)":>8} {"g2(SPY)":>10} {"t2(SPY)":>9} {"R²(EW)":>8} {"R²(SPY)":>9} {"SignFlip":>9}')
for _, r in table.iterrows():
    flip = 'YES' if (np.sign(r['g2_ew']) != np.sign(r['g2_spy'])
                     and not np.isnan(r['g2_ew']) and not np.isnan(r['g2_spy'])) else ''
    print(f'{r["crisis"]:<8} {r["estimator"]:<8}'
          f' {r["g2_ew"]:>10.4f}{_star(r["t2_ew"]):<3}'
          f' {r["t2_ew"]:>8.2f}'
          f' {r["g2_spy"]:>10.4f}{_star(r["t2_spy"]):<3}'
          f' {r["t2_spy"]:>9.2f}'
          f' {r["r2_ew"]:>8.3f} {r["r2_spy"]:>9.3f}'
          f' {flip:>9}')
print()


# ── Figure: γ₁ and γ₂ side-by-side bars per crisis, EW vs SPY ─────────────
crises   = list(CRISIS_PEAKS.keys())
est_list = list(ESTIMATORS.keys())

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey='row')
fig.suptitle(
    'Model D Coefficients: EW vs SPY Market Proxy\n'
    'w_i = α + γ₁·total_var + γ₂·syst_share  |  Crisis peaks',
    fontsize=12
)

for col_idx, crisis in enumerate(crises):
    for row_idx, (coef_key_ew, coef_key_spy, label) in enumerate([
        ('g1_ew', 'g1_spy', 'γ₁  (total_var)'),
        ('g2_ew', 'g2_spy', 'γ₂  (syst_share)'),
    ]):
        ax    = axes[row_idx][col_idx]
        sub   = table[table['crisis'] == crisis]
        x     = np.arange(len(est_list))
        bar_w = 0.35

        vals_ew  = [sub[sub['estimator'] == e][coef_key_ew].values[0]
                    if len(sub[sub['estimator'] == e]) > 0 else np.nan
                    for e in est_list]
        vals_spy = [sub[sub['estimator'] == e][coef_key_spy].values[0]
                    if len(sub[sub['estimator'] == e]) > 0 else np.nan
                    for e in est_list]

        b1 = ax.bar(x - bar_w/2, vals_ew,  width=bar_w, label='EW proxy',
                    color=[EST_COLORS[e] for e in est_list], alpha=0.9)
        b2 = ax.bar(x + bar_w/2, vals_spy, width=bar_w, label='SPY proxy',
                    color=[EST_COLORS[e] for e in est_list], alpha=0.45,
                    hatch='////', edgecolor='black', linewidth=0.5)

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(est_list, fontsize=9)
        if row_idx == 0:
            ax.set_title(crisis, fontsize=11, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)

# shared legend
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor='gray', alpha=0.9, label='EW proxy (solid)'),
    Patch(facecolor='gray', alpha=0.45, hatch='////', label='SPY proxy (hatched)'),
]
fig.legend(handles=legend_els, loc='upper right', fontsize=9)
plt.tight_layout()
out = FIGURES / 'spy_robustness_bars.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved → {out}')
print('\nDone.')
