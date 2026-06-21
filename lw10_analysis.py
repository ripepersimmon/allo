"""
LW-Only 10-Year Analysis (2015-2024)
=====================================
Uses only the Ledoit-Wolf estimator on the most recent 10 years of data.
Crisis periods in range: COVID (peak 2020-04-30), Rates (peak 2023-01-31).

Significant variables from the full-period study:
  - downside_vol   (σ⁻)          9/9 cells, strongest univariate
  - beta                          8/9 cells
  - total_var                     9/9 cells (γ₁ baseline)
  - inv_idio_var   (1/σ²_ε)      7/9 cells, Woodbury direct term
  - syst_share                    6/9 cells
  - avg_corr                      5/9 cells
  - corr_min                      +0.022 Δadj-R² (Exp E, Model K)
  - pc1_var_share + avg_corr      +0.049 Δadj-R² (Exp E, Model L)

Models:
  (D)    w = α + γ₁·total_var + γ₂·syst_share              [full-study baseline]
  (D_σ)  w = α + γ₁·total_var + γ₂·downside_vol            [strongest predictor swap]
  (K)    w = α + γ₁·total_var + γ₂·syst_share + γ₃·corr_min
  (L)    w = α + γ₁·total_var + γ₂·pc1_var_share + γ₃·avg_corr
  (W)    w = α + γ₁·total_var + γ₂·inv_idio_var            [Woodbury]
  (F)    w = α + γ₁·downside_vol + γ₂·inv_idio_var + γ₃·avg_corr + γ₄·corr_min  [full sig.]

Outputs:
  reports/2026-05-26_LW10.md
  results/figures/lw10/  (figures)
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats

from src.data_loader import load_prices_from_parquet, compute_returns, TICKERS
from src.estimators import lw_cov
from src.market import get_market_proxy
from src.analysis import rolling_gmv
from src.portfolio import effective_n, turnover

np.random.seed(42)

# ── config ────────────────────────────────────────────────────────────────────
DATA_START = '2015-01-01'
DATA_END   = '2024-12-31'
WINDOW     = 252

FIGURES = Path('results/figures/lw10')
REPORTS = Path('reports')
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

CRISIS_PEAKS = {
    'COVID': '2020-04-30',
    'Rates': '2023-01-31',
}
CRISIS_RANGES = {
    'COVID': ('2019-10-01', '2020-09-30'),
    'Rates': ('2021-07-01', '2023-01-31'),
}
PRE_RANGES = {
    'COVID': ('2018-01-01', '2019-09-30'),
    'Rates': ('2019-07-01', '2021-06-30'),
}

# ── data ──────────────────────────────────────────────────────────────────────
print('Loading prices (2015-2024)...')
prices  = load_prices_from_parquet('sp500', tickers=TICKERS,
                                    start='2014-01-01',   # 1yr warm-up for rolling
                                    end=DATA_END)
returns = compute_returns(prices, method='log')
# restrict analysis window
returns_10 = returns.loc[DATA_START:]
print(f'Returns (2015-2024): {returns_10.shape[0]} days × {returns_10.shape[1]} assets')


# ── OLS helpers ───────────────────────────────────────────────────────────────
def _ols(y, X):
    n, k = X.shape
    try:
        Q, R    = np.linalg.qr(X)
        beta    = np.linalg.solve(R, Q.T @ y)
        XtX_inv = np.linalg.inv(R) @ np.linalg.inv(R).T
    except np.linalg.LinAlgError:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        XtX_inv = np.linalg.pinv(X.T @ X)
    y_hat  = X @ beta
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 1e-14 else 0.0
    dof    = n - k
    if dof > 0 and ss_res > 1e-14:
        s2    = ss_res / dof
        se    = np.sqrt(np.maximum(np.diag(XtX_inv) * s2, 0))
        tstat = beta / np.where(se > 1e-14, se, np.nan)
        pval  = 2 * (1 - stats.t.cdf(np.abs(tstat), df=dof))
    else:
        se = tstat = pval = np.full(k, np.nan)
    adj_r2 = max(1 - (1 - r2) * (n - 1) / (n - k), 0.0) if n > k else 0.0
    return dict(beta=beta, se=se, tstat=tstat, pval=pval, r2=r2, adj_r2=adj_r2, n=n, k=k)


def star(t):
    if pd.isna(t): return ''
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.960 else ('*' if a > 1.645 else ''))


def gmv_weights_lw(win):
    """LW GMV weights (unconstrained analytical)."""
    cov = lw_cov(win)
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)
    raw = prec @ np.ones(cov.shape[0])
    s   = raw.sum()
    if abs(s) < 1e-10:
        return None, cov
    return raw / s, cov


# ── feature computation ───────────────────────────────────────────────────────
def compute_features(win: pd.DataFrame) -> pd.DataFrame:
    """Compute significant variables for each asset in the estimation window."""
    mkt     = get_market_proxy(win, 'ew', None)
    valid   = mkt.dropna().index.intersection(win.index)
    if len(valid) < 30:
        return pd.DataFrame()
    win     = win.loc[valid]
    mkt     = mkt.loc[valid]
    mkt_var = mkt.var()
    if mkt_var < 1e-14:
        return pd.DataFrame()

    tickers  = list(win.columns)
    n_tickers = len(tickers)
    cov_mat  = win.cov().values.copy()

    # PC1 variance share
    eigval, eigvec = np.linalg.eigh(cov_mat)
    lambda1        = eigval[-1]
    pc1_vec        = eigvec[:, -1]
    diag_vars      = np.diag(cov_mat)
    pc1_var_share  = np.where(
        diag_vars > 1e-14,
        np.clip(lambda1 * pc1_vec**2 / diag_vars, 0, 1),
        0.0
    )

    # Correlation matrix (off-diagonal)
    corr_mat = win.corr().values.copy()
    np.fill_diagonal(corr_mat, np.nan)
    avg_corr_arr = np.nanmean(corr_mat, axis=1)
    corr_min_arr = np.nanmin(corr_mat,  axis=1)

    rows = []
    for i, col in enumerate(tickers):
        r  = win[col]
        tv = float(diag_vars[i])
        if tv < 1e-14:
            continue

        # Beta & variance decomposition
        cov_rm   = float(r.cov(mkt))
        beta_m   = cov_rm / mkt_var
        syst_var = beta_m**2 * mkt_var
        idio_var = max(tv - syst_var, 1e-14)
        raw_r2   = min(beta_m**2 * mkt_var / tv, 1.0)
        syst_share = max(1 - (1 - raw_r2) * (len(win) - 1) / (len(win) - 2), 0.0)

        # Woodbury term
        inv_idio_var = 1.0 / idio_var

        # Downside volatility
        r_arr  = r.values
        neg_r  = r_arr[r_arr < 0]
        down_v = float(np.std(neg_r)) if len(neg_r) > 5 else np.nan

        # VaR / CVaR 5%
        q5    = np.percentile(r_arr, 5)
        tail  = r_arr[r_arr <= q5]
        var5  = -q5
        cvar5 = -float(tail.mean()) if len(tail) > 0 else np.nan

        rows.append(dict(
            ticker        = col,
            total_var     = tv,
            syst_share    = syst_share,
            beta          = beta_m,
            inv_idio_var  = inv_idio_var,
            downside_vol  = down_v,
            var_5pct      = var5,
            cvar_5pct     = cvar5,
            avg_corr      = avg_corr_arr[i],
            corr_min      = corr_min_arr[i],
            pc1_var_share = pc1_var_share[i],
        ))

    return pd.DataFrame(rows).set_index('ticker')


# ── cross-sectional OLS per crisis peak ───────────────────────────────────────
def run_ols_models(feat: pd.DataFrame, w_arr: np.ndarray,
                   tickers: list) -> dict:
    """Run 6 OLS models. Returns dict of model results."""
    common = feat.index.intersection(tickers)
    if len(common) < 8:
        return {}
    f   = feat.loc[common]
    idx = pd.Index(tickers).get_indexer(common)
    wt  = w_arr[idx]
    n   = len(wt)
    ones = np.ones(n)

    tv   = f['total_var'].values
    ss   = f['syst_share'].values
    dv   = f['downside_vol'].values
    iiv  = f['inv_idio_var'].values
    ac   = f['avg_corr'].values
    cm   = f['corr_min'].values
    p1   = f['pc1_var_share'].values
    bt   = f['beta'].values

    results = {}

    # Model D: total_var + syst_share (baseline)
    mask = ~np.isnan(tv + ss + wt)
    if mask.sum() >= 8:
        X = np.column_stack([ones[mask], tv[mask], ss[mask]])
        results['D'] = _ols(wt[mask], X)

    # Model D_σ: total_var + downside_vol
    mask = ~np.isnan(tv + dv + wt)
    if mask.sum() >= 8:
        X = np.column_stack([ones[mask], tv[mask], dv[mask]])
        results['D_sig'] = _ols(wt[mask], X)

    # Model K: total_var + syst_share + corr_min
    mask = ~np.isnan(tv + ss + cm + wt)
    if mask.sum() >= 8:
        X = np.column_stack([ones[mask], tv[mask], ss[mask], cm[mask]])
        results['K'] = _ols(wt[mask], X)

    # Model L: total_var + pc1_var_share + avg_corr
    mask = ~np.isnan(tv + p1 + ac + wt)
    if mask.sum() >= 8:
        X = np.column_stack([ones[mask], tv[mask], p1[mask], ac[mask]])
        results['L'] = _ols(wt[mask], X)

    # Model W: total_var + inv_idio_var (Woodbury)
    mask = ~np.isnan(tv + iiv + wt)
    if mask.sum() >= 8:
        X = np.column_stack([ones[mask], tv[mask], iiv[mask]])
        results['W'] = _ols(wt[mask], X)

    # Model F: downside_vol + inv_idio_var + avg_corr + corr_min
    mask = ~np.isnan(dv + iiv + ac + cm + wt)
    if mask.sum() >= 8:
        X = np.column_stack([ones[mask], dv[mask], iiv[mask], ac[mask], cm[mask]])
        results['F'] = _ols(wt[mask], X)

    return results


# ── weight-shift test ─────────────────────────────────────────────────────────
def permutation_test(w_pre: pd.DataFrame, w_crisis: pd.DataFrame,
                     n_perm: int = 2000) -> tuple[float, float]:
    """Squared L2 distance between mean weight vectors."""
    mean_pre    = w_pre.mean(axis=0)
    mean_crisis = w_crisis.mean(axis=0)
    cols = mean_pre.index.intersection(mean_crisis.index)
    obs  = float(np.sum((mean_pre[cols] - mean_crisis[cols])**2))

    all_w = pd.concat([w_pre[cols], w_crisis[cols]], axis=0)
    n1    = len(w_pre)
    perm_stats = []
    for _ in range(n_perm):
        idx   = np.random.permutation(len(all_w))
        g1    = all_w.iloc[idx[:n1]].mean(axis=0)
        g2    = all_w.iloc[idx[n1:]].mean(axis=0)
        perm_stats.append(float(np.sum((g1 - g2)**2)))
    p = np.mean(np.array(perm_stats) >= obs)
    return obs, p


# ── main analysis ─────────────────────────────────────────────────────────────
print('\n=== LW Cross-Sectional OLS at Crisis Peaks ===')

snap_results = {}   # snap_results[crisis] = dict(feat, w_arr, tickers, ols)
feat_store   = {}   # for reporting

for crisis, peak_date in CRISIS_PEAKS.items():
    end   = pd.Timestamp(peak_date)
    start = end - pd.offsets.BDay(WINDOW)
    win   = returns.loc[start:end].dropna(axis=1)
    tickers = list(win.columns)
    print(f'\n[{crisis}] peak={peak_date}  window={start.date()}→{end.date()}  '
          f'N={len(tickers)} assets')

    w_arr, cov_lw = gmv_weights_lw(win)
    if w_arr is None:
        print(f'  [SKIP] degenerate weights')
        continue

    feat = compute_features(win)
    feat_store[crisis] = feat
    ols_res = run_ols_models(feat, w_arr, tickers)
    snap_results[crisis] = dict(feat=feat, w_arr=w_arr, tickers=tickers,
                                 ols=ols_res, n_assets=len(tickers))

    for mname, res in ols_res.items():
        print(f'  Model {mname}: adj-R²={res["adj_r2"]:.3f}  N={res["n"]}  '
              f'tstat={[f"{t:.2f}" for t in res["tstat"][1:]]}')


# ── GMV weight shift test ─────────────────────────────────────────────────────
print('\n=== GMV Weight Shift Analysis (LW, long-only) ===')

weights_lo = rolling_gmv(returns, lw_cov, window=WINDOW, constrained=True)
# restrict to data from 2015 onwards
weights_lo = weights_lo.loc[DATA_START:]

shift_results = {}
for crisis in CRISIS_PEAKS:
    pre_start, pre_end     = PRE_RANGES[crisis]
    cris_start, cris_end   = CRISIS_RANGES[crisis]

    w_pre    = weights_lo.loc[pre_start:pre_end].dropna(how='all')
    w_crisis = weights_lo.loc[cris_start:cris_end].dropna(how='all')

    if len(w_pre) < 10 or len(w_crisis) < 10:
        print(f'  [{crisis}] insufficient rows — skipping')
        continue

    effn_pre    = w_pre.apply(effective_n, axis=1).median()
    effn_crisis = w_crisis.apply(effective_n, axis=1).median()

    obs, p_perm = permutation_test(w_pre, w_crisis, n_perm=2000)

    # per-asset Welch t-test + BH correction
    from scipy.stats import ttest_ind
    pvals_raw = []
    cols = w_pre.columns.intersection(w_crisis.columns)
    for col in cols:
        _, p = ttest_ind(w_pre[col].dropna(), w_crisis[col].dropna(),
                         equal_var=False)
        pvals_raw.append(p)
    pvals_raw = np.array(pvals_raw)
    # Benjamini-Hochberg
    m     = len(pvals_raw)
    order = np.argsort(pvals_raw)
    rank  = np.empty(m); rank[order] = np.arange(1, m + 1)
    adj_p = pvals_raw * m / rank
    adj_p = np.minimum.accumulate(adj_p[order][::-1])[::-1][np.argsort(order)]
    n_sig = int((adj_p < 0.05).sum())

    # top gainers/losers
    mean_diff = (w_crisis[cols].mean() - w_pre[cols].mean()).sort_values()
    top_losers  = mean_diff.head(5)
    top_gainers = mean_diff.tail(5)[::-1]

    shift_results[crisis] = dict(
        effn_pre=effn_pre, effn_crisis=effn_crisis,
        obs=obs, p_perm=p_perm,
        n_sig=n_sig, n_total=m,
        top_gainers=top_gainers, top_losers=top_losers,
        w_pre=w_pre, w_crisis=w_crisis,
    )
    print(f'  [{crisis}] EffN: pre={effn_pre:.2f} → crisis={effn_crisis:.2f}  '
          f'perm-p={p_perm:.4f}  sig_assets={n_sig}/{m}')


# ── figures ───────────────────────────────────────────────────────────────────

# Fig 1: adj-R² comparison across models for each crisis
model_labels = {
    'D':     'D: total_var\n+syst_share',
    'D_sig': 'Dσ: total_var\n+down_vol',
    'K':     'K: total_var\n+syst_share\n+corr_min',
    'L':     'L: total_var\n+pc1_share\n+avg_corr',
    'W':     'W: total_var\n+inv_idio',
    'F':     'F: down_vol\n+inv_idio\n+avg_corr\n+corr_min',
}
colors_mod = {'D': '#4575b4', 'D_sig': '#74add1', 'K': '#f46d43',
              'L': '#d73027', 'W': '#fdae61', 'F': '#1a9641'}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, crisis in zip(axes, ['COVID', 'Rates']):
    if crisis not in snap_results:
        ax.set_title(f'{crisis} (no data)')
        continue
    ols = snap_results[crisis]['ols']
    mnames = [m for m in model_labels if m in ols]
    adj_r2s = [ols[m]['adj_r2'] for m in mnames]
    bars = ax.bar(range(len(mnames)), adj_r2s,
                  color=[colors_mod[m] for m in mnames], width=0.6)
    ax.set_xticks(range(len(mnames)))
    ax.set_xticklabels([model_labels[m] for m in mnames], fontsize=7)
    ax.set_title(f'{crisis}', fontweight='bold')
    ax.set_ylabel('adj-R²')
    ax.set_ylim(0, max(adj_r2s) * 1.3 + 0.02)
    for bar, v in zip(bars, adj_r2s):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

fig.suptitle('LW (2015-2024) — Cross-Sectional adj-R² by Model', fontweight='bold')
plt.tight_layout()
fig.savefig(FIGURES / 'lw10_adjr2.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nFig saved: lw10_adjr2.png')


# Fig 2: coefficient heatmap — γ and t-stat for selected models
model_coef_map = {
    'D':     {'labels': ['intercept', 'total_var', 'syst_share'], 'display': 'D'},
    'D_sig': {'labels': ['intercept', 'total_var', 'downside_vol'], 'display': 'Dσ'},
    'K':     {'labels': ['intercept', 'total_var', 'syst_share', 'corr_min'], 'display': 'K'},
    'L':     {'labels': ['intercept', 'total_var', 'pc1_share', 'avg_corr'], 'display': 'L'},
    'W':     {'labels': ['intercept', 'total_var', 'inv_idio_var'], 'display': 'W'},
    'F':     {'labels': ['intercept', 'downside_vol', 'inv_idio_var', 'avg_corr', 'corr_min'], 'display': 'F'},
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, crisis in zip(axes, ['COVID', 'Rates']):
    if crisis not in snap_results:
        ax.set_visible(False)
        continue
    ols = snap_results[crisis]['ols']
    rows, row_labels = [], []
    for mname, info in model_coef_map.items():
        if mname not in ols:
            continue
        res = ols[mname]
        for j, lbl in enumerate(info['labels'][1:], start=1):  # skip intercept
            t = res['tstat'][j]
            rows.append(t)
            row_labels.append(f"{info['display']}: {lbl}")

    data = np.array(rows).reshape(len(rows), 1)
    vmax = np.nanmax(np.abs(data)) + 0.1
    im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks([])
    ax.set_title(f'{crisis} — t-statistics', fontweight='bold')
    for i, (t, lbl) in enumerate(zip(rows, row_labels)):
        s = star(t)
        ax.text(0, i, f'{t:.2f}{s}', ha='center', va='center',
                color='white' if abs(t) > vmax * 0.6 else 'black', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle('LW (2015-2024) — t-statistics per Variable per Model', fontweight='bold')
plt.tight_layout()
fig.savefig(FIGURES / 'lw10_tstat_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig saved: lw10_tstat_heatmap.png')


# Fig 3: Effective N time series (2015-2024)
fig, ax = plt.subplots(figsize=(13, 4))
effn_ts = weights_lo.apply(effective_n, axis=1)
ax.plot(effn_ts.index, effn_ts.values, color='#377eb8', lw=1.2, label='Effective N (LW)')
crisis_colors = {'COVID': '#e41a1c', 'Rates': '#ff7f00'}
for crisis, (cs, ce) in CRISIS_RANGES.items():
    ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce),
               alpha=0.15, color=crisis_colors[crisis], label=crisis)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.set_ylabel('Effective N (1/Σwᵢ²)')
ax.set_title('LW Long-Only GMV — Effective N (2015-2024)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(FIGURES / 'lw10_effn.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig saved: lw10_effn.png')


# Fig 4: top gainers/losers bar for each crisis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, crisis in zip(axes, ['COVID', 'Rates']):
    if crisis not in shift_results:
        ax.set_visible(False)
        continue
    sr = shift_results[crisis]
    gainers = sr['top_gainers']
    losers  = sr['top_losers']
    tickers_all = list(gainers.index) + list(losers.index)
    vals_all    = list(gainers.values) + list(losers.values)
    colors_bar  = ['#2ca25f' if v > 0 else '#de2d26' for v in vals_all]
    ax.barh(range(len(tickers_all)), [v * 100 for v in vals_all],
            color=colors_bar, edgecolor='white')
    ax.set_yticks(range(len(tickers_all)))
    ax.set_yticklabels(tickers_all, fontsize=9)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Δ weight (pp)')
    ax.set_title(f'{crisis} — Top 5 Gainers & Losers (LW)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
fig.savefig(FIGURES / 'lw10_weight_shift.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig saved: lw10_weight_shift.png')


# ── write report ──────────────────────────────────────────────────────────────
MODEL_LABELS_FULL = {
    'D':     'w = α + γ₁·total_var + γ₂·syst_share',
    'D_sig': 'w = α + γ₁·total_var + γ₂·downside_vol',
    'K':     'w = α + γ₁·total_var + γ₂·syst_share + γ₃·corr_min',
    'L':     'w = α + γ₁·total_var + γ₂·pc1_share + γ₃·avg_corr',
    'W':     'w = α + γ₁·total_var + γ₂·inv_idio_var',
    'F':     'w = α + γ₁·downside_vol + γ₂·inv_idio_var + γ₃·avg_corr + γ₄·corr_min',
}
MODEL_COEF_NAMES = {
    'D':     ['intercept', 'total_var', 'syst_share'],
    'D_sig': ['intercept', 'total_var', 'downside_vol'],
    'K':     ['intercept', 'total_var', 'syst_share', 'corr_min'],
    'L':     ['intercept', 'total_var', 'pc1_share', 'avg_corr'],
    'W':     ['intercept', 'total_var', 'inv_idio_var'],
    'F':     ['intercept', 'downside_vol', 'inv_idio_var', 'avg_corr', 'corr_min'],
}


def fmt_coef(res, names):
    parts = []
    for j in range(1, len(names)):
        b = res['beta'][j]
        t = res['tstat'][j]
        s = star(t)
        parts.append(f'{names[j]}={b:.4f}{s} (t={t:.2f})')
    return '  |  '.join(parts)


lines = []
lines.append(f"# LW 추정기 단독 — 최근 10년 분석 보고서")
lines.append(f"")
lines.append(f"**작성일**: 2026-05-26  ")
lines.append(f"**데이터**: S&P 100, 2015-01-01 ~ 2024-12-31  ")
lines.append(f"**추정기**: Ledoit-Wolf (LW) 단독  ")
lines.append(f"**분석 대상 위기**: COVID (2019-10 ~ 2020-09), 금리위기 (2021-07 ~ 2023-01)  ")
lines.append(f"**분석 변수**: 전기간 연구에서 유의하게 나온 변수만 사용  ")
lines.append(f"")
lines.append(f"---")
lines.append(f"")
lines.append(f"## 1. 분석 설계")
lines.append(f"")
lines.append(f"### 1.1 데이터 범위")
lines.append(f"")
lines.append(f"최근 10년(2015-2024)으로 한정하면 전기간 연구(2000-2024)에서 정의된 세 위기 중  ")
lines.append(f"GFC(2007-2009)는 범위 밖이며, **COVID와 금리위기만 포함**된다.")
lines.append(f"")
lines.append(f"| 위기 | Crisis 기간 | Pre-Crisis 기간 | 고점(snapshot) | 포함 여부 |")
lines.append(f"|------|------------|----------------|:-------------:|:---------:|")
lines.append(f"| GFC | 2007-01-01 ~ 2009-06-30 | 2005-2006 | 2009-03-31 | **제외** |")
lines.append(f"| COVID | 2019-10-01 ~ 2020-09-30 | 2018-01-01 ~ 2019-09-30 | 2020-04-30 | ✓ |")
lines.append(f"| Rates | 2021-07-01 ~ 2023-01-31 | 2019-07-01 ~ 2021-06-30 | 2023-01-31 | ✓ |")
lines.append(f"")
lines.append(f"### 1.2 사용 변수 선정 근거")
lines.append(f"")
lines.append(f"전기간 분석(2000-2024, 3추정기 × 3위기 = 9셀)에서 유의하게 확인된 변수만 포함한다:")
lines.append(f"")
lines.append(f"| 변수 | 전기간 유의 셀 | 기대 부호 | 선정 근거 |")
lines.append(f"|------|:------------:|:--------:|----------|")
lines.append(f"| downside_vol (σ⁻) | **9/9** | − | 가장 강한 단변량 예측 변수 |")
lines.append(f"| beta | 8/9 | − | 고베타 기피 일관 |")
lines.append(f"| total_var | 9/9 (γ₁) | − | 기준선 모형 필수 |")
lines.append(f"| inv_idio_var (1/σ²_ε) | 7/9 | **+** | Woodbury 이론 항 |")
lines.append(f"| syst_share | 6/9 | − | GFC·COVID 메커니즘 확인 |")
lines.append(f"| avg_corr | 5/9 | − | 상관 패널티 일관 |")
lines.append(f"| corr_min | +0.022 Δadj-R² | − | 최선 헤지 파트너 존재 여부 |")
lines.append(f"| pc1_var_share | +0.049 Δadj-R² (Model L) | − | avg_corr와 결합 시 최선 |")
lines.append(f"")
lines.append(f"**제외 변수**: log_dolvol (0/9), skewness (0/9), ex_kurtosis (0/9), idio_var (0/9)")
lines.append(f"")
lines.append(f"### 1.3 OLS 모형 명세")
lines.append(f"")
lines.append(f"| 모형 | 수식 | 출처 |")
lines.append(f"|------|------|------|")
for mname, formula in MODEL_LABELS_FULL.items():
    src = {'D': '전기간 기준선', 'D_sig': '최강 단변량 교체',
           'K': '실험 E Model K', 'L': '실험 E Model L (최선)',
           'W': 'Woodbury 이론', 'F': '유의 변수 통합'}
    lines.append(f"| ({mname}) | {formula} | {src[mname]} |")

lines.append(f"")
lines.append(f"---")
lines.append(f"")
lines.append(f"## 2. GMV 비중 이동 분석")
lines.append(f"")

for crisis in ['COVID', 'Rates']:
    if crisis not in shift_results:
        continue
    sr = shift_results[crisis]
    lines.append(f"### 2.{list(CRISIS_PEAKS.keys()).index(crisis)+1} {crisis}")
    lines.append(f"")
    n_assets = sr['n_total']
    lines.append(f"| 항목 | 값 |")
    lines.append(f"|------|-----|")
    lines.append(f"| pre-crisis Effective N (중앙값) | {sr['effn_pre']:.2f} |")
    lines.append(f"| crisis Effective N (중앙값) | {sr['effn_crisis']:.2f} |")
    lines.append(f"| EffN 변화 | {sr['effn_crisis'] - sr['effn_pre']:+.2f} ({(sr['effn_crisis']/sr['effn_pre']-1)*100:+.1f}%) |")
    lines.append(f"| Permutation test p-value | {sr['p_perm']:.4f} |")
    lines.append(f"| BH 5% 보정 후 유의 자산 수 | {sr['n_sig']}/{n_assets} ({sr['n_sig']/n_assets*100:.1f}%) |")
    lines.append(f"")
    lines.append(f"**상위 비중 증가 자산 (수혜):**")
    lines.append(f"")
    lines.append(f"| 자산 | Δ비중 (pp) |")
    lines.append(f"|------|----------:|")
    for ticker, dw in sr['top_gainers'].items():
        lines.append(f"| {ticker} | {dw*100:+.2f} |")
    lines.append(f"")
    lines.append(f"**상위 비중 감소 자산 (피해):**")
    lines.append(f"")
    lines.append(f"| 자산 | Δ비중 (pp) |")
    lines.append(f"|------|----------:|")
    for ticker, dw in sr['top_losers'].items():
        lines.append(f"| {ticker} | {dw*100:+.2f} |")
    lines.append(f"")

lines.append(f"---")
lines.append(f"")
lines.append(f"## 3. 횡단면 OLS 결과")
lines.append(f"")
lines.append(f"*\\* p<.10  \\*\\* p<.05  \\*\\*\\* p<.01*")
lines.append(f"")

for crisis in ['COVID', 'Rates']:
    if crisis not in snap_results:
        continue
    ols = snap_results[crisis]['ols']
    n_a = snap_results[crisis]['n_assets']
    lines.append(f"### 3.{list(CRISIS_PEAKS.keys()).index(crisis)+1} {crisis} (고점: {CRISIS_PEAKS[crisis]}, N={n_a})")
    lines.append(f"")

    # adj-R² table
    lines.append(f"#### adj-R² 비교")
    lines.append(f"")
    lines.append(f"| 모형 | adj-R² | N | 기준선 D 대비 Δ |")
    lines.append(f"|------|:------:|:-:|:--------------:|")
    r2_D = ols['D']['adj_r2'] if 'D' in ols else np.nan
    for mname in MODEL_LABELS_FULL:
        if mname not in ols:
            continue
        r2 = ols[mname]['adj_r2']
        delta = r2 - r2_D if mname != 'D' else ''
        delta_str = f'{delta:+.3f}' if delta != '' else '—'
        lines.append(f"| ({mname}) | {r2:.3f} | {ols[mname]['n']} | {delta_str} |")
    lines.append(f"")

    # coefficient table
    lines.append(f"#### 계수 상세")
    lines.append(f"")
    for mname in MODEL_LABELS_FULL:
        if mname not in ols:
            continue
        res = ols[mname]
        names = MODEL_COEF_NAMES[mname]
        lines.append(f"**Model {mname}** (adj-R²={res['adj_r2']:.3f})")
        lines.append(f"")
        lines.append(f"| 변수 | 계수 | t-stat | p-value |")
        lines.append(f"|------|-----:|:------:|:-------:|")
        for j, nm in enumerate(names):
            b = res['beta'][j]
            t = res['tstat'][j]
            p = res['pval'][j]
            s = star(t)
            lines.append(f"| {nm} | {b:.4e} | {t:.2f}{s} | {p:.4f} |")
        lines.append(f"")

lines.append(f"---")
lines.append(f"")
lines.append(f"## 4. 전기간 결과와의 비교")
lines.append(f"")
lines.append(f"### 4.1 Model D (기준선): γ₂(syst_share) 부호 일관성")
lines.append(f"")
lines.append(f"| 위기 | 기간 | 추정기 | γ₂(syst_share) | 유의 |")
lines.append(f"|------|------|--------|:--------------:|:----:|")
lines.append(f"| GFC | 2000-2024 | LW | −0.171 | *** |")
lines.append(f"| COVID | 2000-2024 | LW | −0.165 | *** |")
lines.append(f"| Rates | 2000-2024 | LW | −0.076 | * |")

for crisis in ['COVID', 'Rates']:
    if crisis not in snap_results or 'D' not in snap_results[crisis]['ols']:
        continue
    res = snap_results[crisis]['ols']['D']
    b   = res['beta'][2]
    t   = res['tstat'][2]
    s   = star(t)
    lines.append(f"| {crisis} | **2015-2024** | **LW** | **{b:.3f}** | **{s if s else 'n.s.'}** |")

lines.append(f"")
lines.append(f"### 4.2 Model L (최선 확장): adj-R² 비교")
lines.append(f"")
lines.append(f"| 위기 | 기간 | adj-R²(D) | adj-R²(L) | Δ |")
lines.append(f"|------|------|:---------:|:---------:|:-:|")
lines.append(f"| COVID | 2000-2024 | 0.209 | 0.351 | +0.142 |")
lines.append(f"| Rates | 2000-2024 | 0.060 | 0.150 | +0.090 |")

for crisis in ['COVID', 'Rates']:
    if crisis not in snap_results:
        continue
    ols = snap_results[crisis]['ols']
    r2d = ols['D']['adj_r2'] if 'D' in ols else np.nan
    r2l = ols['L']['adj_r2'] if 'L' in ols else np.nan
    delta = r2l - r2d if not (np.isnan(r2d) or np.isnan(r2l)) else np.nan
    lines.append(f"| {crisis} | **2015-2024** | **{r2d:.3f}** | **{r2l:.3f}** | **{delta:+.3f}** |")

lines.append(f"")
lines.append(f"---")
lines.append(f"")
lines.append(f"## 5. 핵심 발견")
lines.append(f"")

# Collect summary findings dynamically
findings = []

# GMV shift findings
for crisis in ['COVID', 'Rates']:
    if crisis not in shift_results:
        continue
    sr = shift_results[crisis]
    pct = (sr['effn_crisis'] / sr['effn_pre'] - 1) * 100
    direction = "감소 (집중)" if pct < 0 else "증가 (분산)"
    findings.append(
        f"**비중 이동**: {crisis} 위기 — Effective N {sr['effn_pre']:.1f} → {sr['effn_crisis']:.1f} "
        f"({pct:+.1f}%, {direction}). "
        f"Permutation p={sr['p_perm']:.4f}, BH-보정 유의 자산 {sr['n_sig']}/{sr['n_total']}."
    )

# OLS findings
best_models = {}
for crisis in ['COVID', 'Rates']:
    if crisis not in snap_results:
        continue
    ols = snap_results[crisis]['ols']
    best_m = max(ols, key=lambda m: ols[m]['adj_r2'])
    best_models[crisis] = (best_m, ols[best_m]['adj_r2'])

for crisis, (best_m, best_r2) in best_models.items():
    findings.append(
        f"**최선 모형**: {crisis}에서 Model {best_m} (adj-R²={best_r2:.3f})이 가장 높은 설명력. "
        f"기준선 D 대비 Δadj-R²={best_r2 - snap_results[crisis]['ols'].get('D', {}).get('adj_r2', best_r2):+.3f}."
    )

# syst_share sign check
for crisis in ['COVID', 'Rates']:
    if crisis not in snap_results or 'D' not in snap_results[crisis]['ols']:
        continue
    res = snap_results[crisis]['ols']['D']
    b   = res['beta'][2]
    t   = res['tstat'][2]
    s   = star(t)
    sign_ok = "음수 유지" if b < 0 else "양수 반전"
    findings.append(
        f"**syst_share 부호**: {crisis} γ₂={b:.3f} (t={t:.2f}{s}) — 전기간 대비 {sign_ok}."
    )

for i, f in enumerate(findings, 1):
    lines.append(f"{i}. {f}")
    lines.append(f"")

lines.append(f"---")
lines.append(f"")
lines.append(f"## 6. 한계 및 해석 주의사항")
lines.append(f"")
lines.append(f"1. **셀 수 축소**: 10년 + LW 단독으로 분석 셀이 9개(3위기×3추정기)에서 **2개**로 감소. ")
lines.append(f"   통계적 일반화에 주의가 필요하다.")
lines.append(f"2. **GFC 부재**: 가장 극단적인 위기(2008-2009 금융위기)가 제외되어, ")
lines.append(f"   폭락형 위기에서의 syst_share 메커니즘(6/6 유의) 검증 불가.")
lines.append(f"3. **기간 단축 효과**: 추정 창 252일이 동일하더라도 COVID pre-crisis(2018-)가 ")
lines.append(f"   데이터 범위(2015-) 내에 있어 warm-up 자료로는 2014년부터 사용.")
lines.append(f"4. **선택 편의**: 전기간 유의 변수를 골라 같은 기간 하위 집합에 적용하므로 ")
lines.append(f"   과적합 가능성이 있다. 결과는 in-sample 확인으로 해석해야 한다.")
lines.append(f"")
lines.append(f"---")
lines.append(f"")
lines.append(f"## 부록 — 산출 파일")
lines.append(f"")
lines.append(f"| 파일 | 내용 |")
lines.append(f"|------|------|")
lines.append(f"| `results/figures/lw10/lw10_adjr2.png` | 모형별 adj-R² 비교 |")
lines.append(f"| `results/figures/lw10/lw10_tstat_heatmap.png` | 변수별 t-통계량 히트맵 |")
lines.append(f"| `results/figures/lw10/lw10_effn.png` | Effective N 시계열 (2015-2024) |")
lines.append(f"| `results/figures/lw10/lw10_weight_shift.png` | 위기별 비중 이동 상위 자산 |")
lines.append(f"")
lines.append(f"---")
lines.append(f"")
lines.append(f"*분석 코드: `lw10_analysis.py` | 분석 환경: Python 3.x, statsmodels, scikit-learn, cvxpy*")

report_text = '\n'.join(lines)
out_path = REPORTS / '2026-05-26_LW10.md'
out_path.write_text(report_text, encoding='utf-8')
print(f'\nReport saved → {out_path}')
print('\n=== Done ===')
