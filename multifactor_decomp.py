"""
Multi-Factor Variance Decomposition vs. GMV Weight
====================================================
Replaces the single equal-weighted market factor in Model D with Fama-French
3-factor (FF3) and 5-factor (FF5) time-series R² as the measure of "how
systematic" each asset is.

For each asset i in a 252-day window:
  ff3_syst_share_i = adj-R² from regressing r_i on (Mkt-RF, SMB, HML)
  ff5_syst_share_i = adj-R² from regressing r_i on (Mkt-RF, SMB, HML, RMW, CMA)
  avg_corr_i       = mean off-diagonal pairwise correlation within the window

Models compared at each (crisis, estimator) cell:
  (D)  w = α + γ₁·total_var + γ₂·mkt_syst_share          [1-factor baseline]
  (G3) w = α + γ₁·total_var + γ₂·ff3_syst_share           [FF3 replaces mkt]
  (G5) w = α + γ₁·total_var + γ₂·ff5_syst_share           [FF5 replaces mkt]
  (H3) w = α + γ₁·total_var + γ₂·ff3_syst_share + γ₃·avg_corr
  (H5) w = α + γ₁·total_var + γ₂·ff5_syst_share + γ₃·avg_corr

Outputs:
    reports/multifactor_decomp_table.csv
    reports/multifactor_decomp_report.md
    results/figures/multifactor_decomp_r2.png
    results/figures/multifactor_decomp_gamma.png
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from scipy import stats

from src.data_loader import load_prices_from_parquet, compute_returns, TICKERS
from src.estimators import sample_cov, lw_cov, gerber_cov
from src.market import get_market_proxy
from src.ff49 import load_ff_factors

WINDOW  = 252
FIGURES = Path('results/figures')
REPORTS = Path('reports')
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

ESTIMATORS = {'Sample': sample_cov, 'LW': lw_cov, 'Gerber': gerber_cov}
EST_COLORS = {'Sample': '#e41a1c', 'LW': '#377eb8', 'Gerber': '#4daf4a'}
EST_LIST   = list(ESTIMATORS.keys())

CRISIS_PEAKS = {
    'GFC':   '2009-03-31',
    'COVID': '2020-04-30',
    'Rates': '2023-01-31',
}

# ── data ──────────────────────────────────────────────────────────────────────
print('Loading prices...')
prices  = load_prices_from_parquet('sp500', tickers=TICKERS,
                                   start='2000-01-01', end='2024-12-31')
returns = compute_returns(prices, method='log')
print(f'Returns: {returns.shape[0]} days × {returns.shape[1]} assets')

print('Loading FF3 factors...')
ff3 = load_ff_factors('F-F_Research_Data_Factors_daily.csv',
                      '2000-01-01', '2024-12-31')
print(f'FF3: {ff3.shape[0]} days × columns {ff3.columns.tolist()}')

print('Loading FF5 factors...')
ff5 = load_ff_factors('F-F_Research_Data_5_Factors_2x3_daily.csv',
                      '2000-01-01', '2024-12-31')
print(f'FF5: {ff5.shape[0]} days × columns {ff5.columns.tolist()}\n')


# ── helpers ───────────────────────────────────────────────────────────────────

def get_window(end_date: str) -> pd.DataFrame:
    end   = pd.Timestamp(end_date)
    start = end - pd.offsets.BDay(WINDOW)
    return returns.loc[start:end].dropna(axis=1)


def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS via QR; X must include intercept column."""
    n, k = X.shape
    try:
        Q, R  = np.linalg.qr(X)
        beta  = np.linalg.solve(R, Q.T @ y)
        R_inv = np.linalg.inv(R)
        XtX_inv = R_inv @ R_inv.T
    except np.linalg.LinAlgError:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        XtX_inv = np.linalg.pinv(X.T @ X)

    y_hat  = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 1e-14 else 0.0
    dof    = n - k
    if dof > 0 and ss_res > 1e-14:
        sigma2 = ss_res / dof
        se     = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0))
        tstat  = beta / np.where(se > 1e-14, se, np.nan)
        pval   = 2 * (1 - stats.t.cdf(np.abs(tstat), df=dof))
    else:
        se = tstat = pval = np.full(k, np.nan)
    return dict(beta=beta, se=se, tstat=tstat, pval=pval, r2=r2, n=n)


def _adj_r2(r2: float, n: int, k: int) -> float:
    """Adjusted R²; clamped to [0, 1]."""
    if n <= k:
        return 0.0
    return max(1 - (1 - r2) * (n - 1) / (n - k), 0.0)


def _ff_syst_share(r: pd.Series, factors: pd.DataFrame) -> float:
    """Adjusted R² of regressing asset return r on factor DataFrame (+ intercept)."""
    idx = r.index.intersection(factors.index)
    if len(idx) < factors.shape[1] + 5:
        return 0.0
    y = r.loc[idx].values
    X = np.column_stack([np.ones(len(idx)), factors.loc[idx].values])
    beta   = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat  = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_raw = 1 - ss_res / ss_tot if ss_tot > 1e-14 else 0.0
    return _adj_r2(r2_raw, len(idx), X.shape[1])


def gmv_weights(cov: np.ndarray):
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)
    raw   = prec @ np.ones(cov.shape[0])
    total = raw.sum()
    if abs(total) < 1e-10:
        return None
    return raw / total


def compute_features(win: pd.DataFrame) -> pd.DataFrame:
    """Per-asset features for cross-sectional OLS.

    Columns: total_var, mkt_syst_share, ff3_syst_share, ff5_syst_share, avg_corr
    """
    mkt   = get_market_proxy(win, 'ew', None)
    valid = mkt.dropna().index.intersection(win.index)
    if len(valid) < 30:
        return pd.DataFrame()
    win   = win.loc[valid]
    mkt   = mkt.loc[valid]
    mkt_var = mkt.var()
    if mkt_var < 1e-14:
        return pd.DataFrame()

    # align factor DataFrames to this window
    ff3_win = ff3.reindex(win.index).dropna()
    ff5_win = ff5.reindex(win.index).dropna()

    # avg pairwise correlation (per asset)
    corr_mat = win.corr().values.copy()
    np.fill_diagonal(corr_mat, np.nan)
    avg_corr_arr = np.nanmean(corr_mat, axis=1)

    rows = []
    for i, col in enumerate(win.columns):
        r         = win[col]
        total_var = r.var()
        if total_var < 1e-14:
            continue

        # single-factor (EW market) syst_share — adj-R² for comparability with FF3/FF5
        beta_m         = r.cov(mkt) / mkt_var
        raw_r2_mkt     = min(beta_m ** 2 * mkt_var / total_var, 1.0)
        mkt_syst_share = _adj_r2(raw_r2_mkt, len(r), 2)

        # FF3 and FF5 multi-factor syst_shares (adj-R² of time-series regression)
        ff3_share = _ff_syst_share(r, ff3_win)
        ff5_share = _ff_syst_share(r, ff5_win)

        rows.append({
            'ticker':          col,
            'total_var':       total_var,
            'mkt_syst_share':  mkt_syst_share,
            'ff3_syst_share':  ff3_share,
            'ff5_syst_share':  ff5_share,
            'avg_corr':        avg_corr_arr[i],
        })

    return pd.DataFrame(rows).set_index('ticker')


def run_models(features: pd.DataFrame, w_vals: np.ndarray,
               tickers: pd.Index) -> dict | None:
    """Run Models D, G3, G5, H3, H5 for one (crisis, estimator) cell."""
    common = features.index.intersection(tickers)
    if len(common) < 6:
        return None

    feat = features.loc[common]
    idx  = pd.Index(tickers).get_indexer(common)
    wt   = w_vals[idx]

    tv   = feat['total_var'].values
    ms   = feat['mkt_syst_share'].values
    f3   = feat['ff3_syst_share'].values
    f5   = feat['ff5_syst_share'].values
    ac   = feat['avg_corr'].values
    ones = np.ones(len(wt))

    rD  = _ols(wt, np.column_stack([ones, tv, ms]))
    rG3 = _ols(wt, np.column_stack([ones, tv, f3]))
    rG5 = _ols(wt, np.column_stack([ones, tv, f5]))
    rH3 = _ols(wt, np.column_stack([ones, tv, f3, ac]))
    rH5 = _ols(wt, np.column_stack([ones, tv, f5, ac]))

    def _g(r, i): return r['beta'][i], r['tstat'][i], r['pval'][i]

    g2_D,  g2_D_t,  p2_D  = _g(rD,  2)
    g2_G3, g2_G3_t, p2_G3 = _g(rG3, 2)
    g2_G5, g2_G5_t, p2_G5 = _g(rG5, 2)
    g2_H3, g2_H3_t, p2_H3 = _g(rH3, 2)
    g2_H5, g2_H5_t, p2_H5 = _g(rH5, 2)
    g3_H3, g3_H3_t, p3_H3 = _g(rH3, 3)
    g3_H5, g3_H5_t, p3_H5 = _g(rH5, 3)

    n = rD['n']
    return {
        'r2_D':  _adj_r2(rD['r2'],  n, 3), 'r2_G3': _adj_r2(rG3['r2'], n, 3),
        'r2_G5': _adj_r2(rG5['r2'], n, 3), 'r2_H3': _adj_r2(rH3['r2'], n, 4),
        'r2_H5': _adj_r2(rH5['r2'], n, 4),
        'n': n,
        'g2_D':  g2_D,  'g2_D_t':  g2_D_t,  'p2_D':  p2_D,
        'g2_G3': g2_G3, 'g2_G3_t': g2_G3_t, 'p2_G3': p2_G3,
        'g2_G5': g2_G5, 'g2_G5_t': g2_G5_t, 'p2_G5': p2_G5,
        'g2_H3': g2_H3, 'g2_H3_t': g2_H3_t, 'p2_H3': p2_H3,
        'g2_H5': g2_H5, 'g2_H5_t': g2_H5_t, 'p2_H5': p2_H5,
        'g3_H3': g3_H3, 'g3_H3_t': g3_H3_t, 'p3_H3': p3_H3,
        'g3_H5': g3_H5, 'g3_H5_t': g3_H5_t, 'p3_H5': p3_H5,
    }


# ── snapshot analysis ─────────────────────────────────────────────────────────

print('Running cross-sectional OLS at crisis peaks...')
records = []

for crisis, peak_date in CRISIS_PEAKS.items():
    win      = get_window(peak_date)
    features = compute_features(win)
    if features.empty:
        print(f'  {crisis}: skipped (empty features)')
        continue

    for est_name, est_fn in ESTIMATORS.items():
        try:
            cov   = est_fn(win)
            raw_w = gmv_weights(cov)
            if raw_w is None:
                continue
            w = pd.Series(raw_w, index=win.columns)
        except Exception:
            continue

        res = run_models(features, w.values, w.index)
        if res is None:
            continue
        res.update({'crisis': crisis, 'estimator': est_name})
        records.append(res)

    # print avg FF syst_share for context
    f3_mean = features['ff3_syst_share'].mean()
    f5_mean = features['ff5_syst_share'].mean()
    print(f'  {crisis} done  n={len(features)}  '
          f'avg ff3_share={f3_mean:.3f}  avg ff5_share={f5_mean:.3f}')

df = pd.DataFrame(records).set_index(['crisis', 'estimator'])
print()

# ── save CSV ──────────────────────────────────────────────────────────────────
csv_path = REPORTS / 'multifactor_decomp_table.csv'
df.to_csv(csv_path, float_format='%.6f')
print(f'Saved → {csv_path}')

# ── console summary ───────────────────────────────────────────────────────────
def star(t):
    if np.isnan(t): return ''
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.960 else ('*' if a > 1.645 else ''))

print('\n=== R² Comparison ===')
hdr = f'{"Crisis":<8} {"Est":<8} {"R²(D)":>6} {"R²(G3)":>7} {"R²(G5)":>7} ' \
      f'{"R²(H3)":>7} {"R²(H5)":>7}  {"ΔG3":>5} {"ΔG5":>5} {"ΔH3":>5} {"ΔH5":>5}'
print(hdr)
for (crisis, est), row in df.iterrows():
    d, g3, g5, h3, h5 = row['r2_D'], row['r2_G3'], row['r2_G5'], row['r2_H3'], row['r2_H5']
    print(f'{crisis:<8} {est:<8} {d:6.3f} {g3:7.3f} {g5:7.3f} {h3:7.3f} {h5:7.3f} '
          f' {g3-d:+5.3f} {g5-d:+5.3f} {h3-d:+5.3f} {h5-d:+5.3f}')

print('\n=== γ₂ (syst_share coef) across models ===')
print(f'{"Crisis":<8} {"Est":<8} '
      f'{"γ₂(D)":>9} {"γ₂(G3)":>10} {"γ₂(G5)":>10} {"γ₂(H3)":>10} {"γ₂(H5)":>10}')
for (crisis, est), row in df.iterrows():
    def fmt(key_g, key_t, r=row):
        return f'{r[key_g]:8.4f}{star(r[key_t]):<3}'
    print(f'{crisis:<8} {est:<8} '
          f'{fmt("g2_D","g2_D_t")} {fmt("g2_G3","g2_G3_t")} '
          f'{fmt("g2_G5","g2_G5_t")} {fmt("g2_H3","g2_H3_t")} '
          f'{fmt("g2_H5","g2_H5_t")}')

print('\n=== γ₃ (avg_corr coef) in H models ===')
print(f'{"Crisis":<8} {"Est":<8} {"γ₃(H3)":>10} {"γ₃(H5)":>10}')
for (crisis, est), row in df.iterrows():
    def fmt3(key_g, key_t, r=row):
        return f'{r[key_g]:8.4f}{star(r[key_t]):<3}'
    print(f'{crisis:<8} {est:<8} '
          f'{fmt3("g3_H3","g3_H3_t")} {fmt3("g3_H5","g3_H5_t")}')


# ── Figure 1: R² bar chart ────────────────────────────────────────────────────
crises       = list(CRISIS_PEAKS.keys())
model_keys   = ['r2_D', 'r2_G3', 'r2_G5', 'r2_H3', 'r2_H5']
model_labels = ['(D) EW mkt', '(G3) FF3', '(G5) FF5', '(H3) FF3+corr', '(H5) FF5+corr']
model_colors = ['#2166ac', '#d7191c', '#e87a35', '#1a9641', '#4dac26']

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
bar_w = 0.16
x = np.arange(len(EST_LIST))

for ax, crisis in zip(axes, crises):
    for mi, (mk, ml, mc) in enumerate(zip(model_keys, model_labels, model_colors)):
        vals = [float(df.loc[(crisis, e), mk]) if (crisis, e) in df.index else 0.0
                for e in EST_LIST]
        offset = (mi - 2) * bar_w
        bars = ax.bar(x + offset, vals, width=bar_w, label=ml, color=mc, alpha=0.85,
                      edgecolor='white', linewidth=0.4)
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=5)

    ax.set_xticks(x); ax.set_xticklabels(EST_LIST, fontsize=10)
    ax.set_title(crisis, fontsize=12, fontweight='bold')
    ax.set_ylabel('adj-R²' if crisis == 'GFC' else '')
    ax.set_ylim(0, df[model_keys].max().max() * 1.3)
    ax.grid(axis='y', alpha=0.3)

axes[0].legend(fontsize=7.5, loc='upper right')
fig.suptitle(
    'adj-R² Comparison: Single-Factor (D) vs FF3 (G3) vs FF5 (G5) vs +avg_corr (H)\n'
    'w_i = α + γ₁·total_var + γ₂·syst_share  |  crisis peaks',
    fontsize=10)
plt.tight_layout()
out1 = FIGURES / 'multifactor_decomp_r2.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved → {out1}')


# ── Figure 2: γ₂ comparison across models ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=False)
gamma_models = [
    ('g2_D',  'g2_D_t',  '(D) EW',   '#2166ac', None),
    ('g2_G3', 'g2_G3_t', '(G3) FF3', '#d7191c', None),
    ('g2_G5', 'g2_G5_t', '(G5) FF5', '#e87a35', None),
    ('g2_H3', 'g2_H3_t', '(H3) FF3+corr', '#1a9641', '////'),
    ('g2_H5', 'g2_H5_t', '(H5) FF5+corr', '#4dac26', '////'),
]
bar_w = 0.16

for ax, crisis in zip(axes, crises):
    for mi, (gk, tk, label, color, hatch) in enumerate(gamma_models):
        vals = [float(df.loc[(crisis, e), gk]) if (crisis, e) in df.index else np.nan
                for e in EST_LIST]
        tvals = [float(df.loc[(crisis, e), tk]) if (crisis, e) in df.index else np.nan
                 for e in EST_LIST]
        offset = (mi - 2) * bar_w
        bars = ax.bar(x + offset, vals, width=bar_w, label=label, color=color,
                      alpha=0.80, hatch=hatch, edgecolor='black', linewidth=0.3)
        for bar, v, t in zip(bars, vals, tvals):
            s = star(t)
            if s and not np.isnan(t):
                y_pos = bar.get_y() + bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                        s, ha='center', va='bottom' if v >= 0 else 'top', fontsize=7)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(EST_LIST, fontsize=10)
    ax.set_title(crisis, fontsize=12, fontweight='bold')
    if crisis == 'GFC':
        ax.set_ylabel('γ₂  (syst_share coefficient)', fontsize=9)
    ax.tick_params(axis='y', labelsize=8)

axes[0].legend(fontsize=7, loc='lower left', ncol=1)
fig.suptitle(
    'γ₂ on syst_share: Single-Factor vs FF3 vs FF5 (with and without avg_corr)\n'
    '* p<.10  ** p<.05  *** p<.01',
    fontsize=9)
plt.tight_layout()
out2 = FIGURES / 'multifactor_decomp_gamma.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved → {out2}')


# ── report ────────────────────────────────────────────────────────────────────
def _fmt(g, t):
    return f'{g:.4f}{"***" if abs(t)>2.576 else "**" if abs(t)>1.960 else "*" if abs(t)>1.645 else ""}'

r2_rows, g2_rows, g3_rows = [], [], []
for crisis in crises:
    for est in EST_LIST:
        if (crisis, est) not in df.index:
            continue
        row = df.loc[(crisis, est)]
        d, g3, g5, h3, h5 = row['r2_D'], row['r2_G3'], row['r2_G5'], row['r2_H3'], row['r2_H5']
        r2_rows.append(
            f'| {crisis} | {est} | {d:.3f} | {g3:.3f} ({g3-d:+.3f}) | '
            f'{g5:.3f} ({g5-d:+.3f}) | {h3:.3f} ({h3-d:+.3f}) | '
            f'{h5:.3f} ({h5-d:+.3f}) | {int(row["n"])} |'
        )
        g2_rows.append(
            f'| {crisis} | {est} | {_fmt(row["g2_D"], row["g2_D_t"])} | '
            f'{_fmt(row["g2_G3"], row["g2_G3_t"])} | '
            f'{_fmt(row["g2_G5"], row["g2_G5_t"])} | '
            f'{_fmt(row["g2_H3"], row["g2_H3_t"])} | '
            f'{_fmt(row["g2_H5"], row["g2_H5_t"])} |'
        )
        g3_rows.append(
            f'| {crisis} | {est} | {_fmt(row["g3_H3"], row["g3_H3_t"])} | '
            f'{_fmt(row["g3_H5"], row["g3_H5_t"])} |'
        )

avg_delta = {mk: (df[mk] - df['r2_D']).mean() for mk in ['r2_G3','r2_G5','r2_H3','r2_H5']}
best_model = max(avg_delta, key=avg_delta.get)

report = f"""# Multi-Factor Decomposition: FF3 and FF5 vs. Single-Factor

**Date**: {pd.Timestamp.today().strftime('%Y-%m-%d')}
**Estimators**: Sample, Ledoit-Wolf (LW), Gerber
**Window**: {WINDOW} trading days

---

## 1. Specification

For each asset *i* in the 252-day window, we compute:

| Symbol | Definition |
|--------|-----------|
| `mkt_syst_share` | adj-R² from regressing r_i on EW market return (k=2) |
| `ff3_syst_share` | adj-R² from regressing r_i on Mkt-RF, SMB, HML (k=4) |
| `ff5_syst_share` | adj-R² from regressing r_i on Mkt-RF, SMB, HML, RMW, CMA (k=6) |
| `avg_corr` | mean off-diagonal pairwise correlation in the window |

All syst_share measures use adj-R² so they are on a comparable scale. adj-R² penalty is small at n={WINDOW} (≤ 2 pp difference from raw R² for k≤6).

Cross-sectional models:
- **(D)**  w = α + γ₁·total_var + γ₂·mkt_syst_share
- **(G3)** w = α + γ₁·total_var + γ₂·ff3_syst_share
- **(G5)** w = α + γ₁·total_var + γ₂·ff5_syst_share
- **(H3)** w = α + γ₁·total_var + γ₂·ff3_syst_share + γ₃·avg_corr
- **(H5)** w = α + γ₁·total_var + γ₂·ff5_syst_share + γ₃·avg_corr

---

## 2. adj-R² Results

| Crisis | Est | adj-R²(D) | adj-R²(G3) Δ | adj-R²(G5) Δ | adj-R²(H3) Δ | adj-R²(H5) Δ | N |
|--------|-----|-----------|-------------|-------------|-------------|-------------|---|
{chr(10).join(r2_rows)}

Average adj-R² gain vs Model D: G3={avg_delta['r2_G3']:+.3f}, G5={avg_delta['r2_G5']:+.3f}, H3={avg_delta['r2_H3']:+.3f}, H5={avg_delta['r2_H5']:+.3f}
**Best model on average**: {best_model.replace('r2_', '')}

---

## 3. γ₂ Coefficient Table (syst_share)

| Crisis | Est | γ₂(D) | γ₂(G3) | γ₂(G5) | γ₂(H3) | γ₂(H5) |
|--------|-----|-------|--------|--------|--------|--------|
{chr(10).join(g2_rows)}

*Significance: * p<.10  ** p<.05  *** p<.01 (two-sided, asymptotic z-thresholds: 1.645, 1.960, 2.576)*

---

## 4. γ₃ Coefficient Table (avg_corr in H models)

| Crisis | Est | γ₃(H3) | γ₃(H5) |
|--------|-----|--------|--------|
{chr(10).join(g3_rows)}

*avg_corr is the mean off-diagonal pairwise correlation for that asset's 252-day window.*

---

## 5. Figures

| Figure | File |
|--------|------|
| adj-R² bars | `multifactor_decomp_r2.png` |
| γ₂ comparison | `multifactor_decomp_gamma.png` |

---

*Analysis code: `multifactor_decomp.py`*
"""

rpt_path = REPORTS / 'multifactor_decomp_report.md'
rpt_path.write_text(report, encoding='utf-8')
print(f'Saved → {rpt_path}')
print('\nDone.')
