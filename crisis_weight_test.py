"""
Statistical test: pre-crisis vs crisis weight allocation change
Tests:
  1. Per-asset Welch t-test + Benjamini-Hochberg FDR correction
  2. Portfolio-level permutation test (squared L2 distance of mean vectors)
  3. Effective N Mann-Whitney U test
  4. Weight distribution shift (mean / std / skewness)
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind

from src.data_loader import load_prices_from_parquet, compute_returns, TICKERS
from src.estimators import sample_cov, lw_cov, gerber_cov
from src.portfolio import gmv_long_only, effective_n
from src.analysis import rolling_gmv

np.random.seed(42)

# ── config ────────────────────────────────────────────────────────────────────
WINDOW  = 252
FIGURES = Path('results/figures')
REPORTS = Path('reports')
FIGURES.mkdir(parents=True, exist_ok=True)

ESTIMATORS = {'Sample': sample_cov, 'LW': lw_cov, 'Gerber': gerber_cov}
EST_COLORS = {'Sample': '#e41a1c', 'LW': '#377eb8', 'Gerber': '#4daf4a'}

PERIODS = {
    'GFC':   {'pre': ('2005-01-01', '2006-12-31'), 'crisis': ('2007-01-01', '2009-06-30')},
    'COVID': {'pre': ('2018-01-01', '2019-09-30'), 'crisis': ('2019-10-01', '2020-09-30')},
    'Rates': {'pre': ('2019-07-01', '2021-06-30'), 'crisis': ('2021-07-01', '2023-01-31')},
}

# ── data ──────────────────────────────────────────────────────────────────────
print('Loading data...')
prices  = load_prices_from_parquet('sp500', tickers=TICKERS,
                                   start='2000-01-01', end='2024-12-31')
returns = compute_returns(prices, method='log')
print(f'Returns: {returns.shape}\n')

# ── rolling weights ───────────────────────────────────────────────────────────
print('Computing rolling GMV weights (long-only, window=252)...')
weights = {}
for est_name, est_fn in ESTIMATORS.items():
    print(f'  {est_name}', flush=True)
    weights[est_name] = rolling_gmv(returns, est_fn, window=WINDOW, constrained=True)
print()


# ── helper: BH correction ─────────────────────────────────────────────────────
def bh_correction(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    thresholds = ranks / n * alpha
    reject = pvals <= thresholds
    # monotone: reject all p-values up to the largest significant one
    if reject.any():
        max_k = ranks[reject].max()
        reject = ranks <= max_k
    return reject


# ── Test 1: per-asset Welch t-test ────────────────────────────────────────────
def per_asset_ttest(w_pre: pd.DataFrame, w_crisis: pd.DataFrame):
    """
    For each asset: Welch t-test comparing weight distributions.
    Returns DataFrame with t-stat, p-value, Cohen's d, BH-significant flag.
    """
    common = w_pre.columns.intersection(w_crisis.columns)
    records = []
    for col in common:
        a = w_pre[col].values
        b = w_crisis[col].values
        t, p = ttest_ind(a, b, equal_var=False)
        # Cohen's d (pooled std)
        pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
        d = (b.mean() - a.mean()) / pooled_std if pooled_std > 0 else 0.0
        records.append({'ticker': col, 't': t, 'p': p, 'd': d,
                        'pre_mean': a.mean(), 'crisis_mean': b.mean(),
                        'delta_mean': b.mean() - a.mean()})
    df = pd.DataFrame(records).set_index('ticker')
    df['bh_reject'] = bh_correction(df['p'].values)
    return df


# ── Test 2: portfolio-level permutation test ──────────────────────────────────
def permutation_test(w_pre: pd.DataFrame, w_crisis: pd.DataFrame,
                     n_perm: int = 2000) -> tuple[float, float]:
    """
    H0: mean weight vector same in pre-crisis and crisis.
    Statistic: squared L2 distance between mean weight vectors.
    """
    common = w_pre.columns.intersection(w_crisis.columns)
    a = w_pre[common].values   # (T_pre, n)
    b = w_crisis[common].values  # (T_crisis, n)
    pooled = np.vstack([a, b])
    n_a = len(a)

    observed = np.sum((a.mean(0) - b.mean(0)) ** 2)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        idx = np.random.permutation(len(pooled))
        null[i] = np.sum((pooled[idx[:n_a]].mean(0) - pooled[idx[n_a:]].mean(0)) ** 2)

    p_val = (null >= observed).mean()
    return float(observed), float(p_val)


# ── Test 3: Effective N Mann-Whitney U ────────────────────────────────────────
def effn_test(w_pre: pd.DataFrame, w_crisis: pd.DataFrame) -> dict:
    eff_pre    = w_pre.apply(effective_n, axis=1).values
    eff_crisis = w_crisis.apply(effective_n, axis=1).values
    stat, p = mannwhitneyu(eff_pre, eff_crisis, alternative='two-sided')
    return {
        'pre_mean': eff_pre.mean(), 'pre_std': eff_pre.std(),
        'crisis_mean': eff_crisis.mean(), 'crisis_std': eff_crisis.std(),
        'U': stat, 'p': p,
    }


# ── Test 4: weight distribution moments ───────────────────────────────────────
def distribution_moments(w: pd.DataFrame) -> dict:
    vals = w.values.flatten()
    return {
        'mean': vals.mean(), 'std': vals.std(),
        'skew': float(pd.Series(vals).skew()),
        'kurt': float(pd.Series(vals).kurt()),
        'max':  vals.max(), 'min': vals.min(),
    }


# ── run all tests ─────────────────────────────────────────────────────────────
results = {}          # (crisis, estimator) → dict of test outputs
ttest_tables = {}     # (crisis, estimator) → per-asset DataFrame

for crisis_name, segs in PERIODS.items():
    pre_start, pre_end       = segs['pre']
    crisis_start, crisis_end = segs['crisis']

    for est_name, w_full in weights.items():
        key = (crisis_name, est_name)
        print(f'Testing {crisis_name} / {est_name}...', flush=True)

        w_pre    = w_full.loc[pre_start:pre_end]
        w_crisis = w_full.loc[crisis_start:crisis_end]

        # per-asset t-test
        tt = per_asset_ttest(w_pre, w_crisis)
        ttest_tables[key] = tt

        # permutation test
        obs, p_perm = permutation_test(w_pre, w_crisis, n_perm=2000)

        # effective N
        en = effn_test(w_pre, w_crisis)

        # distribution moments
        m_pre    = distribution_moments(w_pre)
        m_crisis = distribution_moments(w_crisis)

        results[key] = {
            'n_assets': len(tt),
            'n_sig_bh': int(tt['bh_reject'].sum()),
            'pct_sig': tt['bh_reject'].mean() * 100,
            'n_increase': int((tt.loc[tt['bh_reject'], 'delta_mean'] > 0).sum()),
            'n_decrease': int((tt.loc[tt['bh_reject'], 'delta_mean'] < 0).sum()),
            'perm_stat': obs, 'perm_p': p_perm,
            'effn_pre': en['pre_mean'], 'effn_crisis': en['crisis_mean'],
            'effn_p': en['p'],
            'std_pre': m_pre['std'], 'std_crisis': m_crisis['std'],
            'skew_pre': m_pre['skew'], 'skew_crisis': m_crisis['skew'],
        }

print()


# ── Figure 1: summary heatmap — % significant assets ─────────────────────────
def plot_summary_heatmap():
    crises = list(PERIODS.keys())
    ests   = list(ESTIMATORS.keys())

    metrics = {
        '유의 자산 비율 (%)\n(BH-corrected)': 'pct_sig',
        'Permutation\np-value': 'perm_p',
        'Effective N\n(pre)': 'effn_pre',
        'Effective N\n(crisis)': 'effn_crisis',
        'Effective N\n변화 p-value': 'effn_p',
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.5))
    for ax, (title, key) in zip(axes, metrics.items()):
        mat = np.array([[results[(c, e)][key] for e in ests] for c in crises])
        im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(range(len(ests))); ax.set_xticklabels(ests, fontsize=9)
        ax.set_yticks(range(len(crises))); ax.set_yticklabels(crises, fontsize=9)
        for i in range(len(crises)):
            for j in range(len(ests)):
                ax.text(j, i, f'{mat[i,j]:.3f}' if mat[i,j] < 1 else f'{mat[i,j]:.1f}',
                        ha='center', va='center', fontsize=8, color='black')
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Pre-crisis vs Crisis 검정 요약', fontsize=12)
    plt.tight_layout()
    out = FIGURES / 'crisis_test_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


# ── Figure 2: per-asset volcano plot (t vs delta_mean) ────────────────────────
def plot_volcano_grid():
    crises = list(PERIODS.keys())
    ests   = list(ESTIMATORS.keys())
    fig, axes = plt.subplots(len(crises), len(ests),
                             figsize=(5 * len(ests), 4 * len(crises)),
                             sharex=False, sharey=False)

    for r, crisis_name in enumerate(crises):
        for c, est_name in enumerate(ests):
            ax  = axes[r][c]
            tt  = ttest_tables[(crisis_name, est_name)]
            sig = tt['bh_reject']

            ax.scatter(tt.loc[~sig, 'delta_mean'], -np.log10(tt.loc[~sig, 'p'].clip(1e-10)),
                       s=12, alpha=0.4, color='gray', linewidths=0)
            ax.scatter(tt.loc[sig & (tt['delta_mean'] > 0), 'delta_mean'],
                       -np.log10(tt.loc[sig & (tt['delta_mean'] > 0), 'p'].clip(1e-10)),
                       s=18, alpha=0.8, color='#e41a1c', linewidths=0, label='↑ 유의')
            ax.scatter(tt.loc[sig & (tt['delta_mean'] < 0), 'delta_mean'],
                       -np.log10(tt.loc[sig & (tt['delta_mean'] < 0), 'p'].clip(1e-10)),
                       s=18, alpha=0.8, color='#377eb8', linewidths=0, label='↓ 유의')

            ax.axvline(0, color='k', linewidth=0.6, linestyle='--')
            ax.set_xlabel('Δ mean weight (crisis − pre)', fontsize=9)
            ax.set_ylabel('−log₁₀(p)', fontsize=9)

            n_up   = int((sig & (tt['delta_mean'] > 0)).sum())
            n_down = int((sig & (tt['delta_mean'] < 0)).sum())
            ax.set_title(f'{est_name} / {crisis_name}  (↑{n_up} ↓{n_down})', fontsize=10)
            if n_up + n_down > 0:
                ax.legend(fontsize=8, markerscale=1.2)

    fig.suptitle('Volcano Plot — 자산별 비중 변화 (pre → crisis)\n'
                 '빨강=위기시 비중 증가, 파랑=감소 (BH FDR 5%)',
                 fontsize=12)
    plt.tight_layout()
    out = FIGURES / 'crisis_test_volcano.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


# ── Figure 3: effective N time series with period shading ─────────────────────
def plot_effn_timeseries():
    crises = list(PERIODS.keys())
    fig, axes = plt.subplots(len(crises), 1, figsize=(14, 4 * len(crises)), sharex=False)

    for ax, crisis_name in zip(axes, crises):
        pre_start, pre_end       = PERIODS[crisis_name]['pre']
        crisis_start, crisis_end = PERIODS[crisis_name]['crisis']
        # 전체 표시 구간
        t_start = pd.Timestamp(pre_start) - pd.DateOffset(months=3)
        t_end   = pd.Timestamp(crisis_end) + pd.DateOffset(months=3)

        for est_name, w_full in weights.items():
            sub = w_full.loc[t_start:t_end]
            eff = sub.apply(effective_n, axis=1)
            ax.plot(eff.index, eff.values, label=est_name,
                    color=EST_COLORS[est_name], linewidth=1.5)

        ax.axvspan(pd.Timestamp(pre_start), pd.Timestamp(pre_end),
                   alpha=0.08, color='green', label='pre-crisis')
        ax.axvspan(pd.Timestamp(crisis_start), pd.Timestamp(crisis_end),
                   alpha=0.12, color='red', label='crisis')
        ax.axvline(pd.Timestamp(crisis_start), color='red', linewidth=1.0, linestyle='--')
        ax.set_ylabel('Effective N')
        ax.set_title(f'{crisis_name} — Effective N 추이')
        ax.legend(fontsize=9)

    fig.suptitle('Effective N: pre-crisis vs crisis 구간 비교', fontsize=12)
    plt.tight_layout()
    out = FIGURES / 'crisis_effn_timeseries.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


# ── Figure 4: weight distribution boxplot ────────────────────────────────────
def plot_weight_dist_boxplot():
    crises = list(PERIODS.keys())
    ests   = list(ESTIMATORS.keys())
    fig, axes = plt.subplots(len(crises), len(ests),
                             figsize=(5 * len(ests), 4 * len(crises)))

    for r, crisis_name in enumerate(crises):
        pre_start, pre_end       = PERIODS[crisis_name]['pre']
        crisis_start, crisis_end = PERIODS[crisis_name]['crisis']
        for c, est_name in enumerate(ests):
            ax = axes[r][c]
            w  = weights[est_name]
            vals_pre    = w.loc[pre_start:pre_end].values.flatten()
            vals_crisis = w.loc[crisis_start:crisis_end].values.flatten()
            # remove exact zeros (assets not in universe that period)
            vals_pre    = vals_pre[vals_pre != 0]
            vals_crisis = vals_crisis[vals_crisis != 0]

            bp = ax.boxplot([vals_pre, vals_crisis],
                            labels=['pre', 'crisis'],
                            patch_artist=True,
                            medianprops=dict(color='black', linewidth=1.5),
                            flierprops=dict(marker='.', markersize=2, alpha=0.3))
            bp['boxes'][0].set_facecolor('#aec6e8')
            bp['boxes'][1].set_facecolor('#f4a582')

            stat, p = mannwhitneyu(vals_pre, vals_crisis, alternative='two-sided')
            p_str = f'p={p:.4f}' if p >= 0.0001 else 'p<0.0001'
            ax.set_title(f'{est_name} / {crisis_name}\n(MW {p_str})', fontsize=10)
            ax.set_ylabel('Weight')

    fig.suptitle('Weight 분포 비교: pre-crisis vs crisis\n(파랑=pre, 주황=crisis)', fontsize=12)
    plt.tight_layout()
    out = FIGURES / 'crisis_weight_boxplot.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


# ── run plots ─────────────────────────────────────────────────────────────────
print('Generating figures...')
plot_summary_heatmap()
plot_volcano_grid()
plot_effn_timeseries()
plot_weight_dist_boxplot()


# ── console summary table ─────────────────────────────────────────────────────
print('\n' + '='*80)
print('SUMMARY TABLE')
print('='*80)
rows = []
for (crisis_name, est_name), r in results.items():
    rows.append({
        'Crisis': crisis_name, 'Estimator': est_name,
        'N_sig (BH5%)': r['n_sig_bh'],
        '%_sig': f"{r['pct_sig']:.1f}%",
        'Up/Down': f"↑{r['n_increase']} ↓{r['n_decrease']}",
        'Perm_p': f"{r['perm_p']:.4f}",
        'EffN_pre': f"{r['effn_pre']:.2f}",
        'EffN_crisis': f"{r['effn_crisis']:.2f}",
        'EffN_p': f"{r['effn_p']:.4f}",
    })
df_summary = pd.DataFrame(rows).set_index(['Crisis', 'Estimator'])
print(df_summary.to_string())

# save summary CSV
csv_out = REPORTS / 'crisis_weight_test_summary.csv'
df_summary.to_csv(csv_out)
print(f'\nSummary saved → {csv_out}')

# save top changed assets per (crisis, estimator)
txt_out = REPORTS / 'crisis_weight_test_topassets.txt'
with open(txt_out, 'w') as f:
    for (crisis_name, est_name), tt in ttest_tables.items():
        sig = tt[tt['bh_reject']].sort_values('d', key=abs, ascending=False)
        f.write(f'\n=== {crisis_name} / {est_name}  (n_sig={len(sig)}) ===\n')
        if len(sig) > 0:
            f.write(sig[['pre_mean', 'crisis_mean', 'delta_mean', 't', 'p', 'd']].round(4).to_string())
        f.write('\n')
print(f'Top-changed assets saved → {txt_out}')

print('\nDone.')
