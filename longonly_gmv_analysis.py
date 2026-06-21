"""
Long-Only Global Minimum Variance Portfolio — Comprehensive Experiment
=====================================================================
Three covariance estimators (Sample, LW, Gerber) × three crisis periods
(GFC, COVID, Rates) under long-only constraints (w >= 0, sum(w) = 1).

Outputs (all saved to results/figures/ and reports/):
  1. Rolling weight area charts per crisis × estimator
  2. Effective-N time-series with pre/crisis shading
  3. Realised portfolio variance time-series
  4. Turnover comparison bar-charts
  5. Weight-concentration heatmaps (top-N assets)
  6. Pre-crisis vs crisis boxplots
  7. Volcano plots (per-asset weight shift)
  8. Summary heatmap
  9. Constrained vs unconstrained Effective-N overlay
 10. Top-weight asset evolution line charts
 11. CSV + text summaries in reports/
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
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind

from src.data_loader import load_prices_from_parquet, compute_returns, TICKERS
from src.estimators import sample_cov, lw_cov, gerber_cov
from src.portfolio import gmv_long_only, gmv_unconstrained, effective_n, turnover
from src.analysis import rolling_gmv, CRISIS_PERIODS

np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW  = 252
FIGURES = Path('results/figures/longonly')
REPORTS = Path('reports')
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

ESTIMATORS = {'Sample': sample_cov, 'LW': lw_cov, 'Gerber': gerber_cov}
EST_COLORS = {'Sample': '#e41a1c', 'LW': '#377eb8', 'Gerber': '#4daf4a'}

PERIODS = {
    'GFC':   {'pre': ('2005-01-01', '2006-12-31'), 'crisis': ('2007-01-01', '2009-06-30')},
    'COVID': {'pre': ('2018-01-01', '2019-09-30'), 'crisis': ('2019-10-01', '2020-09-30')},
    'Rates': {'pre': ('2019-07-01', '2021-06-30'), 'crisis': ('2021-07-01', '2023-01-31')},
}

# ── Data ─────────────────────────────────────────────────────────────────────
print('Loading data...')
prices  = load_prices_from_parquet('sp500', tickers=TICKERS,
                                   start='2000-01-01', end='2024-12-31')
returns = compute_returns(prices, method='log')
print(f'Returns: {returns.shape[0]} days × {returns.shape[1]} assets\n')

# ── Rolling GMV (long-only) ───────────────────────────────────────────────────
print('Computing rolling long-only GMV weights (window=252)...')
weights_lo = {}
for est_name, est_fn in ESTIMATORS.items():
    print(f'  {est_name}', flush=True)
    weights_lo[est_name] = rolling_gmv(returns, est_fn, window=WINDOW, constrained=True)
print()

# ── Rolling GMV (unconstrained) for comparison ────────────────────────────────
print('Computing rolling unconstrained GMV weights (window=252)...')
weights_unc = {}
for est_name, est_fn in ESTIMATORS.items():
    print(f'  {est_name}', flush=True)
    weights_unc[est_name] = rolling_gmv(returns, est_fn, window=WINDOW, constrained=False)
print()


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: BH correction
# ═══════════════════════════════════════════════════════════════════════════════
def bh_correction(pvals, alpha=0.05):
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    thresholds = ranks / n * alpha
    reject = pvals <= thresholds
    if reject.any():
        max_k = ranks[reject].max()
        reject = ranks <= max_k
    return reject


def per_asset_ttest(w_pre, w_crisis):
    common = w_pre.columns.intersection(w_crisis.columns)
    records = []
    for col in common:
        a = w_pre[col].values
        b = w_crisis[col].values
        t, p = ttest_ind(a, b, equal_var=False)
        pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
        d = (b.mean() - a.mean()) / pooled_std if pooled_std > 0 else 0.0
        records.append({'ticker': col, 't': t, 'p': p, 'd': d,
                        'pre_mean': a.mean(), 'crisis_mean': b.mean(),
                        'delta_mean': b.mean() - a.mean()})
    df = pd.DataFrame(records).set_index('ticker')
    df['bh_reject'] = bh_correction(df['p'].values)
    return df


def permutation_test(w_pre, w_crisis, n_perm=2000):
    common = w_pre.columns.intersection(w_crisis.columns)
    a = w_pre[common].values
    b = w_crisis[common].values
    pooled = np.vstack([a, b])
    n_a = len(a)
    observed = np.sum((a.mean(0) - b.mean(0)) ** 2)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        idx = np.random.permutation(len(pooled))
        null[i] = np.sum((pooled[idx[:n_a]].mean(0) - pooled[idx[n_a:]].mean(0)) ** 2)
    return float(observed), float((null >= observed).mean())


def effn_test(w_pre, w_crisis):
    ep = w_pre.apply(effective_n, axis=1).values
    ec = w_crisis.apply(effective_n, axis=1).values
    stat, p = mannwhitneyu(ep, ec, alternative='two-sided')
    return {'pre_mean': ep.mean(), 'pre_std': ep.std(),
            'crisis_mean': ec.mean(), 'crisis_std': ec.std(), 'U': stat, 'p': p}


def realised_variance(weights_df, ret_df):
    """Daily portfolio variance = (w . r)^2 summed per day."""
    aligned_r = ret_df.reindex(columns=weights_df.columns, fill_value=0)
    port_ret = (weights_df * aligned_r.reindex(weights_df.index)).sum(axis=1)
    return port_ret ** 2


def portfolio_return(weights_df, ret_df):
    aligned_r = ret_df.reindex(columns=weights_df.columns, fill_value=0)
    return (weights_df * aligned_r.reindex(weights_df.index)).sum(axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Run all statistical tests
# ═══════════════════════════════════════════════════════════════════════════════
results = {}
ttest_tables = {}

for crisis_name, segs in PERIODS.items():
    pre_start, pre_end = segs['pre']
    crisis_start, crisis_end = segs['crisis']
    for est_name, w_full in weights_lo.items():
        key = (crisis_name, est_name)
        print(f'Testing {crisis_name} / {est_name}...', flush=True)
        w_pre    = w_full.loc[pre_start:pre_end]
        w_crisis = w_full.loc[crisis_start:crisis_end]

        tt = per_asset_ttest(w_pre, w_crisis)
        ttest_tables[key] = tt
        obs, p_perm = permutation_test(w_pre, w_crisis, n_perm=2000)
        en = effn_test(w_pre, w_crisis)

        # turnover stats
        sub = w_full.loc[crisis_start:crisis_end]
        to_vals = [turnover(sub.iloc[i-1].values, sub.iloc[i].values)
                   for i in range(1, len(sub))]
        sub_pre = w_full.loc[pre_start:pre_end]
        to_pre = [turnover(sub_pre.iloc[i-1].values, sub_pre.iloc[i].values)
                  for i in range(1, len(sub_pre))]

        # max single weight
        max_w_pre    = sub_pre.max().max()
        max_w_crisis = sub.max().max()

        # portfolio variance
        rv_pre    = realised_variance(sub_pre, returns)
        rv_crisis = realised_variance(sub, returns)

        results[key] = {
            'n_assets': len(tt),
            'n_sig_bh': int(tt['bh_reject'].sum()),
            'pct_sig': tt['bh_reject'].mean() * 100,
            'n_increase': int((tt.loc[tt['bh_reject'], 'delta_mean'] > 0).sum()),
            'n_decrease': int((tt.loc[tt['bh_reject'], 'delta_mean'] < 0).sum()),
            'perm_stat': obs, 'perm_p': p_perm,
            'effn_pre': en['pre_mean'], 'effn_crisis': en['crisis_mean'],
            'effn_delta': en['crisis_mean'] - en['pre_mean'],
            'effn_p': en['p'],
            'turnover_pre': np.mean(to_pre) if to_pre else np.nan,
            'turnover_crisis': np.mean(to_vals) if to_vals else np.nan,
            'max_w_pre': max_w_pre,
            'max_w_crisis': max_w_crisis,
            'rv_pre': rv_pre.mean() * 252,
            'rv_crisis': rv_crisis.mean() * 252,
        }
print()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Rolling weight area charts (long-only)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_rolling_weights():
    for crisis_name, segs in PERIODS.items():
        pre_start = segs['pre'][0]
        crisis_end = segs['crisis'][1]
        # show slightly wider window
        t0 = (pd.Timestamp(pre_start) - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
        t1 = (pd.Timestamp(crisis_end) + pd.DateOffset(months=3)).strftime('%Y-%m-%d')

        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        for ax, (est_name, w_full) in zip(axes, weights_lo.items()):
            sub = w_full.loc[t0:t1]
            # pick top-15 assets by mean weight over the window
            top15 = sub.mean().nlargest(15).index
            rest  = sub.drop(columns=top15).sum(axis=1)
            plot_df = sub[top15].copy()
            plot_df['Other'] = rest
            plot_df.plot.area(ax=ax, stacked=True, legend=False, linewidth=0, alpha=0.85)

            ax.axvspan(pd.Timestamp(segs['pre'][0]), pd.Timestamp(segs['pre'][1]),
                       alpha=0.07, color='green', label='pre-crisis')
            ax.axvspan(pd.Timestamp(segs['crisis'][0]), pd.Timestamp(segs['crisis'][1]),
                       alpha=0.10, color='red', label='crisis')
            ax.axvline(pd.Timestamp(segs['crisis'][0]), color='red', lw=1.2, ls='--')
            ax.set_title(f'{est_name} — Long-Only GMV 비중 추이', fontsize=11)
            ax.set_ylabel('Weight')
            ax.set_ylim(0, 1)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles[:16], labels[:16], loc='upper right',
                   bbox_to_anchor=(1.13, 0.98), fontsize=7)
        fig.suptitle(f'Long-Only GMV 비중 — {crisis_name} 위기', fontsize=14)
        plt.tight_layout()
        out = FIGURES / f'lo_weights_{crisis_name}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'saved → {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Effective N time-series
# ═══════════════════════════════════════════════════════════════════════════════
def fig_effn_timeseries():
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=False)
    for ax, (crisis_name, segs) in zip(axes, PERIODS.items()):
        t0 = (pd.Timestamp(segs['pre'][0]) - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        t1 = (pd.Timestamp(segs['crisis'][1]) + pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        for est_name, w_full in weights_lo.items():
            sub = w_full.loc[t0:t1]
            eff = sub.apply(effective_n, axis=1)
            ax.plot(eff.index, eff.values, label=est_name,
                    color=EST_COLORS[est_name], linewidth=1.8)
        ax.axvspan(pd.Timestamp(segs['pre'][0]), pd.Timestamp(segs['pre'][1]),
                   alpha=0.08, color='green', label='pre-crisis')
        ax.axvspan(pd.Timestamp(segs['crisis'][0]), pd.Timestamp(segs['crisis'][1]),
                   alpha=0.12, color='red', label='crisis')
        ax.axvline(pd.Timestamp(segs['crisis'][0]), color='red', lw=1.0, ls='--')
        ax.set_ylabel('Effective N  (1/HHI)')
        ax.set_title(f'{crisis_name} — Effective N 추이 (Long-Only GMV)')
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.suptitle('Effective N: pre-crisis vs crisis (Long-Only)', fontsize=13)
    plt.tight_layout()
    out = FIGURES / 'lo_effn_timeseries.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved → {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Realised portfolio variance
# ═══════════════════════════════════════════════════════════════════════════════
def fig_realised_variance():
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=False)
    for ax, (crisis_name, segs) in zip(axes, PERIODS.items()):
        t0 = (pd.Timestamp(segs['pre'][0]) - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        t1 = (pd.Timestamp(segs['crisis'][1]) + pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        for est_name, w_full in weights_lo.items():
            sub = w_full.loc[t0:t1]
            rv  = realised_variance(sub, returns).rolling(21).mean()  # 1-month smooth
            ax.plot(rv.index, rv.values * 1e4, label=est_name,
                    color=EST_COLORS[est_name], linewidth=1.6)
        ax.axvspan(pd.Timestamp(segs['crisis'][0]), pd.Timestamp(segs['crisis'][1]),
                   alpha=0.12, color='red', label='crisis')
        ax.axvline(pd.Timestamp(segs['crisis'][0]), color='red', lw=1.0, ls='--')
        ax.set_ylabel('실현분산 (×10⁻⁴, 21일 이동평균)')
        ax.set_title(f'{crisis_name} — 포트폴리오 실현분산 (Long-Only GMV)')
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.suptitle('포트폴리오 실현분산: 위기 전후 (Long-Only GMV)', fontsize=13)
    plt.tight_layout()
    out = FIGURES / 'lo_realised_variance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved → {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Turnover bar chart + line overlay
# ═══════════════════════════════════════════════════════════════════════════════
def fig_turnover():
    crises = list(PERIODS.keys())
    ests   = list(ESTIMATORS.keys())
    x = np.arange(len(crises))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, phase in enumerate(['pre', 'crisis']):
        ax = axes[ax_idx]
        for i, est_name in enumerate(ests):
            vals = []
            for crisis_name in crises:
                key = (crisis_name, est_name)
                vals.append(results[key][f'turnover_{phase}'] * 100)  # ×100 for bps-like
            ax.bar(x + i * width, vals, width,
                   label=est_name, color=EST_COLORS[est_name], alpha=0.85)
        ax.set_xticks(x + width)
        ax.set_xticklabels(crises, fontsize=11)
        ax.set_ylabel('일평균 Turnover (%)')
        ax.set_title(f'{"Pre-crisis" if phase == "pre" else "Crisis"} 구간 평균 Turnover')
        ax.legend(fontsize=9)

    fig.suptitle('Long-Only GMV Turnover 비교 (Pre-crisis vs Crisis)', fontsize=13)
    plt.tight_layout()
    out = FIGURES / 'lo_turnover_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved → {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Top-asset weight evolution (top 10 by crisis mean)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_top_asset_evolution():
    for crisis_name, segs in PERIODS.items():
        t0 = segs['pre'][0]
        t1 = segs['crisis'][1]
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        for ax, (est_name, w_full) in zip(axes, weights_lo.items()):
            sub = w_full.loc[t0:t1]
            top10 = sub.loc[segs['crisis'][0]:segs['crisis'][1]].mean().nlargest(10).index
            for ticker in top10:
                ax.plot(sub.index, sub[ticker].values * 100, linewidth=1.4, label=ticker)
            ax.axvspan(pd.Timestamp(segs['pre'][0]), pd.Timestamp(segs['pre'][1]),
                       alpha=0.07, color='green')
            ax.axvspan(pd.Timestamp(segs['crisis'][0]), pd.Timestamp(segs['crisis'][1]),
                       alpha=0.10, color='red')
            ax.axvline(pd.Timestamp(segs['crisis'][0]), color='red', lw=1.0, ls='--')
            ax.set_ylabel('Weight (%)')
            ax.set_title(f'{est_name} — 위기 구간 상위 10개 자산 비중')
            ax.legend(fontsize=7, ncol=5, loc='upper left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.suptitle(f'{crisis_name} 위기 — 상위 자산 비중 추이 (Long-Only GMV)', fontsize=13)
        plt.tight_layout()
        out = FIGURES / f'lo_top_assets_{crisis_name}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'saved → {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: Pre-crisis vs Crisis weight boxplot
# ═══════════════════════════════════════════════════════════════════════════════
def fig_weight_boxplot():
    crises = list(PERIODS.keys())
    ests   = list(ESTIMATORS.keys())
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for r, crisis_name in enumerate(crises):
        segs = PERIODS[crisis_name]
        for c, est_name in enumerate(ests):
            ax = axes[r][c]
            w  = weights_lo[est_name]
            vp = w.loc[segs['pre'][0]:segs['pre'][1]].values.flatten()
            vc = w.loc[segs['crisis'][0]:segs['crisis'][1]].values.flatten()
            vp = vp[vp > 0.001]
            vc = vc[vc > 0.001]
            bp = ax.boxplot([vp * 100, vc * 100],
                            labels=['pre', 'crisis'],
                            patch_artist=True,
                            medianprops=dict(color='black', lw=1.5),
                            flierprops=dict(marker='.', ms=2, alpha=0.3))
            bp['boxes'][0].set_facecolor('#aec6e8')
            bp['boxes'][1].set_facecolor('#f4a582')
            _, p = mannwhitneyu(vp, vc, alternative='two-sided')
            p_str = f'p={p:.4f}' if p >= 0.0001 else 'p<0.0001'
            ax.set_title(f'{est_name} / {crisis_name}\n(MW {p_str})', fontsize=9)
            ax.set_ylabel('Weight (%)')
    fig.suptitle('비중 분포 비교: pre-crisis vs crisis\n(파랑=pre, 주황=crisis, Long-Only GMV)',
                 fontsize=12)
    plt.tight_layout()
    out = FIGURES / 'lo_weight_boxplot.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved → {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7: Volcano plot
# ═══════════════════════════════════════════════════════════════════════════════
def fig_volcano():
    crises = list(PERIODS.keys())
    ests   = list(ESTIMATORS.keys())
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for r, crisis_name in enumerate(crises):
        for c, est_name in enumerate(ests):
            ax = axes[r][c]
            tt  = ttest_tables[(crisis_name, est_name)]
            sig = tt['bh_reject']
            ax.scatter(tt.loc[~sig, 'delta_mean'] * 100,
                       -np.log10(tt.loc[~sig, 'p'].clip(1e-12)),
                       s=12, alpha=0.35, color='gray', linewidths=0)
            up = sig & (tt['delta_mean'] > 0)
            dn = sig & (tt['delta_mean'] < 0)
            ax.scatter(tt.loc[up, 'delta_mean'] * 100,
                       -np.log10(tt.loc[up, 'p'].clip(1e-12)),
                       s=18, alpha=0.8, color='#e41a1c', linewidths=0, label=f'↑{up.sum()}')
            ax.scatter(tt.loc[dn, 'delta_mean'] * 100,
                       -np.log10(tt.loc[dn, 'p'].clip(1e-12)),
                       s=18, alpha=0.8, color='#377eb8', linewidths=0, label=f'↓{dn.sum()}')
            # label top movers
            for _, row in tt[sig].nlargest(3, 'd').iterrows():
                ax.annotate(row.name,
                            xy=(row['delta_mean'] * 100, -np.log10(max(row['p'], 1e-12))),
                            fontsize=6, ha='center', va='bottom')
            ax.axvline(0, color='k', lw=0.6, ls='--')
            ax.set_xlabel('Δ 비중 (crisis−pre, %pt)', fontsize=8)
            ax.set_ylabel('−log₁₀(p)', fontsize=8)
            ax.set_title(f'{est_name} / {crisis_name}', fontsize=9)
            ax.legend(fontsize=7)
    fig.suptitle('Volcano Plot — 자산별 비중 변화 (Long-Only GMV, BH FDR 5%)\n'
                 '빨강=위기시 비중 증가, 파랑=감소', fontsize=12)
    plt.tight_layout()
    out = FIGURES / 'lo_volcano.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved → {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8: Summary heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def fig_summary_heatmap():
    crises = list(PERIODS.keys())
    ests   = list(ESTIMATORS.keys())
    metrics = {
        '유의 자산\n비율 (%)': 'pct_sig',
        'Perm\np-value': 'perm_p',
        'EffN\n(pre)': 'effn_pre',
        'EffN\n(crisis)': 'effn_crisis',
        'EffN\n변화': 'effn_delta',
        'Max 비중\n(crisis)': 'max_w_crisis',
        'Turnover\n(crisis, %)': 'turnover_crisis',
        'Ann. Var\n(crisis, ×10⁻⁴)': 'rv_crisis',
    }

    def scale(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-12)

    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 4))
    for ax, (title, key) in zip(axes, metrics.items()):
        mat = np.array([[results[(c, e)][key] for e in ests] for c in crises])
        multiplier = 100 if key in ('turnover_crisis', 'turnover_pre') else (
                     1e4 if key == 'rv_crisis' else 1)
        mat_disp = mat * multiplier
        im = ax.imshow(scale(mat), cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(ests))); ax.set_xticklabels(ests, fontsize=8)
        ax.set_yticks(range(len(crises))); ax.set_yticklabels(crises, fontsize=8)
        for i in range(len(crises)):
            for j in range(len(ests)):
                v = mat_disp[i, j]
                txt = f'{v:.1f}' if abs(v) >= 1 else f'{v:.3f}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=7, color='black')
        ax.set_title(title, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Long-Only GMV 종합 요약 (행=위기, 열=추정량)', fontsize=12)
    plt.tight_layout()
    out = FIGURES / 'lo_summary_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved → {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 9: Long-only vs Unconstrained Effective N overlay
# ═══════════════════════════════════════════════════════════════════════════════
def fig_constrained_vs_unconstrained():
    for crisis_name, segs in PERIODS.items():
        t0 = (pd.Timestamp(segs['pre'][0]) - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        t1 = (pd.Timestamp(segs['crisis'][1]) + pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
        for ax, est_name in zip(axes, ESTIMATORS.keys()):
            sub_lo  = weights_lo[est_name].loc[t0:t1]
            sub_unc = weights_unc[est_name].loc[t0:t1]
            en_lo   = sub_lo.apply(effective_n, axis=1)
            en_unc  = sub_unc.apply(effective_n, axis=1)
            ax.plot(en_lo.index, en_lo.values, label='Long-Only',
                    color=EST_COLORS[est_name], linewidth=1.8)
            ax.plot(en_unc.index, en_unc.values, label='Unconstrained',
                    color=EST_COLORS[est_name], linewidth=1.4, ls='--', alpha=0.7)
            ax.axvspan(pd.Timestamp(segs['pre'][0]), pd.Timestamp(segs['pre'][1]),
                       alpha=0.07, color='green')
            ax.axvspan(pd.Timestamp(segs['crisis'][0]), pd.Timestamp(segs['crisis'][1]),
                       alpha=0.10, color='red')
            ax.axvline(pd.Timestamp(segs['crisis'][0]), color='red', lw=1.0, ls='--')
            ax.set_ylabel('Effective N')
            ax.set_title(f'{est_name}')
            ax.legend(fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.suptitle(f'{crisis_name} — Long-Only vs Unconstrained GMV: Effective N 비교',
                     fontsize=13)
        plt.tight_layout()
        out = FIGURES / f'lo_vs_unc_effn_{crisis_name}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'saved → {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 10: Cumulative portfolio return
# ═══════════════════════════════════════════════════════════════════════════════
def fig_cumulative_return():
    for crisis_name, segs in PERIODS.items():
        t0 = (pd.Timestamp(segs['pre'][0]) - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        t1 = (pd.Timestamp(segs['crisis'][1]) + pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        fig, ax = plt.subplots(figsize=(14, 5))
        for est_name, w_full in weights_lo.items():
            sub = w_full.loc[t0:t1]
            pr  = portfolio_return(sub, returns)
            cum = (1 + pr).cumprod()
            cum = cum / cum.iloc[0]  # normalise to 1
            ax.plot(cum.index, cum.values, label=est_name,
                    color=EST_COLORS[est_name], linewidth=1.8)
        # equal-weight baseline
        eq_ret = returns.loc[t0:t1].mean(axis=1)
        eq_cum = (1 + eq_ret).cumprod()
        eq_cum = eq_cum / eq_cum.iloc[0]
        ax.plot(eq_cum.index, eq_cum.values, label='Equal-Weight',
                color='gray', linewidth=1.2, ls=':')
        ax.axvspan(pd.Timestamp(segs['pre'][0]), pd.Timestamp(segs['pre'][1]),
                   alpha=0.07, color='green', label='pre-crisis')
        ax.axvspan(pd.Timestamp(segs['crisis'][0]), pd.Timestamp(segs['crisis'][1]),
                   alpha=0.10, color='red', label='crisis')
        ax.axvline(pd.Timestamp(segs['crisis'][0]), color='red', lw=1.0, ls='--')
        ax.set_ylabel('누적 수익률 (정규화)')
        ax.set_title(f'{crisis_name} — Long-Only GMV 누적 수익률')
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.tight_layout()
        out = FIGURES / f'lo_cumret_{crisis_name}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'saved → {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Run all figures
# ═══════════════════════════════════════════════════════════════════════════════
print('Generating figures...')
fig_rolling_weights()
fig_effn_timeseries()
fig_realised_variance()
fig_turnover()
fig_top_asset_evolution()
fig_weight_boxplot()
fig_volcano()
fig_summary_heatmap()
fig_constrained_vs_unconstrained()
fig_cumulative_return()
print()


# ═══════════════════════════════════════════════════════════════════════════════
# Save numerical summaries
# ═══════════════════════════════════════════════════════════════════════════════
rows = []
for (crisis_name, est_name), r in results.items():
    rows.append({
        '위기': crisis_name, '추정량': est_name,
        '유의자산(BH5%)': r['n_sig_bh'],
        '유의비율(%)': round(r['pct_sig'], 1),
        '↑/↓': f"↑{r['n_increase']} ↓{r['n_decrease']}",
        'Perm_p': round(r['perm_p'], 4),
        'EffN_pre': round(r['effn_pre'], 2),
        'EffN_crisis': round(r['effn_crisis'], 2),
        'EffN_delta': round(r['effn_delta'], 2),
        'EffN_p': f"{r['effn_p']:.2e}",
        'Turnover_pre(%)': round(r['turnover_pre'] * 100, 4),
        'Turnover_crisis(%)': round(r['turnover_crisis'] * 100, 4),
        'MaxW_pre': round(r['max_w_pre'], 4),
        'MaxW_crisis': round(r['max_w_crisis'], 4),
        'AnnVar_pre(×1e4)': round(r['rv_pre'] * 1e4, 4),
        'AnnVar_crisis(×1e4)': round(r['rv_crisis'] * 1e4, 4),
    })
df_summary = pd.DataFrame(rows).set_index(['위기', '추정량'])
csv_out = REPORTS / 'longonly_gmv_summary.csv'
df_summary.to_csv(csv_out)
print(f'Summary saved → {csv_out}')
print(df_summary.to_string())

# Top-changed assets
txt_out = REPORTS / 'longonly_gmv_topassets.txt'
with open(txt_out, 'w') as f:
    for (crisis_name, est_name), tt in ttest_tables.items():
        sig = tt[tt['bh_reject']].sort_values('d', key=abs, ascending=False)
        f.write(f'\n=== {crisis_name} / {est_name}  (n_sig={len(sig)}) ===\n')
        if len(sig) > 0:
            f.write(sig[['pre_mean', 'crisis_mean', 'delta_mean', 't', 'p', 'd']].round(4).to_string())
        f.write('\n')
print(f'Top assets saved → {txt_out}')
print('\nDone.')
