"""
Asset-level syst_share Tracking: pre-crisis vs crisis
======================================================
For each (crisis × estimator), identifies the top-5 beneficiary and top-5
casualty assets (by Cohen's d on GMV weight) and tracks their syst_share
(= β²σ²_m / total_var) across the pre + crisis window.

Connects the weight-shift evidence (crisis_weight_test.py) to the
variance-decomposition narrative (variance_decomp.py): if assets that gained
weight during a crisis also showed rising syst_share (or vice-versa for
casualties), the two independent report streams corroborate each other.

Market proxy = equal-weighted return of all available assets in the 252-day
rolling window (same as all other OLS scripts in this project).

Outputs:
    results/figures/syst_share_tracking/{crisis}_{estimator}.png  (9 figures)
    results/figures/syst_share_tracking/scatter_summary.png
    reports/asset_syst_share_changes.csv
    reports/asset_syst_share_report.md
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

from src.data_loader import load_prices_from_parquet, compute_returns, TICKERS
from src.estimators import sample_cov, lw_cov, gerber_cov
from src.analysis import rolling_gmv

np.random.seed(42)

# ── config ────────────────────────────────────────────────────────────────────
WINDOW  = 252
TOP_N   = 5
STEP    = 5          # trading days between rolling syst_share samples
FIGURES = Path('results/figures/syst_share_tracking')
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

# ── data ──────────────────────────────────────────────────────────────────────
print('Loading data...')
prices  = load_prices_from_parquet('sp500', tickers=TICKERS,
                                   start='2000-01-01', end='2024-12-31')
returns = compute_returns(prices, method='log')
print(f'Returns: {returns.shape[0]} days × {returns.shape[1]} assets')

# ── rolling GMV weights (long-only, constrained=True) ─────────────────────────
print('\nComputing rolling long-only GMV weights (window=252)...')
weights = {}
for est_name, est_fn in ESTIMATORS.items():
    print(f'  {est_name}', flush=True)
    weights[est_name] = rolling_gmv(returns, est_fn, window=WINDOW, constrained=True)
print()


# ── helpers ───────────────────────────────────────────────────────────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return (b.mean() - a.mean()) / pooled if pooled > 1e-10 else np.nan


def select_top_assets(w_pre: pd.DataFrame, w_crisis: pd.DataFrame,
                      n: int = TOP_N) -> tuple:
    """
    Returns (beneficiaries, casualties, d_series, weight_stats).
    beneficiaries: n tickers with largest positive Cohen's d (gained weight).
    casualties:    n tickers with most negative Cohen's d (lost weight).
    weight_stats:  DataFrame[ticker × {pre_mean, crisis_mean, delta}].
    """
    common = w_pre.columns.intersection(w_crisis.columns)
    d_vals, wstats = {}, {}
    for col in common:
        a = w_pre[col].dropna().values
        b = w_crisis[col].dropna().values
        if len(a) < 5 or len(b) < 5:
            continue
        d = cohens_d(a, b)
        if np.isnan(d):
            continue
        d_vals[col]  = d
        wstats[col]  = dict(pre_weight_mean=a.mean(),
                            crisis_weight_mean=b.mean(),
                            weight_delta=b.mean() - a.mean())
    d_series     = pd.Series(d_vals)
    weight_stats = pd.DataFrame(wstats).T
    beneficiaries = d_series[d_series > 0].nlargest(n).index.tolist()
    casualties    = d_series[d_series < 0].nsmallest(n).index.tolist()
    if len(beneficiaries) < n or len(casualties) < n:
        print(f'    WARNING: fewer than {n} tickers on one side '
              f'(bene={len(beneficiaries)}, casu={len(casualties)})')
    return beneficiaries, casualties, d_series, weight_stats


def compute_syst_share_series(tickers: list, span_start: str,
                               span_end: str, step: int = STEP) -> pd.DataFrame:
    """
    Rolling syst_share for each ticker over [span_start, span_end].
    Window = WINDOW trading days, stepped every `step` trading days.
    Market proxy = equal-weighted return of all available assets in each window.
    Returns DataFrame indexed by date, columns = tickers.
    """
    t0 = pd.Timestamp(span_start)
    t1 = pd.Timestamp(span_end)
    all_dates    = returns.loc[t0:t1].index
    sample_dates = all_dates[::step]

    records = {}
    for date in sample_dates:
        win_start = date - pd.offsets.BDay(WINDOW)
        win       = returns.loc[win_start:date].dropna(axis=1)
        if win.shape[0] < WINDOW // 2:
            records[date] = {tk: np.nan for tk in tickers}
            continue
        mkt     = win.mean(axis=1)
        mkt_var = mkt.var()
        row = {}
        for tk in tickers:
            if tk not in win.columns or mkt_var < 1e-14:
                row[tk] = np.nan
                continue
            r = win[tk]
            total_var = r.var()
            if total_var < 1e-14:
                row[tk] = np.nan
                continue
            beta      = r.cov(mkt) / mkt_var
            syst_var  = beta ** 2 * mkt_var
            row[tk]   = syst_var / total_var
        records[date] = row

    df = pd.DataFrame(records).T
    df.index = pd.to_datetime(df.index)
    return df.reindex(columns=tickers)


# ── Pass 1: select top assets per (crisis, estimator) ────────────────────────
print('Selecting top assets per (crisis × estimator)...')
selected = {}
for crisis_name, segs in PERIODS.items():
    pre_start, pre_end       = segs['pre']
    crisis_start, crisis_end = segs['crisis']
    for est_name, w_full in weights.items():
        w_pre    = w_full.loc[pre_start:pre_end]
        w_crisis = w_full.loc[crisis_start:crisis_end]
        bene, casu, d_series, wstat = select_top_assets(w_pre, w_crisis)
        selected[(crisis_name, est_name)] = {
            'beneficiaries': bene,
            'casualties':    casu,
            'd_series':      d_series,
            'wstat':         wstat,
        }
        print(f'  {crisis_name}/{est_name}: '
              f'bene={bene}  casu={casu}')
print()


# ── Pass 2: compute rolling syst_share per crisis (union of tickers) ──────────
print('Computing rolling syst_share series per crisis...')
syst_share_data = {}
for crisis_name, segs in PERIODS.items():
    pre_start, pre_end       = segs['pre']
    crisis_start, crisis_end = segs['crisis']
    tickers_needed = set()
    for est_name in ESTIMATORS:
        info = selected[(crisis_name, est_name)]
        tickers_needed.update(info['beneficiaries'] + info['casualties'])
    tickers_needed = sorted(tickers_needed)
    print(f'  {crisis_name}: {len(tickers_needed)} unique tickers '
          f'→ {tickers_needed}')
    syst_share_data[crisis_name] = compute_syst_share_series(
        tickers_needed, pre_start, crisis_end
    )
print()


# ── Build summary CSV ─────────────────────────────────────────────────────────
print('Building summary CSV...')
csv_rows = []
for crisis_name, segs in PERIODS.items():
    pre_start, pre_end       = segs['pre']
    crisis_start, crisis_end = segs['crisis']
    ss_df = syst_share_data[crisis_name]

    for est_name in ESTIMATORS:
        info     = selected[(crisis_name, est_name)]
        d_series = info['d_series']
        wstat    = info['wstat']

        for side, tickers in [('beneficiary', info['beneficiaries']),
                               ('casualty',    info['casualties'])]:
            for tk in tickers:
                ws     = wstat.loc[tk] if tk in wstat.index else pd.Series(dtype=float)
                pre_w  = float(ws.get('pre_weight_mean',   np.nan))
                cri_w  = float(ws.get('crisis_weight_mean', np.nan))
                dw     = float(ws.get('weight_delta',       np.nan))
                cd     = float(d_series.get(tk, np.nan))
                if tk in ss_df.columns:
                    col     = ss_df[tk]
                    ss_pre  = float(col.loc[pre_start:pre_end].mean())
                    ss_cri  = float(col.loc[crisis_start:crisis_end].mean())
                    ss_d    = ss_cri - ss_pre
                else:
                    ss_pre = ss_cri = ss_d = np.nan

                csv_rows.append({
                    'crisis':             crisis_name,
                    'estimator':          est_name,
                    'ticker':             tk,
                    'side':               side,
                    'pre_weight_mean':    pre_w,
                    'crisis_weight_mean': cri_w,
                    'weight_delta':       dw,
                    'cohens_d_weight':    cd,
                    'pre_syst_share':     ss_pre,
                    'crisis_syst_share':  ss_cri,
                    'syst_share_delta':   ss_d,
                })

csv_df = pd.DataFrame(csv_rows)
csv_path = REPORTS / 'asset_syst_share_changes.csv'
csv_df.to_csv(csv_path, index=False, float_format='%.6f')
print(f'Saved → {csv_path}  ({len(csv_df)} rows)')
print()


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _palette(cmap_name: str, n: int) -> list:
    cmap = matplotlib.colormaps[cmap_name]
    return [cmap(0.35 + 0.50 * i / max(n - 1, 1)) for i in range(n)]


def _shade_periods(ax, pre_start, pre_end, crisis_start, crisis_end):
    ax.axvspan(pd.Timestamp(pre_start),    pd.Timestamp(pre_end),
               alpha=0.07, color='green',  zorder=0)
    ax.axvspan(pd.Timestamp(crisis_start), pd.Timestamp(crisis_end),
               alpha=0.12, color='red',    zorder=0)
    ax.axvline(pd.Timestamp(crisis_start), color='#d62728',
               linewidth=1.2, linestyle='--', zorder=1)


# ── Figure 1: 9 per-cell time-series figures ──────────────────────────────────
print('Generating per-cell time-series figures...')

for crisis_name, segs in PERIODS.items():
    pre_start, pre_end       = segs['pre']
    crisis_start, crisis_end = segs['crisis']
    ss_df = syst_share_data[crisis_name]

    for est_name in ESTIMATORS:
        info = selected[(crisis_name, est_name)]
        bene = info['beneficiaries']
        casu = info['casualties']

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True
        )
        fig.suptitle(
            f'syst_share 추이 — {est_name} / {crisis_name}\n'
            f'(상단: weight 수혜 상위 {TOP_N}, 하단: 피해 상위 {TOP_N})',
            fontsize=12
        )

        for ax, tickers, label, cmap_name, patch_color in [
            (ax_top, bene, f'수혜 (beneficiary, top-{TOP_N})',
             'Greens', '#2ca02c'),
            (ax_bot, casu, f'피해 (casualty, top-{TOP_N})',
             'Reds',   '#d62728'),
        ]:
            _shade_periods(ax, pre_start, pre_end, crisis_start, crisis_end)

            if tickers:
                colors = _palette(cmap_name, len(tickers))
                for tk, color in zip(tickers, colors):
                    if tk not in ss_df.columns:
                        continue
                    series = ss_df[tk].dropna()
                    ax.plot(series.index, series.values, label=tk,
                            color=color, linewidth=1.8, zorder=2)

            ax.set_ylabel('syst_share  (β²σ²_m / total_var)', fontsize=9)
            ax.set_title(label, fontsize=10)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8, ncol=max(1, min(len(tickers), 5)), loc='upper left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(),
                     rotation=30, ha='right', fontsize=8)

        xl = (pd.Timestamp(pre_start) - pd.DateOffset(months=1),
              pd.Timestamp(crisis_end) + pd.DateOffset(months=1))
        ax_top.set_xlim(xl)   # ax_bot shares x-axis

        plt.tight_layout()
        out = FIGURES / f'{crisis_name}_{est_name}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved → {out}')

print()


# ── Figure 2: 3×3 scatter summary ────────────────────────────────────────────
print('Generating scatter summary figure...')

crises = list(PERIODS.keys())
ests   = list(ESTIMATORS.keys())
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle(
    'Δsyst_share (crisis − pre mean)  vs  Δweight\n'
    '초록=weight 수혜, 빨강=피해  (top-5 per cell, 9 = 3 crises × 3 estimators)',
    fontsize=12
)

for r, crisis_name in enumerate(crises):
    for c, est_name in enumerate(ests):
        ax   = axes[r][c]
        sub  = csv_df[(csv_df['crisis'] == crisis_name) &
                      (csv_df['estimator'] == est_name)].copy()

        for side, color in [('beneficiary', '#2ca02c'), ('casualty', '#d62728')]:
            part = sub[sub['side'] == side].dropna(
                subset=['syst_share_delta', 'weight_delta']
            )
            if part.empty:
                continue
            ax.scatter(part['syst_share_delta'], part['weight_delta'],
                       color=color, s=55, alpha=0.85, zorder=3)
            for _, row in part.iterrows():
                ax.annotate(
                    row['ticker'],
                    (row['syst_share_delta'], row['weight_delta']),
                    fontsize=6, ha='center', va='bottom',
                    xytext=(0, 3), textcoords='offset points'
                )

        ax.axhline(0, color='gray', linewidth=0.6, linestyle=':')
        ax.axvline(0, color='gray', linewidth=0.6, linestyle=':')
        ax.set_xlabel('Δsyst_share', fontsize=8)
        ax.set_ylabel('Δweight',     fontsize=8)
        ax.set_title(f'{est_name} / {crisis_name}', fontsize=9)
        ax.tick_params(labelsize=7)

plt.tight_layout()
out = FIGURES / 'scatter_summary.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved → {out}\n')


# ── Markdown report ───────────────────────────────────────────────────────────
print('Writing markdown report...')

def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


marquee = ['VZ', 'JNJ', 'LMT']

lines = [
    '# Asset-level syst_share Tracking Report\n',
    '## Overview\n',
    'This report bridges two parallel evidence streams:\n',
    '1. **Weight-shift evidence** (`crisis_weight_test.py`): which assets '
    'gained or lost GMV weight during each crisis.\n',
    '2. **Variance-decomposition evidence** (`variance_decomp.py`): the '
    'aggregate cross-sectional relationship between syst_share and GMV weight.\n\n',
    'For each (crisis × estimator) cell the **top-5 beneficiaries** and '
    '**top-5 casualties** (by Cohen\'s d on weight) are identified, and their '
    '`syst_share = β²σ²_m / total_var` is tracked with a rolling 252-day window '
    '(equal-weighted market proxy, stepped every 5 trading days).\n\n',
    '---\n',
    '## Marquee Asset Patterns\n',
    'JNJ, LMT, and VZ appear across multiple crises as high-signal movers:\n\n',
]

for tk in marquee:
    sub = csv_df[csv_df['ticker'] == tk][
        ['crisis', 'estimator', 'side', 'cohens_d_weight',
         'weight_delta', 'pre_syst_share', 'crisis_syst_share',
         'syst_share_delta']
    ].round(4)
    if sub.empty:
        continue
    lines.append(f'### {tk}\n\n')
    lines.append(_fmt(sub) + '\n\n')

lines.append('---\n\n## Full Results by Crisis\n\n')
for crisis_name in PERIODS:
    sub = csv_df[csv_df['crisis'] == crisis_name][
        ['estimator', 'side', 'ticker', 'cohens_d_weight',
         'weight_delta', 'pre_syst_share', 'crisis_syst_share',
         'syst_share_delta']
    ].round(4)
    lines.append(f'### {crisis_name}\n\n')
    lines.append(_fmt(sub) + '\n\n')

report_path = REPORTS / 'asset_syst_share_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'Saved → {report_path}')

print('\nDone.')
