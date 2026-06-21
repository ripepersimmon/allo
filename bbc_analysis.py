"""
BBC Precision-Matrix Analysis — Estimator-Wise
Kim et al. (2025) Algorithm 1: Bidirectional Block Construction

Runs for GFC / COVID / Rates crisis peaks, compares Sample, LW, Gerber.
Saves figures to results/figures/.
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import load_prices_from_parquet, compute_returns, TICKERS
from src.estimators import sample_cov, lw_cov, gerber_cov, bbc_permutation

# ── config ────────────────────────────────────────────────────────────────────
WINDOW  = 252
FIGURES = Path('results/figures')
FIGURES.mkdir(parents=True, exist_ok=True)

ESTIMATORS = {'Sample': sample_cov, 'LW': lw_cov, 'Gerber': gerber_cov}

CRISES = {
    'GFC':   '2009-03-31',
    'COVID': '2020-04-30',
    'Rates': '2023-01-31',
}

# ── data ──────────────────────────────────────────────────────────────────────
print('Loading prices...')
prices  = load_prices_from_parquet('sp500', tickers=TICKERS,
                                   start='2000-01-01', end='2024-12-31')
returns = compute_returns(prices, method='log')
print(f'Returns: {returns.shape[0]} days × {returns.shape[1]} assets\n')


# ── helpers ───────────────────────────────────────────────────────────────────

def estimation_window(crisis_end: str):
    end = pd.Timestamp(crisis_end)
    start = end - pd.offsets.BDay(WINDOW)
    win = returns.loc[start:end].dropna(axis=1)
    return win


def plot_bbc_panel(win: pd.DataFrame, crisis_name: str):
    """3 rows (estimators) × 3 cols: raw precision | BBC-permuted | GMV weights."""
    tickers = win.columns.tolist()
    n       = len(tickers)

    fig, axes = plt.subplots(
        3, 3, figsize=(18, 15),
        gridspec_kw={'width_ratios': [4, 4, 1.5]}
    )

    for row, (est_name, est_fn) in enumerate(ESTIMATORS.items()):
        cov  = est_fn(win)
        prec = np.linalg.pinv(cov)
        pi   = bbc_permutation(prec)
        labels_perm = [tickers[i] for i in pi]

        # unconstrained GMV weight ∝ row-sum of precision
        ones  = np.ones(n)
        raw_w = prec @ ones
        w_gmv = raw_w / raw_w.sum()
        w_perm = w_gmv[pi]

        pmax        = np.percentile(np.abs(prec), 99)
        prec_clip   = np.clip(prec, -pmax, pmax)
        prec_perm   = prec_clip[np.ix_(pi, pi)]

        # col 0: raw precision
        ax0 = axes[row][0]
        im0 = ax0.imshow(prec_clip, cmap='RdBu_r', aspect='auto',
                         vmin=-pmax, vmax=pmax)
        ax0.set_title(f'{est_name} — Raw Precision', fontsize=11)
        ax0.set_xticks([]); ax0.set_yticks([])
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # col 1: BBC-permuted precision
        ax1 = axes[row][1]
        im1 = ax1.imshow(prec_perm, cmap='RdBu_r', aspect='auto',
                         vmin=-pmax, vmax=pmax)
        ax1.set_title(f'{est_name} — BBC-Permuted', fontsize=11)
        ax1.set_xticks([]); ax1.set_yticks([])
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # col 2: GMV weight bar (BBC order, top→bottom)
        ax2 = axes[row][2]
        colors = ['#e41a1c' if w >= 0 else '#377eb8' for w in w_perm]
        ax2.barh(range(n), w_perm, color=colors, height=0.8)
        ax2.axvline(0, color='k', linewidth=0.8)
        ax2.set_yticks([])
        ax2.set_xlabel('GMV weight')
        ax2.set_title(f'{est_name} — Weights\n(BBC order)', fontsize=11)
        ax2.invert_yaxis()

    fig.suptitle(f'BBC Precision Analysis — {crisis_name}  (window end: {CRISES[crisis_name]})',
                 fontsize=13)
    plt.tight_layout()
    out = FIGURES / f'bbc_{crisis_name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  saved → {out}')
    plt.close()


def plot_rowsums(crisis_name: str, win: pd.DataFrame):
    """Precision row-sums in BBC order, one subplot per estimator."""
    n   = win.shape[1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (est_name, est_fn) in zip(axes, ESTIMATORS.items()):
        cov      = est_fn(win)
        prec     = np.linalg.pinv(cov)
        pi       = bbc_permutation(prec)
        row_sums = prec.sum(axis=1)[pi]

        colors = ['#e41a1c' if s >= 0 else '#377eb8' for s in row_sums]
        ax.bar(range(n), row_sums, color=colors, width=0.9)
        ax.axhline(0, color='k', linewidth=0.7)
        ax.set_title(est_name, fontsize=11)
        ax.set_xlabel('Stock index (BBC order)')
        ax.set_ylabel('Row-sum of Σ⁻¹  (∝ GMV weight)')
        ax.set_xlim(-0.5, n - 0.5)

    fig.suptitle(f'Precision Row-Sums in BBC Order — {crisis_name}', fontsize=12)
    plt.tight_layout()
    out = FIGURES / f'bbc_rowsums_{crisis_name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  saved → {out}')
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

for crisis_name, crisis_end in CRISES.items():
    print(f'[{crisis_name}]  window ending {crisis_end}')
    win = estimation_window(crisis_end)
    print(f'  {win.shape[0]} days, {win.shape[1]} assets')
    plot_bbc_panel(win, crisis_name)
    plot_rowsums(crisis_name, win)
    print()

print('Done.')
