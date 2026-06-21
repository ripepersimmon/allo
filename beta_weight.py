"""
Beta vs. GMV-Weight Relationship — Estimator-Wise
Market proxy: equal-weighted return of all assets in the estimation window.
GMV: unconstrained analytical solution (w ∝ Σ⁻¹ 1).
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats

from src.data_loader import load_prices_from_parquet, compute_returns, TICKERS
from src.estimators import sample_cov, lw_cov, gerber_cov
from src.market import get_market_proxy

# ── proxy argument (parsed before data load) ─────────────────────────────────
import argparse as _argparse
_parser = _argparse.ArgumentParser(description='Beta vs GMV-weight analysis')
_parser.add_argument('--proxy', choices=['ew', 'spy'], default='ew',
                     help='Market proxy: ew = equal-weighted (default), spy = SPY')
PROXY  = _parser.parse_known_args()[0].proxy
SUFFIX = f'_{PROXY}' if PROXY != 'ew' else ''
del _argparse, _parser

# ── config ────────────────────────────────────────────────────────────────────
WINDOW  = 252
FIGURES = Path('results/figures')
FIGURES.mkdir(parents=True, exist_ok=True)

ESTIMATORS = {'Sample': sample_cov, 'LW': lw_cov, 'Gerber': gerber_cov}
EST_COLORS = {'Sample': '#e41a1c', 'LW': '#377eb8', 'Gerber': '#4daf4a'}

# crisis peak windows (same as bbc_analysis)
CRISES = {
    'GFC':   '2009-03-31',
    'COVID': '2020-04-30',
    'Rates': '2023-01-31',
}

# rolling correlation: full crisis date ranges
CRISIS_RANGES = {
    'GFC':   ('2007-01-01', '2009-06-30'),
    'COVID': ('2019-10-01', '2020-09-30'),
    'Rates': ('2021-07-01', '2023-01-31'),
}

# ── data ──────────────────────────────────────────────────────────────────────
print('Loading prices...')
prices  = load_prices_from_parquet('sp500', tickers=TICKERS,
                                   start='2000-01-01', end='2024-12-31')
returns = compute_returns(prices, method='log')
print(f'Returns: {returns.shape[0]} days × {returns.shape[1]} assets\n')

SPY_RETURNS = None
if PROXY == 'spy':
    from src.market import load_spy_returns
    SPY_RETURNS = load_spy_returns(start='2000-01-01', end='2024-12-31')
    print(f'Market proxy: SPY ({len(SPY_RETURNS)} return observations)')


# ── core functions ────────────────────────────────────────────────────────────

def get_window(end_date: str) -> pd.DataFrame:
    end   = pd.Timestamp(end_date)
    start = end - pd.offsets.BDay(WINDOW)
    return returns.loc[start:end].dropna(axis=1)


def compute_betas(win: pd.DataFrame, spy_returns=None) -> pd.Series:
    """OLS beta of each asset against the configured market proxy."""
    mkt = get_market_proxy(win, PROXY, spy_returns)
    mkt_var = mkt.var()
    betas = {}
    for col in win.columns:
        betas[col] = win[col].cov(mkt) / mkt_var
    return pd.Series(betas)


def gmv_weights(cov: np.ndarray) -> np.ndarray:
    """Unconstrained analytical GMV: w ∝ Σ⁻¹ 1."""
    prec = np.linalg.pinv(cov)
    raw  = prec @ np.ones(cov.shape[0])
    return raw / raw.sum()


# ── Plot 1: scatter grid — beta vs weight, one snapshot per crisis ────────────

def plot_scatter_grid():
    n_crises = len(CRISES)
    n_est    = len(ESTIMATORS)
    fig, axes = plt.subplots(
        n_crises, n_est,
        figsize=(5 * n_est, 4.5 * n_crises),
        sharex=False, sharey=False,
    )

    for r, (crisis_name, crisis_end) in enumerate(CRISES.items()):
        win   = get_window(crisis_end)
        betas = compute_betas(win, SPY_RETURNS)

        for c, (est_name, est_fn) in enumerate(ESTIMATORS.items()):
            ax = axes[r][c]
            cov = est_fn(win)
            w   = pd.Series(gmv_weights(cov), index=win.columns)

            # align betas and weights on common tickers
            common = betas.index.intersection(w.index)
            b, wt  = betas[common].values, w[common].values

            # color by weight sign
            col = np.where(wt >= 0, '#e41a1c', '#377eb8')
            ax.scatter(b, wt, c=col, s=18, alpha=0.7, linewidths=0)
            ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
            ax.axvline(1, color='gray', linewidth=0.6, linestyle=':')

            rho, pval = stats.pearsonr(b, wt)
            ax.text(0.05, 0.95, f'ρ = {rho:.2f}  (p={pval:.3f})',
                    transform=ax.transAxes, fontsize=9,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

            # OLS fit line
            m, intercept, *_ = stats.linregress(b, wt)
            x_fit = np.linspace(b.min(), b.max(), 100)
            ax.plot(x_fit, m * x_fit + intercept, 'k-', linewidth=1.2)

            if r == 0:
                ax.set_title(est_name, fontsize=12, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'{crisis_name}\nGMV weight', fontsize=10)
            if r == n_crises - 1:
                ax.set_xlabel('Market Beta (β)', fontsize=10)

    fig.suptitle(
        'Market Beta vs. GMV Weight  |  EW market proxy  |  Unconstrained\n'
        '● red = long,  ● blue = short,  vertical dotted = β=1',
        fontsize=12
    )
    plt.tight_layout()
    out = FIGURES / f'beta_weight_scatter{SUFFIX}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


# ── Plot 2: rolling beta–weight correlation through each crisis ───────────────

def compute_rolling_corr(crisis_name: str, est_name: str, est_fn) -> pd.Series:
    start, end = CRISIS_RANGES[crisis_name]
    dates = returns.loc[start:end].index
    corrs = {}
    for date in dates:
        win_end   = date
        win_start = win_end - pd.offsets.BDay(WINDOW)
        win = returns.loc[win_start:win_end].dropna(axis=1)
        if win.shape[0] < WINDOW // 2 or win.shape[1] < 5:
            continue
        betas = compute_betas(win, SPY_RETURNS)
        try:
            cov = est_fn(win)
            w   = pd.Series(gmv_weights(cov), index=win.columns)
        except Exception:
            continue
        common = betas.index.intersection(w.index)
        if len(common) < 5:
            continue
        rho, _ = stats.pearsonr(betas[common].values, w[common].values)
        corrs[date] = rho
    return pd.Series(corrs)


def plot_rolling_corr():
    fig, axes = plt.subplots(len(CRISIS_RANGES), 1,
                             figsize=(13, 4 * len(CRISIS_RANGES)), sharex=False)

    for ax, (crisis_name, (c_start, c_end)) in zip(axes, CRISIS_RANGES.items()):
        for est_name, est_fn in ESTIMATORS.items():
            print(f'  rolling corr: {crisis_name} / {est_name}', flush=True)
            s = compute_rolling_corr(crisis_name, est_name, est_fn)
            ax.plot(s.index, s.values,
                    label=est_name, color=EST_COLORS[est_name], linewidth=1.6)

        ax.axhline(0, color='k', linewidth=0.7, linestyle='--')
        ax.set_ylabel('Pearson ρ(β, w)', fontsize=10)
        ax.set_title(f'{crisis_name} — rolling corr(Market Beta, GMV Weight)', fontsize=11)
        ax.set_ylim(-1, 1)
        ax.legend(fontsize=9)

    fig.suptitle(
        'Rolling Pearson ρ between Market Beta and GMV Weight\n'
        f'(estimation window = {WINDOW} trading days, EW market proxy, unconstrained GMV)',
        fontsize=12
    )
    plt.tight_layout()
    out = FIGURES / f'beta_weight_rolling_corr{SUFFIX}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

print('Plot 1: scatter grid (3 crises × 3 estimators)...')
plot_scatter_grid()

print('\nPlot 2: rolling beta–weight correlation (slow, ~few minutes)...')
plot_rolling_corr()

print('\nDone.')
