"""
Decay-Form Intervention OLS Analysis
=====================================
Per-asset GMV weight panel regressed on asset characteristics (market beta,
average correlation, liquidity, momentum) plus a geometric-decay crisis
intervention term (Box-Tiao transfer function: effect = omega * delta^(t - T0)).

Usage (from repo root, venv active):
    python intervention_analysis.py

Outputs:
    reports/intervention_ols_{Sample,LW,Gerber}.txt
    reports/intervention_coefficients.csv
    reports/intervention_delta_selection.csv
    results/figures/intervention_coef_comparison.png
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.data_loader import (
    load_prices_from_parquet, compute_returns, load_dollar_volume, TICKERS,
)
from src.estimators import sample_cov, lw_cov, gerber_cov
from src.portfolio import gmv_long_only
from src.analysis import CRISIS_PERIODS
from src.market import get_market_proxy

# ── config ────────────────────────────────────────────────────────────────────
WINDOW   = 252
LEAD_IN  = 60            # trading days before each crisis onset to include
FIGURES  = Path('results/figures')
REPORTS  = Path('reports')
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

ESTIMATORS = {'Sample': sample_cov, 'LW': lw_cov, 'Gerber': gerber_cov}
EST_COLORS = {'Sample': '#e41a1c', 'LW': '#377eb8', 'Gerber': '#4daf4a'}

import argparse as _argparse
_parser = _argparse.ArgumentParser(description='Intervention OLS analysis')
_parser.add_argument('--proxy', choices=['ew', 'spy'], default='ew')
PROXY  = _parser.parse_known_args()[0].proxy
SUFFIX = f'_{PROXY}' if PROXY != 'ew' else ''
del _argparse, _parser

# delta grid by half-life in trading days: delta = 0.5**(1/halflife)
HALFLIFE_GRID = [5, 10, 21, 42, 63]
DELTA_GRID    = [round(0.5 ** (1 / hl), 4) for hl in HALFLIFE_GRID]

# ── data ──────────────────────────────────────────────────────────────────────
print('Loading prices and dollar volume...')
prices  = load_prices_from_parquet('sp500', tickers=TICKERS,
                                   start='2000-01-01', end='2024-12-31')
returns = compute_returns(prices, method='log')
dvol    = load_dollar_volume('sp500', tickers=TICKERS,
                             start='2000-01-01', end='2024-12-31')
print(f'Returns: {returns.shape[0]} days × {returns.shape[1]} assets')
print(f'Dollar volume: {dvol.shape[0]} days × {dvol.shape[1]} assets\n')

SPY_RETURNS = None
if PROXY == 'spy':
    from src.market import load_spy_returns
    SPY_RETURNS = load_spy_returns(start='2000-01-01', end='2024-12-31')
    print(f'Market proxy: SPY ({len(SPY_RETURNS)} return observations)')

# precompute position map once
POS_MAP = {d: k for k, d in enumerate(returns.index)}


# ── estimation dates (crisis windows + lead-in) ───────────────────────────────

def get_estimation_dates() -> pd.Index:
    """Union of returns dates within [T0 - LEAD_IN bdays, T1] for each crisis."""
    keep = pd.Index([], dtype='datetime64[ns]')
    for t0_str, t1_str in CRISIS_PERIODS.values():
        t0 = pd.Timestamp(t0_str) - pd.offsets.BDay(LEAD_IN)
        t1 = pd.Timestamp(t1_str)
        mask = (returns.index >= t0) & (returns.index <= t1)
        keep = keep.union(returns.index[mask])
    return keep.sort_values()

EST_DATES = get_estimation_dates()
print(f'Estimation dates: {len(EST_DATES)} '
      f'(crisis windows + {LEAD_IN}d lead-in per crisis)\n')


# ── characteristic builders ───────────────────────────────────────────────────

def compute_betas(win: pd.DataFrame, spy_returns=None) -> pd.Series:
    """OLS beta vs configured market proxy."""
    mkt = get_market_proxy(win, PROXY, spy_returns)
    mkt_var = mkt.var()
    return pd.Series(
        {col: win[col].cov(mkt) / mkt_var for col in win.columns}
    )


def compute_avg_corr(win: pd.DataFrame) -> pd.Series:
    """Mean pairwise correlation of each asset with the rest of the universe."""
    C = win.corr()
    n = len(C)
    return (C.sum(axis=1) - 1) / (n - 1)


def compute_liquidity(win: pd.DataFrame) -> pd.Series:
    """log(1 + mean daily dollar volume) over the window."""
    cols = win.columns
    shared_idx = dvol.index.intersection(win.index)
    if len(shared_idx) == 0:
        return pd.Series(0.0, index=cols)
    available = [c for c in cols if c in dvol.columns]
    dv = dvol.loc[shared_idx, available].mean()
    liq = np.log1p(dv)
    full = pd.Series(np.nan, index=cols)
    full[available] = liq.values
    full.fillna(full.median(), inplace=True)
    return full


def compute_momentum(win: pd.DataFrame) -> pd.Series:
    """Sum of log returns over window = log cumulative return."""
    return win.sum()


# ── combined panel builder (weights + characteristics in one pass) ────────────

def build_panel(est_fn) -> pd.DataFrame:
    """
    For each date t in EST_DATES:
      - slice the same trailing WINDOW returns used by rolling_gmv (lookahead-safe)
      - compute GMV weights via est_fn + gmv_long_only
      - compute asset characteristics from the same window
    Keeps optimizer-assigned zero weights (informative; dropping = selection bias).
    """
    rows = []
    total = len(EST_DATES)
    for idx_num, t in enumerate(EST_DATES):
        if t not in POS_MAP:
            continue
        i = POS_MAP[t]
        if i < WINDOW:
            continue
        win = returns.iloc[i - WINDOW : i].dropna(axis=1)
        if win.shape[1] < 5:
            continue

        # GMV weights (long-only, same as crisis_weight_test.py)
        try:
            cov      = est_fn(win)
            w_active = gmv_long_only(cov)
        except Exception:
            w_active = np.full(win.shape[1], 1.0 / win.shape[1])

        w_map = dict(zip(win.columns, w_active))

        # characteristics
        betas = compute_betas(win, SPY_RETURNS)
        acorr = compute_avg_corr(win)
        liq   = compute_liquidity(win)
        mom   = compute_momentum(win)

        for tk in win.columns:
            rows.append({
                'date':      t,
                'ticker':    tk,
                'weight':    float(w_map.get(tk, 0.0)),
                'beta':      float(betas[tk]),
                'avg_corr':  float(acorr[tk]),
                'liquidity': float(liq[tk]),
                'momentum':  float(mom[tk]),
            })

        if (idx_num + 1) % 100 == 0 or (idx_num + 1) == total:
            print(f'    {idx_num+1}/{total} dates processed', flush=True)

    return pd.DataFrame(rows)


# ── z-score standardisation ───────────────────────────────────────────────────

def standardise(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    for col in ['beta', 'avg_corr', 'liquidity', 'momentum']:
        mu, sd = p[col].mean(), p[col].std()
        p[f'z_{col}'] = (p[col] - mu) / (sd if sd > 0 else 1.0)
    return p


# ── decay intervention regressor ──────────────────────────────────────────────

def crisis_decay_series(dates: pd.Index, delta: float) -> pd.Series:
    """
    CrisisDecay_t = delta^(trading_bars since T0)  for T0 <= t <= T1, else 0.
    Trading-clock distance keeps consistency with the 252-bar window.
    """
    out = pd.Series(0.0, index=dates)
    for t0_str, t1_str in CRISIS_PERIODS.values():
        t0, t1 = pd.Timestamp(t0_str), pd.Timestamp(t1_str)
        win_dates = dates[(dates >= t0) & (dates <= t1)]
        if len(win_dates) == 0:
            continue
        # win_dates are drawn from returns.index, so exact lookup is safe
        r_pos  = np.array([returns.index.get_loc(d) for d in win_dates])
        r_pos0 = returns.index.get_loc(win_dates[0])
        steps  = (r_pos - r_pos0).astype(float)
        assert (steps >= 0).all(), f"Negative decay step for crisis {t0_str}"
        out.loc[win_dates] = delta ** steps
    return out


# ── OLS ───────────────────────────────────────────────────────────────────────

FORMULA = (
    "weight ~ z_beta + z_avg_corr + z_liquidity + z_momentum + decay"
    " + decay:z_beta + decay:z_avg_corr + decay:z_liquidity + decay:z_momentum"
)

def fit_ols(panel: pd.DataFrame):
    return smf.ols(FORMULA, data=panel).fit(
        cov_type='cluster',
        cov_kwds={'groups': panel['date']},
    )

def compute_vif(panel: pd.DataFrame) -> pd.Series:
    X = panel[['z_beta', 'z_avg_corr', 'z_liquidity', 'z_momentum']].dropna()
    X = pd.concat([pd.Series(1.0, index=X.index, name='const'), X], axis=1)
    return pd.Series({
        col: variance_inflation_factor(X.values, i)
        for i, col in enumerate(X.columns)
    })


# ── main estimation loop ──────────────────────────────────────────────────────

results   = {}
delta_log = []
coef_rows = []

for est_name, est_fn in ESTIMATORS.items():
    print(f'[{est_name}] Building panel (weights + characteristics on {len(EST_DATES)} dates)...')
    raw_panel = build_panel(est_fn)
    zero_share = (raw_panel['weight'] == 0).mean()
    print(f'  Panel rows: {len(raw_panel):,}  (zero-weight share: {zero_share:.1%})')

    assert raw_panel[['beta','avg_corr','liquidity','momentum']].isna().sum().sum() == 0
    assert raw_panel['weight'].between(0, 1).all()

    panel = standardise(raw_panel)

    vif = compute_vif(panel)
    print(f'  VIFs: {vif.drop("const").round(2).to_dict()}')

    print(f'  Grid-searching delta...')
    best_ssr, best_delta, best_res, best_panel = np.inf, None, None, None
    unique_dates = pd.Index(panel['date'].unique()).sort_values()

    for hl, delta in zip(HALFLIFE_GRID, DELTA_GRID):
        date_decay = crisis_decay_series(unique_dates, delta)
        panel_d    = panel.copy()
        panel_d['decay'] = panel_d['date'].map(date_decay).fillna(0.0)
        res = fit_ols(panel_d)
        delta_log.append({
            'estimator': est_name, 'halflife': hl, 'delta': delta,
            'r2': res.rsquared, 'ssr': res.ssr,
        })
        if res.ssr < best_ssr:
            best_ssr, best_delta, best_res, best_panel = res.ssr, (hl, delta), res, panel_d

    print(f'  Best delta: {best_delta[1]} (half-life {best_delta[0]} td), '
          f'R²={best_res.rsquared:.4f}')

    out_txt = REPORTS / f'intervention_ols_{est_name}{SUFFIX}.txt'
    with open(out_txt, 'w') as f:
        f.write(f'Estimator: {est_name}\n')
        f.write(f'Selected delta: {best_delta[1]}  (half-life {best_delta[0]} trading days)\n')
        f.write(f'N observations: {int(best_res.nobs)}\n\n')
        f.write(str(best_res.summary()))
        f.write('\n\nVIF:\n')
        f.write(vif.round(3).to_string())
    print(f'  Saved: {out_txt}\n')

    results[est_name] = (best_res, best_delta)
    for term in best_res.params.index:
        coef_rows.append({
            'estimator': est_name, 'term': term,
            'coef': best_res.params[term], 'se': best_res.bse[term],
            't': best_res.tvalues[term],   'p':  best_res.pvalues[term],
        })

# ── reports ───────────────────────────────────────────────────────────────────

coef_df = pd.DataFrame(coef_rows)
coef_df.to_csv(REPORTS / f'intervention_coefficients{SUFFIX}.csv', index=False)
print(f'Saved: {REPORTS / f"intervention_coefficients{SUFFIX}.csv"}')

delta_df = pd.DataFrame(delta_log)
delta_df.to_csv(REPORTS / f'intervention_delta_selection{SUFFIX}.csv', index=False)
print(f'Saved: {REPORTS / f"intervention_delta_selection{SUFFIX}.csv"}')


# ── coefficient comparison figure ────────────────────────────────────────────

TERM_LABELS = {
    'Intercept':          'Intercept',
    'z_beta':             'β: beta',
    'z_avg_corr':         'β: avg corr',
    'z_liquidity':        'β: liquidity',
    'z_momentum':         'β: momentum',
    'decay':              'γ: decay',
    'z_beta:decay':       'θ: decay×beta',
    'z_avg_corr:decay':   'θ: decay×avg_corr',
    'z_liquidity:decay':  'θ: decay×liquidity',
    'z_momentum:decay':   'θ: decay×momentum',
}

def normalise_term(t: str) -> str:
    if ':' in t:
        return ':'.join(sorted(t.split(':')))
    return t

PLOT_TERMS = [t for t in TERM_LABELS if t != 'Intercept']
est_names  = list(ESTIMATORS.keys())
n_terms    = len(PLOT_TERMS)
n_est      = len(est_names)
bar_width  = 0.25
x          = np.arange(n_terms)

fig, ax = plt.subplots(figsize=(14, 6))

for k, est_name in enumerate(est_names):
    res, (hl, delta) = results[est_name]
    params  = res.params.rename(normalise_term)
    bse     = res.bse.rename(normalise_term)
    coefs   = [params.get(normalise_term(t), np.nan) for t in PLOT_TERMS]
    errors  = [1.96 * bse.get(normalise_term(t), np.nan) for t in PLOT_TERMS]
    offset  = (k - (n_est - 1) / 2) * bar_width

    ax.bar(x + offset, coefs, bar_width,
           label=f'{est_name} (δ={delta}, HL={hl}td)',
           color=EST_COLORS[est_name], alpha=0.85, edgecolor='white')
    ax.errorbar(x + offset, coefs, yerr=errors,
                fmt='none', color='black', capsize=3, linewidth=1.1)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels([TERM_LABELS[t] for t in PLOT_TERMS],
                   rotation=30, ha='right', fontsize=10)
ax.set_ylabel('OLS coefficient (z-scored regressors)', fontsize=11)
ax.set_title(
    'Decay Intervention OLS — Per-Asset GMV Weight\n'
    'Sample / LW / Gerber  |  bars = coef,  error bars = 95% CI (cluster-robust by date)',
    fontsize=12,
)
ax.legend(fontsize=10)
plt.tight_layout()
fig_path = FIGURES / f'intervention_coef_comparison{SUFFIX}.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {fig_path}')

# ── summary table ─────────────────────────────────────────────────────────────

print('\n' + '='*70)
print('COEFFICIENT COMPARISON (cluster-robust t-stats)')
print('='*70)
pivot   = coef_df.pivot_table(index='term', columns='estimator', values='coef')
piv_t   = coef_df.pivot_table(index='term', columns='estimator', values='t')
ordered = [t for t in TERM_LABELS if t in pivot.index]
print('\n--- Coefficients ---')
print(pivot.loc[ordered].round(5).to_string())
print('\n--- t-statistics ---')
print(piv_t.loc[ordered].round(3).to_string())
print('\n--- Selected delta & R² ---')
for est_name in est_names:
    res, (hl, delta) = results[est_name]
    print(f'  {est_name:8s}  delta={delta}  half-life={hl:2d}td  R²={res.rsquared:.4f}  N={int(res.nobs):,}')
print('\nDone.')
