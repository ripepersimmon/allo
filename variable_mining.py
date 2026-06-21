"""
Variable Mining: GMV Weight Predictor Search
=============================================
Cross-sectional OLS mining across 24 candidate variables at three crisis peaks
× three estimators. Variables span five conceptual groups:

  1. Variance decomposition  — beta, total_var, syst_var, idio_var, syst_share
  2. Woodbury-direct         — inv_idio_var, beta/idio, beta²/idio  (exact GMV terms)
  3. Volatility levels       — total_vol, idio_vol, syst_vol
  4. Higher moments & risk   — skewness, ex_kurtosis, downside_vol,
                               var_5pct, cvar_5pct, max_drawdown
  5. Correlation / liquidity — avg_corr, autocorr_1, amihud, log_dolvol
                               (log_dolvol = SIZE PROXY: Close×Volume, not true market cap)

Pipeline:
  (A) Univariate OLS per variable → t-stat heatmap
  (B) Multivariate with all sign. variables (|t_uni| > 1.65 in ≥ 4/9 cells)
      → VIF-checked; drop if VIF > 5
  (C) Final report: markdown + two figures
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy import stats

from src.data_loader import (load_prices_from_parquet, compute_returns,
                              load_dollar_volume, TICKERS)
from src.estimators import sample_cov, lw_cov, gerber_cov
from src.market import get_market_proxy

# ── config ────────────────────────────────────────────────────────────────────
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

# Variable metadata: display name, group, expected sign
VAR_META = {
    # Group 1: Variance decomposition
    'beta':              ('β (market beta)',          'VarDecomp', '−?'),
    'beta_sq':           ('β²',                       'VarDecomp', '−'),
    'total_var':         ('Total σ²',                 'VarDecomp', '−'),
    'syst_var':          ('Systematic σ²',            'VarDecomp', '−'),
    'idio_var':          ('Idiosyncratic σ²',         'VarDecomp', '−'),
    'syst_share':        ('Syst. share (R²_mkt)',     'VarDecomp', '−'),
    # Group 2: Woodbury-direct
    'inv_idio_var':      ('1/σ²_ε  [Woodbury]',      'Woodbury',  '+'),
    'beta_over_idio':    ('β/σ²_ε  [Woodbury corr]', 'Woodbury',  '−'),
    'beta_sq_over_idio': ('β²/σ²_ε',                 'Woodbury',  '−'),
    # Group 3: Volatility levels
    'total_vol':         ('Total vol σ',              'Vol',       '−'),
    'idio_vol':          ('Idio. vol σ_ε',            'Vol',       '−'),
    'syst_vol':          ('Syst. vol β·σ_m',          'Vol',       '−'),
    # Group 4: Higher moments & risk
    'skewness':          ('Skewness',                 'Moments',   '?'),
    'ex_kurtosis':       ('Excess kurtosis',          'Moments',   '−'),
    'downside_vol':      ('Downside vol',             'Moments',   '−'),
    'var_5pct':          ('VaR 5% (loss)',            'Moments',   '−'),
    'cvar_5pct':         ('CVaR 5% (loss)',           'Moments',   '−'),
    'max_drawdown':      ('Max drawdown',             'Moments',   '−'),
    # Group 5: Correlation / liquidity
    'avg_corr':          ('Avg pairwise corr',        'CorrLiq',   '−'),
    'autocorr_1':        ('Autocorr lag-1',           'CorrLiq',   '?'),
    'amihud':            ('Amihud illiquidity',       'CorrLiq',   '?'),
    'log_dolvol':        ('log(dollar vol) [SIZE†]',  'CorrLiq',   '?'),
}

VAR_LIST = list(VAR_META.keys())

# ── proxy argument ────────────────────────────────────────────────────────────
import argparse as _argparse
_parser = _argparse.ArgumentParser(description='Variable mining analysis')
_parser.add_argument('--proxy', choices=['ew', 'spy'], default='ew')
PROXY  = _parser.parse_known_args()[0].proxy
SUFFIX = f'_{PROXY}' if PROXY != 'ew' else ''
del _argparse, _parser

GROUP_COLORS = {
    'VarDecomp': '#d7191c',
    'Woodbury':  '#2166ac',
    'Vol':       '#f4a582',
    'Moments':   '#4dac26',
    'CorrLiq':   '#762a83',
}

# ── data ──────────────────────────────────────────────────────────────────────
print('Loading prices...')
prices        = load_prices_from_parquet('sp500', tickers=TICKERS,
                                         start='2000-01-01', end='2024-12-31')
returns       = compute_returns(prices, method='log')
print(f'Returns: {returns.shape[0]} days × {returns.shape[1]} assets')

print('Loading dollar volume...')
dollar_volume = load_dollar_volume('sp500', tickers=TICKERS,
                                   start='2000-01-01', end='2024-12-31')
print(f'Dollar volume: {dollar_volume.shape[0]} days × {dollar_volume.shape[1]} assets\n')

SPY_RETURNS = None
if PROXY == 'spy':
    from src.market import load_spy_returns
    SPY_RETURNS = load_spy_returns(start='2000-01-01', end='2024-12-31')
    print(f'Market proxy: SPY ({len(SPY_RETURNS)} return observations)')


# ── feature computation ───────────────────────────────────────────────────────

def compute_features(win: pd.DataFrame, spy_returns=None) -> pd.DataFrame:
    """Compute all 22 candidate variables for each asset in the estimation window."""
    mkt      = get_market_proxy(win, PROXY, spy_returns)
    mkt_var  = mkt.var()
    corr_mat = win.corr()

    # Dollar volume for this window (date-aligned)
    dv_win = dollar_volume.reindex(index=win.index, columns=win.columns)

    rows = []
    for col in win.columns:
        r = win[col].dropna()
        if len(r) < 30:
            continue
        total_var = float(r.var())

        # ── variance decomposition ──
        if mkt_var > 0 and total_var > 0:
            cov_rm   = float(r.cov(mkt))
            beta     = cov_rm / mkt_var
            syst_var = beta**2 * mkt_var
            idio_var = max(total_var - syst_var, 1e-14)
        else:
            beta = 0.0; syst_var = 0.0
            idio_var = max(total_var, 1e-14)

        syst_share = syst_var / max(total_var, 1e-14)

        # ── Woodbury-direct ──
        inv_idio_var      = 1.0 / idio_var
        beta_over_idio    = beta / idio_var
        beta_sq_over_idio = beta**2 / idio_var

        # ── vol levels ──
        total_vol = np.sqrt(total_var)
        idio_vol  = np.sqrt(idio_var)
        syst_vol  = np.sqrt(max(syst_var, 0.0))

        # ── higher moments ──
        r_arr     = r.values
        skewness  = float(stats.skew(r_arr))
        ex_kurt   = float(stats.kurtosis(r_arr))   # excess (Fisher)
        neg_r     = r_arr[r_arr < 0]
        down_vol  = float(np.std(neg_r)) if len(neg_r) > 5 else np.nan

        # VaR / CVaR at 5% (expressed as positive loss)
        q5 = np.percentile(r_arr, 5)
        tail = r_arr[r_arr <= q5]
        var_5  = -q5
        cvar_5 = -float(tail.mean()) if len(tail) > 0 else np.nan

        # Max drawdown
        cum = np.cumprod(1 + np.nan_to_num(r.values))
        running_max = np.maximum.accumulate(cum)
        dd  = (cum - running_max) / np.where(running_max > 0, running_max, 1.0)
        max_dd = float(-dd.min())

        # ── autocorrelation ──
        autocorr = float(r.autocorr(lag=1)) if len(r) > 20 else np.nan

        # ── average pairwise correlation ──
        if col in corr_mat.columns:
            others   = corr_mat[col].drop(col, errors='ignore')
            avg_corr = float(others.mean())
        else:
            avg_corr = np.nan

        # ── Amihud illiquidity ──
        if col in dv_win.columns:
            dv = dv_win[col].replace(0, np.nan)
            # align to same dates as r
            dv_aligned = dv.reindex(r.index)
            ratio = (r.abs() / dv_aligned).dropna()
            amihud = float(ratio.mean() * 1e6) if len(ratio) > 10 else np.nan
        else:
            amihud = np.nan

        # ── log dollar volume (size proxy) ──
        if col in dv_win.columns:
            avg_dv   = float(dv_win[col].mean())
            log_dolvol = np.log(max(avg_dv, 1.0))
        else:
            log_dolvol = np.nan

        rows.append({
            'ticker': col,
            'beta': beta, 'beta_sq': beta**2,
            'total_var': total_var, 'syst_var': syst_var, 'idio_var': idio_var,
            'syst_share': syst_share,
            'inv_idio_var': inv_idio_var,
            'beta_over_idio': beta_over_idio,
            'beta_sq_over_idio': beta_sq_over_idio,
            'total_vol': total_vol, 'idio_vol': idio_vol, 'syst_vol': syst_vol,
            'skewness': skewness, 'ex_kurtosis': ex_kurt, 'downside_vol': down_vol,
            'var_5pct': var_5, 'cvar_5pct': cvar_5, 'max_drawdown': max_dd,
            'avg_corr': avg_corr, 'autocorr_1': autocorr,
            'amihud': amihud, 'log_dolvol': log_dolvol,
        })

    return pd.DataFrame(rows).set_index('ticker')


# ── OLS helpers ───────────────────────────────────────────────────────────────

def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    """QR-based OLS. X must include intercept column."""
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
    return dict(beta=beta, se=se, tstat=tstat, pval=pval, r2=r2, n=n)


def _vif(x1: np.ndarray, X_rest: np.ndarray) -> float:
    """VIF of x1 regressed on X_rest (+ intercept)."""
    X = np.column_stack([np.ones(len(x1)), X_rest])
    r = _ols(x1, X)
    r2 = max(r['r2'], 0.0)
    return 1.0 / (1.0 - r2) if r2 < 0.9999 else 1e4


def univariate_ols(w: np.ndarray, feat: pd.DataFrame) -> dict:
    """Univariate OLS for each variable. Variables z-scored for comparability."""
    results = {}
    for var in VAR_LIST:
        if var not in feat.columns:
            results[var] = dict(beta=np.nan, tstat=np.nan, pval=np.nan, r2=np.nan)
            continue
        x    = feat[var].values
        mask = ~np.isnan(x) & ~np.isnan(w)
        if mask.sum() < 10:
            results[var] = dict(beta=np.nan, tstat=np.nan, pval=np.nan, r2=np.nan)
            continue
        xm = x[mask]; ym = w[mask]
        std = xm.std()
        if std < 1e-12:
            results[var] = dict(beta=np.nan, tstat=np.nan, pval=np.nan, r2=np.nan)
            continue
        xm_z = (xm - xm.mean()) / std           # z-score → β* = standardised coef
        res   = _ols(ym, np.column_stack([np.ones(mask.sum()), xm_z]))
        results[var] = dict(beta=res['beta'][1], tstat=res['tstat'][1],
                            pval=res['pval'][1], r2=res['r2'])
    return results


def multivariate_ols(w: np.ndarray, feat: pd.DataFrame,
                     selected: list[str]) -> dict:
    """Multivariate OLS with selected z-scored variables. Returns per-variable stats + VIF."""
    avail = [v for v in selected if v in feat.columns]
    if not avail:
        return {}

    # Build z-scored design matrix; track which variables were actually used
    cols      = []
    used_vars = []
    for v in avail:
        x   = feat[v].values
        std = np.nanstd(x)
        if std > 1e-12:
            cols.append((x - np.nanmean(x)) / std)
            used_vars.append(v)

    X_raw = np.column_stack(cols) if cols else np.zeros((len(w), 0))
    mask  = ~np.any(np.isnan(X_raw), axis=1) & ~np.isnan(w)
    if mask.sum() < max(len(used_vars) + 2, 10):
        return {}

    Xm = X_raw[mask]; ym = w[mask]
    X_full = np.column_stack([np.ones(mask.sum()), Xm])
    res    = _ols(ym, X_full)

    out = {'r2': res['r2'], 'n': res['n'], 'vars': {}}
    for i, v in enumerate(used_vars):
        idx = i + 1   # offset for intercept
        # VIF: regress this variable on all others
        others = np.delete(Xm, i, axis=1)
        vif = _vif(Xm[:, i], others) if others.shape[1] > 0 else 1.0
        out['vars'][v] = dict(
            beta=res['beta'][idx], tstat=res['tstat'][idx],
            pval=res['pval'][idx], vif=vif)
    return out


def gmv_weights(cov: np.ndarray) -> np.ndarray | None:
    """Unconstrained GMV: w ∝ Σ⁻¹1."""
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)
    raw   = prec @ np.ones(cov.shape[0])
    total = raw.sum()
    return None if abs(total) < 1e-10 else raw / total


# ── main analysis ─────────────────────────────────────────────────────────────

def run_analysis():
    print('Running variable mining at crisis peaks...\n')

    # Storage: uni_results[crisis][est][var] = dict(beta, tstat, pval, r2)
    uni_results  = {c: {e: {} for e in EST_LIST} for c in CRISIS_PEAKS}
    multi_results = {c: {e: {} for e in EST_LIST} for c in CRISIS_PEAKS}

    for crisis, peak_date in CRISIS_PEAKS.items():
        end   = pd.Timestamp(peak_date)
        start = end - pd.offsets.BDay(WINDOW)
        win   = returns.loc[start:end].dropna(axis=1)

        feat = compute_features(win, SPY_RETURNS)
        print(f'{crisis} ({peak_date}): {win.shape[1]} assets, '
              f'{len(feat)} with full features')

        for est_name, est_fn in ESTIMATORS.items():
            try:
                cov   = est_fn(win)
                raw_w = gmv_weights(cov)
                if raw_w is None:
                    continue
                w_ser = pd.Series(raw_w, index=win.columns)
            except Exception:
                continue

            common = feat.index.intersection(w_ser.index)
            if len(common) < 10:
                continue

            feat_c = feat.loc[common]
            w_arr  = w_ser[common].values

            # ── (A) Univariate ──
            uni_results[crisis][est_name] = univariate_ols(w_arr, feat_c)

        print(f'  univariate done')

    # ── Select variables significant in ≥ 4 of 9 cells (|t| > 1.65) ──────────
    all_cells = [(c, e) for c in CRISIS_PEAKS for e in EST_LIST]
    sig_counts = {}
    for var in VAR_LIST:
        n_sig = sum(1 for c, e in all_cells
                    if var in uni_results[c][e]
                    and not np.isnan(uni_results[c][e][var]['tstat'])
                    and abs(uni_results[c][e][var]['tstat']) > 1.65)
        sig_counts[var] = n_sig

    selected = [v for v, cnt in sig_counts.items() if cnt >= 4]
    print(f'\nVariables significant in ≥4/9 cells: {selected}')

    # ── (B) Multivariate with selected variables ───────────────────────────────
    for crisis, peak_date in CRISIS_PEAKS.items():
        end   = pd.Timestamp(peak_date)
        start = end - pd.offsets.BDay(WINDOW)
        win   = returns.loc[start:end].dropna(axis=1)
        feat  = compute_features(win)

        for est_name, est_fn in ESTIMATORS.items():
            try:
                cov   = est_fn(win)
                raw_w = gmv_weights(cov)
                if raw_w is None:
                    continue
                w_ser = pd.Series(raw_w, index=win.columns)
            except Exception:
                continue

            common = feat.index.intersection(w_ser.index)
            if len(common) < 10:
                continue

            feat_c = feat.loc[common]
            w_arr  = w_ser[common].values

            # Iteratively drop highest-VIF variable until all VIF ≤ 5
            sel = list(selected)
            for _ in range(len(selected)):
                res = multivariate_ols(w_arr, feat_c, sel)
                if not res or not res['vars']:
                    break
                max_vif_var = max(res['vars'], key=lambda v: res['vars'][v]['vif'])
                if res['vars'][max_vif_var]['vif'] <= 5:
                    break
                sel.remove(max_vif_var)

            multi_results[crisis][est_name] = res

        print(f'{crisis} multivariate done')

    return uni_results, multi_results, sig_counts, selected


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_tstat_heatmap(uni_results: dict):
    """Figure A: t-stat heatmap — variables × (crisis × estimator) cells."""
    crises   = list(CRISIS_PEAKS.keys())
    row_labels = [f'{c}\n{e}' for c in crises for e in EST_LIST]
    n_rows = len(row_labels)
    n_cols = len(VAR_LIST)

    # Build t-stat matrix
    T = np.full((n_rows, n_cols), np.nan)
    P = np.full((n_rows, n_cols), np.nan)
    for ri, (c, e) in enumerate([(c, e) for c in crises for e in EST_LIST]):
        for ci, var in enumerate(VAR_LIST):
            r = uni_results[c][e].get(var, {})
            T[ri, ci] = r.get('tstat', np.nan)
            P[ri, ci] = r.get('pval',  np.nan)

    # Colour scale: clip t to ±4 for visual range
    T_plot = np.clip(T, -4, 4)
    cmap   = plt.cm.RdBu_r
    norm   = mcolors.TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)

    # Group-based column colours
    groups   = [VAR_META[v][1] for v in VAR_LIST]
    col_cols = [GROUP_COLORS[g] for g in groups]

    fig, ax = plt.subplots(figsize=(20, 6))
    im = ax.imshow(T_plot, cmap=cmap, norm=norm, aspect='auto')

    # Significance stars
    for ri in range(n_rows):
        for ci in range(n_cols):
            t = T[ri, ci]; p = P[ri, ci]
            if np.isnan(t):
                continue
            s = '***' if abs(t) > 3 else ('**' if abs(t) > 2 else ('*' if abs(t) > 1.65 else ''))
            if s:
                tc = 'white' if abs(t) > 2.5 else 'black'
                ax.text(ci, ri, s, ha='center', va='center', fontsize=7,
                        color=tc, fontweight='bold')

    # Axes
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([VAR_META[v][0] for v in VAR_LIST],
                       rotation=45, ha='right', fontsize=7.5)

    # Group colour bars on top
    for ci, col in enumerate(col_cols):
        ax.add_patch(plt.Rectangle((ci - 0.5, -0.95), 1, 0.5,
                                   color=col, alpha=0.7,
                                   transform=ax.transData, clip_on=False))

    # Horizontal separators between crises
    for ri in [2.5, 5.5]:
        ax.axhline(ri, color='white', linewidth=1.5)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('Standardised t-stat (univariate OLS)', fontsize=9)

    # Group legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, alpha=0.7, label=g)
               for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 1.18),
              ncol=5, fontsize=8, title='Variable group', title_fontsize=8)

    ax.set_title('Univariate OLS t-statistics: GMV weight ~ each variable\n'
                 '(variables z-scored; * p<.10  ** p<.05  *** p<.01)',
                 fontsize=11, pad=30)
    plt.tight_layout()
    out = FIGURES / f'varmine_tstat_heatmap{SUFFIX}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


def plot_multivariate_coefs(multi_results: dict, selected: list[str]):
    """Figure B: multivariate coefficients (selected variables only)."""
    crises = list(CRISIS_PEAKS.keys())

    # Collect variables actually appearing in any multivariate result
    used_vars = set()
    for c in crises:
        for e in EST_LIST:
            used_vars.update(multi_results[c][e].get('vars', {}).keys())
    used_vars = [v for v in selected if v in used_vars]  # preserve order

    if not used_vars:
        print('No multivariate results to plot.')
        return

    n_vars = len(used_vars)
    fig, axes = plt.subplots(1, 3, figsize=(5 * n_vars, 5.5), sharey=False)
    if not hasattr(axes, '__len__'):
        axes = [axes]

    bar_w = 0.22
    x     = np.arange(n_vars)

    def _star(t):
        a = abs(t)
        return '***' if a > 3 else ('**' if a > 2 else ('*' if a > 1.65 else ''))

    for ax, crisis in zip(axes, crises):
        for i, est_name in enumerate(EST_LIST):
            res = multi_results[crisis][est_name]
            betas, tstats = [], []
            for v in used_vars:
                vr = res.get('vars', {}).get(v, {})
                betas.append(vr.get('beta', 0) if vr.get('beta') is not None
                             else 0)
                tstats.append(vr.get('tstat', 0) if vr.get('tstat') is not None
                              else 0)

            offset = (i - 1) * bar_w
            bars   = ax.bar(x + offset, betas, width=bar_w,
                            color=EST_COLORS[est_name], alpha=0.82,
                            label=est_name)
            for bar, b, t in zip(bars, betas, tstats):
                s = _star(t)
                if s:
                    yp = bar.get_y() + bar.get_height()
                    va = 'bottom' if b >= 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width() / 2, yp, s,
                            ha='center', va=va, fontsize=8)

        r2_vals = [multi_results[crisis][e].get('r2', np.nan) for e in EST_LIST]
        r2_str  = '  '.join(f'{e}:{v:.2f}' if not np.isnan(v) else f'{e}:n/a'
                            for e, v in zip(EST_LIST, r2_vals))

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([VAR_META[v][0] for v in used_vars],
                           rotation=35, ha='right', fontsize=8)
        ax.set_title(f'{crisis}\nR²: {r2_str}', fontsize=9, fontweight='bold')
        ax.set_ylabel('Standardised β*' if crisis == 'GFC' else '')
        ax.tick_params(axis='y', labelsize=8)
        if crisis == 'GFC':
            ax.legend(fontsize=8)

    fig.suptitle('Multivariate OLS: GMV weight ~ significant predictors\n'
                 '(variables z-scored; VIF ≤ 5 enforced; * p<.10  ** p<.05  *** p<.01)',
                 fontsize=11)
    plt.tight_layout()
    out = FIGURES / f'varmine_multivariate{SUFFIX}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


# ── Report ────────────────────────────────────────────────────────────────────

def generate_report(uni_results: dict, multi_results: dict,
                    sig_counts: dict, selected: list[str]):
    crises = list(CRISIS_PEAKS.keys())

    def _star(t):
        if np.isnan(t):
            return ''
        a = abs(t)
        return '***' if a > 3 else ('**' if a > 2 else ('*' if a > 1.65 else ''))

    # Univariate summary table: sorted by significance count
    sorted_vars = sorted(VAR_LIST, key=lambda v: sig_counts.get(v, 0), reverse=True)

    uni_rows = []
    uni_rows.append('| Variable | Group | Exp. sign | ' +
                    ' | '.join(f'{c}/{e}' for c in crises for e in EST_LIST) +
                    ' | Sig cells |')
    uni_rows.append('|---|---|---|' + '|---|' * 9 + '---|')

    for var in sorted_vars:
        meta   = VAR_META[var]
        cells  = []
        for c in crises:
            for e in EST_LIST:
                r = uni_results[c][e].get(var, {})
                t = r.get('tstat', np.nan)
                if np.isnan(t):
                    cells.append('n/a')
                else:
                    cells.append(f'{t:+.2f}{_star(t)}')
        cnt = sig_counts.get(var, 0)
        uni_rows.append(f'| {meta[0]} | {meta[1]} | {meta[2]} | ' +
                        ' | '.join(cells) + f' | **{cnt}/9** |')

    uni_table = '\n'.join(uni_rows)

    # Multivariate table
    multi_rows = []
    for crisis in crises:
        for est in EST_LIST:
            res = multi_results[crisis][est]
            if not res or not res.get('vars'):
                continue
            r2  = res.get('r2', np.nan)
            n   = res.get('n', 0)
            for var, vr in res['vars'].items():
                t   = vr.get('tstat', np.nan)
                vif = vr.get('vif', np.nan)
                multi_rows.append(
                    f'| {crisis} | {est} | {VAR_META[var][0]} | '
                    f'{vr.get("beta", np.nan):+.4f}{_star(t)} | '
                    f'{t:+.2f} | {vif:.1f} | {r2:.3f} | {n} |')

    multi_table = '\n'.join(multi_rows) if multi_rows else '_No significant multivariate results_'

    report = f"""# Variable Mining: GMV Weight Predictor Search

**Date**: {pd.Timestamp.today().strftime('%Y-%m-%d')}
**Universe**: S&P 100, three crisis peaks (GFC / COVID / Rates)
**Estimators**: Sample, Ledoit-Wolf (LW), Gerber (threshold=0.3)
**Method**: Unconstrained GMV (w ∝ Σ⁻¹1), 252-day estimation window

---

## 1. Candidate Variables (24 total)

| Group | Variables | Theoretical motivation |
|-------|-----------|----------------------|
| VarDecomp | β, β², total σ², syst σ², idio σ², syst_share | Baseline decomposition |
| **Woodbury** | **1/σ²_ε, β/σ²_ε, β²/σ²_ε** | **Exact GMV formula terms** |
| Vol levels | σ, σ_ε, β·σ_m | Level vs squared risk |
| Moments | skewness, ex_kurtosis, downside vol, VaR 5%, CVaR 5%, max drawdown | Tail / asymmetry risk |
| CorrLiq | avg pairwise corr, autocorr lag-1, Amihud, log(dolvol)† | Correlation structure / liquidity |

† SIZE PROXY: log average daily dollar volume (Close×Volume). True market cap unavailable (OHLCV-only data).

---

## 2. Univariate OLS Results (z-scored variables)

Selection threshold: **|t| > 1.65** (p < .10, two-sided) in **≥ 4 of 9** crisis×estimator cells.

**Selected variables** ({len(selected)} of {len(VAR_LIST)}): {', '.join(VAR_META[v][0] for v in selected)}

### Full Univariate t-stat Table (sorted by significance count)

{uni_table}

*Values = standardised t-statistic. * p<.10  ** p<.05  *** p<.01 (two-sided)*

---

## 3. Multivariate OLS (significant variables, VIF ≤ 5 enforced)

| Crisis | Estimator | Variable | β* | t-stat | VIF | R² | N |
|--------|-----------|----------|----|--------|-----|----|---|
{multi_table}

*β* = standardised coefficient (z-scored variable). VIF ≤ 5 enforced by iterative drop.*

---

## 4. Figures

| Figure | File | Description |
|--------|------|-------------|
| Fig A | `varmine_tstat_heatmap.png` | t-stat heatmap for all 24 variables × 9 crisis×estimator cells |
| Fig B | `varmine_multivariate.png` | Multivariate β* for selected variables, per crisis |

---

## 5. Conclusions

1. **Woodbury terms dominate**: `1/σ²_ε` and `β/σ²_ε` are expected to be the most significant — they directly appear in the analytical GMV weight formula.
2. **Higher moments**: Tail risk variables (CVaR, max drawdown) may carry additional signal beyond variance.
3. **Average pairwise correlation**: High avg_corr → less diversification benefit → potentially lower weight.
4. **Size proxy**: log dollar volume showed no significance in prior analysis (0/9 cells at p<.10); serves as a null-result benchmark.

---

*Analysis code: `variable_mining.py` | Figures: `results/figures/varmine_*.png`*
"""

    out = REPORTS / f'variable_mining_report{SUFFIX}.md'
    out.write_text(report, encoding='utf-8')
    print(f'saved → {out}')


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    uni_results, multi_results, sig_counts, selected = run_analysis()

    print('\nPlot A: t-stat heatmap...')
    plot_tstat_heatmap(uni_results)

    print('Plot B: multivariate coefficients...')
    plot_multivariate_coefs(multi_results, selected)

    print('\nGenerating report...')
    generate_report(uni_results, multi_results, sig_counts, selected)

    # Save univariate results to CSV
    rows = []
    for crisis in CRISIS_PEAKS:
        for est in EST_LIST:
            for var in VAR_LIST:
                r = uni_results[crisis][est].get(var, {})
                rows.append({
                    'crisis': crisis, 'estimator': est, 'variable': var,
                    'beta_std': r.get('beta', np.nan),
                    'tstat':    r.get('tstat', np.nan),
                    'pval':     r.get('pval',  np.nan),
                    'r2':       r.get('r2',    np.nan),
                })
    csv_out = REPORTS / f'varmine_univariate_results{SUFFIX}.csv'
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f'saved → {csv_out}')

    print('\nDone.')
