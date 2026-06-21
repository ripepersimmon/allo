"""
Variance Decomposition vs. GMV Weight
Decomposes each asset's total variance into systematic (β²σ²_m) and
idiosyncratic (σ²_ε) components, then runs cross-sectional OLS across
three crisis snapshots and rolling windows.

Five comparison models per snapshot:
  (A) beta-only:   w = α + γ·β
  (B) total-var:   w = α + γ·σ²
  (C) decomposed:  w = α + γ₁·syst_var + γ₂·idio_var
  (D) orthogonal:  w = α + γ₁·total_var + γ₂·syst_share   (syst_share = syst_var/total_var)
  (E) + size:      w = α + γ₁·total_var + γ₂·syst_share + γ₃·log_dolvol

Model (D) is the primary variance-decomposition specification.
Model (E) adds a SIZE PROXY: log of average daily dollar volume (Close × Volume)
over the 252-day estimation window. Dollar volume is used as a proxy for
market capitalisation because actual shares-outstanding data are not available
in the source files (OHLCV only). Within S&P 100, dollar volume correlates
strongly with market cap and institutional coverage.
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats

from src.data_loader import load_prices_from_parquet, compute_returns, load_dollar_volume, TICKERS
from src.estimators import sample_cov, lw_cov, gerber_cov
from src.market import get_market_proxy

# ── config ────────────────────────────────────────────────────────────────────
WINDOW  = 252
FIGURES = Path('results/figures')
REPORTS = Path('reports')
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

ESTIMATORS  = {'Sample': sample_cov, 'LW': lw_cov, 'Gerber': gerber_cov}
EST_COLORS  = {'Sample': '#e41a1c', 'LW': '#377eb8', 'Gerber': '#4daf4a'}
EST_LIST    = list(ESTIMATORS.keys())

SUFFIX = ''  # set to '_spy' at runtime via --proxy spy

CRISIS_PEAKS = {
    'GFC':   '2009-03-31',
    'COVID': '2020-04-30',
    'Rates': '2023-01-31',
}
CRISIS_RANGES = {
    'GFC':   ('2007-01-01', '2009-06-30'),
    'COVID': ('2019-10-01', '2020-09-30'),
    'Rates': ('2021-07-01', '2023-01-31'),
}

# ── data ──────────────────────────────────────────────────────────────────────
print('Loading prices...')
prices  = load_prices_from_parquet('sp500', tickers=TICKERS, start='2000-01-01', end='2024-12-31')
returns = compute_returns(prices, method='log')
print(f'Returns: {returns.shape[0]} days × {returns.shape[1]} assets')

print('Loading dollar volume (size proxy: Close × Volume)...')
dollar_volume = load_dollar_volume('sp500', tickers=TICKERS, start='2000-01-01', end='2024-12-31')
print(f'Dollar volume: {dollar_volume.shape[0]} days × {dollar_volume.shape[1]} assets\n')


# ── core helpers ──────────────────────────────────────────────────────────────

def get_window(end_date: str) -> pd.DataFrame:
    end   = pd.Timestamp(end_date)
    start = end - pd.offsets.BDay(WINDOW)
    return returns.loc[start:end].dropna(axis=1)


def get_log_dolvol(dates: pd.DatetimeIndex, tickers: list) -> pd.Series:
    """Average log dollar volume over the given dates for each ticker.

    SIZE PROXY: dollar volume (Close × Volume) is used in place of market
    capitalisation, which is not available in the source OHLCV files.
    Log-transform is applied after averaging; tickers with no valid data
    receive NaN and are excluded from downstream OLS.
    """
    sub = dollar_volume.reindex(index=dates, columns=tickers)
    avg = sub.mean(axis=0)                    # average over 252 days
    log_dv = np.log(avg.clip(lower=1.0))      # clip avoids log(0) for zero-vol days
    return log_dv


def decompose_variance(win: pd.DataFrame,
                       proxy: str = 'ew',
                       spy_returns=None) -> pd.DataFrame:
    """
    Returns a DataFrame (index=ticker) with columns:
      beta, total_var, syst_var, idio_var, r2_mkt
    Market proxy is controlled by `proxy` ('ew' or 'spy').
    """
    mkt = get_market_proxy(win, proxy, spy_returns)
    # ensure win and mkt share the same non-NaN dates (matters for SPY proxy)
    valid = mkt.dropna().index.intersection(win.index)
    if len(valid) < 30:
        return pd.DataFrame()   # caller checks empty decomp and skips
    win   = win.loc[valid]
    mkt   = mkt.loc[valid]
    mkt_var = mkt.var()
    rows = []
    for col in win.columns:
        r         = win[col]
        total_var = r.var()
        if mkt_var > 0 and total_var > 0:
            cov_rm   = r.cov(mkt)
            beta     = cov_rm / mkt_var
            syst_var = beta**2 * mkt_var
            idio_var = max(total_var - syst_var, 0.0)
            r2_mkt   = min((cov_rm**2) / (total_var * mkt_var), 1.0)
        else:
            beta = syst_var = r2_mkt = 0.0
            idio_var = total_var
        rows.append(dict(ticker=col, beta=beta, total_var=total_var,
                         syst_var=syst_var, idio_var=idio_var, r2_mkt=r2_mkt))
    return pd.DataFrame(rows).set_index('ticker')


def gmv_weights(cov: np.ndarray) -> np.ndarray:
    """Unconstrained analytical GMV: w ∝ Σ⁻¹ 1.

    Tries exact inv first; falls back to pinv only on singularity.
    Returns None if the precision row sums are near-zero (degenerate weights).
    """
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)
    raw = prec @ np.ones(cov.shape[0])
    total = raw.sum()
    if abs(total) < 1e-10:
        return None
    return raw / total


def _ols(y: np.ndarray, X: np.ndarray):
    """OLS with t-stats via QR decomposition (numerically consistent).
    X must include intercept column.
    Uses a single QR path for both coefficients and SEs to avoid
    beta/SE source mismatch on near-rank-deficient inputs.
    """
    n, k = X.shape
    try:
        Q, R  = np.linalg.qr(X)
        beta  = np.linalg.solve(R, Q.T @ y)
        R_inv = np.linalg.inv(R)
        XtX_inv = R_inv @ R_inv.T        # == (X'X)^{-1}, consistent with beta
    except np.linalg.LinAlgError:
        # full fallback: lstsq beta + pinv SE (both from same X'X)
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


def _vif(x1: np.ndarray, x2: np.ndarray) -> float:
    """VIF of x1 regressed on x2 (+ intercept). High VIF = collinearity."""
    X = np.column_stack([np.ones(len(x2)), x2])
    r = _ols(x1, X)
    r2 = max(r['r2'], 0.0)
    return 1.0 / (1.0 - r2) if r2 < 0.9999 else 1e4


def ols_models(w: np.ndarray, beta_vals: np.ndarray,
               syst: np.ndarray, idio: np.ndarray,
               total_var: np.ndarray,
               log_dolvol: np.ndarray | None = None) -> dict:
    """Run all five comparison models and return consolidated stats.

    Model (D) is the primary variance-decomposition spec.
    Model (E) adds log_dolvol as a SIZE PROXY (dollar volume, not true market cap).
    If log_dolvol is None or contains NaN, Model (E) is skipped.
    """
    n = len(w)
    if n < 6:
        return None
    ones       = np.ones(n)
    syst_share = np.where(total_var > 1e-14, syst / total_var, 0.0)

    rA = _ols(w, np.column_stack([ones, beta_vals]))               # (A) beta-only
    rB = _ols(w, np.column_stack([ones, total_var]))                # (B) total-var
    rC = _ols(w, np.column_stack([ones, syst, idio]))               # (C) decomposed
    rD = _ols(w, np.column_stack([ones, total_var, syst_share]))    # (D) orthogonal

    vif_c = _vif(syst, idio)

    result = {
        'r2_A': rA['r2'], 'r2_B': rB['r2'], 'r2_C': rC['r2'], 'r2_D': rD['r2'],
        'gamma1': rC['beta'][1], 'gamma2': rC['beta'][2],
        'tstat_g1': rC['tstat'][1], 'tstat_g2': rC['tstat'][2],
        'pval_g1': rC['pval'][1],   'pval_g2': rC['pval'][2],
        'gD_totalvar': rD['beta'][1], 'gD_systshare': rD['beta'][2],
        'tD_totalvar': rD['tstat'][1], 'tD_systshare': rD['tstat'][2],
        'pD_totalvar': rD['pval'][1],  'pD_systshare': rD['pval'][2],
        'vif_C': vif_c,
        'gamma_beta': rA['beta'][1],
        'tstat_beta': rA['tstat'][1],
        'n': n,
        # model (E) defaults — overwritten below if size data is available
        'r2_E': np.nan,
        'gE_size': np.nan, 'tE_size': np.nan, 'pE_size': np.nan,
        'gE_totalvar': np.nan, 'gE_systshare': np.nan,
        'tE_totalvar': np.nan, 'tE_systshare': np.nan,
        'pE_totalvar': np.nan, 'pE_systshare': np.nan,
    }

    # (E) Model D + size proxy — filter to non-NaN rows so one missing ticker
    # doesn't silently drop Model E for the whole cross-section.
    if log_dolvol is not None:
        valid = ~np.isnan(log_dolvol)
        nv = valid.sum()
        if nv >= 6:
            wv  = w[valid];  tv = total_var[valid];  ssv = syst_share[valid]
            ldv = log_dolvol[valid]
            rE  = _ols(wv, np.column_stack([np.ones(nv), tv, ssv, ldv]))
            result.update({
                'r2_E':         rE['r2'],
                'gE_totalvar':  rE['beta'][1], 'tE_totalvar':  rE['tstat'][1], 'pE_totalvar':  rE['pval'][1],
                'gE_systshare': rE['beta'][2], 'tE_systshare': rE['tstat'][2], 'pE_systshare': rE['pval'][2],
                'gE_size':      rE['beta'][3], 'tE_size':      rE['tstat'][3], 'pE_size':      rE['pval'][3],
            })

    return result


def _run_fe_models(w: np.ndarray, beta_vals: np.ndarray,
                   syst: np.ndarray, idio: np.ndarray,
                   total_var: np.ndarray,
                   log_dolvol: np.ndarray | None,
                   tickers: list) -> dict:
    """Models D and E with GICS sector FE dummies appended to the design matrix.

    Mirrors the ols_models signature so callers can pass the same arrays.
    Sector dummies are imported lazily from src.sectors (drop_first=True → InfoTech is reference).
    All-zero sector columns (sectors absent from this window) are dropped before OLS
    to avoid rank-deficient X.  Returns empty dict when n < 6.
    """
    from src.sectors import get_sector_dummies
    n = len(w)
    if n < 6:
        return {}

    sec = get_sector_dummies(list(tickers), drop_first=True).fillna(0.0)
    # drop sec_Unknown (unrecognised tickers) and all-zero columns (absent sectors)
    if 'sec_Unknown' in sec.columns:
        sec = sec.drop(columns='sec_Unknown')
    sec_vals = sec.values  # shape (n, n_sectors - 1)
    present  = sec_vals.any(axis=0)
    sec_vals = sec_vals[:, present]

    ones       = np.ones(n)
    syst_share = np.where(total_var > 1e-14, syst / total_var, 0.0)

    # ── Model D + FE ─────────────────────────────────────────────────────────
    X_D = np.column_stack([ones, total_var, syst_share, sec_vals])
    rD  = _ols(w, X_D)

    out = {
        'r2_D_fe':          rD['r2'],
        'gD_totalvar_fe':   rD['beta'][1],  'tD_totalvar_fe':  rD['tstat'][1],
        'pD_totalvar_fe':   rD['pval'][1],
        'gD_systshare_fe':  rD['beta'][2],  'tD_systshare_fe': rD['tstat'][2],
        'pD_systshare_fe':  rD['pval'][2],
        'r2_E_fe':          np.nan,
        'gE_systshare_fe':  np.nan,         'tE_systshare_fe': np.nan,
        'pE_systshare_fe':  np.nan,
    }

    # ── Model E + FE (only when size proxy available) ─────────────────────────
    if log_dolvol is not None:
        valid = ~np.isnan(log_dolvol)
        nv    = valid.sum()
        if nv >= 6:
            wv   = w[valid];  tv = total_var[valid];  ssv = syst_share[valid]
            ldv  = log_dolvol[valid]
            secv = sec_vals[valid]
            X_E  = np.column_stack([np.ones(nv), tv, ssv, ldv, secv])
            rE   = _ols(wv, X_E)
            out.update({
                'r2_E_fe':         rE['r2'],
                'gE_systshare_fe': rE['beta'][2], 'tE_systshare_fe': rE['tstat'][2],
                'pE_systshare_fe': rE['pval'][2],
            })

    return out


# ── Snapshot analysis ─────────────────────────────────────────────────────────

def run_snapshot_analysis(proxy: str = 'ew', spy_returns=None, sector_fe: bool = False):
    print('Running snapshot OLS at crisis peaks...')
    records  = []
    fe_rows  = []
    scatter  = {}  # {crisis: {est: dict}}

    for crisis, peak_date in CRISIS_PEAKS.items():
        win    = get_window(peak_date)
        decomp = decompose_variance(win, proxy, spy_returns)
        log_dv = get_log_dolvol(win.index, win.columns.tolist())
        scatter[crisis] = {}

        for est_name, est_fn in ESTIMATORS.items():
            try:
                cov   = est_fn(win)
                raw_w = gmv_weights(cov)
                if raw_w is None:
                    continue
                w = pd.Series(raw_w, index=win.columns)
            except Exception:
                continue

            common = decomp.index.intersection(w.index)
            d, wt  = decomp.loc[common], w[common].values
            ldv    = log_dv.reindex(common).values   # align size to common tickers

            res = ols_models(wt, d['beta'].values, d['syst_var'].values,
                             d['idio_var'].values, d['total_var'].values,
                             log_dolvol=ldv)
            if res is None:
                continue
            res.update({'crisis': crisis, 'estimator': est_name})
            records.append(res)

            scatter[crisis][est_name] = {
                'syst_var': d['syst_var'].values,
                'idio_var': d['idio_var'].values,
                'beta':     d['beta'].values,
                'w':        wt,
            }

            # ── sector FE comparison (optional) ──────────────────────────────
            if sector_fe:
                fe = _run_fe_models(wt, d['beta'].values, d['syst_var'].values,
                                    d['idio_var'].values, d['total_var'].values,
                                    ldv, common.tolist())
                if fe:
                    for model, regs in [
                        ('D', [('total_var',  'gD_totalvar',  'tD_totalvar', 'pD_totalvar',
                                              'gD_totalvar_fe','tD_totalvar_fe','pD_totalvar_fe', 'r2_D', 'r2_D_fe'),
                               ('syst_share', 'gD_systshare', 'tD_systshare', 'pD_systshare',
                                              'gD_systshare_fe','tD_systshare_fe','pD_systshare_fe','r2_D','r2_D_fe')]),
                        ('E',  [('syst_share', 'gE_systshare', 'tE_systshare', 'pE_systshare',
                                              'gE_systshare_fe','tE_systshare_fe','pE_systshare_fe','r2_E','r2_E_fe')]),
                    ]:
                        for (reg, cn, tn, pn, cfe, tfe, pfe, r2n, r2fe) in regs:
                            # skip Model E row if base model E was not fit
                            if model == 'E' and np.isnan(res.get(r2n, np.nan)):
                                continue
                            fe_rows.append({
                                'crisis':    crisis,
                                'estimator': est_name,
                                'model':     model,
                                'regressor': reg,
                                'coef_no_fe': res.get(cn,  np.nan),
                                't_no_fe':    res.get(tn,  np.nan),
                                'p_no_fe':    res.get(pn,  np.nan),
                                'coef_fe':    fe.get(cfe,  np.nan),
                                't_fe':       fe.get(tfe,  np.nan),
                                'p_fe':       fe.get(pfe,  np.nan),
                                'r2_no_fe':   res.get(r2n, np.nan),
                                'r2_fe':      fe.get(r2fe, np.nan),
                            })

        print(f'  {crisis} done')

    df    = pd.DataFrame(records).set_index(['crisis', 'estimator'])
    fe_df = pd.DataFrame(fe_rows) if fe_rows else None
    return df, scatter, fe_df


# ── Rolling analysis ──────────────────────────────────────────────────────────

def run_rolling_analysis(proxy: str = 'ew', spy_returns=None):
    print('\nRunning rolling OLS through crisis periods (every 5 trading days)...')
    rolling = {c: {e: [] for e in ESTIMATORS} for c in CRISIS_RANGES}

    for crisis, (c_start, c_end) in CRISIS_RANGES.items():
        all_dates = returns.loc[c_start:c_end].index
        dates     = all_dates[::5]  # sample every 5 days

        for date in dates:
            win_start = date - pd.offsets.BDay(WINDOW)
            win = returns.loc[win_start:date].dropna(axis=1)
            if win.shape[0] < WINDOW // 2 or win.shape[1] < 10:
                continue

            decomp = decompose_variance(win, proxy, spy_returns)
            log_dv = get_log_dolvol(win.index, win.columns.tolist())

            for est_name, est_fn in ESTIMATORS.items():
                try:
                    cov   = est_fn(win)
                    raw_w = gmv_weights(cov)
                    if raw_w is None:
                        continue
                    w = pd.Series(raw_w, index=win.columns)
                except Exception:
                    continue

                common = decomp.index.intersection(w.index)
                if len(common) < 10:
                    continue
                d, wt = decomp.loc[common], w[common].values
                ldv   = log_dv.reindex(common).values

                res = ols_models(wt, d['beta'].values, d['syst_var'].values,
                                 d['idio_var'].values, d['total_var'].values,
                                 log_dolvol=ldv)
                if res is None:
                    continue
                res['date'] = date
                rolling[crisis][est_name].append(res)

        print(f'  {crisis} done')

    dfs = {}
    for crisis in rolling:
        dfs[crisis] = {}
        for est in rolling[crisis]:
            rows = rolling[crisis][est]
            if rows:
                df = pd.DataFrame(rows).set_index('date')
                df.index = pd.to_datetime(df.index)
                dfs[crisis][est] = df
    return dfs


# ── Plotting ──────────────────────────────────────────────────────────────────

def _star(t: float) -> str:
    """Significance stars from t-statistic (two-sided)."""
    a = abs(t)
    return '***' if a > 3 else ('**' if a > 2 else ('*' if a > 1.65 else ''))


def plot_coef_snapshot(snap_df: pd.DataFrame):
    """Figure 1: γ₁ vs γ₂ at crisis peaks, per estimator and crisis."""
    from matplotlib.patches import Patch
    crises = list(CRISIS_PEAKS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)

    bar_w = 0.25
    x = np.arange(len(EST_LIST))

    for ax, crisis in zip(axes, crises):
        g1s, g2s, ts_g1, ts_g2 = [], [], [], []
        for est in EST_LIST:
            if (crisis, est) in snap_df.index:
                row = snap_df.loc[(crisis, est)]
                g1s.append(row['gamma1']); g2s.append(row['gamma2'])
                ts_g1.append(row['tstat_g1']); ts_g2.append(row['tstat_g2'])
            else:
                g1s.append(0); g2s.append(0); ts_g1.append(0); ts_g2.append(0)

        bars1 = ax.bar(x - bar_w/2, g1s, width=bar_w,
                       color=[EST_COLORS[e] for e in EST_LIST], alpha=0.9)
        bars2 = ax.bar(x + bar_w/2, g2s, width=bar_w,
                       color=[EST_COLORS[e] for e in EST_LIST], alpha=0.45,
                       hatch='//', edgecolor='black', linewidth=0.5)

        # Significance stars placed at bar tops (using actual bar heights)
        for bar, g, t in [(b, g, t) for b, g, t in
                           zip(list(bars1) + list(bars2),
                               g1s + g2s, ts_g1 + ts_g2)]:
            s = _star(t)
            if s:
                y_pos = bar.get_y() + bar.get_height()
                va = 'bottom' if g >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                        s, ha='center', va=va, fontsize=8)

        ax.axhline(0, color='black', linewidth=0.8)
        ax.relim()
        ax.autoscale_view()

        # R²(D) and VIF annotations placed in axes-fraction y space,
        # below the x-axis ticks, independent of data scale.
        for i, est in enumerate(EST_LIST):
            if (crisis, est) in snap_df.index:
                row = snap_df.loc[(crisis, est)]
                ax.text(x[i], -0.22,
                        f'R²(D)={row["r2_D"]:.2f}\nVIF={row["vif_C"]:.1f}',
                        transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=6.5,
                        color='#555', clip_on=False)

        ax.set_xticks(x)
        ax.set_xticklabels(EST_LIST, fontsize=10)
        ax.set_title(crisis, fontsize=12, fontweight='bold')
        ax.set_ylabel('OLS Coefficient (Model C)' if crisis == 'GFC' else '')
        ax.tick_params(axis='y', labelsize=8)

    # legend: both pattern and estimator colors
    legend_elements = [
        Patch(facecolor=EST_COLORS['Sample'],  alpha=0.9, label='Sample'),
        Patch(facecolor=EST_COLORS['LW'],      alpha=0.9, label='LW'),
        Patch(facecolor=EST_COLORS['Gerber'],  alpha=0.9, label='Gerber'),
        Patch(facecolor='gray', alpha=0.9, label='γ₁ (solid = systematic var)'),
        Patch(facecolor='gray', alpha=0.45, hatch='//', label='γ₂ (hatched = idiosyncratic var)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle('Cross-Sectional OLS: GMV Weight ~ α + γ₁·syst_var + γ₂·idio_var  (Model C)\n'
                 'R²(D) = R² of orthogonal spec (total_var + syst_share)  |  '
                 '* p<.10  ** p<.05  *** p<.01', fontsize=9, y=0.99)
    plt.tight_layout()
    out = FIGURES / f'vardec_coef_snapshot{SUFFIX}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


def plot_scatter_decomp(scatter: dict):
    """Figure 2: Scatter of syst_var and idio_var vs weight at GFC peak."""
    crisis = 'GFC'
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for c, est_name in enumerate(EST_LIST):
        if est_name not in scatter[crisis]:
            continue
        sd     = scatter[crisis][est_name]
        color  = EST_COLORS[est_name]
        syst   = sd['syst_var'];  idio = sd['idio_var'];  w = sd['w']
        syst_s = syst * 1e4;  idio_s = idio * 1e4  # scale to ×10⁻⁴

        for row, (x_vals, x_label) in enumerate([(syst_s, 'Systematic Var (×10⁻⁴)'),
                                                   (idio_s,  'Idiosyncratic Var (×10⁻⁴)')]):
            ax = axes[row][c]
            ax.scatter(x_vals, w, c=color, s=18, alpha=0.65, linewidths=0)
            ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')

            # OLS fit
            m, b, r, p, se = stats.linregress(x_vals, w)
            xf = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(xf, m * xf + b, 'k-', linewidth=1.3)
            pstr = f'p={p:.3f}' if p >= 0.001 else 'p<.001'
            ax.text(0.05, 0.95, f'γ={m:.3f}\nt={m/se:.2f}  {pstr}',
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75))

            if row == 0: ax.set_title(est_name, fontsize=12, fontweight='bold')
            if c == 0:   ax.set_ylabel(('Syst. var vs weight' if row == 0
                                         else 'Idio. var vs weight'), fontsize=9)
            ax.set_xlabel(x_label, fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle('GFC Peak (2009-03-31)  |  Variance Components vs. GMV Weight\n'
                 'Unconstrained GMV  |  EW market proxy', fontsize=11)
    plt.tight_layout()
    out = FIGURES / f'vardec_scatter_gfc{SUFFIX}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


def plot_rolling_coef(rolling_dfs: dict):
    """Figure 3: Rolling γ₁ and γ₂ through each crisis (3×2 grid)."""
    crises = list(CRISIS_RANGES.keys())
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex='row')

    col_labels = ['γ₁  (systematic var)', 'γ₂  (idiosyncratic var)']

    for r, crisis in enumerate(crises):
        for c, coef_key in enumerate(['gamma1', 'gamma2']):
            ax = axes[r][c]
            for est_name in EST_LIST:
                df = rolling_dfs[crisis].get(est_name)
                if df is None or df.empty:
                    continue
                ax.plot(df.index, df[coef_key], label=est_name,
                        color=EST_COLORS[est_name], linewidth=1.6,
                        linestyle='-' if c == 0 else '--')
            ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(labelsize=8)
            if r == 0: ax.set_title(col_labels[c], fontsize=11, fontweight='bold')
            if c == 0: ax.set_ylabel(crisis, fontsize=11, fontweight='bold')
            if r == 0 and c == 0: ax.legend(fontsize=8)

    fig.suptitle('Rolling Cross-Sectional OLS Coefficients\n'
                 'w_i = α + γ₁·syst_var_i + γ₂·idio_var_i  |  sampled every 5 trading days',
                 fontsize=11)
    plt.tight_layout()
    out = FIGURES / f'vardec_rolling_coef{SUFFIX}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


def plot_r2_comparison(snap_df: pd.DataFrame):
    """Figure 4: R² across 5 models including size proxy (E).
    Note: R²(C) >= R²(B) by construction (extra df); R²(D) is the fair test.
    R²(E) vs R²(D) shows the marginal contribution of the size proxy.
    """
    crises   = list(CRISIS_PEAKS.keys())
    models   = ['r2_A', 'r2_B', 'r2_C', 'r2_D', 'r2_E']
    m_labels = ['(A) β only', '(B) total σ²', '(C) syst+idio',
                '(D) total+syst_share', '(E) D + size proxy']
    m_colors = ['#d7191c', '#fdae61', '#1a9641', '#2166ac', '#762a83']

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    bar_w = 0.15
    x = np.arange(len(EST_LIST))

    for ax, crisis in zip(axes, crises):
        for mi, (mkey, mlabel, mcol) in enumerate(zip(models, m_labels, m_colors)):
            vals = []
            for est in EST_LIST:
                if (crisis, est) in snap_df.index:
                    v = snap_df.loc[(crisis, est), mkey]
                    vals.append(max(v, 0) if not np.isnan(v) else 0)
                else:
                    vals.append(0)
            offset = (mi - 2) * bar_w
            bars = ax.bar(x + offset, vals, width=bar_w, label=mlabel,
                          color=mcol, alpha=0.8, edgecolor='white', linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=5.5)

        ax.set_xticks(x)
        ax.set_xticklabels(EST_LIST, fontsize=10)
        ax.set_title(crisis, fontsize=12, fontweight='bold')
        ax.set_ylabel('R²' if crisis == 'GFC' else '')
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.grid(axis='y', alpha=0.3)

    axes[0].legend(fontsize=7, loc='upper right')
    fig.suptitle('R² Across Five OLS Models at Crisis Peaks\n'
                 '(E) = Model D + size proxy (log dollar volume)  |  '
                 'R²(E) − R²(D) = marginal size contribution',
                 fontsize=11)
    plt.tight_layout()
    out = FIGURES / f'vardec_r2_comparison{SUFFIX}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


def plot_size_coef_snapshot(snap_df: pd.DataFrame):
    """Figure 6: Size proxy coefficient (γ₃) at crisis peaks.

    Shows:
    - Left panel: γ₃ (log dollar volume) for each estimator × crisis
    - Right panel: syst_share coefficient in Model D vs Model E —
      does adding size absorb the composition effect?
    SIZE PROXY = log average daily dollar volume (Close × Volume).
    """
    from matplotlib.patches import Patch
    crises  = list(CRISIS_PEAKS.keys())
    bar_w   = 0.22
    x       = np.arange(len(crises))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: γ₃ (size coefficient) ──────────────────────────────────────────
    ax = axes[0]
    for i, est_name in enumerate(EST_LIST):
        gvals, tvals = [], []
        for crisis in crises:
            if (crisis, est_name) in snap_df.index:
                row = snap_df.loc[(crisis, est_name)]
                gvals.append(row['gE_size'] if not np.isnan(row['gE_size']) else 0)
                tvals.append(row['tE_size'] if not np.isnan(row['tE_size']) else 0)
            else:
                gvals.append(0); tvals.append(0)
        offset = (i - 1) * bar_w
        bars = ax.bar(x + offset, gvals, width=bar_w, label=est_name,
                      color=EST_COLORS[est_name], alpha=0.85)
        for bar, g, t in zip(bars, gvals, tvals):
            s = _star(t)
            if s:
                y_pos = bar.get_y() + bar.get_height()
                va = 'bottom' if g >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width()/2, y_pos, s,
                        ha='center', va=va, fontsize=9)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(crises, fontsize=11)
    ax.set_ylabel('γ₃ coefficient (size proxy)', fontsize=10)
    ax.set_title('Model E: Size proxy coefficient\n(log avg dollar volume)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)

    # ── Right: syst_share coefficient — Model D vs Model E ───────────────────
    ax2 = axes[1]
    bar_w2 = 0.13
    x2 = np.arange(len(crises))
    model_pairs = [
        ('gD_systshare', '(D) no size', '#2166ac', '-'),
        ('gE_systshare', '(E) + size',  '#762a83', '//'),
    ]
    for mi, (col, label, color, hatch) in enumerate(model_pairs):
        for i, est_name in enumerate(EST_LIST):
            vals = []
            for crisis in crises:
                if (crisis, est_name) in snap_df.index:
                    v = snap_df.loc[(crisis, est_name), col]
                    vals.append(v if not np.isnan(v) else 0)
                else:
                    vals.append(0)
            offset = (i * 2 + mi - 2.5) * bar_w2
            ax2.bar(x2 + offset, vals, width=bar_w2,
                    color=color, alpha=0.75, hatch=hatch if mi else None,
                    edgecolor='black', linewidth=0.4,
                    label=f'{est_name} {label}' if i == 0 else '_nolegend_')

    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(x2); ax2.set_xticklabels(crises, fontsize=11)
    ax2.set_ylabel('γ₂ (syst_share coefficient)', fontsize=10)
    ax2.set_title('syst_share coef: Model D vs Model E\n'
                  '(does size absorb the composition effect?)',
                  fontsize=10, fontweight='bold')
    ax2.legend(fontsize=7.5, loc='lower right', ncol=2)

    fig.suptitle(
        'Size Proxy Analysis  |  SIZE PROXY = log(avg daily dollar volume = Close × Volume)\n'
        'Actual market capitalisation unavailable; dollar volume used as S&P 100 size proxy',
        fontsize=10)
    plt.tight_layout()
    out = FIGURES / f'vardec_size_coef{SUFFIX}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


def plot_rolling_coef_D(rolling_dfs: dict):
    """Figure 3b: Rolling Model D coefficients (primary orthogonal spec)."""
    crises = list(CRISIS_RANGES.keys())
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex='row')

    col_labels = ['γ₁  (total_var)', 'γ₂  (syst_share = R²_mkt)']

    for r, crisis in enumerate(crises):
        for c, coef_key in enumerate(['gD_totalvar', 'gD_systshare']):
            ax = axes[r][c]
            for est_name in EST_LIST:
                df = rolling_dfs[crisis].get(est_name)
                if df is None or df.empty or coef_key not in df.columns:
                    continue
                ax.plot(df.index, df[coef_key], label=est_name,
                        color=EST_COLORS[est_name], linewidth=1.6,
                        linestyle='-' if c == 0 else '--')
            ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(labelsize=8)
            if r == 0: ax.set_title(col_labels[c], fontsize=11, fontweight='bold')
            if c == 0: ax.set_ylabel(crisis, fontsize=11, fontweight='bold')
            if r == 0 and c == 0: ax.legend(fontsize=8)

    fig.suptitle('Rolling Model (D) Coefficients  [PRIMARY SPEC — orthogonal decomposition]\n'
                 'w_i = α + γ₁·total_var_i + γ₂·syst_share_i  |  sampled every 5 trading days',
                 fontsize=11)
    plt.tight_layout()
    out = FIGURES / f'vardec_rolling_coef_D{SUFFIX}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


def plot_rolling_size_coef(rolling_dfs: dict):
    """Figure 7: Rolling size proxy coefficient (γ₃) from Model E."""
    crises = list(CRISIS_RANGES.keys())
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)

    for ax, crisis in zip(axes, crises):
        for est_name in EST_LIST:
            df = rolling_dfs[crisis].get(est_name)
            if df is None or df.empty or 'gE_size' not in df.columns:
                continue
            s = df['gE_size'].dropna()
            if s.empty:
                continue
            ax.plot(s.index, s.values, label=est_name,
                    color=EST_COLORS[est_name], linewidth=1.5)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_ylabel('γ₃ (size proxy coef)', fontsize=10)
        ax.set_title(f'{crisis}', fontsize=11, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)

    fig.suptitle(
        'Rolling Size Proxy Coefficient — Model E\n'
        'SIZE PROXY = log(avg daily dollar volume = Close × Volume)  |  not actual market cap\n'
        'w_i = α + γ₁·total_var + γ₂·syst_share + γ₃·log_dolvol',
        fontsize=10)
    plt.tight_layout()
    out = FIGURES / f'vardec_rolling_size{SUFFIX}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


def plot_ratio_rolling(rolling_dfs: dict):
    """Figure 5: |γ₁|/|γ₂| ratio rolling — when does systematic dominate?"""
    crises = list(CRISIS_RANGES.keys())
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)

    for ax, crisis in zip(axes, crises):
        for est_name in EST_LIST:
            df = rolling_dfs[crisis].get(est_name)
            if df is None or df.empty:
                continue
            ratio = np.abs(df['gamma1']) / (np.abs(df['gamma2']) + 1e-12)
            ratio = ratio.clip(0, 8)  # clip and ylim match at 8
            ax.plot(df.index, ratio, label=est_name,
                    color=EST_COLORS[est_name], linewidth=1.5)

        ax.axhline(1, color='black', linewidth=0.8, linestyle='--', label='ratio=1 (equal)')
        ax.set_ylabel('|γ₁| / |γ₂|', fontsize=10)
        ax.set_title(f'{crisis}  —  Relative sensitivity: systematic vs idiosyncratic variance',
                     fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 8)

    fig.suptitle('Rolling |γ₁|/|γ₂| Ratio\n'
                 'Values > 1 → systematic variance is relatively more penalizing than idiosyncratic',
                 fontsize=11)
    plt.tight_layout()
    out = FIGURES / f'vardec_ratio_rolling{SUFFIX}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


def plot_sector_fe_gamma2(fe_df: pd.DataFrame):
    """Bar chart: γ₂(D) = syst_share coefficient, no-FE vs sector-FE side-by-side."""
    from matplotlib.patches import Patch
    crises   = list(CRISIS_PEAKS.keys())
    sub_D    = fe_df[(fe_df['model'] == 'D') & (fe_df['regressor'] == 'syst_share')]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)
    bar_w = 0.30
    x = np.arange(len(EST_LIST))

    for ax, crisis in zip(axes, crises):
        vals_nfe, vals_fe = [], []
        t_nfe, t_fe = [], []
        for est in EST_LIST:
            row = sub_D[(sub_D['crisis'] == crisis) & (sub_D['estimator'] == est)]
            if len(row) == 1:
                vals_nfe.append(row['coef_no_fe'].values[0])
                vals_fe.append(row['coef_fe'].values[0])
                t_nfe.append(row['t_no_fe'].values[0])
                t_fe.append(row['t_fe'].values[0])
            else:
                vals_nfe.append(np.nan); vals_fe.append(np.nan)
                t_nfe.append(np.nan);   t_fe.append(np.nan)

        b1 = ax.bar(x - bar_w/2, vals_nfe, width=bar_w,
                    color=[EST_COLORS[e] for e in EST_LIST], alpha=0.9, label='No FE')
        b2 = ax.bar(x + bar_w/2, vals_fe, width=bar_w,
                    color=[EST_COLORS[e] for e in EST_LIST], alpha=0.45,
                    hatch='////', edgecolor='black', linewidth=0.5, label='Sector FE')

        for bars, tvals in [(b1, t_nfe), (b2, t_fe)]:
            for bar, t in zip(bars, tvals):
                s = _star(t)
                if s and not np.isnan(t):
                    g = bar.get_height()
                    y_pos = bar.get_y() + g
                    va = 'bottom' if g >= 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                            s, ha='center', va=va, fontsize=8)

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(EST_LIST, fontsize=10)
        ax.set_title(crisis, fontsize=12, fontweight='bold')
        if crisis == 'GFC':
            ax.set_ylabel('γ₂(D)  syst_share coefficient', fontsize=9)
        ax.tick_params(axis='y', labelsize=8)

    legend_els = [
        Patch(facecolor=EST_COLORS['Sample'], alpha=0.9, label='Sample'),
        Patch(facecolor=EST_COLORS['LW'],     alpha=0.9, label='LW'),
        Patch(facecolor=EST_COLORS['Gerber'], alpha=0.9, label='Gerber'),
        Patch(facecolor='gray', alpha=0.9,  label='No sector FE (solid)'),
        Patch(facecolor='gray', alpha=0.45, hatch='////', label='With sector FE (hatched)'),
    ]
    fig.legend(handles=legend_els, loc='upper center', ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(
        'Model D: γ₂ on syst_share — No FE vs Sector FE\n'
        'w_i = α + γ₁·total_var + γ₂·syst_share [+ sector dummies]  |  crisis peaks\n'
        '* p<.10  ** p<.05  *** p<.01',
        fontsize=9, y=0.99)
    plt.tight_layout()
    out = FIGURES / f'vardec_sector_fe_gamma2{SUFFIX}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved → {out}')
    plt.close()


def _append_sector_fe_section(fe_df: pd.DataFrame):
    """Append a 'Sector FE robustness' section to reports/variance_decomp_report{SUFFIX}.md."""
    crises   = list(CRISIS_PEAKS.keys())
    sub_D    = fe_df[(fe_df['model'] == 'D') & (fe_df['regressor'] == 'syst_share')].copy()

    def star(t):
        if np.isnan(t):
            return ''
        a = abs(t)
        return '***' if a > 3 else ('**' if a > 2 else ('*' if a > 1.65 else ''))

    # Build markdown table
    table_rows = []
    for crisis in crises:
        for est in EST_LIST:
            row = sub_D[(sub_D['crisis'] == crisis) & (sub_D['estimator'] == est)]
            if len(row) != 1:
                continue
            r = row.iloc[0]
            survived = (np.sign(r['coef_no_fe']) == np.sign(r['coef_fe'])
                        and not (np.isnan(r['coef_no_fe']) or np.isnan(r['coef_fe'])))
            sig_fe = abs(r['t_fe']) > 1.65 if not np.isnan(r['t_fe']) else False
            table_rows.append(
                f'| {crisis} | {est} '
                f'| {r["coef_no_fe"]:.4f}{star(r["t_no_fe"])} ({r["t_no_fe"]:.2f}) '
                f'| {r["r2_no_fe"]:.3f} '
                f'| {r["coef_fe"]:.4f}{star(r["t_fe"])} ({r["t_fe"]:.2f}) '
                f'| {r["r2_fe"]:.3f} '
                f'| {"✓" if survived else "FLIP"} {"sig" if sig_fe else ""} |'
            )
    table_str = '\n'.join(table_rows)

    # summary stats
    n_flip = sum(
        1 for _, r in sub_D.iterrows()
        if not (np.isnan(r['coef_no_fe']) or np.isnan(r['coef_fe']))
        and np.sign(r['coef_no_fe']) != np.sign(r['coef_fe'])
    )
    n_sig_fe = sum(
        1 for _, r in sub_D.iterrows()
        if not np.isnan(r['t_fe']) and abs(r['t_fe']) > 1.65
    )
    n_total  = len(sub_D)
    r2_gain  = (sub_D['r2_fe'] - sub_D['r2_no_fe']).mean()

    # pre-compute narrative strings to avoid nested f-strings
    flip_msg = ('No sign flips — syst_share sign is robust to sector controls.'
                if n_flip == 0
                else f'{n_flip} sign flip(s) detected — sector controls absorb part of the syst_share signal.')

    rates_sub     = sub_D[sub_D['crisis'] == 'Rates']
    n_sig_rates_nfe = int((rates_sub['t_no_fe'].abs() > 1.65).sum())
    n_sig_rates_fe  = int((rates_sub['t_fe'].abs() > 1.65).sum())
    n_rates         = len(rates_sub)
    if n_sig_rates_fe > n_sig_rates_nfe:
        rates_msg = (f'γ₂(D) strengthens after sector FE: {n_sig_rates_fe}/{n_rates} estimators '
                     f'significant (vs {n_sig_rates_nfe}/{n_rates} without FE); sector membership '
                     f'does not explain away the syst_share signal in the Rates period.')
    elif n_sig_rates_fe == n_sig_rates_nfe:
        rates_msg = (f'γ₂(D) is broadly stable: {n_sig_rates_fe}/{n_rates} estimators significant '
                     f'after sector FE (same as without FE); sector controls neither absorb nor '
                     f'rescue the syst_share signal in the Rates period.')
    else:
        rates_msg = (f'γ₂(D) weakens after sector FE: {n_sig_rates_fe}/{n_rates} estimators '
                     f'significant (vs {n_sig_rates_nfe}/{n_rates} without FE); sector concentration '
                     f'partially explains the weak cross-sectional fit in the Rates period.')

    section = f"""

---

## 8. Sector Fixed-Effects Robustness

**Purpose**: Test whether γ₂(D) — the coefficient on `syst_share` in Model D — survives
GICS sector controls. If sector membership (not idiosyncratic risk) drove the syst_share
signal, adding sector dummies should absorb it. Conversely, if γ₂(D) survives, the
variance-decomposition narrative strengthens.

**Specification**: Model D + sector dummies (11 GICS sectors; InfoTech = reference, 10 dummies).
All-zero sector columns (sectors absent from each 252-day window) are dropped before OLS.

### γ₂(D) Comparison: No FE vs Sector FE

| Crisis | Estimator | γ₂ no-FE (t) | R²(D) | γ₂ with-FE (t) | R²(D+FE) | Sign stable? |
|--------|-----------|-------------|-------|----------------|----------|--------------|
{table_str}

*Significance: * p<.10, ** p<.05, *** p<.01 (two-sided)*

### Summary

- **Sign flips after FE**: {n_flip}/{n_total} cells. {flip_msg}
- **γ₂ significant after FE** (p<.10): {n_sig_fe}/{n_total} cells.
- **Average R² gain from sector FE**: {r2_gain:+.3f} (R²(D+FE) − R²(D)); FE adds explanatory power beyond the variance decomposition.
- **Rates crisis**: {rates_msg}

*Full FE comparison table: `reports/variance_decomp_sector_fe_table{SUFFIX}.csv`*
*Figure: `results/figures/vardec_sector_fe_gamma2{SUFFIX}.png`*
"""

    report_path = REPORTS / f'variance_decomp_report{SUFFIX}.md'
    if not report_path.exists():
        print(f'WARNING: report not found at {report_path}; run without --sector-fe first')
        return
    existing = report_path.read_text(encoding='utf-8')
    if '## 8. Sector Fixed-Effects Robustness' in existing:
        # overwrite: strip the old section and re-append the fresh one
        cutoff = existing.index('\n\n---\n\n## 8. Sector Fixed-Effects Robustness')
        report_path.write_text(existing[:cutoff], encoding='utf-8')
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(section)
    print(f'appended sector FE section → {report_path}')


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(snap_df: pd.DataFrame, rolling_dfs: dict):
    """Write markdown report to reports/variance_decomp_report.md."""

    crises = list(CRISIS_PEAKS.keys())

    # Build snapshot table strings
    def fmt_coef(val, t):
        stars = '***' if abs(t) > 3 else ('**' if abs(t) > 2 else ('*' if abs(t) > 1.65 else ''))
        return f'{val:.4f}{stars}'

    table_rows = []
    for crisis in crises:
        for est in EST_LIST:
            if (crisis, est) not in snap_df.index:
                continue
            row = snap_df.loc[(crisis, est)]
            table_rows.append(
                f'| {crisis} | {est} | {fmt_coef(row["gamma1"], row["tstat_g1"])} '
                f'| {fmt_coef(row["gamma2"], row["tstat_g2"])} '
                f'| {row["r2_A"]:.3f} | {row["r2_B"]:.3f} | {row["r2_C"]:.3f} | {row["r2_D"]:.3f} '
                f'| {row["vif_C"]:.1f} | {int(row["n"])} |'
            )
    table_str = '\n'.join(table_rows)

    # Rolling summary stats per crisis × estimator
    roll_summary = []
    for crisis in crises:
        for est in EST_LIST:
            df = rolling_dfs[crisis].get(est)
            if df is None or df.empty:
                continue
            g1_mean = df['gamma1'].mean()
            g2_mean = df['gamma2'].mean()
            ratio   = (df['gamma1'].abs() / (df['gamma2'].abs() + 1e-12)).clip(0, 20).mean()
            r2_mean = df['r2_C'].mean()
            roll_summary.append(
                f'| {crisis} | {est} | {g1_mean:.4f} | {g2_mean:.4f} '
                f'| {ratio:.2f} | {r2_mean:.3f} |'
            )
    roll_str = '\n'.join(roll_summary)

    # Key findings per crisis
    findings = {}
    for crisis in crises:
        f = []
        for est in EST_LIST:
            if (crisis, est) not in snap_df.index:
                continue
            row = snap_df.loc[(crisis, est)]
            g1, g2 = row['gamma1'], row['gamma2']
            r2_gain = row['r2_C'] - row['r2_A']
            dominant = 'systematic' if abs(g1) > abs(g2) else 'idiosyncratic'
            f.append(f'  - **{est}**: γ₁={g1:.4f} (t={row["tstat_g1"]:.2f}), '
                     f'γ₂={g2:.4f} (t={row["tstat_g2"]:.2f}), '
                     f'R²(decomp)={row["r2_C"]:.3f} (+{r2_gain:+.3f} vs β-only); '
                     f'dominant component = **{dominant} variance**')
        findings[crisis] = '\n'.join(f)

    # Gerber-specific interpretation
    gerber_g1 = {c: snap_df.loc[(c, 'Gerber'), 'gamma1']
                 for c in crises if (c, 'Gerber') in snap_df.index}
    sample_g1 = {c: snap_df.loc[(c, 'Sample'), 'gamma1']
                 for c in crises if (c, 'Sample') in snap_df.index}
    gerber_stronger = sum(abs(gerber_g1[c]) > abs(sample_g1[c])
                         for c in gerber_g1 if c in sample_g1)

    # Size proxy summary
    size_sig = sum(1 for c in crises for e in EST_LIST
                   if (c, e) in snap_df.index
                   and not np.isnan(snap_df.loc[(c, e), 'pE_size'])
                   and snap_df.loc[(c, e), 'pE_size'] < 0.10)
    re_gt_rd = sum(1 for c in crises for e in EST_LIST
                   if (c, e) in snap_df.index
                   and not np.isnan(snap_df.loc[(c, e), 'r2_E'])
                   and snap_df.loc[(c, e), 'r2_E'] > snap_df.loc[(c, e), 'r2_D'])
    size_neg = sum(1 for c in crises for e in EST_LIST
                   if (c, e) in snap_df.index
                   and not np.isnan(snap_df.loc[(c, e), 'gE_size'])
                   and snap_df.loc[(c, e), 'gE_size'] < 0)

    size_table_rows = []
    for crisis in crises:
        for est in EST_LIST:
            if (crisis, est) not in snap_df.index:
                continue
            row = snap_df.loc[(crisis, est)]
            if np.isnan(row['gE_size']):
                continue
            size_table_rows.append(
                f'| {crisis} | {est} | {fmt_coef(row["gE_size"], row["tE_size"])} '
                f'| {fmt_coef(row["gE_systshare"], row["tE_systshare"])} '
                f'| {row["r2_D"]:.3f} | {row["r2_E"]:.3f} '
                f'| {(row["r2_E"] - row["r2_D"]):.3f} |'
            )
    size_table_str = '\n'.join(size_table_rows)

    # Dynamic conclusions
    neg_g1_count = sum(1 for c in crises for e in EST_LIST
                       if (c, e) in snap_df.index and snap_df.loc[(c, e), 'gamma1'] < 0)
    neg_g2_count = sum(1 for c in crises for e in EST_LIST
                       if (c, e) in snap_df.index and snap_df.loc[(c, e), 'gamma2'] < 0)
    total_cells  = sum(1 for c in crises for e in EST_LIST if (c, e) in snap_df.index)
    g1_neg_pct   = neg_g1_count / max(total_cells, 1) * 100
    g2_neg_pct   = neg_g2_count / max(total_cells, 1) * 100
    rc_gt_ra     = sum(1 for c in crises for e in EST_LIST
                       if (c, e) in snap_df.index
                       and snap_df.loc[(c, e), 'r2_C'] > snap_df.loc[(c, e), 'r2_A'])
    rd_gt_rb     = sum(1 for c in crises for e in EST_LIST
                       if (c, e) in snap_df.index
                       and snap_df.loc[(c, e), 'r2_D'] > snap_df.loc[(c, e), 'r2_B'])
    high_vif     = sum(1 for c in crises for e in EST_LIST
                       if (c, e) in snap_df.index and snap_df.loc[(c, e), 'vif_C'] > 5)
    avg_vif      = snap_df['vif_C'].mean() if 'vif_C' in snap_df.columns else float('nan')

    report = f"""# Variance Decomposition Analysis of GMV Weights

**Date**: {pd.Timestamp.today().strftime('%Y-%m-%d')}
**Estimators**: Sample covariance, Ledoit-Wolf (LW), Gerber (threshold=0.3)
**Method**: Unconstrained GMV (w ∝ Σ⁻¹1), equal-weighted market proxy
**Window**: {WINDOW} trading days

---

## 1. Methodology

### Variance Decomposition

For each asset *i*, we run a time-series OLS against the equal-weighted market return *r_m*:

$$r_{{it}} = \\alpha_i + \\beta_i r_{{mt}} + \\varepsilon_{{it}}$$

yielding:

$$\\sigma^2_i = \\underbrace{{\\beta^2_i \\sigma^2_m}}_{{\text{{systematic}}}} + \\underbrace{{\\sigma^2_{{\\varepsilon,i}}}}_{{\text{{idiosyncratic}}}}$$

### Cross-Sectional OLS

At each date, we run four models across all active assets:

| Model | Specification | Purpose |
|-------|--------------|---------|
| (A) β-only | w_i = α + γ · β_i | baseline |
| (B) total-σ² | w_i = α + γ · σ²_i | variance level only |
| (C) decomposed | w_i = α + γ₁ · syst_var_i + γ₂ · idio_var_i | raw decomposition |
| **(D) orthogonal** | **w_i = α + γ₁ · total_var_i + γ₂ · syst_share_i** | **primary spec** |

**Model (D) is the primary specification.** Since syst_var + idio_var = total_var exactly, Model (C) suffers from structural near-collinearity (VIF inflates as cross-sectional β dispersion shrinks in crises). Model (D) orthogonalizes by using total_var (the level) and syst_share = syst_var/total_var (the composition), which are genuinely distinct: VIF is typically much lower. The key test is whether γ₂ in (D) is non-zero — i.e., does the *fraction* of systematic variance matter for weights, beyond total variance alone?

Model (C) is still reported for context; its R² vs. (B) should be interpreted with caution because (C) has an extra free parameter and will mechanically fit at least as well.

### Theoretical Expectation

From the Woodbury identity applied to a single-factor covariance:

$$\\Sigma^{{-1}}\\mathbf{{1}} = D^{{-1}}\\mathbf{{1}} - D^{{-1}}\\beta\\left(\\sigma^{{-2}}_m + \\beta' D^{{-1}}\\beta\\right)^{{-1}}\\beta' D^{{-1}}\\mathbf{{1}}$$

- High-**idiosyncratic** variance → lower GMV weight (D⁻¹ term penalizes it directly)
- High-**systematic** variance → extra penalty via the β correction term
- So both γ₁ < 0 and γ₂ < 0 are expected; **|γ₁| > |γ₂|** if systematic variance is more penalizing

**Gerber hypothesis**: the Gerber statistic filters sub-threshold moves, discarding idiosyncratic small-noise. This makes its estimated covariance more "systematic-like", predicting stronger γ₁ relative to γ₂.

---

## 2. Snapshot Analysis at Crisis Peaks

**Peak dates**: GFC = 2009-03-31, COVID = 2020-04-30, Rates = 2023-01-31

### OLS Results Table

| Crisis | Estimator | γ₁ (syst) | γ₂ (idio) | R²(A) β | R²(B) σ² | R²(C)† | R²(D) | VIF(C) | N |
|--------|-----------|-----------|-----------|---------|----------|--------|-------|--------|---|
{table_str}

*Significance: * p<.10, ** p<.05, *** p<.01 (two-sided)*
*† R²(C) inflated by extra df vs (B); R²(D) is the fair comparison for composition effect.*
*VIF(C) = VIF of syst_var regressed on idio_var in Model (C); values > 5 indicate problematic collinearity.*

### Key Findings by Crisis

#### GFC (2009-03-31)
{findings.get('GFC', 'No data')}

#### COVID (2020-04-30)
{findings.get('COVID', 'No data')}

#### Rates (2023-01-31)
{findings.get('Rates', 'No data')}

---

## 3. Rolling Analysis: Within-Crisis Dynamics

### Average Coefficient Summary

| Crisis | Estimator | E[γ₁] | E[γ₂] | E[|γ₁|/|γ₂|] | E[R²(C)] |
|--------|-----------|--------|--------|--------------|---------|
{roll_str}

**Interpretation**:
- **|γ₁|/|γ₂| > 1** → systematic variance is the more penalizing component on average
- **|γ₁|/|γ₂| < 1** → idiosyncratic variance dominates the weight allocation signal
- Ratio rising over a crisis window → correlation regime shift is increasing the role of systematic risk

---

## 4. Size Proxy Analysis (Model E)

> **Note**: Actual market capitalisation (shares outstanding) is not available in the OHLCV source
> files. **Dollar volume (Close × Volume)** is used as a size proxy.  Within S&P 100, dollar volume
> correlates strongly with market cap and institutional coverage, but results should be interpreted
> with this data limitation in mind.

### Size Proxy OLS Results (Model E = D + log dollar volume)

| Crisis | Estimator | γ₃ (size) | γ₂ (syst_share, E) | R²(D) | R²(E) | ΔR² |
|--------|-----------|-----------|-------------------|-------|-------|-----|
{size_table_str}

*Significance: * p<.10, ** p<.05, *** p<.01 (two-sided)*
*γ₃ = coefficient on log(avg daily dollar volume) = size proxy*
*ΔR² = R²(E) − R²(D): marginal contribution of size proxy beyond variance decomposition*

**Size proxy findings**: γ₃ < 0 in {size_neg}/{total_cells} cells (negative = larger-cap stocks get lower GMV weight).
R²(E) > R²(D) in {re_gt_rd}/{total_cells} cells; size is significant (p<.10) in {size_sig}/{total_cells} cells.

---

## 5. Figures

| Figure | File | Description |
|--------|------|-------------|
| Fig 1 | `vardec_coef_snapshot.png` | Model C: γ₁ and γ₂ bar chart at crisis peaks per estimator |
| Fig 2 | `vardec_scatter_gfc.png` | Scatter of syst/idio var vs weight at GFC peak |
| Fig 3a | `vardec_rolling_coef.png` | Model C: rolling γ₁ and γ₂ time-series through crisis periods |
| Fig 3b | `vardec_rolling_coef_D.png` | **Model D (primary)**: rolling total_var and syst_share coefficients |
| Fig 4 | `vardec_r2_comparison.png` | R² comparison across all five models (A–E) |
| Fig 5 | `vardec_ratio_rolling.png` | Rolling |γ₁|/|γ₂| ratio (Model C) |
| Fig 6 | `vardec_size_coef.png` | Model E: γ₃ (size proxy) at crisis peaks + syst_share D vs E comparison |
| Fig 7 | `vardec_rolling_size.png` | Rolling γ₃ (size proxy coefficient) through crisis periods |

---

## 6. Estimator-Level Interpretation

### Sample Covariance
Uses the full empirical covariance without regularization. In a high-correlation crisis regime, the sample estimator may produce extreme precision matrix entries, amplifying the systematic component's influence on weights. Idiosyncratic estimates are noisy.

### Ledoit-Wolf (LW)
Shrinkage pulls the sample covariance toward a structured target (scaled identity), which **compresses the eigenvalue spread**. This reduces the penalty on large-eigenvalue (systematic) directions, potentially weakening γ₁ relative to Sample. R² improvement from decomposition should be smaller if LW already implicitly handles the decomposition via its shrinkage structure.

### Gerber (threshold=0.3)
The Gerber statistic only counts co-movements exceeding 0.3σ. Small idiosyncratic fluctuations are discarded; only large synchronized moves (systematic in nature) contribute to the correlation estimate.

**Observed result**: Gerber showed {'stronger' if gerber_stronger >= 2 else 'similar or weaker'} |γ₁| relative to Sample in {gerber_stronger}/3 crises, {'consistent with' if gerber_stronger >= 2 else 'partially supporting or not supporting'} the hypothesis that threshold filtering amplifies the systematic-variance signal.

---

## 7. Conclusions

1. **Sign of γ₁ and γ₂**: γ₁ < 0 in {g1_neg_pct:.0f}% of crisis×estimator cells; γ₂ < 0 in {g2_neg_pct:.0f}% — {'broadly consistent with' if min(g1_neg_pct, g2_neg_pct) > 70 else 'partially consistent with'} the Woodbury prediction that both variance components negatively predict GMV weight.

2. **Decomposition adds explanatory power**: R²(C) > R²(A) in {rc_gt_ra}/{total_cells} cells. Critically, R²(D) > R²(B) in {rd_gt_rb}/{total_cells} cells — meaning the *composition* effect (syst_share) explains weight variation beyond total variance level even in the collinearity-corrected specification.

3. **Collinearity in Model (C)**: Average VIF(C) = {avg_vif:.1f}; {high_vif}/{total_cells} cells have VIF > 5. {'Collinearity is a concern — rely on Model (D) for inference.' if avg_vif > 5 else 'Collinearity is moderate — Model (C) and (D) results are broadly consistent.'}

4. **Estimator heterogeneity**: |γ₁|/|γ₂| differs across estimators, especially during active crisis windows. Gerber showed {'stronger' if gerber_stronger >= 2 else 'similar or weaker'} |γ₁| relative to Sample in {gerber_stronger}/3 crises, {'consistent with' if gerber_stronger >= 2 else 'not strongly supporting'} the hypothesis that threshold filtering amplifies the systematic-variance signal.

5. **Crisis dynamics**: The rolling |γ₁|/|γ₂| ratio tends to shift during crisis windows. This is consistent with the view that during market stress, the covariance structure becomes increasingly driven by common factors, changing how estimators translate variance decomposition into weight allocation.

6. **Implication**: The variance decomposition reveals estimator-level heterogeneity that beta-only regressions miss. Choosing an estimator that is "systematic-aware" (Gerber) versus one that treats all variance symmetrically (Sample) leads to materially different weight allocations precisely when crises make systematic risk dominant.

---

*Analysis code: `variance_decomp.py` | Figures: `results/figures/vardec_*.png`*
"""

    out = REPORTS / f'variance_decomp_report{SUFFIX}.md'
    out.write_text(report, encoding='utf-8')
    print(f'saved → {out}')


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Variance decomposition OLS analysis')
    ap.add_argument('--proxy', choices=['ew', 'spy'], default='ew',
                    help='Market proxy: ew = equal-weighted (default), spy = SPY')
    ap.add_argument('--sector-fe', action='store_true', default=False,
                    help='Add GICS sector fixed effects to Models D and E (comparison only; '
                         'base figures are unchanged)')
    args, _ = ap.parse_known_args()
    PROXY     = args.proxy
    SECTOR_FE = args.sector_fe
    SUFFIX    = f'_{PROXY}' if PROXY != 'ew' else ''

    SPY_RETURNS = None
    if PROXY == 'spy':
        from src.market import load_spy_returns
        SPY_RETURNS = load_spy_returns(start='2000-01-01', end='2024-12-31')
        print(f'Market proxy: SPY ({len(SPY_RETURNS)} return observations)')
    else:
        print('Market proxy: equal-weighted (default)')

    if SECTOR_FE:
        print('Sector FE: enabled (GICS 11-sector dummies; InfoTech = reference)')

    snap_df, scatter, fe_df = run_snapshot_analysis(
        proxy=PROXY, spy_returns=SPY_RETURNS, sector_fe=SECTOR_FE)

    print('\nPlot 1: coefficient snapshot...')
    plot_coef_snapshot(snap_df)

    print('Plot 2: scatter decomposition (GFC)...')
    plot_scatter_decomp(scatter)

    print('Plot 4: R² comparison...')
    plot_r2_comparison(snap_df)

    rolling_dfs = run_rolling_analysis(proxy=PROXY, spy_returns=SPY_RETURNS)

    print('\nPlot 3a: rolling coefficients (Model C)...')
    plot_rolling_coef(rolling_dfs)

    print('Plot 3b: rolling coefficients (Model D — primary)...')
    plot_rolling_coef_D(rolling_dfs)

    print('Plot 5: rolling ratio...')
    plot_ratio_rolling(rolling_dfs)

    print('Plot 6: size proxy coefficients (snapshot)...')
    plot_size_coef_snapshot(snap_df)

    print('Plot 7: rolling size proxy coefficient...')
    plot_rolling_size_coef(rolling_dfs)

    print('\nGenerating report...')
    generate_report(snap_df, rolling_dfs)

    if SECTOR_FE and fe_df is not None:
        print('\nSaving sector FE comparison table...')
        fe_csv = REPORTS / f'variance_decomp_sector_fe_table{SUFFIX}.csv'
        fe_df.to_csv(fe_csv, index=False, float_format='%.6f')
        print(f'saved → {fe_csv}')

        print('Plot: sector FE γ₂(D) bar chart...')
        plot_sector_fe_gamma2(fe_df)

        print('Appending sector FE section to report...')
        _append_sector_fe_section(fe_df)

    print('\nSaving snapshot CSV...')
    snap_df.to_csv(REPORTS / f'vardec_snapshot_results{SUFFIX}.csv')
    print(f'saved → {REPORTS}/vardec_snapshot_results{SUFFIX}.csv')

    print('\nDone.')
