"""Crisis case study — how LW-GMV weights shift through VIX-defined crises (Tables 2, 5).

For each VIX episode (src/crises.py) the LW long-only GMV portfolio is tracked from
a calm pre-window to the VIX peak: portfolio weighted-average beta, concentration
(Effective-N, top-5 / low-beta-decile share), and the cross-sectional beta-weight
OLS slope. Descriptive per-episode narrative, not cross-crisis inference.

    python fetch_data.py            # once, if data/ is empty
    python crisis_case.py
    python crisis_case.py --proxy spy
"""
import sys, argparse, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

import statsmodels.api as sm

from src.data_loader import load_prices_from_parquet, compute_returns, load_dollar_volume, TICKERS
from src.estimators import lw_cov
from src.portfolio import gmv_long_only, effective_n
from src.market import get_market_proxy, load_spy_returns
from src.crises import load_vix, detect_vix_crises

WINDOW = 252
PRE_BUF = POST_BUF = 63   # ~3 trading months padding around each episode
STEP = 5                  # sample the timeline weekly
LOWBETA_Q = 0.10
OLS_FEATS = ['beta', 'amihud', 'log_dolvol', 'momentum']

FIGURES = Path('figures/crisis_case')
TABLES = Path('tables')
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

returns = dvol = SPY_RETURNS = None
PROXY = 'ew'


def trailing_window(end_date) -> pd.DataFrame:
    idx = returns.index
    pos = idx.searchsorted(pd.Timestamp(end_date), side='right')
    lo = max(pos - WINDOW, 0)
    if pos - lo < WINDOW // 2:
        return pd.DataFrame()
    return returns.iloc[lo:pos].dropna(axis=1)


def weights_and_beta(win: pd.DataFrame):
    if win.shape[1] < 5:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    try:
        w = pd.Series(gmv_long_only(lw_cov(win)), index=win.columns)
    except Exception:
        w = pd.Series(1.0 / win.shape[1], index=win.columns)
    mkt = get_market_proxy(win, PROXY, SPY_RETURNS)
    valid = mkt.dropna().index.intersection(win.index)
    wv, m = win.loc[valid], mkt.loc[valid]
    mv = m.var()
    beta = pd.Series({c: (wv[c].cov(m) / mv if mv > 0 else 0.0) for c in win.columns})
    return w, beta


def full_snapshot(win: pd.DataFrame) -> pd.DataFrame:
    w, beta = weights_and_beta(win)
    if w.empty:
        return pd.DataFrame()
    dv_win = dvol.reindex(index=win.index, columns=win.columns)
    rows = []
    for c in win.columns:
        r = win[c]
        dv = dv_win[c].replace(0, np.nan) if c in dv_win else pd.Series(dtype=float)
        ratio = (r.abs() / dv.reindex(r.index)).dropna()
        rows.append({
            'ticker': c, 'w': float(w.get(c, 0.0)), 'beta': float(beta.get(c, 0.0)),
            'amihud': float(ratio.mean() * 1e6) if len(ratio) > 10 else np.nan,
            'log_dolvol': float(np.log(max(dv.mean(), 1.0))) if dv.notna().any() else np.nan,
            'momentum': float(r.sum()),
        })
    out = pd.DataFrame(rows).set_index('ticker')
    return out.fillna(out.median())


def timeline_metrics(date) -> dict | None:
    win = trailing_window(date)
    if win.empty:
        return None
    w, beta = weights_and_beta(win)
    if w.empty:
        return None
    w = w / w.sum()
    common = w.index.intersection(beta.index)
    w, beta = w.loc[common], beta.loc[common]
    thr = beta.quantile(LOWBETA_Q)
    return {
        'date': pd.Timestamp(date),
        'port_beta': float((w * beta).sum()),
        'eff_n': float(effective_n(w.values)),
        'top5_share': float(w.nlargest(5).sum()),
        'lowbeta_share': float(w[beta <= thr].sum()),
    }


def beta_slope(snap: pd.DataFrame) -> float:
    """HC3 OLS w ~ feats; return the partial beta coefficient."""
    d = snap.dropna(subset=OLS_FEATS + ['w'])
    if len(d) < 20:
        return np.nan
    res = sm.OLS(d['w'], sm.add_constant(d[OLS_FEATS])).fit(cov_type='HC3')
    return res.params['beta']


def analyse_crisis(cr: pd.Series, vix: pd.Series) -> dict:
    start, end, peak = cr['start'], cr['end'], cr['peak_date']
    idx = returns.index
    p0 = max(idx.searchsorted(start) - PRE_BUF, WINDOW)
    p1 = min(idx.searchsorted(end) + POST_BUF, len(idx) - 1)
    span = idx[p0:p1:STEP]

    tl = pd.DataFrame([m for d in span if (m := timeline_metrics(d))]).set_index('date')

    # PRE = calm window ending ~PRE_BUF days before onset (always separated from the
    # peak window, which matters when VIX peaks right at onset, e.g. China-2015).
    pre_snap = full_snapshot(trailing_window(idx[p0]))
    peak_snap = full_snapshot(trailing_window(peak))

    fig_crisis(cr, tl, vix)
    return {
        'label': cr['label'], 'peak_date': peak.date(), 'peak_vix': cr['peak_vix'],
        'pre_port_beta': pre_snap['w'].mul(pre_snap['beta']).sum() if not pre_snap.empty else np.nan,
        'peak_port_beta': peak_snap['w'].mul(peak_snap['beta']).sum() if not peak_snap.empty else np.nan,
        'pre_eff_n': effective_n(pre_snap['w'].values) if not pre_snap.empty else np.nan,
        'peak_eff_n': effective_n(peak_snap['w'].values) if not peak_snap.empty else np.nan,
        'pre_beta_slope': beta_slope(pre_snap), 'peak_beta_slope': beta_slope(peak_snap),
    }


def fig_crisis(cr: pd.Series, tl: pd.DataFrame, vix: pd.Series):
    if tl.empty:
        return
    s, e, pk = cr['start'], cr['end'], cr['peak_date']
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    vseg = vix.loc[tl.index.min():tl.index.max()]

    ax = axes[0]
    ax.plot(tl.index, tl['port_beta'], color='#377eb8', lw=2, label='Portfolio β (GMV)')
    ax.set_ylabel('Portfolio β'); ax.legend(loc='upper left')
    axv = ax.twinx()
    axv.plot(vseg.index, vseg.values, color='#999', lw=1, alpha=0.7, label='VIX')
    axv.set_ylabel('VIX', color='#999'); axv.legend(loc='upper right')

    ax = axes[1]
    ax.plot(tl.index, tl['eff_n'], color='#4daf4a', lw=2, label='Effective N')
    ax.set_ylabel('Effective N'); ax.legend(loc='upper left')
    axt = ax.twinx()
    axt.plot(tl.index, tl['top5_share'], color='#e41a1c', lw=1.5, ls='--', label='Top-5 weight share')
    axt.set_ylabel('Top-5 share', color='#e41a1c'); axt.legend(loc='upper right')

    ax = axes[2]
    ax.plot(tl.index, tl['lowbeta_share'], color='#984ea3', lw=2, label='Low-β decile weight share')
    ax.set_ylabel('Low-β decile share'); ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    for a in axes:
        a.axvspan(s, e, color='orange', alpha=0.08)
        a.axvline(pk, color='red', ls=':', lw=1)
    fig.suptitle(f"Crisis case: {cr['label']}  (peak {pk.date()}, VIX {cr['peak_vix']})", fontsize=13)
    plt.tight_layout()
    safe = cr['label'].split('(')[0].strip().replace(' / ', '_').replace(' ', '_')
    fig.savefig(FIGURES / f"case_{pk.strftime('%Y%m')}_{safe}.png", dpi=150)
    plt.close(fig)


def fig_summary(summ: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [f"{r.label}\n{r.peak_date}" for r in summ.itertuples()]
    x = np.arange(len(summ))
    for ax, pre, peak, title, ylab in [
        (axes[0], 'pre_port_beta', 'peak_port_beta', 'Portfolio β: pre vs peak', 'Portfolio β'),
        (axes[1], 'pre_eff_n', 'peak_eff_n', 'Effective N: pre vs peak', 'Effective N'),
    ]:
        ax.bar(x - 0.2, summ[pre], 0.4, label='pre', color='#9ecae1')
        ax.bar(x + 0.2, summ[peak], 0.4, label='peak', color='#08519c')
        ax.set_title(title); ax.set_ylabel(ylab); ax.legend()
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    plt.tight_layout(); fig.savefig(FIGURES / 'summary_pre_vs_peak.png', dpi=150); plt.close(fig)


def main():
    global returns, dvol, SPY_RETURNS, PROXY
    ap = argparse.ArgumentParser()
    ap.add_argument('--proxy', choices=['ew', 'spy'], default='ew')
    args = ap.parse_args()
    PROXY = args.proxy

    print('Loading data...')
    prices = load_prices_from_parquet(tickers=TICKERS, start='2000-01-01', end='2024-12-31')
    returns = compute_returns(prices, method='log')
    dvol = load_dollar_volume(tickers=TICKERS, start='2000-01-01', end='2024-12-31')
    SPY_RETURNS = load_spy_returns() if PROXY == 'spy' else None
    vix = load_vix()
    crises = detect_vix_crises(vix)
    print(f'Returns: {returns.shape}  proxy={PROXY}  crises={len(crises)}\n')

    results = []
    for _, cr in crises.iterrows():
        print(f"[{cr['label']:24s} peak {cr['peak_date'].date()}] analysing...", flush=True)
        results.append(analyse_crisis(cr, vix))

    summ = pd.DataFrame(results)
    summ.round(4).to_csv(TABLES / 'crisis_summary.csv', index=False)
    fig_summary(summ)
    print('\nPRE → PEAK summary:')
    print(summ.round(3).to_string(index=False))
    print(f'\nDone. Figures → {FIGURES}/   Tables → {TABLES}/')


if __name__ == '__main__':
    main()
