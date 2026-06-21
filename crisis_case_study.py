"""
Crisis Case Study — how LW-GMV weights shift through VIX-defined crises
======================================================================
For each VIX-detected crisis episode (src/crises.py), a descriptive pre→peak→
recovery case study of the Ledoit-Wolf long-only GMV portfolio:

  (1) Threshold/structure : portfolio weighted-average beta over time — does the
                            portfolio de-risk (lower beta) into the peak?
  (2) Concentration       : Effective-N, top-5 weight share, low-beta-decile share.
  (3) Characteristic shift: w ~ beta + amihud + size + momentum (HC3) in the PRE
                            window vs the PEAK window — does the beta slope steepen
                            and does liquidity (amihud) matter more (flight to
                            liquidity)?
  (4) Named cases         : assets with the largest weight gain / loss (peak − pre).

This is a DESCRIPTIVE case study (per-episode narrative), not cross-crisis
inference — which sidesteps the small-sample / autocorrelation traps.
Theoretical lens: Clarke, de Silva & Thorley (2011) — long-only MVP weight is a
function of beta with a threshold above which securities leave the solution; that
threshold depends on market volatility, so it should move in a crisis.

Run from repo root with the `allo` env active:
    conda activate allo
    python fetch_vix.py            # once, if sp500/VIX.parquet is missing
    python crisis_case_study.py
    python crisis_case_study.py --proxy spy
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

# ── config ────────────────────────────────────────────────────────────────────
WINDOW   = 252
PRE_BUF  = 63          # ~3 months of trading days before episode start
POST_BUF = 63          # ~3 months after episode end
STEP     = 5           # sample the timeline every 5 trading days (weekly)
LOWBETA_Q = 0.10       # low-beta decile
OLS_FEATS = ['beta', 'amihud', 'log_dolvol', 'momentum']

FIGURES = Path('results/figures/crisis_case')
REPORTS = Path('reports')
FIGURES.mkdir(parents=True, exist_ok=True)

# globals
returns = None
dvol = None
SPY_RETURNS = None
PROXY = 'ew'


# ── snapshot primitives ───────────────────────────────────────────────────────

def trailing_window(end_date) -> pd.DataFrame:
    """252-td window up to and including the last trading day <= end_date."""
    idx = returns.index
    pos = idx.searchsorted(pd.Timestamp(end_date), side='right')
    lo = max(pos - WINDOW, 0)
    if pos - lo < WINDOW // 2:
        return pd.DataFrame()
    return returns.iloc[lo:pos].dropna(axis=1)


def weights_and_beta(win: pd.DataFrame):
    """(w, beta) Series for the active universe of `win`."""
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
    """w + OLS features (beta, amihud, log_dolvol, momentum), index=ticker."""
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
    """Concentration/structure metrics of the GMV portfolio at one date."""
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


# ── per-crisis analysis ───────────────────────────────────────────────────────

def ols_coef(snap: pd.DataFrame) -> dict:
    """HC3 OLS w ~ feats; return beta & amihud coefficients."""
    d = snap.dropna(subset=OLS_FEATS + ['w'])
    if len(d) < 20:
        return {f: np.nan for f in ['beta_coef', 'beta_t', 'amihud_coef', 'amihud_t']}
    res = sm.OLS(d['w'], sm.add_constant(d[OLS_FEATS])).fit(cov_type='HC3')
    return {'beta_coef': res.params['beta'], 'beta_t': res.tvalues['beta'],
            'amihud_coef': res.params['amihud'], 'amihud_t': res.tvalues['amihud']}


def analyse_crisis(cr: pd.Series, vix: pd.Series) -> dict:
    start, end, peak = cr['start'], cr['end'], cr['peak_date']
    idx = returns.index
    p0 = max(idx.searchsorted(start) - PRE_BUF, WINDOW)
    p1 = min(idx.searchsorted(end) + POST_BUF, len(idx) - 1)
    span = idx[p0:p1:STEP]

    tl = pd.DataFrame([m for d in span if (m := timeline_metrics(d))]).set_index('date')

    # PRE = calm window ending ~PRE_BUF trading days BEFORE onset (so it is always
    # separated from the peak window — matters when VIX peaks at onset, e.g. China-2015)
    pre_end = idx[p0]
    pre_snap = full_snapshot(trailing_window(pre_end))
    peak_snap = full_snapshot(trailing_window(peak))
    pre_c, peak_c = ols_coef(pre_snap), ols_coef(peak_snap)

    # top movers: Δw = w_peak − w_pre
    movers = pd.DataFrame()
    if not pre_snap.empty and not peak_snap.empty:
        common = pre_snap.index.intersection(peak_snap.index)
        movers = pd.DataFrame({
            'dw': peak_snap.loc[common, 'w'] - pre_snap.loc[common, 'w'],
            'pre_beta': pre_snap.loc[common, 'beta'],
        }).sort_values('dw')

    fig_crisis(cr, tl, vix)
    return {
        'label': cr['label'], 'peak_date': peak.date(), 'peak_vix': cr['peak_vix'],
        'pre_port_beta': pre_snap['w'].mul(pre_snap['beta']).sum() if not pre_snap.empty else np.nan,
        'peak_port_beta': peak_snap['w'].mul(peak_snap['beta']).sum() if not peak_snap.empty else np.nan,
        'pre_eff_n': effective_n(pre_snap['w'].values) if not pre_snap.empty else np.nan,
        'peak_eff_n': effective_n(peak_snap['w'].values) if not peak_snap.empty else np.nan,
        'pre_beta_coef': pre_c['beta_coef'], 'peak_beta_coef': peak_c['beta_coef'],
        'pre_amihud_coef': pre_c['amihud_coef'], 'peak_amihud_coef': peak_c['amihud_coef'],
        'movers': movers,
    }


# ── figures ───────────────────────────────────────────────────────────────────

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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = [f"{r.label}\n{r.peak_date}" for r in summ.itertuples()]
    x = np.arange(len(summ))

    ax = axes[0]
    ax.bar(x - 0.2, summ['pre_port_beta'], 0.4, label='pre', color='#9ecae1')
    ax.bar(x + 0.2, summ['peak_port_beta'], 0.4, label='peak', color='#08519c')
    ax.set_title('Portfolio β: pre vs peak'); ax.set_ylabel('Portfolio β'); ax.legend()
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)

    ax = axes[1]
    ax.bar(x - 0.2, summ['pre_eff_n'], 0.4, label='pre', color='#a1d99b')
    ax.bar(x + 0.2, summ['peak_eff_n'], 0.4, label='peak', color='#006d2c')
    ax.set_title('Effective N: pre vs peak'); ax.set_ylabel('Effective N'); ax.legend()
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)

    ax = axes[2]
    ax.bar(x - 0.2, summ['pre_amihud_coef'], 0.4, label='pre', color='#fdae6b')
    ax.bar(x + 0.2, summ['peak_amihud_coef'], 0.4, label='peak', color='#a63603')
    ax.set_title('Amihud (illiquidity) coef: pre vs peak'); ax.set_ylabel('coef'); ax.legend()
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)

    plt.tight_layout()
    fig.savefig(FIGURES / 'summary_pre_vs_peak.png', dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global returns, dvol, SPY_RETURNS, PROXY
    ap = argparse.ArgumentParser()
    ap.add_argument('--proxy', choices=['ew', 'spy'], default='ew')
    args = ap.parse_args()
    PROXY = args.proxy

    print('Loading data...')
    prices = load_prices_from_parquet('sp500', tickers=TICKERS, start='2000-01-01', end='2024-12-31')
    returns = compute_returns(prices, method='log')
    dvol = load_dollar_volume('sp500', tickers=TICKERS, start='2000-01-01', end='2024-12-31')
    SPY_RETURNS = load_spy_returns() if PROXY == 'spy' else None
    vix = load_vix()
    crises = detect_vix_crises(vix)
    print(f'Returns: {returns.shape}  proxy={PROXY}  crises={len(crises)}\n')

    results = []
    for _, cr in crises.iterrows():
        print(f"[{cr['label']:24s} peak {cr['peak_date'].date()}] analysing...", flush=True)
        results.append(analyse_crisis(cr, vix))

    summ = pd.DataFrame([{k: v for k, v in r.items() if k != 'movers'} for r in results])
    summ.to_csv(REPORTS / 'crisis_case_summary.csv', index=False)
    fig_summary(summ)
    write_report(results, summ, args)
    print(f'\nDone. Figures → {FIGURES}/   Report → {REPORTS}/crisis_case_report.md')


def _md(df: pd.DataFrame, idx_label: str) -> str:
    cols = [idx_label] + [str(c) for c in df.columns]
    out = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for i, row in zip(df.index, df.values):
        out.append('| ' + ' | '.join([str(i)] + [str(v) for v in row]) + ' |')
    return '\n'.join(out)


def write_report(results, summ, args):
    L = []; A = L.append
    A('# Crisis Case Study — LW-GMV weight shifts through VIX-defined crises\n')
    A(f'Covariance: Ledoit-Wolf  |  Market proxy: `{args.proxy}`  |  '
      f'Crises: VIX hysteresis (enter>30 / exit<20), {len(summ)} episodes  |  '
      f'Pre = 252-td window ending ~{PRE_BUF} td before onset (calm baseline); '
      'Peak = 252-td window at VIX peak.\n')
    A('Theoretical lens: Clarke, de Silva & Thorley (2011) — long-only MVP weight is a '
      'function of beta with a volatility-dependent threshold (see '
      '`references_weight_explain.md`). Descriptive per-episode case study.\n')

    A('## Cross-crisis summary — pre vs peak\n')
    disp = summ[['label', 'peak_date', 'peak_vix', 'pre_port_beta', 'peak_port_beta',
                 'pre_eff_n', 'peak_eff_n', 'pre_amihud_coef', 'peak_amihud_coef']].copy()
    disp = disp.round(3).set_index('label')
    A(_md(disp, 'crisis'))
    A('\n- **Portfolio β pre→peak**: does the GMV portfolio de-risk into the crisis peak?\n'
      '- **Effective N pre→peak**: does it concentrate?\n'
      '- **Amihud coef pre→peak**: more negative/positive at peak ⇒ liquidity matters more '
      '(flight to liquidity).\n')

    for r in results:
        A(f"\n## {r['label']} — peak {r['peak_date']} (VIX {r['peak_vix']})\n")
        A(f"- Portfolio β: {r['pre_port_beta']:.3f} → {r['peak_port_beta']:.3f}  |  "
          f"Effective N: {r['pre_eff_n']:.1f} → {r['peak_eff_n']:.1f}\n")
        A(f"- β-weight slope: {r['pre_beta_coef']:.4f} → {r['peak_beta_coef']:.4f}  |  "
          f"amihud coef: {r['pre_amihud_coef']:.3f} → {r['peak_amihud_coef']:.3f}\n")
        mv = r['movers']
        if not mv.empty:
            losers = mv.head(3); gainers = mv.tail(3).iloc[::-1]
            g = ', '.join(f'{t} (+{row.dw*100:.1f}%, β{row.pre_beta:.2f})' for t, row in gainers.iterrows())
            l = ', '.join(f'{t} ({row.dw*100:.1f}%, β{row.pre_beta:.2f})' for t, row in losers.iterrows())
            A(f"- Top weight **gainers**: {g}\n")
            A(f"- Top weight **losers**: {l}\n")
        A(f"- Figure: `results/figures/crisis_case/`\n")

    A('\n## Limitations\n')
    A('- **Descriptive, not inferential** — per-episode characterization, not a test across crises.\n')
    A('- **Survivorship bias** — 2024 S&P 100 universe; GFC episodes miss failed financials '
      '(Lehman, Bear, WaMu), so GFC weight dynamics are partial.\n')
    A('- **Beta endogeneity** — betas are estimated on crisis-contaminated windows (betas '
      'spike/compress in crises); interpret the β-weight relationship descriptively.\n')
    A('- **Static 2024 GICS**; window-overlap between pre and peak snapshots.\n')
    (REPORTS / 'crisis_case_report.md').write_text('\n'.join(L))


if __name__ == '__main__':
    main()
