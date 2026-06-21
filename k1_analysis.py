"""
K-1 Model Analysis: Model K + cross_corr (2015-2024)
=====================================================
Model K-1 adds cross_corr (섹터 간 평균 상관) to Model K:
  (K)   w = α + γ₁·total_var + γ₂·syst_share + γ₃·corr_min
  (K-1) w = α + γ₁·total_var + γ₂·syst_share + γ₃·corr_min + γ₄·cross_corr

Same rolling setup as lw10_full_analysis.py (252-day window, monthly, 2015-2024).
Also runs all LW10 base models for comparison.

Outputs:
  reports/2026-05-27_K-1.md
  results/figures/k1/       (figures)
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
from scipy import stats

from src.data_loader import load_prices_from_parquet, compute_returns, TICKERS
from src.estimators import lw_cov
from src.market import get_market_proxy
from src.portfolio import effective_n
from src.sectors import GICS_SECTORS

np.random.seed(42)

WINDOW  = 252
FIGURES = Path('results/figures/k1')
REPORTS = Path('reports')
FIGURES.mkdir(parents=True, exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
print('Loading data...')
prices  = load_prices_from_parquet('sp500', tickers=TICKERS,
                                    start='2014-01-01', end='2024-12-31')
returns = compute_returns(prices, method='log')
print(f'Full returns: {returns.shape}')


# ── helpers ───────────────────────────────────────────────────────────────────
def _ols(y, X):
    n, k = X.shape
    try:
        Q, R    = np.linalg.qr(X)
        beta    = np.linalg.solve(R, Q.T @ y)
        XtX_inv = np.linalg.inv(R) @ np.linalg.inv(R).T
    except np.linalg.LinAlgError:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        XtX_inv = np.linalg.pinv(X.T @ X)
    yhat   = X @ beta
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 1e-14 else 0.0
    dof    = n - k
    if dof > 0 and ss_res > 1e-14:
        se    = np.sqrt(np.maximum(np.diag(XtX_inv) * ss_res / dof, 0))
        tstat = beta / np.where(se > 1e-14, se, np.nan)
        pval  = 2 * (1 - stats.t.cdf(np.abs(tstat), df=dof))
    else:
        se = tstat = pval = np.full(k, np.nan)
    adj_r2 = max(1 - (1 - r2) * (n - 1) / (n - k), 0.0) if n > k else 0.0
    return dict(beta=beta, tstat=tstat, pval=pval, r2=r2, adj_r2=adj_r2, n=n)


def gmv_lw(win):
    cov = lw_cov(win)
    try:    prec = np.linalg.inv(cov)
    except: prec = np.linalg.pinv(cov)
    raw = prec @ np.ones(cov.shape[0])
    s   = raw.sum()
    return raw / s if abs(s) > 1e-10 else None


def compute_features(win):
    mkt     = get_market_proxy(win, 'ew')
    valid   = mkt.dropna().index.intersection(win.index)
    if len(valid) < 30:
        return pd.DataFrame()
    win     = win.loc[valid]
    mkt     = mkt.loc[valid]
    mkt_var = mkt.var()
    if mkt_var < 1e-14:
        return pd.DataFrame()

    tickers   = list(win.columns)
    cov_mat   = win.cov().values
    diag_vars = np.diag(cov_mat)

    # PC1
    eigval, eigvec = np.linalg.eigh(cov_mat)
    lambda1 = eigval[-1]
    pc1_vec = eigvec[:, -1]
    pc1_share = np.where(diag_vars > 1e-14,
                         np.clip(lambda1 * pc1_vec**2 / diag_vars, 0, 1), 0.0)

    # Correlation matrices
    corr_mat = win.corr().values.copy()
    np.fill_diagonal(corr_mat, np.nan)
    avg_corr_arr = np.nanmean(corr_mat, axis=1)
    corr_min_arr = np.nanmin(corr_mat, axis=1)

    # Sector within/cross correlation
    sector_of  = [GICS_SECTORS.get(t, 'Unknown') for t in tickers]
    within_arr = np.full(len(tickers), np.nan)
    cross_arr  = np.full(len(tickers), np.nan)
    for i, sec in enumerate(sector_of):
        same  = [j for j in range(len(tickers)) if j != i and sector_of[j] == sec]
        other = [j for j in range(len(tickers)) if sector_of[j] != sec]
        within_arr[i] = np.nanmean(corr_mat[i, same])  if same  else avg_corr_arr[i]
        cross_arr[i]  = np.nanmean(corr_mat[i, other]) if other else avg_corr_arr[i]

    rows = []
    for i, col in enumerate(tickers):
        tv = float(diag_vars[i])
        if tv < 1e-14:
            continue
        r      = win[col]
        cov_rm = float(r.cov(mkt))
        beta   = cov_rm / mkt_var
        sv     = beta**2 * mkt_var
        iv     = max(tv - sv, 1e-14)
        raw_r2 = min(beta**2 * mkt_var / tv, 1.0)
        syst_share = max(1 - (1 - raw_r2) * (len(win) - 1) / (len(win) - 2), 0.0)

        r_arr = r.values
        neg_r = r_arr[r_arr < 0]
        dv    = float(np.std(neg_r)) if len(neg_r) > 5 else np.nan

        rows.append(dict(
            ticker        = col,
            total_var     = tv,
            syst_share    = syst_share,
            inv_idio_var  = 1.0 / iv,
            downside_vol  = dv,
            avg_corr      = avg_corr_arr[i],
            corr_min      = corr_min_arr[i],
            pc1_var_share = pc1_share[i],
            within_corr   = within_arr[i],
            cross_corr    = cross_arr[i],
            sector        = sector_of[i],
        ))
    return pd.DataFrame(rows).set_index('ticker')


def run_models(feat, w_arr, tickers):
    common = feat.index.intersection(tickers)
    if len(common) < 8:
        return {}
    f   = feat.loc[common]
    idx = pd.Index(tickers).get_indexer(common)
    wt  = w_arr[idx]
    n   = len(wt)
    o   = np.ones(n)

    tv  = f['total_var'].values
    ss  = f['syst_share'].values
    dv  = f['downside_vol'].values
    iiv = f['inv_idio_var'].values
    ac  = f['avg_corr'].values
    cm  = f['corr_min'].values
    p1  = f['pc1_var_share'].values
    wc  = f['within_corr'].values
    xc  = f['cross_corr'].values

    def _fit(cols):
        mask = ~np.any(np.isnan(np.column_stack(cols + [wt])), axis=1)
        if mask.sum() < 8:
            return None
        X = np.column_stack([o[mask]] + [c[mask] for c in cols])
        return _ols(wt[mask], X)

    sectors    = f['sector'].values
    all_secs   = sorted(set(sectors) - {'InfoTech'})
    sec_dummies = np.column_stack([
        (sectors == s).astype(float) for s in all_secs
    ]) if all_secs else np.zeros((n, 0))

    def _fit_fe(base_cols):
        mask = ~np.any(np.isnan(np.column_stack(base_cols + [wt])), axis=1)
        if mask.sum() < 8:
            return None
        X = np.column_stack([o[mask]] + [c[mask] for c in base_cols] + [sec_dummies[mask]])
        return _ols(wt[mask], X)

    return {
        'D':     _fit([tv, ss]),
        'D_sig': _fit([tv, dv]),
        'K':     _fit([tv, ss, cm]),
        'K1':    _fit([tv, ss, cm, xc]),   # Model K-1: K + cross_corr
        'L':     _fit([tv, p1, ac]),
        'W':     _fit([tv, iiv]),
        'F':     _fit([dv, iiv, ac, cm]),
        'D_FE':  _fit_fe([tv, ss]),
        'L_FE':  _fit_fe([tv, p1, ac]),
        'M':     _fit([tv, wc, xc]),
    }


# ── rolling loop ──────────────────────────────────────────────────────────────
all_dates = returns.loc['2015-01-01':'2024-12-31'].index
monthly   = pd.date_range(all_dates[0] + pd.offsets.BDay(WINDOW),
                           all_dates[-1], freq='BME')

MODEL_NAMES = ['D', 'D_sig', 'K', 'K1', 'L', 'W', 'F', 'D_FE', 'L_FE', 'M']
MODEL_COEF_NAMES = {
    'D':     ['intercept', 'total_var', 'syst_share'],
    'D_sig': ['intercept', 'total_var', 'downside_vol'],
    'K':     ['intercept', 'total_var', 'syst_share', 'corr_min'],
    'K1':    ['intercept', 'total_var', 'syst_share', 'corr_min', 'cross_corr'],
    'L':     ['intercept', 'total_var', 'pc1_share', 'avg_corr'],
    'W':     ['intercept', 'total_var', 'inv_idio_var'],
    'F':     ['intercept', 'downside_vol', 'inv_idio_var', 'avg_corr', 'corr_min'],
    'D_FE':  ['intercept', 'total_var', 'syst_share', '+sector_FE'],
    'L_FE':  ['intercept', 'total_var', 'pc1_share', 'avg_corr', '+sector_FE'],
    'M':     ['intercept', 'total_var', 'within_corr', 'cross_corr'],
}

MODEL_FORMULA = {
    'D':     'w = α + γ₁·total_var + γ₂·syst_share',
    'D_sig': 'w = α + γ₁·total_var + γ₂·downside_vol',
    'K':     'w = α + γ₁·total_var + γ₂·syst_share + γ₃·corr_min',
    'K1':    'w = α + γ₁·total_var + γ₂·syst_share + γ₃·corr_min + γ₄·cross_corr',
    'L':     'w = α + γ₁·total_var + γ₂·pc1_share + γ₃·avg_corr',
    'W':     'w = α + γ₁·total_var + γ₂·inv_idio_var',
    'F':     'w = α + γ₁·downside_vol + γ₂·inv_idio_var + γ₃·avg_corr + γ₄·corr_min',
    'D_FE':  'w = α + γ₁·total_var + γ₂·syst_share + Σ섹터더미',
    'L_FE':  'w = α + γ₁·total_var + γ₂·pc1_share + γ₃·avg_corr + Σ섹터더미',
    'M':     'w = α + γ₁·total_var + γ₂·within_corr + γ₃·cross_corr',
}

BASE_MODELS = ['D', 'D_sig', 'K', 'K1', 'L', 'W', 'F', 'M']
FE_MODELS   = ['D_FE', 'L_FE']

MODEL_COLORS = {
    'D': '#4575b4', 'D_sig': '#74add1',
    'K': '#f46d43', 'K1': '#d73027',
    'L': '#a50026', 'W': '#fdae61',     'F': '#1a9641',
    'D_FE': '#8856a7', 'L_FE': '#df65b0', 'M': '#2ca25f',
}

adjr2_ts = {m: [] for m in MODEL_NAMES}
coef_ts  = {m: [] for m in BASE_MODELS}
tstat_ts = {m: [] for m in BASE_MODELS}
effn_ts    = []
dates_used = []

print(f'Rolling OLS across {len(monthly)} monthly windows...')
for t, date in enumerate(monthly):
    end   = date
    start = end - pd.offsets.BDay(WINDOW)
    win   = returns.loc[start:end].dropna(axis=1)
    if win.shape[1] < 10:
        continue

    w_arr = gmv_lw(win)
    if w_arr is None:
        continue

    w_lo = np.maximum(w_arr, 0)
    s    = w_lo.sum()
    effn_ts.append(effective_n(pd.Series(w_lo / s)) if s > 1e-10 else np.nan)

    feat  = compute_features(win)
    res   = run_models(feat, w_arr, list(win.columns))
    dates_used.append(date)

    for m in MODEL_NAMES:
        r = res.get(m)
        adjr2_ts[m].append(r['adj_r2'] if r else np.nan)

    for m in BASE_MODELS:
        r = res.get(m)
        if r is None:
            coef_ts[m].append(np.full(len(MODEL_COEF_NAMES[m]), np.nan))
            tstat_ts[m].append(np.full(len(MODEL_COEF_NAMES[m]), np.nan))
        else:
            coef_ts[m].append(r['beta'])
            tstat_ts[m].append(r['tstat'])

    if (t + 1) % 20 == 0:
        print(f'  {t+1}/{len(monthly)}  {date.date()}')

dates_idx   = pd.DatetimeIndex(dates_used)
effn_series = pd.Series(effn_ts, index=dates_idx)
adjr2_df    = pd.DataFrame({m: adjr2_ts[m] for m in MODEL_NAMES}, index=dates_idx)

coef_df  = {}
tstat_df = {}
for m in BASE_MODELS:
    names = MODEL_COEF_NAMES[m][1:]
    arr_c = np.array(coef_ts[m])
    arr_t = np.array(tstat_ts[m])
    coef_df[m]  = pd.DataFrame(arr_c[:, 1:len(names)+1], index=dates_idx, columns=names)
    tstat_df[m] = pd.DataFrame(arr_t[:, 1:len(names)+1], index=dates_idx, columns=names)

print(f'\nTotal windows: {len(dates_idx)}')
for m in MODEL_NAMES:
    valid = adjr2_df[m].dropna()
    print(f'  Model {m}: mean adj-R²={valid.mean():.3f}  median={valid.median():.3f}')


# ── aggregate statistics ──────────────────────────────────────────────────────
def summarise(m):
    valid = adjr2_df[m].dropna()
    out   = dict(n_windows=len(valid), mean_r2=float(valid.mean()),
                 median_r2=float(valid.median()))
    if m in BASE_MODELS:
        c_df = coef_df[m].dropna()
        t_df = tstat_df[m].dropna()
        out['mean_coef']  = c_df.mean()
        out['mean_tstat'] = t_df.mean()
        out['pct_sig']    = (t_df.abs() > 1.645).mean() * 100
    else:
        out['mean_coef']  = pd.Series(dtype=float)
        out['mean_tstat'] = pd.Series(dtype=float)
        out['pct_sig']    = pd.Series(dtype=float)
    return out

summary = {m: summarise(m) for m in MODEL_NAMES}


# ── figures ───────────────────────────────────────────────────────────────────

# Fig 1: rolling adj-R² 전체 모형
fig, ax = plt.subplots(figsize=(14, 5))
for m in MODEL_NAMES:
    s = adjr2_df[m].rolling(3, center=True).mean()
    ax.plot(dates_idx, s, label=f'Model {m}', color=MODEL_COLORS[m], lw=1.3)
ax.set_ylabel('adj-R² (3-month smoothed)')
ax.set_title('LW (2015-2024) — Rolling Cross-Sectional adj-R²  [K-1 실험]', fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend(ncol=3, fontsize=9)
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(FIGURES / 'k1_rolling_adjr2.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: k1_rolling_adjr2.png')


# Fig 2: K vs K-1 직접 비교
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax0 = axes[0]
ax0.plot(dates_idx, adjr2_df['K'].rolling(3, center=True).mean(),
         label='K', color=MODEL_COLORS['K'], lw=1.4)
ax0.plot(dates_idx, adjr2_df['K1'].rolling(3, center=True).mean(),
         label='K-1 (+cross_corr)', color=MODEL_COLORS['K1'], lw=1.4, ls='--')
ax0.fill_between(dates_idx,
                  adjr2_df['K'].rolling(3, center=True).mean(),
                  adjr2_df['K1'].rolling(3, center=True).mean(),
                  alpha=0.2, color='gray', label='K-1 − K')
ax0.set_ylabel('adj-R² (3-month smoothed)')
ax0.set_title('Model K vs K-1 — adj-R² 비교', fontweight='bold')
ax0.legend(fontsize=9)
ax0.grid(alpha=0.25)

ax1 = axes[1]
delta = (adjr2_df['K1'] - adjr2_df['K']).rolling(3, center=True).mean()
ax1.plot(dates_idx, delta, color='#d73027', lw=1.2)
ax1.axhline(0, color='black', lw=0.7, ls='--')
ax1.fill_between(dates_idx, 0, delta, where=delta > 0, alpha=0.3, color='#d73027', label='K-1 우위')
ax1.fill_between(dates_idx, 0, delta, where=delta < 0, alpha=0.3, color='#4575b4', label='K 우위')
ax1.set_ylabel('Δadj-R² (K1 − K)')
ax1.set_title('K-1 추가 설명력', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.25)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.xaxis.set_major_locator(mdates.YearLocator())
plt.tight_layout()
fig.savefig(FIGURES / 'k1_vs_k_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: k1_vs_k_comparison.png')


# Fig 3: K-1 모형 rolling 계수
fig, ax = plt.subplots(figsize=(14, 5))
for col in coef_df['K1'].columns:
    ax.plot(dates_idx, coef_df['K1'][col].rolling(3, center=True).mean(),
            label=col, lw=1.2)
ax.axhline(0, color='black', lw=0.7, ls='--')
ax.set_title('Model K-1 — Rolling Coefficients (3-month smoothed)', fontweight='bold')
ax.legend(ncol=4, fontsize=9)
ax.grid(alpha=0.25)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.tight_layout()
fig.savefig(FIGURES / 'k1_rolling_coef.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: k1_rolling_coef.png')


# Fig 4: % windows significant per variable (K, K-1 comparison only)
fig, ax = plt.subplots(figsize=(10, 4))
x_pos = 0
xtick_pos, xtick_lbl = [], []
for m in ['K', 'K1']:
    ps = summary[m]['pct_sig']
    for col in ps.index:
        ax.bar(x_pos, ps[col], color=MODEL_COLORS[m], edgecolor='white', width=0.8)
        xtick_pos.append(x_pos)
        xtick_lbl.append(f'{m}\n{col}')
        x_pos += 1
    x_pos += 0.5
ax.axhline(50, color='black', ls='--', lw=0.8, label='50%')
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_lbl, fontsize=8, rotation=45, ha='right')
ax.set_ylabel('% of windows |t| > 1.645')
ax.set_title('K vs K-1 — 유의 창 비율 비교', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(FIGURES / 'k1_pct_sig.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: k1_pct_sig.png')


# Fig 5: adj-R² 모형 비교 바 차트 (주요 모형만)
fig, ax = plt.subplots(figsize=(12, 4))
compare_models = ['D', 'K', 'K1', 'L', 'F', 'M']
bars = ax.bar(compare_models,
              [summary[m]['mean_r2'] for m in compare_models],
              color=[MODEL_COLORS[m] for m in compare_models],
              edgecolor='white', width=0.6)
for bar, m in zip(bars, compare_models):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.003,
            f'{summary[m]["mean_r2"]:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('평균 adj-R²')
ax.set_title('주요 모형 평균 adj-R² 비교 (LW, 2015-2024)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
# highlight K and K1
for bar, m in zip(bars, compare_models):
    if m in ('K', 'K1'):
        bar.set_edgecolor('black')
        bar.set_linewidth(1.5)
plt.tight_layout()
fig.savefig(FIGURES / 'k1_adjr2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: k1_adjr2_comparison.png')


# Fig 6: K-1 mean t-stat bar
fig, ax = plt.subplots(figsize=(8, 4))
coef_names = list(summary['K1']['mean_tstat'].index)
tvals = [summary['K1']['mean_tstat'][c] for c in coef_names]
colors = ['#d73027' if t < 0 else '#4575b4' for t in tvals]
ax.bar(coef_names, tvals, color=colors, edgecolor='white', width=0.6)
ax.axhline(1.645,  color='gray', ls='--', lw=0.8, label='|t|=1.645')
ax.axhline(-1.645, color='gray', ls='--', lw=0.8)
ax.axhline(0,      color='black', lw=0.7)
ax.set_ylabel('평균 t-통계량')
ax.set_title('Model K-1 — 변수별 평균 t-통계량 (전기간)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
fig.savefig(FIGURES / 'k1_mean_tstat.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: k1_mean_tstat.png')


# ── write report ──────────────────────────────────────────────────────────────
def star(t):
    if pd.isna(t): return ''
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.960 else ('*' if a > 1.645 else ''))

n_win = len(dates_idx)
lines = []
A = lines.append

A("# Model K-1 실험 보고서 — cross_corr 추가 효과")
A("")
A("**작성일**: 2026-05-27  ")
A("**데이터**: S&P 100, 2015-01-01 ~ 2024-12-31  ")
A("**추정기**: Ledoit-Wolf (LW) 단독  ")
A("**분석 방법**: 월별 롤링 252거래일 창 전체 집계 (위기 구분 없음)  ")
A(f"**총 분석 창 수**: {n_win}개  ")
A("**기준 보고서**: 2026-05-26_LW10.md  ")
A("")
A("---")
A("")
A("## 1. 실험 설계")
A("")
A("### 1.1 핵심 질문")
A("")
A("Model K에 섹터 간 평균 상관(`cross_corr`)을 추가(Model K-1)하면 설명력이 개선되는가?  ")
A("Model M(total_var + within_corr + cross_corr)에서 cross_corr의 강한 유의성(평균 t=-3.58***)이  ")
A("syst_share·corr_min과 함께 투입해도 유지되는지 확인한다.")
A("")
A("### 1.2 모형 정의")
A("")
A("| 모형 | 수식 | 비고 |")
A("|------|------|------|")
A(f"| (K) | {MODEL_FORMULA['K']} | 기준 모형 |")
A(f"| (K-1) | {MODEL_FORMULA['K1']} | 실험 모형 (+cross_corr) |")
A("")
A("### 1.3 비교 맥락 (LW10 전체 모형)")
A("")
A("| 모형 | 수식 |")
A("|------|------|")
for m in ['D', 'D_sig', 'K', 'K1', 'L', 'W', 'F']:
    tag = ' ← **실험**' if m == 'K1' else (' ← 기준' if m == 'K' else '')
    A(f"| ({m}) | {MODEL_FORMULA[m]}{tag} |")
A("")
A("---")
A("")
A("## 2. 핵심 결과: K vs K-1")
A("")

k_r2  = summary['K']['mean_r2']
k1_r2 = summary['K1']['mean_r2']
delta_k = k1_r2 - k_r2

A("### 2.1 adj-R² 비교")
A("")
A("| 모형 | 평균 adj-R² | 중앙값 adj-R² | 유효 창 수 | K 대비 Δ |")
A("|------|:-----------:|:------------:|:---------:|:--------:|")
A(f"| (K) | {k_r2:.3f} | {summary['K']['median_r2']:.3f} | {summary['K']['n_windows']} | — |")
A(f"| (K-1) | {k1_r2:.3f} | {summary['K1']['median_r2']:.3f} | {summary['K1']['n_windows']} | {delta_k:+.3f} |")
A("")

if delta_k > 0.005:
    interpretation = f"K-1이 K 대비 평균 adj-R² +{delta_k:.3f} 개선. cross_corr이 syst_share·corr_min 통제 후에도 독립적 설명력을 가진다."
elif delta_k > 0:
    interpretation = f"K-1이 K 대비 소폭 개선(+{delta_k:.3f}). cross_corr의 한계적 기여는 미미하다."
else:
    interpretation = f"K-1이 K 대비 오히려 감소({delta_k:+.3f}). cross_corr 추가 시 다중공선성 또는 과적합 효과."
A(f"**해석**: {interpretation}")
A("")

A("### 2.2 Model K-1 계수 및 t-통계량")
A("")
A("**Model (K-1)**: " + MODEL_FORMULA['K1'])
A("")
A("| 변수 | 평균 계수 | 평균 t-stat | 유의 창 비율 |")
A("|------|----------:|:-----------:|:-----------:|")
for col in summary['K1']['mean_coef'].index:
    mc = summary['K1']['mean_coef'][col]
    mt = summary['K1']['mean_tstat'][col]
    ps = summary['K1']['pct_sig'][col]
    A(f"| {col} | {mc:.4e} | {mt:.2f}{star(mt)} | {ps:.1f}% |")
A("")

A("### 2.3 Model K 계수 참조")
A("")
A("**Model (K)**: " + MODEL_FORMULA['K'])
A("")
A("| 변수 | 평균 계수 | 평균 t-stat | 유의 창 비율 |")
A("|------|----------:|:-----------:|:-----------:|")
for col in summary['K']['mean_coef'].index:
    mc = summary['K']['mean_coef'][col]
    mt = summary['K']['mean_tstat'][col]
    ps = summary['K']['pct_sig'][col]
    A(f"| {col} | {mc:.4e} | {mt:.2f}{star(mt)} | {ps:.1f}% |")
A("")

A("### 2.4 cross_corr 계수 변화 (M → K-1)")
A("")
A("Model M에서의 cross_corr과 Model K-1에서의 cross_corr을 비교한다.")
A("")
A("| 모형 | 수식 | cross_corr 평균 계수 | 평균 t-stat | 유의 창 비율 |")
A("|------|------|--------------------:|:-----------:|:-----------:|")
if 'cross_corr' in summary['M']['mean_coef'].index:
    mc_M = summary['M']['mean_coef']['cross_corr']
    mt_M = summary['M']['mean_tstat']['cross_corr']
    ps_M = summary['M']['pct_sig']['cross_corr']
    A(f"| (M) | {MODEL_FORMULA['M']} | {mc_M:.4e} | {mt_M:.2f}{star(mt_M)} | {ps_M:.1f}% |")
if 'cross_corr' in summary['K1']['mean_coef'].index:
    mc_K1 = summary['K1']['mean_coef']['cross_corr']
    mt_K1 = summary['K1']['mean_tstat']['cross_corr']
    ps_K1 = summary['K1']['pct_sig']['cross_corr']
    A(f"| (K-1) | {MODEL_FORMULA['K1']} | {mc_K1:.4e} | {mt_K1:.2f}{star(mt_K1)} | {ps_K1:.1f}% |")
A("")

A("---")
A("")
A("## 3. 전체 모형 adj-R² 비교")
A("")
A("| 모형 | 평균 adj-R² | 중앙값 adj-R² | 유효 창 수 | 섹터 FE |")
A("|------|:-----------:|:------------:|:---------:|:-------:|")
for m in MODEL_NAMES:
    s  = summary[m]
    fe = '✓' if m in FE_MODELS else ''
    tag = ' **← 실험**' if m == 'K1' else (' ← 기준' if m == 'K' else '')
    A(f"| ({m}){tag} | {s['mean_r2']:.3f} | {s['median_r2']:.3f} | {s['n_windows']} | {fe} |")
A("")

A("---")
A("")
A("## 4. 핵심 발견")
A("")

best_m  = max(MODEL_NAMES, key=lambda m: summary[m]['mean_r2'])
worst_m = min(MODEL_NAMES, key=lambda m: summary[m]['mean_r2'])

findings = []

# K vs K-1
mt_xc_K1 = summary['K1']['mean_tstat'].get('cross_corr', np.nan)
ps_xc_K1 = summary['K1']['pct_sig'].get('cross_corr', np.nan)
mc_xc_K1 = summary['K1']['mean_coef'].get('cross_corr', np.nan)
findings.append(
    f"**K-1 핵심 결과**: cross_corr 추가 시 adj-R² {delta_k:+.3f} ({k_r2:.3f} → {k1_r2:.3f}). "
    f"cross_corr 평균 t={mt_xc_K1:.2f}{star(mt_xc_K1)}, 유의 창 비율 {ps_xc_K1:.1f}%. "
    + ("syst_share·corr_min 통제 후에도 cross_corr이 독립적 음의 효과를 유지한다."
       if mc_xc_K1 < 0 and abs(mt_xc_K1) > 1.645 else
       "syst_share·corr_min과의 상관으로 인해 cross_corr 독립 효과가 약화된다.")
)

# corr_min stability
mt_cm_K  = summary['K']['mean_tstat'].get('corr_min', np.nan)
mt_cm_K1 = summary['K1']['mean_tstat'].get('corr_min', np.nan)
ps_cm_K  = summary['K']['pct_sig'].get('corr_min', np.nan)
ps_cm_K1 = summary['K1']['pct_sig'].get('corr_min', np.nan)
findings.append(
    f"**corr_min 안정성**: K에서 평균 t={mt_cm_K:.2f}{star(mt_cm_K)} (유의 창 {ps_cm_K:.1f}%) → "
    f"K-1에서 평균 t={mt_cm_K1:.2f}{star(mt_cm_K1)} (유의 창 {ps_cm_K1:.1f}%). "
    + ("cross_corr 추가 후에도 corr_min 유의성이 유지 — 두 변수는 독립적 정보를 담는다."
       if abs(mt_cm_K1) > 1.645 else
       "cross_corr 추가로 corr_min 유의성이 감소 — 두 변수 간 정보 중복 가능성.")
)

# syst_share stability
mt_ss_K  = summary['K']['mean_tstat'].get('syst_share', np.nan)
mt_ss_K1 = summary['K1']['mean_tstat'].get('syst_share', np.nan)
findings.append(
    f"**syst_share 안정성**: K 평균 t={mt_ss_K:.2f}{star(mt_ss_K)} → "
    f"K-1 평균 t={mt_ss_K1:.2f}{star(mt_ss_K1)}. "
    f"{'cross_corr 추가 후에도 부호·유의성 유지.' if mt_ss_K1 * mt_ss_K > 0 else '부호 또는 유의성 변화 감지.'}"
)

# Model M cross_corr vs K-1 cross_corr
if 'cross_corr' in summary['M']['mean_coef'].index:
    mt_M = summary['M']['mean_tstat']['cross_corr']
    findings.append(
        f"**cross_corr 맥락 의존성**: Model M(total_var 맥락)에서 평균 t={mt_M:.2f}{star(mt_M)} "
        f"→ Model K-1(syst_share·corr_min 맥락)에서 평균 t={mt_xc_K1:.2f}{star(mt_xc_K1)}. "
        + ("cross_corr 효과가 맥락 의존적으로 약화 — syst_share가 섹터 간 상관 정보의 일부를 흡수."
           if abs(mt_xc_K1) < abs(mt_M) else
           "cross_corr 효과가 맥락과 무관하게 강건 — syst_share와 직교적 정보를 보유.")
    )

# Best model
findings.append(
    f"**전체 최선 모형**: Model {best_m} — 평균 adj-R²={summary[best_m]['mean_r2']:.3f}. "
    f"K-1({k1_r2:.3f})은 F({summary['F']['mean_r2']:.3f})에 비해 "
    f"{'상회' if k1_r2 > summary['F']['mean_r2'] else str(round(k1_r2 - summary['F']['mean_r2'], 3)) + ''}."
)

for i, f in enumerate(findings, 1):
    A(f"{i}. {f}")
    A("")

A("---")
A("")
A("## 5. LW10 기준 모형 결과 비교")
A("")
A("| 모형 | 이번 실험 평균 adj-R² | LW10 보고서 평균 adj-R² | 차이 |")
A("|------|:--------------------:|:----------------------:|:----:|")
LW10_REF = {'D': 0.150, 'D_sig': 0.063, 'K': 0.198, 'L': 0.203,
             'W': 0.044, 'F': 0.286, 'D_FE': 0.166, 'L_FE': 0.189, 'M': 0.172}
for m in ['D', 'K', 'K1', 'L', 'F', 'M']:
    ref = LW10_REF.get(m, '—')
    cur = summary[m]['mean_r2']
    if isinstance(ref, float):
        A(f"| ({m}) | {cur:.3f} | {ref:.3f} | {cur - ref:+.3f} |")
    else:
        A(f"| ({m}) | {cur:.3f} | {ref} | — |")
A("")
A("*LW10 수치는 `2026-05-26_LW10.md` 기준. 동일 데이터·동일 코드 재실행이므로 차이는 부동소수점 수준 이내여야 한다.*")
A("")
A("---")
A("")
A("## 6. 한계")
A("")
A("1. **선택 편의**: cross_corr은 Model M 결과를 보고 투입 — in-sample 확인 성격.")
A("2. **다중공선성**: avg_corr / cross_corr / corr_min은 상호 상관이 높을 수 있어 VIF 검토 필요.")
A("3. **비제약 GMV**: 음수 비중 허용 — 실제 운용 제약과 다를 수 있음.")
A("4. **GFC 제외**: 2015-2024 데이터에 GFC(2007-2009)가 없어 극단 위기 행동 미확인.")
A("")
A("---")
A("")
A("## 부록 — 산출 파일")
A("")
A("| 파일 | 내용 |")
A("|------|------|")
A("| `results/figures/k1/k1_rolling_adjr2.png` | 전체 모형 롤링 adj-R² |")
A("| `results/figures/k1/k1_vs_k_comparison.png` | K vs K-1 adj-R² 직접 비교 + Δ |")
A("| `results/figures/k1/k1_rolling_coef.png` | K-1 롤링 계수 시계열 |")
A("| `results/figures/k1/k1_pct_sig.png` | K vs K-1 유의 창 비율 |")
A("| `results/figures/k1/k1_adjr2_comparison.png` | 주요 모형 adj-R² 바 차트 |")
A("| `results/figures/k1/k1_mean_tstat.png` | K-1 변수별 평균 t-통계량 |")
A("")
A("---")
A("")
A("*분석 코드: `k1_analysis.py` | 분석 환경: Python 3.x, numpy, scipy, scikit-learn*")

report_text = '\n'.join(lines)
out_path = REPORTS / '2026-05-27_K-1.md'
out_path.write_text(report_text, encoding='utf-8')
print(f'\nReport saved → {out_path}')
print('=== Done ===')
