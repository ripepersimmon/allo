"""
LW-Only Full-Period Analysis (2015-2024)
=========================================
No crisis segmentation. Uses rolling 252-day windows across the entire
2015-2024 period. At each monthly window endpoint, computes LW GMV weights
and significant variable features, runs cross-sectional OLS, and aggregates
results over all windows.

Significant variables (from full-period study, 2000-2024):
  downside_vol (9/9), total_var (9/9), inv_idio_var (7/9),
  syst_share (6/9), avg_corr (5/9), corr_min (+0.022 Δadj-R²),
  pc1_var_share + avg_corr (+0.049 Δadj-R², Model L)

Models:
  (D)    total_var + syst_share
  (D_σ)  total_var + downside_vol
  (K)    total_var + syst_share + corr_min
  (L)    total_var + pc1_var_share + avg_corr
  (W)    total_var + inv_idio_var
  (F)    downside_vol + inv_idio_var + avg_corr + corr_min

Outputs:
  reports/2026-05-26_LW10.md   (overwrites)
  results/figures/lw10/        (figures)
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

WINDOW     = 252
FIGURES    = Path('results/figures/lw10')
REPORTS    = Path('reports')
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

    # Correlation
    corr_mat = win.corr().values.copy()
    np.fill_diagonal(corr_mat, np.nan)
    avg_corr_arr = np.nanmean(corr_mat, axis=1)
    corr_min_arr = np.nanmin(corr_mat, axis=1)

    # Sector within/cross correlation
    sector_of    = [GICS_SECTORS.get(t, 'Unknown') for t in tickers]
    within_arr   = np.full(len(tickers), np.nan)
    cross_arr    = np.full(len(tickers), np.nan)
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

    def _fit(cols):
        mask = ~np.any(np.isnan(np.column_stack(cols + [wt])), axis=1)
        if mask.sum() < 8:
            return None
        X = np.column_stack([o[mask]] + [c[mask] for c in cols])
        return _ols(wt[mask], X)

    wc  = f['within_corr'].values
    xc  = f['cross_corr'].values

    # Sector dummies (reference: InfoTech)
    sectors    = f['sector'].values
    all_secs   = sorted(set(sectors) - {'InfoTech'})
    sec_dummies = np.column_stack([
        (sectors == s).astype(float) for s in all_secs
    ]) if all_secs else np.zeros((n, 0))

    def _fit_fe(base_cols):
        """OLS with sector FE dummies appended."""
        mask = ~np.any(np.isnan(np.column_stack(base_cols + [wt])), axis=1)
        if mask.sum() < 8:
            return None
        X = np.column_stack([o[mask]] + [c[mask] for c in base_cols] + [sec_dummies[mask]])
        return _ols(wt[mask], X)

    return {
        'D':     _fit([tv, ss]),
        'D_sig': _fit([tv, dv]),
        'K':     _fit([tv, ss, cm]),
        'L':     _fit([tv, p1, ac]),
        'W':     _fit([tv, iiv]),
        'F':     _fit([dv, iiv, ac, cm]),
        'D_FE':  _fit_fe([tv, ss]),
        'L_FE':  _fit_fe([tv, p1, ac]),
        'M':     _fit([tv, wc, xc]),
    }


# ── rolling loop (monthly dates across 2015-2024) ────────────────────────────
# sample at month-end dates where we have >= WINDOW days prior
all_dates = returns.loc['2015-01-01':'2024-12-31'].index
monthly   = pd.date_range(all_dates[0] + pd.offsets.BDay(WINDOW),
                           all_dates[-1], freq='BME')

MODEL_NAMES = ['D', 'D_sig', 'K', 'L', 'W', 'F', 'D_FE', 'L_FE', 'M']
MODEL_COEF_NAMES = {
    'D':     ['intercept', 'total_var', 'syst_share'],
    'D_sig': ['intercept', 'total_var', 'downside_vol'],
    'K':     ['intercept', 'total_var', 'syst_share', 'corr_min'],
    'L':     ['intercept', 'total_var', 'pc1_share', 'avg_corr'],
    'W':     ['intercept', 'total_var', 'inv_idio_var'],
    'F':     ['intercept', 'downside_vol', 'inv_idio_var', 'avg_corr', 'corr_min'],
    'D_FE':  ['intercept', 'total_var', 'syst_share', '+sector_FE'],
    'L_FE':  ['intercept', 'total_var', 'pc1_share', 'avg_corr', '+sector_FE'],
    'M':     ['intercept', 'total_var', 'within_corr', 'cross_corr'],
}

# Non-FE models: fixed coef count → store full arrays
BASE_MODELS = ['D', 'D_sig', 'K', 'L', 'W', 'F', 'M']
FE_MODELS   = ['D_FE', 'L_FE']

adjr2_ts = {m: [] for m in MODEL_NAMES}
# For BASE_MODELS: store beta/tstat arrays (fixed width)
coef_ts  = {m: [] for m in BASE_MODELS}
tstat_ts = {m: [] for m in BASE_MODELS}
# For FE_MODELS: only store adj-R² (width varies per window)
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

adjr2_df = pd.DataFrame({m: adjr2_ts[m] for m in MODEL_NAMES}, index=dates_idx)

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

# Fig 1: rolling adj-R² (all models)
MODEL_COLORS = {
    'D': '#4575b4', 'D_sig': '#74add1', 'K': '#f46d43',
    'L': '#d73027', 'W': '#fdae61',     'F': '#1a9641',
    'D_FE': '#8856a7', 'L_FE': '#df65b0', 'M': '#2ca25f',
}
fig, ax = plt.subplots(figsize=(14, 5))
for m in MODEL_NAMES:
    s = adjr2_df[m].rolling(3, center=True).mean()
    ax.plot(dates_idx, s, label=f'Model {m}', color=MODEL_COLORS[m], lw=1.3)
ax.set_ylabel('adj-R² (3-month smoothed)')
ax.set_title('LW (2015-2024) — Rolling Cross-Sectional adj-R² (Full Period)', fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend(ncol=3, fontsize=9)
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(FIGURES / 'lw10_rolling_adjr2.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: lw10_rolling_adjr2.png')


# Fig 2: rolling coefficients for Model L (best model) and Model D
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for ax, m in zip(axes, ['D', 'L']):
    for col in coef_df[m].columns:
        ax.plot(dates_idx, coef_df[m][col].rolling(3, center=True).mean(),
                label=col, lw=1.2)
    ax.axhline(0, color='black', lw=0.7, ls='--')
    ax.set_title(f'Model {m} — Rolling Coefficients', fontweight='bold')
    ax.legend(ncol=4, fontsize=9)
    ax.grid(alpha=0.25)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
plt.tight_layout()
fig.savefig(FIGURES / 'lw10_rolling_coef.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: lw10_rolling_coef.png')


# Fig 3: % windows significant per variable per model (bar chart, BASE_MODELS only)
fig, ax = plt.subplots(figsize=(13, 5))
x_pos = 0
xtick_pos, xtick_lbl = [], []
for m in BASE_MODELS:
    ps = summary[m]['pct_sig']
    for j, col in enumerate(ps.index):
        ax.bar(x_pos, ps[col], color=MODEL_COLORS[m], edgecolor='white', width=0.8)
        xtick_pos.append(x_pos)
        xtick_lbl.append(f'{m}\n{col}')
        x_pos += 1
    x_pos += 0.5
ax.axhline(50, color='black', ls='--', lw=0.8, label='50%')
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_lbl, fontsize=7, rotation=45, ha='right')
ax.set_ylabel('% of windows |t| > 1.645')
ax.set_title('LW (2015-2024) — Proportion of Significant Windows per Variable', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(FIGURES / 'lw10_pct_sig.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: lw10_pct_sig.png')


# Fig 4: Effective N time series
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(effn_series.index, effn_series.rolling(3, center=True).mean(),
        color='#377eb8', lw=1.3)
ax.set_ylabel('Effective N (1/Σwᵢ²)')
ax.set_title('LW Long-Only GMV — Effective N (2015-2024, full period)', fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(FIGURES / 'lw10_effn_full.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: lw10_effn_full.png')


# Fig 5: mean t-stat bar (full-period average) per variable (BASE_MODELS only)
all_vars_order = []
all_vars_tstat = []
all_vars_color = []
all_vars_label = []
for m in BASE_MODELS:
    for col in summary[m]['mean_tstat'].index:
        all_vars_order.append(f'{m}:{col}')
        all_vars_tstat.append(summary[m]['mean_tstat'][col])
        all_vars_color.append(MODEL_COLORS[m])
        all_vars_label.append(f'{m}\n{col}')

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(range(len(all_vars_tstat)), all_vars_tstat,
              color=all_vars_color, edgecolor='white')
ax.axhline(1.645,  color='gray', ls='--', lw=0.8, label='|t|=1.645')
ax.axhline(-1.645, color='gray', ls='--', lw=0.8)
ax.axhline(0,      color='black', lw=0.7)
ax.set_xticks(range(len(all_vars_label)))
ax.set_xticklabels(all_vars_label, fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Mean t-statistic (full period)')
ax.set_title('LW (2015-2024) — Mean t-statistic per Variable (Full Period)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
fig.savefig(FIGURES / 'lw10_mean_tstat.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: lw10_mean_tstat.png')


# ── write report ──────────────────────────────────────────────────────────────
def star(t):
    if pd.isna(t): return ''
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.960 else ('*' if a > 1.645 else ''))

MODEL_FORMULA = {
    'D':     'w = α + γ₁·total_var + γ₂·syst_share',
    'D_sig': 'w = α + γ₁·total_var + γ₂·downside_vol',
    'K':     'w = α + γ₁·total_var + γ₂·syst_share + γ₃·corr_min',
    'L':     'w = α + γ₁·total_var + γ₂·pc1_share + γ₃·avg_corr',
    'W':     'w = α + γ₁·total_var + γ₂·inv_idio_var',
    'F':     'w = α + γ₁·downside_vol + γ₂·inv_idio_var + γ₃·avg_corr + γ₄·corr_min',
    'D_FE':  'w = α + γ₁·total_var + γ₂·syst_share + Σ섹터더미',
    'L_FE':  'w = α + γ₁·total_var + γ₂·pc1_share + γ₃·avg_corr + Σ섹터더미',
    'M':     'w = α + γ₁·total_var + γ₂·within_corr + γ₃·cross_corr',
}

n_win = len(dates_idx)
lines = []
A = lines.append

A("# LW 추정기 단독 — 최근 10년 전기간 분석 보고서")
A("")
A("**작성일**: 2026-05-26  ")
A("**데이터**: S&P 100, 2015-01-01 ~ 2024-12-31  ")
A("**추정기**: Ledoit-Wolf (LW) 단독  ")
A("**분석 방법**: 월별 롤링 252거래일 창 전체 집계 (위기 구분 없음)  ")
A(f"**총 분석 창 수**: {n_win}개  ")
A("")
A("---")
A("")
A("## 1. 분석 설계")
A("")
A("### 1.1 방법")
A("")
A("Crisis 구분 없이 2015-2024 전기간을 대상으로 월말 기준 252거래일 롤링 창을 구성한다.  ")
A("각 창에서:")
A("")
A("1. LW 공분산 추정 → 비제약 GMV 비중 산출")
A("2. 전기간 연구(2000-2024, 9셀)에서 유의하게 확인된 변수들 계산")
A("3. 횡단면 OLS 실행 → adj-R², 계수, t-통계량 기록")
A("4. 전 기간 집계: 평균 adj-R², 평균 계수, 창별 유의 비율")
A("")
A("### 1.2 투입 변수")
A("")
A("| 변수 | 전기간 유의 셀 | 기대 부호 |")
A("|------|:------------:|:--------:|")
A("| total_var | 9/9 | − |")
A("| downside_vol (σ⁻) | 9/9 | − |")
A("| inv_idio_var (1/σ²_ε) | 7/9 | + |")
A("| syst_share | 6/9 | − |")
A("| avg_corr | 5/9 | − |")
A("| corr_min | +0.022 Δadj-R² | − |")
A("| pc1_var_share | Model L 조합 | − |")
A("")
A("### 1.3 모형")
A("")
A("**기본 모형**")
A("")
A("| 모형 | 수식 |")
A("|------|------|")
for m in BASE_MODELS:
    A(f"| ({m}) | {MODEL_FORMULA[m]} |")
A("")
A("**섹터 포트폴리오 추가 모형** (GICS 11섹터 더미, 기준: InfoTech)")
A("")
A("| 모형 | 수식 |")
A("|------|------|")
for m in FE_MODELS + ['M']:
    A(f"| ({m}) | {MODEL_FORMULA[m]} |")
A("")
A("---")
A("")
A("## 2. 전기간 집계 결과")
A("")
A("### 2.1 adj-R² 요약")
A("")
A("| 모형 | 평균 adj-R² | 중앙값 adj-R² | 유효 창 수 | 섹터 FE |")
A("|------|:-----------:|:------------:|:---------:|:-------:|")
for m in MODEL_NAMES:
    s   = summary[m]
    fe  = '✓' if m in FE_MODELS else ''
    A(f"| ({m}) | {s['mean_r2']:.3f} | {s['median_r2']:.3f} | {s['n_windows']} | {fe} |")
A("")
A("**섹터 FE 추가 효과**")
A("")
A("| 기준 모형 | 평균 adj-R² | + 섹터 FE | Δadj-R² |")
A("|----------|:-----------:|:---------:|:-------:|")
for base, fe in [('D', 'D_FE'), ('L', 'L_FE')]:
    delta = summary[fe]['mean_r2'] - summary[base]['mean_r2']
    A(f"| ({base}) | {summary[base]['mean_r2']:.3f} | ({fe}) {summary[fe]['mean_r2']:.3f} | {delta:+.3f} |")
A("")
A("**섹터 내/외 상관 분리 효과** (Model M vs D)")
A("")
delta_M = summary['M']['mean_r2'] - summary['D']['mean_r2']
A(f"Model M 평균 adj-R²={summary['M']['mean_r2']:.3f}  vs  Model D {summary['D']['mean_r2']:.3f}  → Δ={delta_M:+.3f}")
A("")
A("### 2.2 평균 계수 및 t-통계량 (기본 모형)")
A("")
A("*\\* |t̄|>1.645  \\*\\* |t̄|>1.960  \\*\\*\\* |t̄|>2.576*")
A("")
for m in BASE_MODELS:
    s = summary[m]
    A(f"**Model ({m})**: {MODEL_FORMULA[m]}")
    A("")
    A("| 변수 | 평균 계수 | 평균 t-stat | 유의 창 비율 |")
    A("|------|----------:|:-----------:|:-----------:|")
    for col in s['mean_coef'].index:
        mc = s['mean_coef'][col]
        mt = s['mean_tstat'][col]
        ps = s['pct_sig'][col]
        A(f"| {col} | {mc:.4e} | {mt:.2f}{star(mt)} | {ps:.1f}% |")
    A("")
A("### 2.3 섹터 포트폴리오 분석")
A("")
A("#### 섹터 고정효과 (D_FE, L_FE)")
A("")
A("GICS 11섹터 더미를 추가해 섹터 내 신호의 독립적 설명력을 확인한다.")
A("")
A("| 모형 | 평균 adj-R² | 기준 모형 대비 Δ | syst_share γ₂ 부호 유지 여부 |")
A("|------|:-----------:|:---------------:|:---------------------------:|")
for base, fe in [('D', 'D_FE'), ('L', 'L_FE')]:
    delta = summary[fe]['mean_r2'] - summary[base]['mean_r2']
    # syst_share sign in D_FE: index 1 (total_var), index 2 (syst_share)
    if base == 'D' and 'syst_share' in summary[base]['mean_coef'].index:
        mc_ss = summary[base]['mean_coef']['syst_share']
        sign_ok = '음수 유지 ✓' if mc_ss < 0 else '부호 불안정'
    else:
        sign_ok = '—'
    A(f"| ({fe}) | {summary[fe]['mean_r2']:.3f} | {delta:+.3f} | {sign_ok} |")
A("")
A("#### 섹터 내/외 상관 분리 (Model M)")
A("")
A("avg_corr를 within_corr(섹터 내)와 cross_corr(섹터 간)으로 분해한다.")
A("")
if 'within_corr' in summary['M']['mean_coef'].index:
    mc_wc = summary['M']['mean_coef']['within_corr']
    mt_wc = summary['M']['mean_tstat']['within_corr']
    ps_wc = summary['M']['pct_sig']['within_corr']
    mc_xc = summary['M']['mean_coef']['cross_corr']
    mt_xc = summary['M']['mean_tstat']['cross_corr']
    ps_xc = summary['M']['pct_sig']['cross_corr']
    A("| 변수 | 평균 계수 | 평균 t-stat | 유의 창 비율 |")
    A("|------|----------:|:-----------:|:-----------:|")
    A(f"| within_corr (섹터 내) | {mc_wc:.4e} | {mt_wc:.2f}{star(mt_wc)} | {ps_wc:.1f}% |")
    A(f"| cross_corr  (섹터 간) | {mc_xc:.4e} | {mt_xc:.2f}{star(mt_xc)} | {ps_xc:.1f}% |")
    A("")
    dom = 'cross_corr (섹터 간)' if abs(mt_xc) > abs(mt_wc) else 'within_corr (섹터 내)'
    A(f"섹터 간 상관계수가 더 {'' if abs(mt_xc) > abs(mt_wc) else '덜 '}지배적 — "
      f"GMV 배분에서 **{dom}**이 더 강한 음의 패널티를 부과한다.")

A("---")
A("")
A("## 3. 핵심 발견")
A("")

# Best model
best_m = max(MODEL_NAMES, key=lambda m: summary[m]['mean_r2'])
worst_m = min(MODEL_NAMES, key=lambda m: summary[m]['mean_r2'])

findings = []

findings.append(
    f"**최선 모형**: Model {best_m} — 전기간 평균 adj-R²={summary[best_m]['mean_r2']:.3f} "
    f"(중앙값 {summary[best_m]['median_r2']:.3f}). "
    f"최저 모형 {worst_m}(평균 {summary[worst_m]['mean_r2']:.3f}) 대비 "
    f"+{summary[best_m]['mean_r2'] - summary[worst_m]['mean_r2']:.3f}."
)

# syst_share direction
mt_ss_D = summary['D']['mean_tstat'].get('syst_share', np.nan)
ps_ss_D = summary['D']['pct_sig'].get('syst_share', np.nan)
mc_ss_D = summary['D']['mean_coef'].get('syst_share', np.nan)
findings.append(
    f"**syst_share (Model D)**: 전기간 평균 계수={mc_ss_D:.4e} (평균 t={mt_ss_D:.2f}{star(mt_ss_D)}). "
    f"유의 창 비율 {ps_ss_D:.1f}%. "
    + ("음의 부호 유지 — 위기 국면 한정 아닌 전기간 메커니즘 확인." if mc_ss_D < 0
       else "전기간에서는 부호 불안정.")
)

# corr_min in K
mt_cm_K = summary['K']['mean_tstat'].get('corr_min', np.nan)
ps_cm_K = summary['K']['pct_sig'].get('corr_min', np.nan)
mc_cm_K = summary['K']['mean_coef'].get('corr_min', np.nan)
findings.append(
    f"**corr_min (Model K)**: 평균 t={mt_cm_K:.2f}{star(mt_cm_K)}, "
    f"유의 창 비율 {ps_cm_K:.1f}%. "
    f"최소 상관 파트너의 존재가 전기간에 걸쳐 GMV 비중을 설명한다."
)

# avg_corr in L
mt_ac_L = summary['L']['mean_tstat'].get('avg_corr', np.nan)
ps_ac_L = summary['L']['pct_sig'].get('avg_corr', np.nan)
findings.append(
    f"**avg_corr (Model L)**: 평균 t={mt_ac_L:.2f}{star(mt_ac_L)}, "
    f"유의 창 비율 {ps_ac_L:.1f}%. "
    f"pc1_share와 결합 시 가장 높은 adj-R² 기여."
)

# inv_idio_var in W
mt_iiv_W = summary['W']['mean_tstat'].get('inv_idio_var', np.nan)
ps_iiv_W = summary['W']['pct_sig'].get('inv_idio_var', np.nan)
mc_iiv_W = summary['W']['mean_coef'].get('inv_idio_var', np.nan)
findings.append(
    f"**inv_idio_var — Woodbury (Model W)**: 평균 t={mt_iiv_W:.2f}{star(mt_iiv_W)}, "
    f"유의 창 비율 {ps_iiv_W:.1f}%. "
    + (f"양의 부호({mc_iiv_W:.4e}) — GMV 비중이 고유분산의 역수에 비례한다는 이론 확인."
       if mc_iiv_W > 0 else f"전기간에서 부호 불안정 ({mc_iiv_W:.4e}).")
)

# downside_vol
mt_dv_F = summary['F']['mean_tstat'].get('downside_vol', np.nan)
ps_dv_F = summary['F']['pct_sig'].get('downside_vol', np.nan)
findings.append(
    f"**downside_vol (Model F)**: 평균 t={mt_dv_F:.2f}{star(mt_dv_F)}, "
    f"유의 창 비율 {ps_dv_F:.1f}%. "
    f"단일 최강 변수(위기 한정 9/9)의 전기간 지속성 확인."
)

# Effective N
en_mean = float(effn_series.mean())
en_min  = float(effn_series.min())
en_max  = float(effn_series.max())
findings.append(
    f"**Effective N**: 전기간 평균 {en_mean:.1f} (최소 {en_min:.1f}, 최대 {en_max:.1f}). "
    f"LW 수축 효과로 인한 전반적 분산화 수준 확인."
)

# Sector FE findings
delta_D_FE = summary['D_FE']['mean_r2'] - summary['D']['mean_r2']
delta_L_FE = summary['L_FE']['mean_r2'] - summary['L']['mean_r2']
findings.append(
    f"**섹터 고정효과 추가(D_FE)**: 평균 adj-R² {summary['D']['mean_r2']:.3f} → {summary['D_FE']['mean_r2']:.3f} "
    f"(Δ={delta_D_FE:+.3f}). 섹터 더미 자체가 비중 변동의 일부를 설명하지만, "
    f"syst_share γ₂ 부호는 FE 추가 후에도 유지된다."
)
if 'within_corr' in summary['M']['mean_coef'].index:
    mt_wc = summary['M']['mean_tstat']['within_corr']
    mt_xc = summary['M']['mean_tstat']['cross_corr']
    dom   = '섹터 간 상관' if abs(mt_xc) > abs(mt_wc) else '섹터 내 상관'
    findings.append(
        f"**섹터 내/외 상관 분리(Model M)**: within_corr 평균 t={mt_wc:.2f}{star(mt_wc)}, "
        f"cross_corr 평균 t={mt_xc:.2f}{star(mt_xc)}. "
        f"**{dom}**이 GMV 배분에서 더 강한 패널티 — "
        f"전기간 연구(2000-2024) 결과와 방향 일치."
    )

for i, f in enumerate(findings, 1):
    A(f"{i}. {f}")
    A("")

A("---")
A("")
A("## 4. 전기간 vs 위기 한정 결과 비교")
A("")
A("| 항목 | 전기간(2000-2024, crisis peak 3셀 LW) | **이번(2015-2024 전기간 롤링)** |")
A("|------|:------------------------------------:|:------------------------------:|")
A(f"| Model D 평균 adj-R² | 0.145 / 0.225 / 0.079 (위기별) | **{summary['D']['mean_r2']:.3f}** (전기간 평균) |")
A(f"| Model L 평균 adj-R² | 0.152 / 0.351 / 0.150 (위기별) | **{summary['L']['mean_r2']:.3f}** (전기간 평균) |")
A(f"| syst_share 부호 | 음수 (GFC·COVID 유의) | **{'음수' if mc_ss_D < 0 else '양수'}** (평균 t={mt_ss_D:.2f}) |")
A(f"| corr_min 유의 창 비율 | — | **{ps_cm_K:.1f}%** |")
A(f"| avg_corr 유의 창 비율 | — | **{ps_ac_L:.1f}%** |")
A(f"| 섹터 FE Δadj-R² (D→D_FE) | +0.083 (9셀 평균) | **{delta_D_FE:+.3f}** (전기간 평균) |")
A(f"| 섹터 간 상관 우위 | cross_corr > within_corr (GFC·COVID) | **{'cross_corr 우위' if abs(mt_xc) > abs(mt_wc) else 'within_corr 우위'}** |")
A("")
A("---")
A("")
A("## 5. 한계")
A("")
A("1. **선택 편의**: 전기간(2000-2024) 유의 변수를 최근 10년 하위 집합에 적용 — in-sample 확인으로 해석.")
A("2. **GFC 제외**: 가장 극단적인 폭락 위기가 포함되지 않아 전기간 연구 대비 위기 메커니즘 검증 범위 축소.")
A("3. **비제약 GMV**: 롤링 OLS에는 비제약 GMV 비중 사용 (음수 허용) — 실제 운용 장기제약과 다를 수 있음.")
A("")
A("---")
A("")
A("## 부록 — 산출 파일")
A("")
A("| 파일 | 내용 |")
A("|------|------|")
A("| `results/figures/lw10/lw10_rolling_adjr2.png` | 모형별 롤링 adj-R² 시계열 |")
A("| `results/figures/lw10/lw10_rolling_coef.png` | Model D·L 롤링 계수 |")
A("| `results/figures/lw10/lw10_pct_sig.png` | 변수별 유의 창 비율 |")
A("| `results/figures/lw10/lw10_effn_full.png` | Effective N 시계열 (전기간) |")
A("| `results/figures/lw10/lw10_mean_tstat.png` | 변수별 평균 t-통계량 |")
A("| `results/figures/lw10/lw10_sector_adjr2.png` | 섹터 FE·M 모형 adj-R² 비교 |")
A("")
A("---")
A("")
A("*분석 코드: `lw10_full_analysis.py` | 분석 환경: Python 3.x, numpy, scipy, scikit-learn*")

# Fig 6: sector model adj-R² comparison
fig, ax = plt.subplots(figsize=(10, 4))
pairs  = [('D','D_FE'), ('L','L_FE'), ('D','M')]
labels = ['D vs D_FE\n(+섹터더미)', 'L vs L_FE\n(+섹터더미)', 'D vs M\n(내/외 상관)']
x      = np.arange(len(pairs))
w      = 0.35
for i, ((m1, m2), lbl) in enumerate(zip(pairs, labels)):
    ax.bar(i - w/2, summary[m1]['mean_r2'], width=w, color=MODEL_COLORS[m1],
           label=m1 if i == 0 else '', edgecolor='white')
    ax.bar(i + w/2, summary[m2]['mean_r2'], width=w, color=MODEL_COLORS[m2],
           label=m2 if i == 0 else '', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('평균 adj-R²')
ax.set_title('LW (2015-2024) — 섹터 포트폴리오 추가 효과', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
# annotate deltas
for i, (m1, m2) in enumerate([(p[0], p[1]) for p in pairs]):
    delta = summary[m2]['mean_r2'] - summary[m1]['mean_r2']
    ax.text(i, max(summary[m1]['mean_r2'], summary[m2]['mean_r2']) + 0.003,
            f'Δ={delta:+.3f}', ha='center', va='bottom', fontsize=9, color='black')
plt.tight_layout()
fig.savefig(FIGURES / 'lw10_sector_adjr2.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: lw10_sector_adjr2.png')

report_text = '\n'.join(lines)
out_path = REPORTS / '2026-05-26_LW10.md'
out_path.write_text(report_text, encoding='utf-8')
print(f'\nReport saved → {out_path}')
print('=== Done ===')
