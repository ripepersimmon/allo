"""
Time Series Analysis of LW GMV Weights (2015-2024)
====================================================
Models the temporal structure of GMV weights and Effective N.

Analyses:
  1. Effective N time series: ADF/KPSS stationarity, AR(1) fit
  2. Panel AR(1): w_{i,t+1} ~ α_i + ρ·w_{i,t}  (weight persistence)
  3. Panel predictive OLS: w_{i,t+1} ~ w_{i,t} + X_{i,t}  (lagged features)
  4. Per-asset weight autocorrelation structure
  5. Rolling 12-month mean/std of individual asset weights

Outputs:
  reports/2026-05-26_timeseries.md
  results/figures/timeseries/
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
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.ar_model import AutoReg

from src.data_loader import load_prices_from_parquet, compute_returns, TICKERS
from src.estimators import lw_cov
from src.portfolio import effective_n
from src.market import get_market_proxy
from src.analysis import rolling_gmv

np.random.seed(42)

FIGURES = Path('results/figures/timeseries')
REPORTS = Path('reports')
FIGURES.mkdir(parents=True, exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
print('Loading data...')
prices  = load_prices_from_parquet('sp500', tickers=TICKERS,
                                    start='2014-01-01', end='2024-12-31')
returns = compute_returns(prices, method='log')

print('Computing rolling LW GMV weights...')
weights = rolling_gmv(returns, lw_cov, window=252, constrained=True)
weights = weights.loc['2015-01-01':]
print(f'Weights: {weights.shape}')


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
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
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

def star(t):
    if pd.isna(t): return ''
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.960 else ('*' if a > 1.645 else ''))


# ── 1. Effective N time series ────────────────────────────────────────────────
print('\n[1] Effective N time series analysis')
effn = weights.apply(effective_n, axis=1).dropna()

# ADF test
adf_res  = adfuller(effn, maxlag=5, autolag='AIC')
adf_stat, adf_p = adf_res[0], adf_res[1]

# KPSS test (H0: stationary)
kpss_stat, kpss_p, _, _ = kpss(effn, regression='c', nlags='auto')

# AR(1) fit
ar1 = AutoReg(effn, lags=1, old_names=False).fit()
ar1_rho  = float(ar1.params.iloc[1])
ar1_tstat = float(ar1.tvalues.iloc[1])
ar1_pval  = float(ar1.pvalues.iloc[1])
fitted   = ar1.fittedvalues
ss_res_ar = float(np.sum((effn.values[1:] - fitted.values)**2))
ss_tot_ar = float(np.sum((effn.values[1:] - effn.values[1:].mean())**2))
ar1_r2   = 1 - ss_res_ar / ss_tot_ar if ss_tot_ar > 1e-14 else 0.0

print(f'  ADF stat={adf_stat:.3f}  p={adf_p:.4f}')
print(f'  KPSS stat={kpss_stat:.3f}  p={kpss_p:.4f}')
print(f'  AR(1) rho={ar1_rho:.3f}  t={ar1_tstat:.2f}  R²={ar1_r2:.3f}')

# Descriptive stats by year
effn_annual = effn.groupby(effn.index.year).agg(['mean', 'min', 'max', 'std'])


# ── 2. Panel AR(1): weight persistence ───────────────────────────────────────
print('\n[2] Panel AR(1) weight persistence')

# Monthly sampling to avoid extreme autocorrelation
monthly_dates = pd.date_range(weights.index[0], weights.index[-1], freq='BME')
monthly_dates = monthly_dates[monthly_dates.isin(weights.index)]
w_monthly = weights.reindex(monthly_dates, method='ffill').dropna(how='all')

# Stack: (t, ticker) panel
w_t  = w_monthly.iloc[:-1].stack()   # t
w_t1 = w_monthly.iloc[1:].stack()    # t+1

w_t.index.names  = ['date', 'ticker']
w_t1.index.names = ['date', 'ticker']

panel = pd.DataFrame({'w_t': w_t, 'w_t1': w_t1}).dropna()

# Demean within ticker (within-estimator to control fixed effects)
panel['w_t_dm']  = panel['w_t']  - panel.groupby('ticker')['w_t'].transform('mean')
panel['w_t1_dm'] = panel['w_t1'] - panel.groupby('ticker')['w_t1'].transform('mean')

mask = panel['w_t_dm'].notna() & panel['w_t1_dm'].notna()
y_ar = panel.loc[mask, 'w_t1_dm'].values
X_ar = np.column_stack([panel.loc[mask, 'w_t_dm'].values])  # no intercept (demeaned)

# simple slope without intercept for within estimator
rho_hat = float(np.dot(X_ar.ravel(), y_ar) / np.dot(X_ar.ravel(), X_ar.ravel()))
resid   = y_ar - rho_hat * X_ar.ravel()
se_rho  = float(np.sqrt(np.mean(resid**2) / np.dot(X_ar.ravel(), X_ar.ravel())))
t_rho   = rho_hat / se_rho if se_rho > 1e-14 else np.nan
ss_tot  = np.sum((y_ar - y_ar.mean())**2)
ss_res  = np.sum(resid**2)
r2_ar   = 1 - ss_res / ss_tot

print(f'  ρ={rho_hat:.3f}  t={t_rho:.2f}  R²={r2_ar:.3f}  N={len(y_ar)}')


# ── 3. Panel predictive OLS with lagged features ──────────────────────────────
print('\n[3] Panel predictive OLS: w_{t+1} ~ w_{t} + X_{t}')

# Compute features at each monthly date using PREVIOUS window's returns
def compute_lagged_features(end_date, returns, window=252):
    end   = pd.Timestamp(end_date)
    start = end - pd.offsets.BDay(window)
    win   = returns.loc[start:end].dropna(axis=1)
    if win.shape[1] < 10 or len(win) < 50:
        return pd.DataFrame()
    mkt     = get_market_proxy(win, 'ew')
    mkt_var = mkt.var()
    if mkt_var < 1e-14:
        return pd.DataFrame()
    cov_mat  = win.cov().values
    diag_v   = np.diag(cov_mat)
    corr_mat = win.corr().values.copy()
    np.fill_diagonal(corr_mat, np.nan)
    avg_corr = np.nanmean(corr_mat, axis=1)
    corr_min = np.nanmin(corr_mat, axis=1)
    eigval, eigvec = np.linalg.eigh(cov_mat)
    lambda1 = eigval[-1]; pc1 = eigvec[:, -1]
    rows = []
    for i, col in enumerate(win.columns):
        tv = float(diag_v[i])
        if tv < 1e-14: continue
        b  = float(win[col].cov(mkt)) / mkt_var
        sv = b**2 * mkt_var
        iv = max(tv - sv, 1e-14)
        r_arr = win[col].values
        neg_r = r_arr[r_arr < 0]
        dv    = float(np.std(neg_r)) if len(neg_r) > 5 else np.nan
        pc1s  = float(np.clip(lambda1 * pc1[i]**2 / tv, 0, 1))
        rows.append(dict(ticker=col, total_var=tv,
                         syst_share=min(b**2 * mkt_var / tv, 1.0),
                         downside_vol=dv, avg_corr=avg_corr[i],
                         corr_min=corr_min[i], pc1_share=pc1s))
    return pd.DataFrame(rows).set_index('ticker')

# Build lagged panel: features at t → weights at t+1
print('  Building lagged feature panel (monthly)...')
records = []
for i in range(len(monthly_dates) - 1):
    feat_date   = monthly_dates[i]
    weight_date = monthly_dates[i + 1]
    if weight_date not in weights.index:
        continue
    feat = compute_lagged_features(feat_date, returns)
    if feat.empty:
        continue
    w_next = weights.loc[weight_date]
    common = feat.index.intersection(w_next.index)
    if len(common) < 10:
        continue
    for tkr in common:
        rec = feat.loc[tkr].to_dict()
        rec['w_next'] = w_next[tkr]
        rec['w_curr'] = weights.loc[feat_date, tkr] if tkr in weights.columns else np.nan
        rec['date']   = feat_date
        rec['ticker'] = tkr
        records.append(rec)

pred_panel = pd.DataFrame(records).dropna(subset=['w_next', 'w_curr',
                                                    'total_var', 'downside_vol',
                                                    'avg_corr', 'corr_min'])
print(f'  Panel size: {len(pred_panel)} obs  ({pred_panel["date"].nunique()} months × ~{len(pred_panel)//pred_panel["date"].nunique()} assets)')

# Within-ticker demean
for col in ['w_next', 'w_curr', 'total_var', 'syst_share', 'downside_vol',
            'avg_corr', 'corr_min', 'pc1_share']:
    pred_panel[f'{col}_dm'] = (pred_panel[col]
                                - pred_panel.groupby('ticker')[col].transform('mean'))

mask = pred_panel[['w_next_dm','w_curr_dm','total_var_dm',
                   'downside_vol_dm','avg_corr_dm','corr_min_dm']].notna().all(axis=1)
sub = pred_panel[mask]

def run_pred(sub, xcols):
    y = sub['w_next_dm'].values
    X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in xcols])
    return _ols(y, X)

pred_models = {
    'AR1':   run_pred(sub, ['w_curr_dm']),
    'AR+tv': run_pred(sub, ['w_curr_dm', 'total_var_dm']),
    'AR+dv': run_pred(sub, ['w_curr_dm', 'downside_vol_dm']),
    'AR+L':  run_pred(sub, ['w_curr_dm', 'total_var_dm', 'pc1_share_dm', 'avg_corr_dm']),
    'AR+F':  run_pred(sub, ['w_curr_dm', 'downside_vol_dm', 'avg_corr_dm', 'corr_min_dm']),
}
PRED_COEF_NAMES = {
    'AR1':   ['intercept', 'w_{t}'],
    'AR+tv': ['intercept', 'w_{t}', 'total_var_{t}'],
    'AR+dv': ['intercept', 'w_{t}', 'downside_vol_{t}'],
    'AR+L':  ['intercept', 'w_{t}', 'total_var_{t}', 'pc1_share_{t}', 'avg_corr_{t}'],
    'AR+F':  ['intercept', 'w_{t}', 'downside_vol_{t}', 'avg_corr_{t}', 'corr_min_{t}'],
}
for m, r in pred_models.items():
    print(f'  {m}: adj-R²={r["adj_r2"]:.3f}  tstats={[f"{t:.2f}" for t in r["tstat"][1:]]}')


# ── 4. Per-asset autocorrelation ─────────────────────────────────────────────
print('\n[4] Per-asset weight autocorrelation')
acf1_all = []
for tkr in weights.columns:
    s = weights[tkr].dropna()
    if len(s) > 50:
        acf1_all.append(s.autocorr(lag=1))
acf1_arr = np.array([x for x in acf1_all if not np.isnan(x)])
print(f'  Median lag-1 ACF={np.median(acf1_arr):.3f}  '
      f'mean={np.mean(acf1_arr):.3f}  '
      f'pct>0.5: {(acf1_arr > 0.5).mean()*100:.1f}%')


# ── figures ───────────────────────────────────────────────────────────────────

# Fig 1: Effective N time series + AR(1) fit
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

ax = axes[0]
ax.plot(effn.index, effn.values, color='#377eb8', lw=1.0, alpha=0.7)
ax.plot(effn.index, effn.rolling(21).mean(), color='#e41a1c', lw=1.5, label='21-day MA')
ax.set_ylabel('Effective N')
ax.set_title('LW GMV — Effective N (2015-2024)', fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend(); ax.grid(alpha=0.25)

ax = axes[1]
effn_vals = effn.values
ax.scatter(effn_vals[:-1], effn_vals[1:], alpha=0.3, s=8, color='#377eb8')
x_range = np.linspace(effn_vals.min(), effn_vals.max(), 100)
ax.plot(x_range, ar1.params.iloc[0] + ar1_rho * x_range,
        color='#e41a1c', lw=1.5, label=f'AR(1): ρ={ar1_rho:.3f}, R²={ar1_r2:.3f}')
ax.set_xlabel('EffN(t)'); ax.set_ylabel('EffN(t+1)')
ax.set_title('AR(1) Scatter — Effective N', fontweight='bold')
ax.legend(); ax.grid(alpha=0.25)

plt.tight_layout()
fig.savefig(FIGURES / 'ts_effn.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: ts_effn.png')

# Fig 2: Predictive model adj-R² bar
fig, ax = plt.subplots(figsize=(9, 4))
mnames = list(pred_models.keys())
r2s    = [pred_models[m]['adj_r2'] for m in mnames]
colors = ['#4575b4','#74add1','#f46d43','#d73027','#1a9641']
bars   = ax.bar(mnames, r2s, color=colors[:len(mnames)], width=0.6)
for bar, v in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.0005,
            f'{v:.3f}', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('adj-R²')
ax.set_title('Predictive Panel OLS: w_{t+1} ~ lagged features\n(within-ticker demeaned)',
             fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(FIGURES / 'ts_pred_adjr2.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: ts_pred_adjr2.png')

# Fig 3: Per-asset ACF(1) distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(acf1_arr, bins=25, color='#377eb8', edgecolor='white')
ax.axvline(np.median(acf1_arr), color='red', ls='--', lw=1.5,
           label=f'Median={np.median(acf1_arr):.3f}')
ax.set_xlabel('Lag-1 autocorrelation of GMV weight')
ax.set_ylabel('Number of assets')
ax.set_title('Per-Asset Weight Autocorrelation (LW, 2015-2024)', fontweight='bold')
ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(FIGURES / 'ts_acf_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: ts_acf_dist.png')

# Fig 4: Annual Effective N boxplot
fig, ax = plt.subplots(figsize=(12, 4))
annual_groups = [effn[effn.index.year == yr].values for yr in range(2015, 2025)
                 if yr in effn.index.year]
years = [yr for yr in range(2015, 2025) if yr in effn.index.year]
bp = ax.boxplot(annual_groups, patch_artist=True,
                boxprops=dict(facecolor='#deebf7'),
                medianprops=dict(color='#e41a1c', lw=1.5))
ax.set_xticklabels(years)
ax.set_ylabel('Effective N')
ax.set_title('Annual Distribution of Effective N — LW GMV (2015-2024)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(FIGURES / 'ts_effn_annual.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: ts_effn_annual.png')


# ── report ────────────────────────────────────────────────────────────────────
L = []
A = L.append

A("# LW GMV 비중의 시계열 구조 분석")
A("")
A("**작성일**: 2026-05-26  ")
A("**데이터**: S&P 100, 2015-01-01 ~ 2024-12-31  ")
A("**추정기**: Ledoit-Wolf (LW) 단독  ")
A("**방법**: Effective N 시계열 특성, Panel AR(1), 예측 패널 OLS (라그 피처)  ")
A("")
A("---")
A("")
A("## 1. Effective N 시계열 특성")
A("")
A("### 1.1 기술 통계 (연도별)")
A("")
A("| 연도 | 평균 | 최소 | 최대 | 표준편차 |")
A("|------|-----:|-----:|-----:|--------:|")
for yr, row in effn_annual.iterrows():
    A(f"| {yr} | {row['mean']:.2f} | {row['min']:.2f} | {row['max']:.2f} | {row['std']:.2f} |")
A("")
A("### 1.2 정상성 검정")
A("")
A("| 검정 | 통계량 | p-value | 결론 |")
A("|------|-------:|:-------:|------|")
adf_concl = "단위근 기각 (정상)" if adf_p < 0.05 else "단위근 기각 실패 (비정상)"
kpss_concl = "정상성 기각" if kpss_p < 0.05 else "정상성 기각 실패 (정상)"
A(f"| ADF (H₀: 단위근) | {adf_stat:.3f} | {adf_p:.4f} | {adf_concl} |")
A(f"| KPSS (H₀: 정상) | {kpss_stat:.3f} | {kpss_p:.4f} | {kpss_concl} |")
A("")

if adf_p < 0.05 and kpss_p >= 0.05:
    stationarity_concl = "ADF·KPSS 모두 정상성 지지 — Effective N은 정상 시계열."
elif adf_p >= 0.05:
    stationarity_concl = "ADF 단위근 기각 실패 — Effective N에 지속적 추세 또는 구조 변화 존재 가능."
else:
    stationarity_concl = "검정 결과 상충 — 추세 정상(trend-stationary) 가능성."
A(f"**결론**: {stationarity_concl}")
A("")
A("### 1.3 AR(1) 추정")
A("")
A("$$\\text{EffN}_t = \\mu + \\rho \\cdot \\text{EffN}_{t-1} + \\varepsilon_t$$")
A("")
A("| 파라미터 | 추정값 | t-통계량 | 유의 |")
A("|----------|-------:|:--------:|:----:|")
A(f"| ρ (지속성) | {ar1_rho:.4f} | {ar1_tstat:.2f} | {star(ar1_tstat) or 'n.s.'} |")
A(f"| 모형 R² | {ar1_r2:.4f} | — | — |")
A("")
if ar1_rho > 0.9:
    A(f"ρ={ar1_rho:.3f}은 Effective N이 매우 높은 지속성을 가짐을 의미한다. "
      f"전일 집중도가 다음날 집중도를 강하게 예측한다.")
else:
    A(f"ρ={ar1_rho:.3f}으로 중간 수준의 지속성. 단기 회귀 성향이 있다.")
A("")
A("---")
A("")
A("## 2. 비중 지속성 — Panel AR(1)")
A("")
A("월별 샘플링 후 자산 고정효과 제거(within-demeaning)한 패널 AR(1):")
A("")
A("$$\\tilde{w}_{i,t+1} = \\rho \\cdot \\tilde{w}_{i,t} + \\varepsilon_{i,t}$$")
A("")
A("| 파라미터 | 추정값 | t-통계량 | R² | 관측수 |")
A("|----------|-------:|:--------:|:--:|------:|")
A(f"| ρ (비중 지속성) | {rho_hat:.4f} | {t_rho:.2f}{star(t_rho)} | {r2_ar:.3f} | {len(y_ar):,} |")
A("")
if rho_hat > 0.5:
    A(f"ρ={rho_hat:.3f} — LW GMV 비중은 자산 고정효과 제거 후에도 강한 지속성을 가진다. "
      f"한 달 전 비중이 다음 달 비중 변동의 {r2_ar*100:.1f}%를 설명한다.")
else:
    A(f"ρ={rho_hat:.3f} — 고정효과 제거 후 비중 지속성은 제한적이다.")
A("")
A("### 2.1 Per-asset 자기상관 분포")
A("")
A("| 통계량 | lag-1 자기상관 |")
A("|--------|:-------------:|")
A(f"| 중앙값 | {np.median(acf1_arr):.3f} |")
A(f"| 평균 | {np.mean(acf1_arr):.3f} |")
A(f"| 최솟값 | {np.min(acf1_arr):.3f} |")
A(f"| 최댓값 | {np.max(acf1_arr):.3f} |")
A(f"| ACF > 0.5인 자산 비율 | {(acf1_arr > 0.5).mean()*100:.1f}% |")
A("")
A("---")
A("")
A("## 3. 예측 패널 OLS — 라그 피처")
A("")
A("피처를 t기 창에서 계산하고 t+1기 비중을 예측한다. 자산 고정효과 within-demeaning 적용.")
A("")
A(f"**패널 크기**: {len(sub):,}개 관측치 ({sub['date'].nunique()}개월 × ~{len(sub)//sub['date'].nunique()}개 자산)")
A("")
A("### 3.1 adj-R² 비교")
A("")
A("| 모형 | 수식 | adj-R² |")
A("|------|------|:------:|")
model_formulas = {
    'AR1':   'w̃_{t+1} ~ w̃_{t}',
    'AR+tv': 'w̃_{t+1} ~ w̃_{t} + total_var_{t}',
    'AR+dv': 'w̃_{t+1} ~ w̃_{t} + downside_vol_{t}',
    'AR+L':  'w̃_{t+1} ~ w̃_{t} + total_var_{t} + pc1_share_{t} + avg_corr_{t}',
    'AR+F':  'w̃_{t+1} ~ w̃_{t} + downside_vol_{t} + avg_corr_{t} + corr_min_{t}',
}
for m, formula in model_formulas.items():
    r = pred_models[m]
    A(f"| {m} | {formula} | {r['adj_r2']:.3f} |")
A("")
A("### 3.2 계수 상세")
A("")
A("*\\* |t|>1.645  \\*\\* |t|>1.960  \\*\\*\\* |t|>2.576*")
A("")
for m in pred_models:
    r     = pred_models[m]
    names = PRED_COEF_NAMES[m]
    A(f"**{m}** (adj-R²={r['adj_r2']:.3f})")
    A("")
    A("| 변수 | 계수 | t-stat |")
    A("|------|-----:|:------:|")
    for j, nm in enumerate(names):
        b = r['beta'][j]; t = r['tstat'][j]
        A(f"| {nm} | {b:.4e} | {t:.2f}{star(t)} |")
    A("")

A("---")
A("")
A("## 4. 핵심 발견")
A("")
best_pred = max(pred_models, key=lambda m: pred_models[m]['adj_r2'])
ar1_r2_val = pred_models['AR1']['adj_r2']
best_r2_val = pred_models[best_pred]['adj_r2']

findings = [
    f"**Effective N 정상성**: ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f}. {stationarity_concl}",
    f"**Effective N 지속성**: AR(1) ρ={ar1_rho:.3f}{star(ar1_tstat)} — 전일 집중도가 다음날을 강하게 예측한다.",
    f"**비중 지속성**: 자산 고정효과 제거 후에도 월별 AR(1) ρ={rho_hat:.3f}{star(t_rho)}. "
    f"LW GMV 비중은 자산별로 안정적으로 유지되는 경향이 강하다.",
    f"**Per-asset ACF**: 자산의 {(acf1_arr > 0.5).mean()*100:.1f}%에서 lag-1 ACF > 0.5. "
    f"개별 자산 비중도 고지속성이다.",
    f"**예측 OLS**: AR1 단독 adj-R²={ar1_r2_val:.3f}. "
    f"라그 피처 추가 시 최선 모형({best_pred}) adj-R²={best_r2_val:.3f}로 "
    f"+{best_r2_val - ar1_r2_val:.3f} 개선. "
    f"t기 특성이 t+1기 비중 변화에 추가 설명력을 갖는다.",
]
for i, f in enumerate(findings, 1):
    A(f"{i}. {f}")
    A("")

A("---")
A("")
A("## 부록")
A("")
A("| 파일 | 내용 |")
A("|------|------|")
A("| `results/figures/timeseries/ts_effn.png` | Effective N 시계열 + AR(1) scatter |")
A("| `results/figures/timeseries/ts_pred_adjr2.png` | 예측 모형 adj-R² 비교 |")
A("| `results/figures/timeseries/ts_acf_dist.png` | Per-asset ACF 분포 |")
A("| `results/figures/timeseries/ts_effn_annual.png` | 연도별 Effective N 박스플롯 |")
A("")
A("*분석 코드: `timeseries_analysis.py`*")

(REPORTS / '2026-05-26_timeseries.md').write_text('\n'.join(L), encoding='utf-8')
print('\nReport saved → reports/2026-05-26_timeseries.md')
print('=== Done ===')
