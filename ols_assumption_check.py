"""
OLS Assumption Verification
2026-05-26
Data: S&P 100, 2015-2024, LW estimator
Rolling window: 252 trading days, month-end, ~108 windows
Model D: w = α + γ₁·total_var + γ₂·syst_share
Model F: w = α + γ₁·downside_vol + γ₂·inv_idio_var + γ₃·avg_corr + γ₄·corr_min
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset, het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.data_loader import load_prices_from_parquet, compute_returns, load_dollar_volume, TICKERS
from src.estimators import lw_cov
from src.market import get_market_proxy

WINDOW = 252
REPORTS = Path('reports')
REPORTS.mkdir(parents=True, exist_ok=True)

# ── data loading ──────────────────────────────────────────────────────────────
print('Loading data 2015-2024...')
prices  = load_prices_from_parquet('sp500', tickers=TICKERS, start='2013-01-01', end='2024-12-31')
returns = compute_returns(prices, method='log')
dolvol  = load_dollar_volume('sp500', tickers=TICKERS, start='2013-01-01', end='2024-12-31')
print(f'Returns: {returns.shape}')

# ── helper: GMV unconstrained (LW) ────────────────────────────────────────────
def gmv_weights(cov: np.ndarray) -> np.ndarray | None:
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)
    raw = prec @ np.ones(cov.shape[0])
    total = raw.sum()
    if abs(total) < 1e-10:
        return None
    return raw / total


# ── helper: feature computation ──────────────────────────────────────────────
def compute_features(win: pd.DataFrame) -> pd.DataFrame:
    mkt     = get_market_proxy(win, 'ew', None)
    mkt_var = mkt.var()
    corr_mat = win.corr()
    dv_win  = dolvol.reindex(index=win.index, columns=win.columns)

    rows = []
    for col in win.columns:
        r = win[col].dropna()
        if len(r) < 30:
            continue
        total_var = float(r.var())

        if mkt_var > 0 and total_var > 0:
            cov_rm   = float(r.cov(mkt))
            beta     = cov_rm / mkt_var
            syst_var = beta**2 * mkt_var
            idio_var = max(total_var - syst_var, 1e-14)
        else:
            beta = 0.0; syst_var = 0.0
            idio_var = max(total_var, 1e-14)

        syst_share   = syst_var / max(total_var, 1e-14)
        inv_idio_var = 1.0 / idio_var

        # downside vol
        r_arr  = r.values
        neg_r  = r_arr[r_arr < 0]
        down_vol = float(np.std(neg_r)) if len(neg_r) > 5 else np.nan

        # avg_corr (mean pairwise excl. self)
        if col in corr_mat.columns:
            others   = corr_mat[col].drop(col, errors='ignore')
            avg_corr = float(others.mean())
            corr_min = float(others.min())
        else:
            avg_corr = np.nan
            corr_min = np.nan

        rows.append({
            'ticker': col,
            'total_var': total_var,
            'syst_share': syst_share,
            'downside_vol': down_vol,
            'inv_idio_var': inv_idio_var,
            'avg_corr': avg_corr,
            'corr_min': corr_min,
        })
    return pd.DataFrame(rows).set_index('ticker')


# ── generate month-end rolling windows ───────────────────────────────────────
all_month_ends = returns.loc['2015-01-01':'2024-12-31'].resample('ME').last().index
window_dates   = []
for d in all_month_ends:
    loc = returns.index.searchsorted(d)
    if loc >= WINDOW:
        window_dates.append(d)
print(f'Valid rolling windows: {len(window_dates)}')


# ── per-window test records ───────────────────────────────────────────────────
records_D = []
records_F = []

for t_idx, end_date in enumerate(window_dates):
    if t_idx % 20 == 0:
        print(f'  Window {t_idx+1}/{len(window_dates)}: {end_date.date()}')

    loc = returns.index.searchsorted(end_date)
    win = returns.iloc[loc - WINDOW: loc].dropna(axis=1)
    if win.shape[1] < 10 or win.shape[0] < WINDOW // 2:
        continue

    # LW covariance
    try:
        cov = lw_cov(win)
        w_arr = gmv_weights(cov)
        if w_arr is None:
            continue
        w = pd.Series(w_arr, index=win.columns)
    except Exception:
        continue

    # features
    try:
        feat = compute_features(win)
    except Exception:
        continue

    common = feat.index.intersection(w.index)
    if len(common) < 10:
        continue

    feat_c = feat.loc[common]
    w_c    = w[common].values

    def run_tests(y, X_df, model_label):
        """Run all OLS assumption tests. X_df has no intercept yet."""
        n = len(y)
        if n < 8:
            return None
        # drop NaN rows
        df_tmp = X_df.copy()
        df_tmp['_y'] = y
        df_tmp = df_tmp.dropna()
        if len(df_tmp) < 8:
            return None
        y_ = df_tmp['_y'].values
        X_ = df_tmp.drop(columns='_y').values
        var_names = list(df_tmp.drop(columns='_y').columns)

        X_sm = sm.add_constant(X_, has_constant='add')
        try:
            ols_res = sm.OLS(y_, X_sm).fit()
        except Exception:
            return None

        resid   = ols_res.resid
        fitted  = ols_res.fittedvalues
        n_obs   = len(resid)
        k_vars  = X_.shape[1]

        rec = {'date': end_date, 'n': n_obs, 'model': model_label}

        # ── 1. Linearity: RESET ───────────────────────────────────────────────
        try:
            reset_res = linear_reset(ols_res, power=2, use_f=True)
            rec['reset_pval'] = float(reset_res.pvalue)
        except Exception:
            rec['reset_pval'] = np.nan

        # ── 3. Exogeneity: residual mean ──────────────────────────────────────
        rec['resid_mean'] = float(resid.mean())

        # ── 4. Homoscedasticity: BP + White ───────────────────────────────────
        try:
            _, bp_p, _, _ = het_breuschpagan(resid, X_sm)
            rec['bp_pval'] = float(bp_p)
        except Exception:
            rec['bp_pval'] = np.nan
        try:
            _, white_p, _, _ = het_white(resid, X_sm)
            rec['white_pval'] = float(white_p)
        except Exception:
            rec['white_pval'] = np.nan

        # ── 5. No autocorrelation: Durbin-Watson ─────────────────────────────
        try:
            rec['dw_stat'] = float(durbin_watson(resid))
        except Exception:
            rec['dw_stat'] = np.nan

        # ── 8. Normality: Jarque-Bera ─────────────────────────────────────────
        try:
            jb_stat, jb_p = stats.jarque_bera(resid)
            rec['jb_pval']     = float(jb_p)
            rec['resid_skew']  = float(stats.skew(resid))
            rec['resid_kurt']  = float(stats.kurtosis(resid))
        except Exception:
            rec['jb_pval']     = np.nan
            rec['resid_skew']  = np.nan
            rec['resid_kurt']  = np.nan

        # ── VIF ───────────────────────────────────────────────────────────────
        X_vif = sm.add_constant(X_, has_constant='add')
        for j, vname in enumerate(var_names):
            try:
                vif_val = variance_inflation_factor(X_vif, j + 1)  # +1 skips const
            except Exception:
                vif_val = np.nan
            rec[f'vif_{vname}'] = float(vif_val)

        # ── HC3 robust re-estimation ──────────────────────────────────────────
        try:
            ols_hc3 = sm.OLS(y_, X_sm).fit(cov_type='HC3')
            for j, vname in enumerate(var_names):
                rec[f'hc3_t_{vname}']    = float(ols_hc3.tvalues[j + 1])
                rec[f'hc3_pval_{vname}'] = float(ols_hc3.pvalues[j + 1])
            rec['ols_r2'] = float(ols_res.rsquared)
        except Exception:
            pass

        return rec

    # Model D: total_var + syst_share
    Xd = feat_c[['total_var', 'syst_share']].copy()
    rec_d = run_tests(w_c, Xd, 'D')
    if rec_d:
        records_D.append(rec_d)

    # Model K: total_var + syst_share + corr_min
    Xf = feat_c[['total_var', 'syst_share', 'corr_min']].copy()
    rec_f = run_tests(w_c, Xf, 'K')
    if rec_f:
        records_F.append(rec_f)

df_D = pd.DataFrame(records_D).set_index('date') if records_D else pd.DataFrame()
df_F = pd.DataFrame(records_F).set_index('date') if records_F else pd.DataFrame()
print(f'\nModel D windows: {len(df_D)}, Model F windows: {len(df_F)}')


# ── aggregation helpers ───────────────────────────────────────────────────────
def agg(series, alpha=0.05):
    s = series.dropna()
    if len(s) == 0:
        return dict(mean=np.nan, median=np.nan, sig_pct=np.nan, n=0)
    return dict(
        mean   = float(s.mean()),
        median = float(s.median()),
        sig_pct = float((s < alpha).mean() * 100),
        n      = len(s),
    )

def agg_dw(series):
    s = series.dropna()
    if len(s) == 0:
        return dict(mean=np.nan, median=np.nan, pct_low=np.nan, pct_high=np.nan)
    return dict(
        mean     = float(s.mean()),
        median   = float(s.median()),
        pct_low  = float((s < 1.5).mean() * 100),
        pct_high = float((s > 2.5).mean() * 100),
    )

def agg_vif(df, col):
    if col not in df.columns:
        return dict(mean=np.nan, max=np.nan, pct_high=np.nan)
    s = df[col].dropna()
    if len(s) == 0:
        return dict(mean=np.nan, max=np.nan, pct_high=np.nan)
    return dict(
        mean     = float(s.mean()),
        max      = float(s.max()),
        pct_high = float((s > 5).mean() * 100),
    )

def verdict(sig_pct, thresh=25.0):
    if np.isnan(sig_pct):
        return '❓'
    if sig_pct < thresh:
        return '✅'
    elif sig_pct < 50:
        return '⚠️'
    return '❌'


# ── compute summary stats ─────────────────────────────────────────────────────
def compute_summary(df, label):
    s = {}
    s['label'] = label
    s['n_windows'] = len(df)

    # 1. Linearity (RESET)
    s['reset'] = agg(df['reset_pval'])

    # 3. Exogeneity (residual mean t-test)
    if 'resid_mean' in df.columns:
        rm = df['resid_mean'].dropna()
        if len(rm) > 5:
            t_stat, p_val = stats.ttest_1samp(rm, 0)
            s['exog'] = dict(mean=float(rm.mean()), std=float(rm.std()),
                             tstat=float(t_stat), pval=float(p_val))
        else:
            s['exog'] = dict(mean=np.nan, std=np.nan, tstat=np.nan, pval=np.nan)
    else:
        s['exog'] = dict(mean=np.nan, std=np.nan, tstat=np.nan, pval=np.nan)

    # 4. Homoscedasticity
    s['bp']    = agg(df['bp_pval'])
    s['white'] = agg(df['white_pval'])

    # 5. No autocorrelation (DW)
    s['dw'] = agg_dw(df['dw_stat'])

    # 8. Normality (JB)
    s['jb'] = agg(df['jb_pval'])
    if 'resid_skew' in df.columns:
        s['skew_mean'] = float(df['resid_skew'].dropna().mean())
        s['kurt_mean'] = float(df['resid_kurt'].dropna().mean())
    else:
        s['skew_mean'] = s['kurt_mean'] = np.nan

    return s

sumD = compute_summary(df_D, 'Model D')
sumF = compute_summary(df_F, 'Model K')


# ── VIF aggregation ───────────────────────────────────────────────────────────
vif_D = {
    'total_var':  agg_vif(df_D, 'vif_total_var'),
    'syst_share': agg_vif(df_D, 'vif_syst_share'),
}
vif_F = {
    'total_var':  agg_vif(df_F, 'vif_total_var'),
    'syst_share': agg_vif(df_F, 'vif_syst_share'),
    'corr_min':   agg_vif(df_F, 'vif_corr_min'),
}


# ── HC3 robustness check ──────────────────────────────────────────────────────
def hc3_summary(df, var_names):
    rows = []
    for v in var_names:
        t_col = f'hc3_t_{v}'
        p_col = f'hc3_pval_{v}'
        if t_col in df.columns:
            t_s = df[t_col].dropna()
            p_s = df[p_col].dropna()
            rows.append({
                'var': v,
                'mean_t': float(t_s.mean()),
                'sig_pct': float((p_s < 0.05).mean() * 100) if len(p_s) > 0 else np.nan,
            })
    return rows

hc3_D = hc3_summary(df_D, ['total_var', 'syst_share'])
hc3_F = hc3_summary(df_F, ['total_var', 'syst_share', 'corr_min'])


# ── write report ──────────────────────────────────────────────────────────────
def fmt(x, dec=3):
    if np.isnan(x):
        return '—'
    return f'{x:.{dec}f}'


lines = []
lines.append('# OLS 가정 검증 보고서')
lines.append('')
lines.append('**작성일**: 2026-05-26')
lines.append(f'**데이터**: S&P 100, 2015-2024')
lines.append(f'**추정기**: LW | **롤링 창**: {len(window_dates)}개')
lines.append(f'**Model D** 실행 창: {sumD["n_windows"]}개 | **Model K** 실행 창: {sumF["n_windows"]}개')
lines.append('')
lines.append('---')
lines.append('')
lines.append('## 요약 테이블')
lines.append('')
lines.append('| 가정 | 검정 방법 | 모형 | 평균 p-value | 유의(p<0.05) 창 비율 | 판정 |')
lines.append('|------|----------|------|:-----------:|:------------------:|:----:|')

for label, s in [('D', sumD), ('K', sumF)]:
    v = verdict(s['reset']['sig_pct'])
    lines.append(f'| 1. 선형성 | RESET | {label} | {fmt(s["reset"]["mean"])} | {fmt(s["reset"]["sig_pct"],1)}% | {v} |')

for label, s in [('D', sumD), ('K', sumF)]:
    if np.isnan(s['exog']['pval']):
        v = '❓'
    elif s['exog']['pval'] > 0.05:
        v = '✅'
    else:
        v = '❌'
    lines.append(f'| 3. 외생성 | 잔차 평균 t-test | {label} | {fmt(s["exog"]["pval"])} (t-test p) | mean={fmt(s["exog"]["mean"],5)} | {v} |')

for label, s in [('D', sumD), ('K', sumF)]:
    v = verdict(s['bp']['sig_pct'])
    lines.append(f'| 4. 등분산 (BP) | Breusch-Pagan | {label} | {fmt(s["bp"]["mean"])} | {fmt(s["bp"]["sig_pct"],1)}% | {v} |')
    v = verdict(s['white']['sig_pct'])
    lines.append(f'| 4. 등분산 (White) | White Test | {label} | {fmt(s["white"]["mean"])} | {fmt(s["white"]["sig_pct"],1)}% | {v} |')

for label, s in [('D', sumD), ('K', sumF)]:
    lines.append(f'| 5. 무상관 | Durbin-Watson | {label} | DW={fmt(s["dw"]["mean"])} | DW<1.5: {fmt(s["dw"]["pct_low"],1)}% | {"✅" if s["dw"]["pct_low"] < 20 else "⚠️"} |')

for label, s in [('D', sumD), ('K', sumF)]:
    v = verdict(s['jb']['sig_pct'], thresh=50)
    lines.append(f'| 8. 정규성 | Jarque-Bera | {label} | {fmt(s["jb"]["mean"])} | {fmt(s["jb"]["sig_pct"],1)}% | {v} |')

lines.append('')
lines.append('---')
lines.append('')

# ── Section 1: Linearity ──────────────────────────────────────────────────────
lines.append('## 1. 선형성 (Ramsey RESET Test)')
lines.append('')
lines.append('**H₀**: 선형 모형으로 충분 (기각 시 비선형성 존재)')
lines.append('')
lines.append('| 모형 | 평균 p-value | 중앙값 p-value | 유의(p<0.05) 창 비율 | 판정 |')
lines.append('|------|:-----------:|:-------------:|:------------------:|:----:|')
for label, s in [('D', sumD), ('K', sumF)]:
    r = s['reset']
    v = verdict(r['sig_pct'])
    lines.append(f'| Model {label} | {fmt(r["mean"])} | {fmt(r["median"])} | {fmt(r["sig_pct"],1)}% | {v} |')
lines.append('')
lines.append('**해석**:')
for label, s in [('D', sumD), ('K', sumF)]:
    r = s['reset']
    lines.append(f'- Model {label}: 유의 창 {fmt(r["sig_pct"],1)}%. '
                 + ('선형성 가정 대체로 충족.' if r['sig_pct'] < 25 else
                    '상당 비율의 창에서 비선형성 존재 가능 — 비선형 변환 고려 권장.' if r['sig_pct'] < 50 else
                    '선형성 가정 심각히 위반. 비선형 변환 또는 분위수 회귀 고려 필요.'))
lines.append('')

# ── Section 3: Exogeneity ─────────────────────────────────────────────────────
lines.append('## 2. 외생성 E[ε|X]=0 (잔차 평균 t-test)')
lines.append('')
lines.append('**H₀**: 창별 잔차 평균 = 0')
lines.append('')
lines.append('| 모형 | 잔차 평균의 평균 | 잔차 평균의 std | 단측 t-statistic | p-value | 판정 |')
lines.append('|------|:--------------:|:-------------:|:---------------:|:-------:|:----:|')
for label, s in [('D', sumD), ('K', sumF)]:
    e = s['exog']
    v = '✅' if not np.isnan(e['pval']) and e['pval'] > 0.05 else '⚠️'
    lines.append(f'| Model {label} | {fmt(e["mean"],6)} | {fmt(e["std"],6)} | {fmt(e["tstat"],3)} | {fmt(e["pval"],4)} | {v} |')
lines.append('')
lines.append('**참고**: OLS 잔차 합 = 0이므로 창별 잔차 평균은 정의상 0. '
             'OLS intercept 포함 시 이 조건은 항상 충족됨 — 외생성 위반은 잔차 vs. X 패턴으로 별도 확인 필요.')
lines.append('')

# ── Section 4: Homoscedasticity ───────────────────────────────────────────────
lines.append('## 3. 등분산성 (Homoscedasticity)')
lines.append('')
lines.append('**H₀**: 오차항의 분산이 일정 (기각 시 이분산 존재)')
lines.append('')
lines.append('### Breusch-Pagan Test')
lines.append('')
lines.append('| 모형 | 평균 p-value | 중앙값 p-value | 유의(p<0.05) 창 비율 | 판정 |')
lines.append('|------|:-----------:|:-------------:|:------------------:|:----:|')
for label, s in [('D', sumD), ('K', sumF)]:
    r = s['bp']
    v = verdict(r['sig_pct'])
    lines.append(f'| Model {label} | {fmt(r["mean"])} | {fmt(r["median"])} | {fmt(r["sig_pct"],1)}% | {v} |')
lines.append('')
lines.append('### White Test')
lines.append('')
lines.append('| 모형 | 평균 p-value | 중앙값 p-value | 유의(p<0.05) 창 비율 | 판정 |')
lines.append('|------|:-----------:|:-------------:|:------------------:|:----:|')
for label, s in [('D', sumD), ('K', sumF)]:
    r = s['white']
    v = verdict(r['sig_pct'])
    lines.append(f'| Model {label} | {fmt(r["mean"])} | {fmt(r["median"])} | {fmt(r["sig_pct"],1)}% | {v} |')
lines.append('')
lines.append('### HC3 Robust 재추정 (이분산 대응)')
lines.append('')
lines.append('이분산이 탐지된 창 비율이 높은 경우 HC3 robust standard error로 재추정하여 계수 유의성을 비교함.')
lines.append('')
lines.append('**Model D HC3**:')
lines.append('')
lines.append('| 변수 | 평균 HC3 t-stat | HC3 유의(p<0.05) 창 비율 |')
lines.append('|------|:--------------:|:----------------------:|')
for row in hc3_D:
    lines.append(f'| {row["var"]} | {fmt(row["mean_t"],3)} | {fmt(row["sig_pct"],1)}% |')
lines.append('')
lines.append('**Model K HC3**:')
lines.append('')
lines.append('| 변수 | 평균 HC3 t-stat | HC3 유의(p<0.05) 창 비율 |')
lines.append('|------|:--------------:|:----------------------:|')
for row in hc3_F:
    lines.append(f'| {row["var"]} | {fmt(row["mean_t"],3)} | {fmt(row["sig_pct"],1)}% |')
lines.append('')

# ── Section 5: No Autocorrelation ─────────────────────────────────────────────
lines.append('## 4. 오차 무상관 (Durbin-Watson)')
lines.append('')
lines.append('**기준**: DW ≈ 2 (무상관), < 1.5 (양의 자기상관), > 2.5 (음의 자기상관)')
lines.append('')
lines.append('| 모형 | 평균 DW | 중앙값 DW | DW < 1.5 비율 | DW > 2.5 비율 | 판정 |')
lines.append('|------|:-------:|:--------:|:------------:|:------------:|:----:|')
for label, s in [('D', sumD), ('K', sumF)]:
    d = s['dw']
    v = '✅' if d['pct_low'] < 20 else '⚠️'
    lines.append(f'| Model {label} | {fmt(d["mean"],3)} | {fmt(d["median"],3)} | {fmt(d["pct_low"],1)}% | {fmt(d["pct_high"],1)}% | {v} |')
lines.append('')
lines.append('**참고**: 횡단면 OLS에서 DW는 자산 간 잔차 순서(알파벳 또는 임의)에 의존하므로 '
             '시계열 자기상관의 해석이 직접 적용되지 않음. 낮은 DW는 잔차 간 공간적 패턴(예: '
             '섹터 군집 효과)을 시사할 수 있음.')
lines.append('')

# ── Section 8: Normality ──────────────────────────────────────────────────────
lines.append('## 5. 잔차 정규성 (Jarque-Bera Test)')
lines.append('')
lines.append('**H₀**: 잔차가 정규분포를 따름 (기각 시 비정규성 존재)')
lines.append('')
lines.append('| 모형 | 평균 p-value | 중앙값 p-value | 유의(p<0.05) 창 비율 | 평균 왜도 | 평균 첨도(초과) | 판정 |')
lines.append('|------|:-----------:|:-------------:|:------------------:|:--------:|:--------------:|:----:|')
for label, s in [('D', sumD), ('K', sumF)]:
    r = s['jb']
    v = verdict(r['sig_pct'], thresh=50)
    lines.append(f'| Model {label} | {fmt(r["mean"])} | {fmt(r["median"])} | {fmt(r["sig_pct"],1)}% | '
                 f'{fmt(s["skew_mean"],3)} | {fmt(s["kurt_mean"],3)} | {v} |')
lines.append('')
lines.append('**정규성 위반 시 방어 논리**:')
lines.append('- 표본 크기 N ≈ 90–100 (창별 S&P 100 활성 종목): 중심극한정리(CLT)에 의해 OLS 계수 추정량은 '
             '점근적으로 정규분포를 따르므로 t-통계량 추론은 여전히 유효.')
lines.append('- 이분산이 동시에 존재하면 HC3 robust SE 사용 권장 (위 섹션 참조).')
lines.append('- Bootstrap confidence interval로 추가 검증 가능.')
lines.append('')

# ── Section 6: Multicollinearity ─────────────────────────────────────────────
lines.append('## 6. 다중공선성 (VIF)')
lines.append('')
lines.append('**기준**: VIF < 5 (허용), 5–10 (주의), > 10 (심각)')
lines.append('')
lines.append('### Model D (total_var + syst_share)')
lines.append('')
lines.append('| 변수 | 평균 VIF | 최대 VIF | VIF > 5 창 비율 |')
lines.append('|------|:--------:|:--------:|:--------------:|')
for vname, vv in vif_D.items():
    lines.append(f'| {vname} | {fmt(vv["mean"],2)} | {fmt(vv["max"],2)} | {fmt(vv["pct_high"],1)}% |')
lines.append('')

d_avg_vif = np.nanmean([vif_D[v]['mean'] for v in vif_D])
lines.append(f'Model D 평균 VIF = {d_avg_vif:.2f}. '
             + ('불완전 공선성 수준 낮음 — total_var와 syst_share는 수직에 가까운 관계.' if d_avg_vif < 5 else
                'total_var와 syst_share 간 공선성 주의 필요.'))
lines.append('')
lines.append('### Model K (total_var + syst_share + corr_min)')
lines.append('')
lines.append('| 변수 | 평균 VIF | 최대 VIF | VIF > 5 창 비율 |')
lines.append('|------|:--------:|:--------:|:--------------:|')
for vname, vv in vif_F.items():
    lines.append(f'| {vname} | {fmt(vv["mean"],2)} | {fmt(vv["max"],2)} | {fmt(vv["pct_high"],1)}% |')
lines.append('')

f_avg_vif = np.nanmean([vif_F[v]['mean'] for v in vif_F])
lines.append(f'Model K 평균 VIF = {f_avg_vif:.2f}. '
             + ('변수 간 공선성 허용 수준 — total_var, syst_share, corr_min 간 선형 종속성 낮음.' if f_avg_vif < 5 else
                'corr_min이 syst_share 또는 total_var와 공선성 의심 — corr_min 제거 또는 직교화 고려.'))
lines.append('')

# ── 종합 판정 ─────────────────────────────────────────────────────────────────
lines.append('---')
lines.append('')
lines.append('## 종합 판정')
lines.append('')
lines.append('| 가정 | Model D | Model K | 논문 대응 방안 |')
lines.append('|------|:-------:|:-------:|--------------|')

def vd(s, key, is_dw=False, is_exog=False):
    if is_exog:
        return '✅' if not np.isnan(s['exog']['pval']) and s['exog']['pval'] > 0.05 else '⚠️'
    if is_dw:
        return '✅' if s['dw']['pct_low'] < 20 else '⚠️'
    r = s[key]
    return verdict(r['sig_pct'])

lin_D = vd(sumD, 'reset')
lin_F = vd(sumF, 'reset')
exo_D = vd(sumD, None, is_exog=True)
exo_F = vd(sumF, None, is_exog=True)
bp_D  = vd(sumD, 'bp')
bp_F  = vd(sumF, 'bp')
dw_D  = vd(sumD, None, is_dw=True)
dw_F  = vd(sumF, None, is_dw=True)
jb_D  = verdict(sumD['jb']['sig_pct'], thresh=50)
jb_F  = verdict(sumF['jb']['sig_pct'], thresh=50)

lines.append(f'| 선형성 | {lin_D} | {lin_F} | 비선형 위반 시: 변수 로그/제곱근 변환 또는 분위수 회귀 |')
lines.append(f'| 외생성 | {exo_D} | {exo_F} | OLS intercept 포함 시 잔차 평균 = 0 by construction |')
lines.append(f'| 등분산 | {bp_D} | {bp_F} | HC3 robust SE 사용 (계수 불변, SE만 조정) |')
lines.append(f'| 무상관 | {dw_D} | {dw_F} | 횡단면 OLS — 섹터 FE 추가로 공간 군집 효과 흡수 |')
lines.append(f'| 정규성 | {jb_D} | {jb_F} | N≈100이므로 CLT 근거 점근 추론 유효; Bootstrap CI 추가 |')
lines.append(f'| 다중공선성 | VIF≈{d_avg_vif:.1f} | VIF≈{f_avg_vif:.1f} | VIF>5 시 corr_min 제거 또는 PCA 사용 |')
lines.append('')
lines.append('### 핵심 권고사항')
lines.append('')
lines.append('1. **이분산 대응**: BP/White 유의 창 비율이 높으면 HC3 robust SE를 기본 리포팅 기준으로 사용.')
lines.append('2. **정규성 위반**: N ≈ 90–100 조건 하 CLT 근거로 추론 가능; 논문에 명시적 방어 논리 기재 권장.')
lines.append('3. **선형성 위반**: RESET 유의 창 비율이 높으면 total_var, corr_min의 로그/제곱근 변환 시도.')
lines.append('4. **다중공선성**: Model K에서 syst_share와 corr_min의 VIF > 5 창이 많으면 corr_min을 직교화하거나 Model D로 후퇴 고려.')
lines.append('5. **횡단면 OLS 한계**: 자산 간 오차 상관(섹터 군집)은 DW로 완전히 포착되지 않으므로 섹터 FE 또는 clustered SE 사용 고려.')
lines.append('')
lines.append('---')
lines.append('')
lines.append('*분석 코드: `ols_assumption_check.py`*')

report_text = '\n'.join(lines)
out_path = REPORTS / '2026-05-26_assumption.md'
out_path.write_text(report_text, encoding='utf-8')
print(f'\nReport saved → {out_path}')
