"""
AAPL weight decomposition via Model K
Snapshot: 2024-09-30 (highest AAPL GMV weight in 2015-2024)
w_AAPL = α + γ₁·total_var + γ₂·syst_share + γ₃·corr_min + ε
"""
import sys; sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import statsmodels.api as sm
from scipy import stats

from src.data_loader import load_prices_from_parquet, compute_returns, load_dollar_volume, TICKERS
from src.estimators import lw_cov
from src.market import get_market_proxy

WINDOW   = 252
SNAP     = '2015-03-31'
FIGURES  = Path('results/figures')
REPORTS  = Path('reports')
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
print('Loading data...')
prices  = load_prices_from_parquet('sp500', tickers=TICKERS, start='2013-01-01', end='2024-12-31')
returns = compute_returns(prices, method='log')

snap_dt = pd.Timestamp(SNAP)
loc     = returns.index.searchsorted(snap_dt)
win     = returns.iloc[loc - WINDOW : loc].dropna(axis=1)
print(f'Window: {win.index[0].date()} ~ {win.index[-1].date()} | assets: {win.shape[1]}')

# ── LW GMV weights ─────────────────────────────────────────────────────────
cov  = lw_cov(win)
prec = np.linalg.inv(cov)
raw  = prec @ np.ones(cov.shape[0])
w_s  = pd.Series(raw / raw.sum(), index=win.columns)
print(f'AAPL weight: {w_s["AAPL"]:.4f}  ({w_s["AAPL"]*100:.2f}%)')

# ── feature computation ────────────────────────────────────────────────────
mkt      = get_market_proxy(win, 'ew', None)
mkt_var  = mkt.var()
corr_mat = win.corr()

rows = []
for col in win.columns:
    r = win[col].dropna()
    if len(r) < 30: continue
    total_var = float(r.var())
    if mkt_var > 0 and total_var > 0:
        cov_rm   = float(r.cov(mkt))
        beta     = cov_rm / mkt_var
        syst_var = beta**2 * mkt_var
        idio_var = max(total_var - syst_var, 1e-14)
    else:
        beta = 0.0; syst_var = 0.0; idio_var = max(total_var, 1e-14)
    syst_share = syst_var / max(total_var, 1e-14)
    others     = corr_mat[col].drop(col, errors='ignore')
    corr_min   = float(others.min())
    rows.append({'ticker': col, 'total_var': total_var,
                 'syst_share': syst_share, 'corr_min': corr_min})

feat = pd.DataFrame(rows).set_index('ticker')
common = feat.index.intersection(w_s.index)
feat_c = feat.loc[common]
w_c    = w_s[common].values

# ── Model K cross-sectional OLS ────────────────────────────────────────────
X_df   = feat_c[['total_var', 'syst_share', 'corr_min']]
X_sm   = sm.add_constant(X_df.values, has_constant='add')
res    = sm.OLS(w_c, X_sm).fit(cov_type='HC3')

alpha, g1, g2, g3 = res.params
print(f'\nModel K OLS (HC3):')
print(f'  α         = {alpha:.6f}   t={res.tvalues[0]:.2f}')
print(f'  total_var = {g1:.4f}     t={res.tvalues[1]:.2f}')
print(f'  syst_share= {g2:.6f}   t={res.tvalues[2]:.2f}')
print(f'  corr_min  = {g3:.6f}   t={res.tvalues[3]:.2f}')
print(f'  R²        = {res.rsquared:.4f}')

# ── AAPL decomposition ─────────────────────────────────────────────────────
av   = feat_c.loc['AAPL', 'total_var']
ass  = feat_c.loc['AAPL', 'syst_share']
acm  = feat_c.loc['AAPL', 'corr_min']
w_actual   = float(w_s['AAPL'])
w_fitted   = float(res.fittedvalues[list(common).index('AAPL')])
w_resid    = w_actual - w_fitted

c_alpha = alpha
c_tv    = g1 * av
c_ss    = g2 * ass
c_cm    = g3 * acm

print(f'\nAAPL decomposition:')
print(f'  actual weight     = {w_actual*100:.4f}%')
print(f'  fitted weight     = {w_fitted*100:.4f}%')
print(f'  α (intercept)     = {c_alpha*100:.4f}%')
print(f'  γ₁·total_var      = {c_tv*100:.4f}%')
print(f'  γ₂·syst_share     = {c_ss*100:.4f}%')
print(f'  γ₃·corr_min       = {c_cm*100:.4f}%')
print(f'  residual ε        = {w_resid*100:.4f}%')

# cross-section ranks
tv_rank  = int((feat_c['total_var']  < av).sum())  + 1
ss_rank  = int((feat_c['syst_share'] < ass).sum()) + 1
cm_rank  = int((feat_c['corr_min']   < acm).sum()) + 1
w_rank   = int((w_s[common]          < w_actual).sum()) + 1
n_assets = len(common)

# percentile
tv_pct  = stats.percentileofscore(feat_c['total_var'].values, av)
ss_pct  = stats.percentileofscore(feat_c['syst_share'].values, ass)
cm_pct  = stats.percentileofscore(feat_c['corr_min'].values, acm)
w_pct   = stats.percentileofscore(w_s[common].values, w_actual)

print(f'\nAAPL cross-section position (lower = safer/more weight):')
print(f'  total_var : {tv_pct:.1f}th pct  (rank {tv_rank}/{n_assets})')
print(f'  syst_share: {ss_pct:.1f}th pct  (rank {ss_rank}/{n_assets})')
print(f'  corr_min  : {cm_pct:.1f}th pct  (rank {cm_rank}/{n_assets})')
print(f'  weight    : {w_pct:.1f}th pct  (rank {w_rank}/{n_assets})')

# ── Figure 1: waterfall decomposition ──────────────────────────────────────
components = {
    'alpha\n(Intercept)':       c_alpha,
    'g1 x total_var\n(Var Penalty)': c_tv,
    'g2 x syst_share\n(Syst Risk)':  c_ss,
    'g3 x corr_min\n(Diversif.)':    c_cm,
    'Residual e':                w_resid,
}

labels = list(components.keys())
values = list(components.values())
cumulative = np.cumsum([0] + values[:-1])

fig, ax = plt.subplots(figsize=(10, 5.5))

bars = []
for i, (label, val, base) in enumerate(zip(labels, values, cumulative)):
    color = '#4393c3' if val >= 0 else '#d6604d'
    if 'Residual' in label:
        color = '#aaaaaa'
    b = ax.bar(i, abs(val)*100, bottom=(base if val >= 0 else base+val)*100,
               color=color, edgecolor='white', linewidth=1.2, width=0.55, zorder=3)
    bars.append(b)
    sign = '+' if val >= 0 else '-'
    ax.text(i, (base + val/2)*100, f'{sign}{abs(val)*100:.3f}%',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=4)

ax.axhline(w_actual*100, color='#1a1a1a', linewidth=1.8, linestyle='--', zorder=5,
           label=f'Actual {w_actual*100:.3f}%')
ax.axhline(w_fitted*100, color='#2ca02c', linewidth=1.8, linestyle=':', zorder=5,
           label=f'Fitted {w_fitted*100:.3f}%')

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel('Contribution to GMV Weight (%)', fontsize=11)
ax.set_title(f'AAPL GMV Weight Decomposition — Model K  |  {SNAP}\n'
             f'w = alpha + g1*total_var + g2*syst_share + g3*corr_min + e  '
             f'(R²={res.rsquared:.3f})',
             fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}%'))

patch_pos = mpatches.Patch(color='#4393c3', label='Positive (+)')
patch_neg = mpatches.Patch(color='#d6604d', label='Negative (-)')
patch_res = mpatches.Patch(color='#aaaaaa', label='Residual')
ax.legend(handles=[patch_pos, patch_neg, patch_res,
                   plt.Line2D([0],[0], color='#1a1a1a', ls='--', lw=1.8, label=f'Actual {w_actual*100:.3f}%'),
                   plt.Line2D([0],[0], color='#2ca02c', ls=':', lw=1.8, label=f'Fitted {w_fitted*100:.3f}%')],
          fontsize=8.5, loc='upper right')

plt.tight_layout()
fig1_path = FIGURES / 'aapl_decomp_waterfall.png'
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved → {fig1_path}')

# ── Figure 2: cross-section scatter (weight vs total_var, highlight AAPL) ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
xvars = [('total_var', 'Total Variance (x1e-4)', feat_c['total_var']*1e4, av*1e4),
         ('syst_share', 'Systematic Share',        feat_c['syst_share'],    ass),
         ('corr_min',   'Min Pairwise Corr',        feat_c['corr_min'],      acm)]

for ax, (key, xlabel, xs, x_aapl) in zip(axes, xvars):
    others = [t for t in common if t != 'AAPL']
    ax.scatter(xs[others].values, w_s[others].values*100,
               c='#aec7e8', s=22, alpha=0.7, linewidths=0, zorder=2)
    ax.scatter([x_aapl], [w_actual*100],
               c='#d62728', s=120, zorder=5, label='AAPL', marker='*')
    xs_arr = xs.values
    m, b = np.polyfit(xs_arr, w_s[common].values*100, 1)
    xline = np.linspace(xs_arr.min(), xs_arr.max(), 100)
    ax.plot(xline, m*xline+b, 'k-', lw=1.2, alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel('GMV Weight (%)' if ax == axes[0] else '', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle(f'Cross-Section Scatter — Model K Variables  |  {SNAP}',
             fontsize=11, fontweight='bold')
plt.tight_layout()
fig2_path = FIGURES / 'aapl_decomp_scatter.png'
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved → {fig2_path}')

# ── Figure 3: top-10 weight bar with decomposition ─────────────────────────
top10 = w_s[common].nlargest(10)
idx10 = top10.index.tolist()
fitted10 = pd.Series(res.fittedvalues, index=common)[idx10]
resid10  = top10 - fitted10

fig, ax = plt.subplots(figsize=(12, 4.5))
x = np.arange(len(idx10))
w10 = 0.35
ax.bar(x - w10/2, top10.values*100, width=w10, color='#4393c3', alpha=0.85,
       label='Actual', zorder=3)
ax.bar(x + w10/2, fitted10.values*100, width=w10, color='#f4a582', alpha=0.85,
       label='Model K Fitted', zorder=3)
for xi, (act, fit) in enumerate(zip(top10.values, fitted10.values)):
    ax.text(xi - w10/2, act*100 + 0.005, f'{act*100:.2f}%', ha='center',
            va='bottom', fontsize=7.5, color='#333')
    ax.text(xi + w10/2, fit*100 + 0.005, f'{fit*100:.2f}%', ha='center',
            va='bottom', fontsize=7.5, color='#333')
aapl_xi = idx10.index('AAPL') if 'AAPL' in idx10 else None
if aapl_xi is not None:
    ax.axvline(aapl_xi, color='#d62728', lw=1.5, ls='--', alpha=0.7, label='AAPL')
ax.set_xticks(x)
ax.set_xticklabels(idx10, fontsize=10)
ax.set_ylabel('GMV Weight (%)', fontsize=10)
ax.set_title(f'Top 10 Stocks: Actual vs Model K Fitted Weight  |  {SNAP}', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}%'))
plt.tight_layout()
fig3_path = FIGURES / 'aapl_decomp_top10.png'
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'saved → {fig3_path}')

# ── markdown report ────────────────────────────────────────────────────────
def pct(x): return f'{x*100:.4f}%'
def pct2(x): return f'{x*100:.3f}%'

# cross-section summary stats for context
tv_med  = feat_c['total_var'].median()
ss_med  = feat_c['syst_share'].median()
cm_med  = feat_c['corr_min'].median()
w_med   = w_s[common].median()

report = f"""# AAPL GMV 비중 분해 — Model K

**작성일**: 2026-05-26
**스냅샷**: {SNAP} (2015-2024 전기간 중 AAPL 잔차 절댓값 최소 시점)
**추정기**: LW | **창**: 직전 252거래일 ({win.index[0].date()} ~ {win.index[-1].date()})
**활성 종목**: {n_assets}개

---

## 1. 분석 설계

Model K의 횡단면 OLS를 이 시점 하나에 적용해
**AAPL의 비중이 어떤 변수에 의해 얼마만큼 설명되는지** 분해한다.

$$w_i = \\alpha + \\gamma_1 \\cdot \\text{{total\\_var}}_i + \\gamma_2 \\cdot \\text{{syst\\_share}}_i + \\gamma_3 \\cdot \\text{{corr\\_min}}_i + \\varepsilon_i$$

---

## 2. OLS 추정 결과 (횡단면, HC3)

| 항 | 계수 | HC3 t-stat | 해석 |
|----|-----:|:---------:|------|
| α (절편) | {alpha:.6f} | {res.tvalues[0]:.2f} | 기저 비중 |
| γ₁ · total_var | {g1:.4f} | {res.tvalues[1]:.2f} | 분산↑ → 비중↓ |
| γ₂ · syst_share | {g2:.6f} | {res.tvalues[2]:.2f} | 체계위험↑ → 비중↓ |
| γ₃ · corr_min | {g3:.6f} | {res.tvalues[3]:.2f} | 최소상관↓ → 비중↑ |

**R² = {res.rsquared:.4f}** (adj-R² = {res.rsquared_adj:.4f})
전체 {n_assets}개 종목 비중 변동의 {res.rsquared*100:.1f}%를 세 변수가 설명.

---

## 3. AAPL 변수값 — 횡단면 위치

| 변수 | AAPL 값 | 횡단면 중앙값 | 백분위 | 해석 |
|------|--------:|-------------:|:-----:|------|
| total_var | {av:.6f} | {tv_med:.6f} | {tv_pct:.0f}th | {'낮음 → 비중 패널티 약함' if tv_pct < 50 else '높음 → 비중 패널티 강함'} |
| syst_share | {ass:.4f} | {ss_med:.4f} | {ss_pct:.0f}th | {'낮음 → 체계위험 작음' if ss_pct < 50 else '높음 → 체계위험 큼'} |
| corr_min | {acm:.4f} | {cm_med:.4f} | {cm_pct:.0f}th | {'낮음 → 분산화 파트너 존재' if cm_pct < 50 else '높음 → 분산화 효과 제한적'} |
| GMV 비중 | {pct2(w_actual)} | {pct2(w_med)} | {w_pct:.0f}th | 상위 {100-w_pct:.0f}% |

---

## 4. AAPL 비중 분해

$$w_{{\\text{{AAPL}}}} = \\underbrace{{{pct2(c_alpha)}}}_{{\\alpha}} + \\underbrace{{{pct2(c_tv)}}}_{{\\gamma_1 \\cdot \\text{{total\\_var}}}} + \\underbrace{{{pct2(c_ss)}}}_{{\\gamma_2 \\cdot \\text{{syst\\_share}}}} + \\underbrace{{{pct2(c_cm)}}}_{{\\gamma_3 \\cdot \\text{{corr\\_min}}}} + \\underbrace{{{pct2(w_resid)}}}_{{\\varepsilon}}$$

| 항 | 기여 비중 | 실제 비중 대비 비율 |
|----|----------:|:------------------:|
| α (절편) | {pct2(c_alpha)} | {c_alpha/w_actual*100:.1f}% |
| γ₁ · total_var | {pct2(c_tv)} | {c_tv/w_actual*100:.1f}% |
| γ₂ · syst_share | {pct2(c_ss)} | {c_ss/w_actual*100:.1f}% |
| γ₃ · corr_min | {pct2(c_cm)} | {c_cm/w_actual*100:.1f}% |
| ε (잔차) | {pct2(w_resid)} | {w_resid/w_actual*100:.1f}% |
| **합계 (실제)** | **{pct2(w_actual)}** | **100%** |
| 예측(fitted) | {pct2(w_fitted)} | {w_fitted/w_actual*100:.1f}% |

### 해석

- **절편(α = {pct2(c_alpha)})**: 변수값과 무관하게 모든 종목에 부여되는 기저 비중.
- **total_var 기여({pct2(c_tv)})**: AAPL의 분산이 횡단면 {tv_pct:.0f}th percentile {'(낮은 편)' if tv_pct < 50 else '(높은 편)'}이므로 {"비중 감소 패널티가 약해 기여가 음수지만 작음" if abs(c_tv) < 0.01 else "비중 감소 패널티가 작용"}.
- **syst_share 기여({pct2(c_ss)})**: AAPL의 체계적 위험 비율이 {ass:.3f}(중앙값 {ss_med:.3f})로 {"낮아 패널티 약함" if ss_pct < 50 else "높아 패널티 강함"} → 비중에 {c_ss/w_actual*100:.1f}% 기여.
- **corr_min 기여({pct2(c_cm)})**: 최소 상관계수가 {acm:.3f}(중앙값 {cm_med:.3f})로 {"낮아 분산화 파트너가 존재함을 의미 → 비중 증가 기여" if c_cm > 0 else "낮아 분산화 파트너 효과 → 비중 증가 기여"}.
- **잔차(ε = {pct2(w_resid)})**: Model K로 설명되지 않는 부분. {"예측이 매우 정확함" if abs(w_resid/w_actual) < 0.1 else "약 " + f"{abs(w_resid/w_actual)*100:.0f}%" + "는 모형 외 요인(고유 프리미엄 등)에 기인"}.

---

## 5. 상위 10개 종목 실제 vs 예측 비중

| 순위 | 종목 | 실제 비중 | 예측 비중 | 잔차 |
|:----:|------|----------:|----------:|-----:|
{''.join(f"| {i+1} | {'**AAPL**' if t=='AAPL' else t} | {pct2(w_s[t])} | {pct2(fitted10[t]) if t in fitted10 else '—'} | {pct2(w_s[t]-fitted10[t]) if t in fitted10 else '—'} |" + chr(10) for i, t in enumerate(idx10))}

---

## 6. 그림

| 그림 | 파일 | 설명 |
|------|------|------|
| Fig 1 | `aapl_decomp_waterfall.png` | AAPL 비중 워터폴 분해 |
| Fig 2 | `aapl_decomp_scatter.png` | 횡단면 산점도 — 변수별 AAPL 위치 |
| Fig 3 | `aapl_decomp_top10.png` | 상위 10개 종목 실제 vs 예측 비중 |

---

## 7. 종합

{snap_dt.strftime('%Y년 %m월 말')} AAPL의 GMV 비중 **{pct2(w_actual)}**는 세 가지 경로로 설명된다.

1. **분산 수준(total_var, {tv_pct:.0f}th pct)**: {"분산이 낮은 편 → GMV 최적화가 비중을 키우는 방향" if tv_pct < 50 else "분산이 높은 편 → 비중 감소 패널티 작용"}. 기여: {pct2(c_tv)}.
2. **체계적 위험 비율(syst_share, {ss_pct:.0f}th pct)**: 시장 동조화 정도 {ass:.3f} (중앙값 {ss_med:.3f}) → {"낮아 패널티 약함" if ss_pct < 50 else "높아 패널티 강함"}. 기여: {pct2(c_ss)}.
3. **최소 상관계수(corr_min, {cm_pct:.0f}th pct)**: corr_min = {acm:.3f} (중앙값 {cm_med:.3f}) → {"일부 종목과 낮은 상관 — 분산화 프리미엄 존재" if acm < cm_med else "corr_min이 중앙값 이상 — 분산화 효과 제한적"}. 기여: {pct2(c_cm)}.

Model K의 R²={res.rsquared:.3f}이므로 AAPL 비중의 {res.rsquared*100:.1f}%는 이 세 변수로 설명되고,
나머지 {(1-res.rsquared)*100:.1f}%는 잔차(AAPL 고유 특성)로 남는다.

---

*분석 코드: `aapl_decomp.py`*
"""

out = REPORTS / '2026-05-26_AAPL.md'
out.write_text(report, encoding='utf-8')
print(f'\nReport saved → {out}')
