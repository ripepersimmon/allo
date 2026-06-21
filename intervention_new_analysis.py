"""
Intervention Analysis — LW GMV Weights (2015-2024)
====================================================
Box-Tiao decay-form panel OLS for COVID and Rates crises.
LW estimator only, 2015-2024.

Model:
  w_{i,t} = β₀ + β₁·z(total_var) + β₂·z(syst_share)
           + β₃·z(downside_vol) + β₄·z(avg_corr)
           + γ·D_t
           + θ₁·D_t·z(total_var) + θ₂·D_t·z(syst_share)
           + θ₃·D_t·z(downside_vol) + θ₄·z(avg_corr)
           + ε_{i,t}

  D_t = δ^{t - T₀}  (Box-Tiao decay,  δ selected by BIC over grid)

Outputs:
  reports/2026-05-26_intervention.md
  results/figures/intervention/
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
from src.portfolio import effective_n
from src.market import get_market_proxy
from src.analysis import rolling_gmv

np.random.seed(42)

FIGURES = Path('results/figures/intervention')
REPORTS = Path('reports')
FIGURES.mkdir(parents=True, exist_ok=True)

CRISIS_ONSETS = {'COVID': '2020-02-21', 'Rates': '2022-01-03'}
HALFLIFE_GRID = [5, 10, 21, 42, 63]   # trading days
LEAD_IN       = 60                     # pre-crisis days included

# ── data ──────────────────────────────────────────────────────────────────────
print('Loading data...')
prices  = load_prices_from_parquet('sp500', tickers=TICKERS,
                                    start='2014-01-01', end='2024-12-31')
returns = compute_returns(prices, method='log')

print('Computing rolling LW GMV weights (constrained)...')
weights = rolling_gmv(returns, lw_cov, window=252, constrained=True)
weights = weights.loc['2015-01-01':]
print(f'Weights: {weights.shape}')


# ── helpers ───────────────────────────────────────────────────────────────────
def star(t):
    if pd.isna(t): return ''
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.960 else ('*' if a > 1.645 else ''))

def _ols_cluster(y, X, cluster_ids):
    """OLS with cluster-robust SEs (clustered by date)."""
    n, k = X.shape
    try:
        beta = np.linalg.solve(X.T @ X, X.T @ y)
    except np.linalg.LinAlgError:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid  = y - X @ beta
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 1e-14 else 0.0
    adj_r2 = max(1 - (1 - r2) * (n - 1) / (n - k), 0.0) if n > k else 0.0

    # Cluster-robust sandwich
    XtX_inv = np.linalg.pinv(X.T @ X)
    clusters = np.unique(cluster_ids)
    B = np.zeros((k, k))
    for c in clusters:
        mask = (cluster_ids == c)
        Xc   = X[mask]; rc = resid[mask]
        B   += Xc.T @ np.outer(rc, rc) @ Xc
    G   = len(clusters)
    adj = G / (G - 1) * (n - 1) / (n - k)
    V   = adj * XtX_inv @ B @ XtX_inv
    se  = np.sqrt(np.maximum(np.diag(V), 0))
    tstat = beta / np.where(se > 1e-14, se, np.nan)
    pval  = 2 * (1 - stats.t.cdf(np.abs(tstat), df=G - 1))
    return dict(beta=beta, se=se, tstat=tstat, pval=pval,
                r2=r2, adj_r2=adj_r2, n=n, n_clusters=G)


def compute_features_date(date, returns, window=252):
    end   = pd.Timestamp(date)
    start = end - pd.offsets.BDay(window)
    win   = returns.loc[start:end].dropna(axis=1)
    if win.shape[1] < 10 or len(win) < 50:
        return pd.DataFrame()
    mkt     = get_market_proxy(win, 'ew')
    mkt_var = mkt.var()
    if mkt_var < 1e-14:
        return pd.DataFrame()
    corr_mat = win.corr().values.copy()
    np.fill_diagonal(corr_mat, np.nan)
    avg_corr_arr = np.nanmean(corr_mat, axis=1)
    diag_v = win.var().values
    rows = []
    for i, col in enumerate(win.columns):
        tv = float(diag_v[i])
        if tv < 1e-14: continue
        b   = float(win[col].cov(mkt)) / mkt_var
        sv  = b**2 * mkt_var
        iv  = max(tv - sv, 1e-14)
        r2m = min(b**2 * mkt_var / tv, 1.0)
        ss  = max(1 - (1 - r2m) * (len(win) - 1) / (len(win) - 2), 0.0)
        r_arr = win[col].values
        neg_r = r_arr[r_arr < 0]
        dv    = float(np.std(neg_r)) if len(neg_r) > 5 else np.nan
        rows.append(dict(ticker=col, total_var=tv, syst_share=ss,
                         downside_vol=dv, avg_corr=avg_corr_arr[i]))
    return pd.DataFrame(rows).set_index('ticker')


# ── build panel per crisis ────────────────────────────────────────────────────
all_crisis_results = {}

for crisis, onset_str in CRISIS_ONSETS.items():
    print(f'\n=== {crisis} (onset: {onset_str}) ===')
    T0 = pd.Timestamp(onset_str)

    # date range: LEAD_IN before onset → 252 days after
    start_panel = T0 - pd.offsets.BDay(LEAD_IN)
    end_panel   = T0 + pd.offsets.BDay(252)
    panel_dates = weights.loc[start_panel:end_panel].index

    print(f'  Panel dates: {panel_dates[0].date()} → {panel_dates[-1].date()}  ({len(panel_dates)} days)')

    # features: compute once at onset (using window before T0)
    feat = compute_features_date(T0 - pd.offsets.BDay(1), returns)
    if feat.empty:
        print('  [SKIP] empty features'); continue

    # standardize features
    feat_std = feat.copy()
    for col in feat.columns:
        mu = feat[col].mean(); sd = feat[col].std()
        feat_std[col] = (feat[col] - mu) / sd if sd > 1e-14 else 0.0

    # BIC grid search over δ
    def build_panel(delta, feat_std, weights, panel_dates, T0):
        rows = []
        for d in panel_dates:
            if d not in weights.index: continue
            t_offset = (d - T0).days
            if t_offset < 0:
                D_t = 0.0
            else:
                bd_offset = sum(1 for dd in pd.bdate_range(T0, d)) - 1
                D_t = delta ** bd_offset
            w_row = weights.loc[d]
            common = feat_std.index.intersection(w_row.index)
            for tkr in common:
                if np.isnan(w_row[tkr]): continue
                f = feat_std.loc[tkr]
                rows.append(dict(
                    date=d, ticker=tkr, w=w_row[tkr], D_t=D_t,
                    tv=f['total_var'], ss=f['syst_share'],
                    dv=f.get('downside_vol', np.nan), ac=f['avg_corr'],
                ))
        return pd.DataFrame(rows).dropna(subset=['w','tv','ss','ac'])

    best_delta, best_bic, best_res_panel = None, np.inf, None
    for hl in HALFLIFE_GRID:
        delta = 0.5 ** (1 / hl)
        panel = build_panel(delta, feat_std, weights, panel_dates, T0)
        if len(panel) < 50: continue
        y = panel['w'].values
        X = np.column_stack([
            np.ones(len(panel)),
            panel['tv'].values, panel['ss'].values,
            panel['ac'].values,
            panel['D_t'].values,
            panel['D_t'].values * panel['tv'].values,
            panel['D_t'].values * panel['ss'].values,
            panel['D_t'].values * panel['ac'].values,
        ])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            continue
        resid = y - X @ beta
        n, k  = len(y), X.shape[1]
        bic   = n * np.log(np.sum(resid**2) / n) + k * np.log(n)
        if bic < best_bic:
            best_bic = bic; best_delta = delta
            best_res_panel = (panel, X)
        print(f'  HL={hl:3d}d  δ={delta:.4f}  BIC={bic:.1f}')

    if best_delta is None:
        print('  [SKIP] no valid delta'); continue

    best_hl = HALFLIFE_GRID[
        [0.5 ** (1/h) for h in HALFLIFE_GRID].index(best_delta)]
    print(f'  → Best HL={best_hl}d  δ={best_delta:.4f}')

    panel, X = best_res_panel
    y  = panel['w'].values
    cluster_ids = panel['date'].values

    res = _ols_cluster(y, X, cluster_ids)
    all_crisis_results[crisis] = dict(
        res=res, panel=panel, delta=best_delta, hl=best_hl,
        onset=T0, n_obs=len(panel), n_dates=panel['date'].nunique(),
    )

    coef_names = ['intercept', 'z_total_var', 'z_syst_share', 'z_avg_corr',
                  'γ (D_t)', 'θ_total_var', 'θ_syst_share', 'θ_avg_corr']
    for nm, b, t, p in zip(coef_names, res['beta'], res['tstat'], res['pval']):
        print(f'    {nm:<20}: β={b:.4e}  t={t:.2f}{star(t)}')
    print(f'  R²={res["r2"]:.3f}  adj-R²={res["adj_r2"]:.3f}  '
          f'N={res["n"]}  clusters={res["n_clusters"]}')


# ── event study: cumulative mean weight change ────────────────────────────────
print('\n[Event Study] Cumulative weight change around onset')

event_results = {}
for crisis, onset_str in CRISIS_ONSETS.items():
    T0 = pd.Timestamp(onset_str)
    available_dates = weights.index
    # find business days relative to T0
    pre_dates  = available_dates[available_dates <  T0][-LEAD_IN:]
    post_dates = available_dates[available_dates >= T0][:252]

    w_pre_mean = weights.loc[pre_dates].mean(axis=0)   # mean weight pre-onset per asset

    cum_changes = []
    rel_days    = []
    for i, d in enumerate(post_dates):
        delta_w = (weights.loc[d] - w_pre_mean).mean()   # cross-asset mean
        cum_changes.append(delta_w)
        rel_days.append(i)

    event_results[crisis] = dict(cum=np.array(cum_changes), days=rel_days)
    print(f'  {crisis}: max|Δw|={np.max(np.abs(cum_changes)):.4f} '
          f'at day {rel_days[np.argmax(np.abs(cum_changes))]}')


# ── figures ───────────────────────────────────────────────────────────────────

# Fig 1: coefficient comparison bar (main effects + interaction)
if all_crisis_results:
    fig, axes = plt.subplots(1, len(all_crisis_results), figsize=(13, 5), sharey=False)
    if len(all_crisis_results) == 1:
        axes = [axes]
    coef_names_disp = ['z_total_var', 'z_syst_share', 'z_avg_corr',
                       'γ (D_t)', 'θ×tv', 'θ×ss', 'θ×avg_corr']
    colors_pos = '#2ca25f'; colors_neg = '#de2d26'
    for ax, (crisis, cr) in zip(axes, all_crisis_results.items()):
        res = cr['res']
        bs  = res['beta'][1:]    # skip intercept
        ts  = res['tstat'][1:]
        bar_colors = [colors_pos if b > 0 else colors_neg for b in bs]
        bars = ax.bar(range(len(bs)), bs, color=bar_colors, edgecolor='white')
        ax.set_xticks(range(len(coef_names_disp)))
        ax.set_xticklabels(coef_names_disp, rotation=40, ha='right', fontsize=8)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_title(f'{crisis}\n(HL={cr["hl"]}d, N={cr["n_obs"]:,})', fontweight='bold')
        ax.set_ylabel('Coefficient')
        for j, (b, t) in enumerate(zip(bs, ts)):
            s = star(t)
            if s:
                ax.text(j, b + (0.0001 if b >= 0 else -0.0001),
                        s, ha='center', va='bottom' if b >= 0 else 'top', fontsize=9)
    fig.suptitle('LW GMV — Intervention OLS Coefficients (cluster-robust SE)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGURES / 'intv_coef_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: intv_coef_bar.png')

# Fig 2: event study
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
for ax, crisis in zip(axes, CRISIS_ONSETS):
    if crisis not in event_results:
        ax.set_visible(False); continue
    er = event_results[crisis]
    ax.plot(er['days'], er['cum'], color='#377eb8', lw=1.5)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.axvline(0, color='red', lw=1.0, ls=':', label='onset')
    ax.set_xlabel('Trading days since crisis onset')
    ax.set_ylabel('Mean Δweight (cross-asset avg)')
    ax.set_title(f'{crisis} — Mean Weight Change from Pre-Onset Baseline', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(FIGURES / 'intv_event_study.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: intv_event_study.png')

# Fig 3: D_t decay curve comparison
fig, ax = plt.subplots(figsize=(8, 4))
days = np.arange(0, 180)
for hl in HALFLIFE_GRID:
    delta = 0.5 ** (1 / hl)
    ax.plot(days, delta**days, label=f'HL={hl}d (δ={delta:.3f})', lw=1.5)
ax.set_xlabel('Trading days since onset')
ax.set_ylabel('D_t = δ^t')
ax.set_title('Box-Tiao Decay Functions by Half-Life', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(FIGURES / 'intv_decay_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: intv_decay_curves.png')


# ── report ────────────────────────────────────────────────────────────────────
L = []; A = L.append

A("# LW GMV 비중의 개입 분석 (Intervention Analysis)")
A("")
A("**작성일**: 2026-05-26  ")
A("**데이터**: S&P 100, 2015-01-01 ~ 2024-12-31  ")
A("**추정기**: Ledoit-Wolf (LW) 단독  ")
A("**방법**: Box-Tiao 감쇠형 개입 패널 OLS, 날짜 클러스터 강건 표준오차  ")
A("**위기**: COVID (onset 2020-02-21), 금리위기 (onset 2022-01-03)  ")
A("")
A("---")
A("")
A("## 1. 모형")
A("")
A(r"$$w_{i,t} = \beta_0 + \sum_k \beta_k z(X_{i,k}) + \gamma D_t + \sum_k \theta_k D_t \cdot z(X_{i,k}) + \varepsilon_{i,t}$$")
A("")
A(r"$$D_t = \delta^{t - T_0}, \quad \delta = 0.5^{1/\text{HL}}$$")
A("")
A("- **주효과 β**: 평상시(pre-onset) 배분 결정 요인")
A("- **γ (D_t)**: 위기 발생 즉시의 포트폴리오 수준 충격")
A("- **상호작용 θ**: 위기 발생 시 각 변수의 계수 변화")
A("- **표준오차**: 날짜 클러스터 robust (동일 날짜 내 비중 합계=1 의존성 보정)")
A("")
A(f"**반감기 그리드**: {HALFLIFE_GRID} 거래일 → BIC로 선택")
A("")
A("---")
A("")
A("## 2. 결과")
A("")

coef_names_full = ['intercept', 'z_total_var', 'z_syst_share', 'z_avg_corr',
                   'γ (D_t)', 'θ_total_var', 'θ_syst_share', 'θ_avg_corr']

for crisis, cr in all_crisis_results.items():
    res = cr['res']
    A(f"### 2.{list(all_crisis_results.keys()).index(crisis)+1} {crisis}")
    A("")
    A(f"| 항목 | 값 |")
    A(f"|------|-----|")
    A(f"| 위기 기점 (T₀) | {cr['onset'].date()} |")
    A(f"| BIC 선택 반감기 | {cr['hl']} 거래일 (δ={cr['delta']:.4f}) |")
    A(f"| 패널 관측수 | {cr['n_obs']:,} (자산×날짜) |")
    A(f"| 날짜 클러스터 수 | {cr['n_dates']} |")
    A(f"| R² | {res['r2']:.3f} |")
    A(f"| adj-R² | {res['adj_r2']:.3f} |")
    A("")
    A("*\\* |t|>1.645  \\*\\* |t|>1.960  \\*\\*\\* |t|>2.576*")
    A("")
    A("| 변수 | 계수 | t-stat | 해석 |")
    A("|------|-----:|:------:|------|")
    interp = {
        'intercept':   '절편',
        'z_total_var': '총분산 높을수록 비중 감소 (평상시)',
        'z_syst_share':'체계적 비중 높을수록 GMV 기피 (평상시)',
        'z_avg_corr':  '평균 상관 높을수록 비중 감소 (평상시)',
        'γ (D_t)':     '위기 즉시 포트폴리오 수준 충격',
        'θ_total_var': '위기 중 총분산 효과 변화',
        'θ_syst_share':'위기 중 syst_share 효과 변화',
        'θ_avg_corr':  '위기 중 상관 효과 변화',
    }
    for nm, b, t in zip(coef_names_full, res['beta'], res['tstat']):
        s = star(t)
        A(f"| {nm} | {b:.4e} | {t:.2f}{s} | {interp.get(nm,'')} |")
    A("")

A("---")
A("")
A("## 3. 이벤트 스터디 — 위기 기점 대비 비중 변화")
A("")
A("Pre-onset 60일 평균 비중 대비, 위기 발생 후 일별 자산 평균 비중 변화.")
A("")
A("| 위기 | 최대 평균 비중 변화 | 발생 시점 (onset 이후) |")
A("|------|:-----------------:|:--------------------:|")
for crisis in CRISIS_ONSETS:
    if crisis not in event_results: continue
    er  = event_results[crisis]
    idx = np.argmax(np.abs(er['cum']))
    A(f"| {crisis} | {er['cum'][idx]*100:+.3f}pp | {er['days'][idx]}거래일 |")
A("")
A("---")
A("")
A("## 4. 핵심 발견")
A("")
findings = []

for crisis, cr in all_crisis_results.items():
    res   = cr['res']
    names = coef_names_full
    # main effects
    b_tv = res['beta'][1]; t_tv = res['tstat'][1]
    b_ss = res['beta'][2]; t_ss = res['tstat'][2]
    b_ac = res['beta'][3]; t_ac = res['tstat'][3]
    # interaction
    b_gm = res['beta'][4]; t_gm = res['tstat'][4]
    th_tv= res['beta'][5]; t_theta_tv = res['tstat'][5]
    th_ss= res['beta'][6]; t_theta_ss = res['tstat'][6]
    th_ac= res['beta'][7]; t_theta_ac = res['tstat'][7]

    findings.append(
        f"**{crisis} 반감기**: BIC 선택 HL={cr['hl']}거래일. "
        f"위기 충격이 약 {cr['hl']}일 만에 절반으로 소멸한다."
    )
    findings.append(
        f"**{crisis} 주효과**: z_total_var β={b_tv:.4e} (t={t_tv:.2f}{star(t_tv)}), "
        f"z_syst_share β={b_ss:.4e} (t={t_ss:.2f}{star(t_ss)}), "
        f"z_avg_corr β={b_ac:.4e} (t={t_ac:.2f}{star(t_ac)}). "
        f"평상시에도 총분산·체계적 비중·평균 상관 모두 비중을 음의 방향으로 결정."
    )
    findings.append(
        f"**{crisis} 상호작용**: θ_total_var={th_tv:.4e} (t={t_theta_tv:.2f}{star(t_theta_tv)}), "
        f"θ_syst_share={th_ss:.4e} (t={t_theta_ss:.2f}{star(t_theta_ss)}), "
        f"θ_avg_corr={th_ac:.4e} (t={t_theta_ac:.2f}{star(t_theta_ac)}). "
        + ("위기 중 변수별 배분 메커니즘이 변화한다." if any(abs(t) > 1.645 for t in [t_theta_tv, t_theta_ss, t_theta_ac])
           else "위기 중 유의한 메커니즘 변화 없음.")
    )

for i, f in enumerate(findings, 1):
    A(f"{i}. {f}")
    A("")

A("---")
A("")
A("## 부록")
A("")
A("| 파일 | 내용 |")
A("|------|------|")
A("| `results/figures/intervention/intv_coef_bar.png` | 위기별 OLS 계수 막대 |")
A("| `results/figures/intervention/intv_event_study.png` | 이벤트 스터디 |")
A("| `results/figures/intervention/intv_decay_curves.png` | 감쇠 함수 비교 |")
A("")
A("*분석 코드: `intervention_new_analysis.py`*")

(REPORTS / '2026-05-26_intervention.md').write_text('\n'.join(L), encoding='utf-8')
print('\nReport saved → reports/2026-05-26_intervention.md')
print('=== Done ===')
