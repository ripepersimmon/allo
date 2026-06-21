"""
ppt_figures.py
==============
PPT-ready (16:9) versions of all key experiment figures with experiment
labels.  Reads from existing CSV results — no re-analysis.

Experiments
-----------
  A  분산분해 기준선    (vardec_snapshot_results.csv)
  B  SPY 강건성 검증   (spy_robustness_table.csv)
  C  섹터 고정효과     (variance_decomp_sector_fe_table.csv)
  D  다중요인 FF3/FF5  (multifactor_decomp_table.csv)

Output: results/figures/ppt/
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

OUT   = Path('results/figures/ppt')
OUT.mkdir(parents=True, exist_ok=True)
TODAY = datetime.date.today().strftime('%Y-%m-%d')

CRISES = ['GFC', 'COVID', 'Rates']
ESTS   = ['Sample', 'LW', 'Gerber']
CRISIS_LABEL = {
    'GFC':   'GFC\n(금융위기)',
    'COVID': 'COVID\n(코로나)',
    'Rates': '금리위기\n(Rates)',
}

# ── PPT-wide global style ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':          ['NanumGothic', 'DejaVu Sans'],
    'font.size':            13,
    'axes.titlesize':       15,
    'axes.titleweight':     'bold',
    'axes.labelsize':       13,
    'xtick.labelsize':      12,
    'ytick.labelsize':      12,
    'legend.fontsize':      11,
    'legend.framealpha':    0.85,
    'figure.titlesize':     16,
    'figure.titleweight':   'bold',
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.grid':            True,
    'grid.alpha':           0.25,
    'axes.axisbelow':       True,
})

# ── palette ───────────────────────────────────────────────────────────────────
EST_CLR = {'Sample': '#e41a1c', 'LW': '#377eb8', 'Gerber': '#4daf4a'}

EXP_META = {
    'A': ('실험 A | 분산분해 기준선',     '#1f497d'),
    'B': ('실험 B | SPY 강건성 검증',     '#375623'),
    'C': ('실험 C | 섹터 고정효과',       '#7b2c3d'),
    'D': ('실험 D | 다중요인 FF3/FF5',   '#4a3069'),
}


# ── helpers ───────────────────────────────────────────────────────────────────
def add_badge(fig, exp: str):
    label, color = EXP_META[exp]
    fig.text(0.008, 0.995, label,
             transform=fig.transFigure, fontsize=11, fontweight='bold',
             color='white', va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.35', facecolor=color,
                       alpha=0.92, edgecolor='none'))
    fig.text(0.999, 0.004, TODAY, transform=fig.transFigure,
             fontsize=8, color='#999999', va='bottom', ha='right')


def star(t):
    if pd.isna(t): return ''
    a = abs(t)
    return '***' if a > 2.576 else ('**' if a > 1.960 else ('*' if a > 1.645 else ''))



def three_panel_fig(sharey=True):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=sharey,
                             gridspec_kw={'wspace': 0.10 if sharey else 0.18})
    return fig, axes


def save(fig, path: Path):
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {path.name}')


# ═══════════════════════════════════════════════════════════════════════════════
# 실험 A  분산분해 기준선
# ═══════════════════════════════════════════════════════════════════════════════
print('\n[실험 A] 분산분해 기준선')
vd = pd.read_csv('reports/vardec_snapshot_results.csv').set_index(['crisis', 'estimator'])

# ── A1: R²(D) vs R²(E) ───────────────────────────────────────────────────────
fig, axes = three_panel_fig(sharey=True)
x, bw = np.arange(len(ESTS)), 0.32

for ax, crisis in zip(axes, CRISES):
    r2d = [vd.loc[(crisis, e), 'r2_D'] for e in ESTS]
    r2e = [vd.loc[(crisis, e), 'r2_E'] for e in ESTS]
    bd = ax.bar(x - bw/2, r2d, width=bw, label='Model D', color='#2166ac', alpha=0.85)
    be = ax.bar(x + bw/2, r2e, width=bw, label='Model E (+log vol)', color='#ef8a62', alpha=0.85)
    for bar, v in list(zip(bd, r2d)) + list(zip(be, r2e)):
        if v > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.004,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_title(CRISIS_LABEL[crisis])
    ax.set_xticks(x); ax.set_xticklabels(ESTS)
    ax.set_ylim(0, max(max(r2d), max(r2e)) * 1.38)
    if crisis == 'GFC':
        ax.set_ylabel('R²')
        ax.legend(loc='upper right')

fig.suptitle(
    'Model D:  w = α + γ₁·total_var + γ₂·syst_share\n'
    'Model E:  + γ₃·log(dollar_volume)   |   위기 고점 횡단면 OLS'
)
add_badge(fig, 'A')
save(fig, OUT / 'A1_vardec_r2_DE.png')

# ── A2: γ₂ (syst_share) coefficient ──────────────────────────────────────────
fig, axes = three_panel_fig(sharey=False)

for ax, crisis in zip(axes, CRISES):
    g2d = [vd.loc[(crisis, e), 'gD_systshare'] for e in ESTS]
    t2d = [vd.loc[(crisis, e), 'tD_systshare'] for e in ESTS]
    g2e = [vd.loc[(crisis, e), 'gE_systshare'] for e in ESTS]
    t2e = [vd.loc[(crisis, e), 'tE_systshare'] for e in ESTS]
    bd = ax.bar(x - bw/2, g2d, width=bw, label='γ₂(D)', color='#2166ac', alpha=0.85)
    be = ax.bar(x + bw/2, g2e, width=bw, label='γ₂(E)', color='#ef8a62', alpha=0.85)
    for bar, v, t in list(zip(bd, g2d, t2d)) + list(zip(be, g2e, t2e)):
        s = star(t)
        if s:
            sign = 1 if v >= 0 else -1
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_y() + bar.get_height() + sign * 0.001,
                    s, ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(CRISIS_LABEL[crisis])
    ax.set_xticks(x); ax.set_xticklabels(ESTS)
    if crisis == 'GFC':
        ax.set_ylabel('γ₂  (syst_share 계수)')
        ax.legend(loc='lower left')

fig.suptitle(
    'γ₂: syst_share 계수   |   * p<.10  ** p<.05  *** p<.01\n'
    '음수 → 체계적 비중 높은 자산일수록 GMV 비중 감소'
)
add_badge(fig, 'A')
save(fig, OUT / 'A2_vardec_gamma2.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 실험 B  SPY 강건성 검증
# ═══════════════════════════════════════════════════════════════════════════════
print('\n[실험 B] SPY 강건성 검증')
spy = pd.read_csv('reports/spy_robustness_table.csv').set_index(['crisis', 'estimator'])

# ── B1: R²(D) EW vs SPY ───────────────────────────────────────────────────────
fig, axes = three_panel_fig(sharey=True)
for ax, crisis in zip(axes, CRISES):
    rew  = [spy.loc[(crisis, e), 'r2_ew']  for e in ESTS]
    rspy = [spy.loc[(crisis, e), 'r2_spy'] for e in ESTS]
    bew  = ax.bar(x - bw/2, rew,  width=bw, label='EW 시장 (기존)', color='#2166ac', alpha=0.85)
    bsp  = ax.bar(x + bw/2, rspy, width=bw, label='SPY 대체',       color='#d6604d', alpha=0.85)
    for bar, v in list(zip(bew, rew)) + list(zip(bsp, rspy)):
        if v > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.004,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_title(CRISIS_LABEL[crisis])
    ax.set_xticks(x); ax.set_xticklabels(ESTS)
    ax.set_ylim(0, max(max(rew), max(rspy)) * 1.38)
    if crisis == 'GFC':
        ax.set_ylabel('R²')
        ax.legend(loc='upper right')

fig.suptitle(
    '시장 프록시 강건성:  EW 동일가중 vs SPY ETF  |  Model D R²\n'
    '부호 반전 여부가 핵심 — 반전 없으면 EW 사용의 내생성 문제 제한적'
)
add_badge(fig, 'B')
save(fig, OUT / 'B1_spy_r2.png')

# ── B2: γ₁ and γ₂ EW vs SPY ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 9),
                         gridspec_kw={'wspace': 0.18, 'hspace': 0.42})

for col, crisis in enumerate(CRISES):
    for row, (coef, tew_col, tspy_col, gew_col, gspy_col, ylabel) in enumerate([
        (1, 't1_ew', 't1_spy', 'g1_ew', 'g1_spy', 'γ₁ (total_var)'),
        (2, 't2_ew', 't2_spy', 'g2_ew', 'g2_spy', 'γ₂ (syst_share)'),
    ]):
        ax = axes[row, col]
        gew  = [spy.loc[(crisis, e), gew_col]  for e in ESTS]
        gspy = [spy.loc[(crisis, e), gspy_col] for e in ESTS]
        tew  = [spy.loc[(crisis, e), tew_col]  for e in ESTS]
        tspy = [spy.loc[(crisis, e), tspy_col] for e in ESTS]
        bew  = ax.bar(x - bw/2, gew,  width=bw, label='EW', color='#2166ac', alpha=0.85)
        bsp  = ax.bar(x + bw/2, gspy, width=bw, label='SPY', color='#d6604d', alpha=0.85)
        for bar, v, t in list(zip(bew, gew, tew)) + list(zip(bsp, gspy, tspy)):
            s = star(t)
            if s:
                sign = 1 if v >= 0 else -1
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_y() + bar.get_height() + sign * 0.0002,
                        s, ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(ESTS, fontsize=11)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        if row == 0:
            ax.set_title(CRISIS_LABEL[crisis], fontsize=13)
        if col == 0 and row == 0:
            ax.legend(fontsize=10, loc='lower left')

fig.suptitle(
    'EW vs SPY 계수 비교  (γ₁: total_var, γ₂: syst_share)\n'
    '* p<.10  ** p<.05  *** p<.01'
)
add_badge(fig, 'B')
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / 'B2_spy_coefs.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  → B2_spy_coefs.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 실험 C  섹터 고정효과
# ═══════════════════════════════════════════════════════════════════════════════
print('\n[실험 C] 섹터 고정효과')
fe_raw = pd.read_csv('reports/variance_decomp_sector_fe_table.csv')
fe = (fe_raw[(fe_raw['model'] == 'D') & (fe_raw['regressor'] == 'syst_share')]
      .set_index(['crisis', 'estimator']))

# ── C1: γ₂(D) no-FE vs with-FE ───────────────────────────────────────────────
fig, axes = three_panel_fig(sharey=False)
for ax, crisis in zip(axes, CRISES):
    g_nof = [fe.loc[(crisis, e), 'coef_no_fe'] for e in ESTS]
    t_nof = [fe.loc[(crisis, e), 't_no_fe']    for e in ESTS]
    g_fe  = [fe.loc[(crisis, e), 'coef_fe']    for e in ESTS]
    t_fe  = [fe.loc[(crisis, e), 't_fe']       for e in ESTS]
    bn = ax.bar(x - bw/2, g_nof, width=bw, label='FE 없음',  color='#4393c3', alpha=0.85)
    bf = ax.bar(x + bw/2, g_fe,  width=bw, label='섹터 FE', color='#d6604d',
                alpha=0.85, hatch='////', edgecolor='white', linewidth=0.4)
    for bar, v, t in list(zip(bn, g_nof, t_nof)) + list(zip(bf, g_fe, t_fe)):
        s = star(t)
        if s:
            sign = 1 if v >= 0 else -1
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_y() + bar.get_height() + sign * 0.0002,
                    s, ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(CRISIS_LABEL[crisis])
    ax.set_xticks(x); ax.set_xticklabels(ESTS)
    if crisis == 'GFC':
        ax.set_ylabel('γ₂  (syst_share 계수)')
        ax.legend(loc='lower left')

fig.suptitle(
    'γ₂(D) 섹터 고정효과 전/후  |  GICS 11개 섹터 더미 (InfoTech 기준)\n'
    '* p<.10  ** p<.05  *** p<.01  |  부호 반전 없음 → syst_share 신호 robust'
)
add_badge(fig, 'C')
save(fig, OUT / 'C1_sector_fe_gamma2.png')

# ── C2: R² no-FE vs with-FE ───────────────────────────────────────────────────
fig, axes = three_panel_fig(sharey=True)
for ax, crisis in zip(axes, CRISES):
    r_nof = [fe.loc[(crisis, e), 'r2_no_fe'] for e in ESTS]
    r_fe  = [fe.loc[(crisis, e), 'r2_fe']    for e in ESTS]
    bn = ax.bar(x - bw/2, r_nof, width=bw, label='FE 없음',  color='#4393c3', alpha=0.85)
    bf = ax.bar(x + bw/2, r_fe,  width=bw, label='섹터 FE', color='#d6604d',
                alpha=0.85, hatch='////', edgecolor='white', linewidth=0.4)
    for bar, v in list(zip(bn, r_nof)) + list(zip(bf, r_fe)):
        if v > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.004,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_title(CRISIS_LABEL[crisis])
    ax.set_xticks(x); ax.set_xticklabels(ESTS)
    ax.set_ylim(0, max(max(r_nof), max(r_fe)) * 1.42)
    if crisis == 'GFC':
        ax.set_ylabel('R²')
        ax.legend(loc='upper right')

fig.suptitle(
    'Model D R²: 섹터 FE 추가 전/후  |  R² 증가 = 섹터 더미의 설명력\n'
    '섹터 FE 추가 후에도 γ₂ 유효 → 섹터 집중 아닌 syst_share 자체의 신호'
)
add_badge(fig, 'C')
save(fig, OUT / 'C2_sector_fe_r2.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 실험 D  다중요인 FF3 / FF5
# ═══════════════════════════════════════════════════════════════════════════════
print('\n[실험 D] 다중요인 분해 (FF3/FF5)')
mf = pd.read_csv('reports/multifactor_decomp_table.csv').set_index(['crisis', 'estimator'])

MODEL_KEYS  = ['r2_D',  'r2_G3', 'r2_G5', 'r2_H3', 'r2_H5']
MODEL_LBLS  = ['(D) EW', '(G3) FF3', '(G5) FF5', '(H3) FF3+corr', '(H5) FF5+corr']
MODEL_CLRS  = ['#2166ac', '#d73027', '#fc8d59', '#1a9641', '#4dac26']
G2_KEYS     = ['g2_D',  'g2_G3', 'g2_G5', 'g2_H3', 'g2_H5']
G2T_KEYS    = ['g2_D_t','g2_G3_t','g2_G5_t','g2_H3_t','g2_H5_t']

n_models = len(MODEL_KEYS)
bw_5 = 0.14
x5 = np.arange(len(ESTS))

# ── D1: adj-R² 5개 모형 비교 ─────────────────────────────────────────────────
fig, axes = three_panel_fig(sharey=True)
for ax, crisis in zip(axes, CRISES):
    for mi, (mk, ml, mc) in enumerate(zip(MODEL_KEYS, MODEL_LBLS, MODEL_CLRS)):
        vals = [mf.loc[(crisis, e), mk] if (crisis, e) in mf.index else 0.0 for e in ESTS]
        offset = (mi - 2) * bw_5
        bars = ax.bar(x5 + offset, vals, width=bw_5, label=ml, color=mc,
                      alpha=0.85, edgecolor='white', linewidth=0.3)
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_title(CRISIS_LABEL[crisis])
    ax.set_xticks(x5); ax.set_xticklabels(ESTS)
    ax.set_ylim(0, mf[MODEL_KEYS].max().max() * 1.42)
    if crisis == 'GFC':
        ax.set_ylabel('adj-R²')
        ax.legend(fontsize=9, loc='upper right')

fig.suptitle(
    'adj-R² 비교: 단일요인 (D) vs FF3 (G3) vs FF5 (G5) vs +avg_corr (H3/H5)\n'
    'w_i = α + γ₁·total_var + γ₂·syst_share  |  위기 고점 횡단면 OLS'
)
add_badge(fig, 'D')
save(fig, OUT / 'D1_ff_r2.png')

# ── D2: γ₂ (syst_share 계수) 5개 모형 ────────────────────────────────────────
fig, axes = three_panel_fig(sharey=False)
HATCH = [None, None, None, '////', '////']
for ax, crisis in zip(axes, CRISES):
    for mi, (gk, tk, ml, mc, ht) in enumerate(
            zip(G2_KEYS, G2T_KEYS, MODEL_LBLS, MODEL_CLRS, HATCH)):
        vals  = [mf.loc[(crisis, e), gk] if (crisis, e) in mf.index else np.nan for e in ESTS]
        tvals = [mf.loc[(crisis, e), tk] if (crisis, e) in mf.index else np.nan for e in ESTS]
        offset = (mi - 2) * bw_5
        bars = ax.bar(x5 + offset, vals, width=bw_5, label=ml, color=mc,
                      alpha=0.82, hatch=ht, edgecolor='black', linewidth=0.3)
        for bar, v, t in zip(bars, vals, tvals):
            s = star(t)
            if s and not pd.isna(t):
                sign = 1 if v >= 0 else -1
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_y() + bar.get_height() + sign * 0.0002,
                        s, ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(CRISIS_LABEL[crisis])
    ax.set_xticks(x5); ax.set_xticklabels(ESTS)
    if crisis == 'GFC':
        ax.set_ylabel('γ₂  (syst_share 계수)')
        ax.legend(fontsize=8.5, loc='lower left', ncol=1)
    ax.tick_params(axis='y', labelsize=10)

fig.suptitle(
    'γ₂: syst_share 계수 비교  |  * p<.10  ** p<.05  *** p<.01\n'
    'avg_corr 추가(H3/H5) 시 γ₂ 유의성 소멸 → syst_share–avg_corr 다중공선'
)
add_badge(fig, 'D')
save(fig, OUT / 'D2_ff_gamma2.png')

# ── D3: γ₃ (avg_corr 계수) H3 vs H5 ─────────────────────────────────────────
fig, axes = three_panel_fig(sharey=False)
for ax, crisis in zip(axes, CRISES):
    g3h3  = [mf.loc[(crisis, e), 'g3_H3']   if (crisis, e) in mf.index else np.nan for e in ESTS]
    t3h3  = [mf.loc[(crisis, e), 'g3_H3_t'] if (crisis, e) in mf.index else np.nan for e in ESTS]
    g3h5  = [mf.loc[(crisis, e), 'g3_H5']   if (crisis, e) in mf.index else np.nan for e in ESTS]
    t3h5  = [mf.loc[(crisis, e), 'g3_H5_t'] if (crisis, e) in mf.index else np.nan for e in ESTS]
    bh3 = ax.bar(x - bw/2, g3h3, width=bw, label='γ₃(H3) FF3+corr',
                 color='#1a9641', alpha=0.85)
    bh5 = ax.bar(x + bw/2, g3h5, width=bw, label='γ₃(H5) FF5+corr',
                 color='#4dac26', alpha=0.85, hatch='////', edgecolor='black', linewidth=0.3)
    for bar, v, t in list(zip(bh3, g3h3, t3h3)) + list(zip(bh5, g3h5, t3h5)):
        s = star(t)
        if s and not pd.isna(t):
            sign = 1 if v >= 0 else -1
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_y() + bar.get_height() + sign * 0.005,
                    s, ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(CRISIS_LABEL[crisis])
    ax.set_xticks(x); ax.set_xticklabels(ESTS)
    if crisis == 'GFC':
        ax.set_ylabel('γ₃  (avg_corr 계수)')
        ax.legend(loc='lower left')
    ax.tick_params(axis='y', labelsize=10)

fig.suptitle(
    'γ₃: avg_corr 계수 (H 모형)  |  * p<.10  ** p<.05  *** p<.01\n'
    '전 셀 음수 → 평균 상관 높은 자산 = 낮은 GMV 비중  (분산효과 감소 반영)'
)
add_badge(fig, 'D')
save(fig, OUT / 'D3_ff_gamma3_avgcorr.png')

# ── D4: adj-R² 델타 요약 (기준선 D 대비) ─────────────────────────────────────
delta_keys  = ['r2_G3', 'r2_G5', 'r2_H3', 'r2_H5']
delta_lbls  = ['ΔG3\n(FF3)', 'ΔG5\n(FF5)', 'ΔH3\n(FF3+corr)', 'ΔH5\n(FF5+corr)']
delta_clrs  = ['#d73027', '#fc8d59', '#1a9641', '#4dac26']

fig, axes = three_panel_fig(sharey=True)
xd = np.arange(len(delta_keys))
bwd = 0.22

for ax, crisis in zip(axes, CRISES):
    for ei, est in enumerate(ESTS):
        if (crisis, est) not in mf.index: continue
        row = mf.loc[(crisis, est)]
        deltas = [row[mk] - row['r2_D'] for mk in delta_keys]
        offset = (ei - 1) * bwd
        bars = ax.bar(xd + offset, deltas, width=bwd, label=est,
                      color=EST_CLR[est], alpha=0.82)
        for bar, v in zip(bars, deltas):
            if abs(v) > 0.003:
                sign = 1 if v >= 0 else -1
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_y() + bar.get_height() + sign * 0.001,
                        f'{v:+.3f}', ha='center',
                        va='bottom' if v >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.9)
    ax.set_title(CRISIS_LABEL[crisis])
    ax.set_xticks(xd); ax.set_xticklabels(delta_lbls, fontsize=11)
    if crisis == 'GFC':
        ax.set_ylabel('Δadj-R²  (vs Model D)')
        ax.legend(fontsize=10, loc='upper right')

fig.suptitle(
    'adj-R² 증감: 각 모형 vs 기준선(D)  |  양수 = 설명력 개선\n'
    'FF3/FF5 단독 대체 → 개선 없음 | avg_corr 추가(H3/H5) → 일관된 개선'
)
add_badge(fig, 'D')
save(fig, OUT / 'D4_ff_r2_delta.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 실험 종합 요약 슬라이드용
# ═══════════════════════════════════════════════════════════════════════════════
print('\n[종합] R² 실험 비교 요약')

# 9개 셀 평균 R² 요약 (Exp A, D)
fig, ax = plt.subplots(figsize=(12, 5.5))

summary = {}
for label, key in [('D (EW)', 'r2_D'), ('G3 (FF3)', 'r2_G3'),
                   ('G5 (FF5)', 'r2_G5'), ('H3 (FF3+corr)', 'r2_H3'),
                   ('H5 (FF5+corr)', 'r2_H5')]:
    summary[label] = mf[key].mean()

# add Model E from vardec
summary['E (EW+size)'] = vd['r2_E'].mean()

order  = ['D (EW)', 'E (EW+size)', 'G3 (FF3)', 'G5 (FF5)', 'H3 (FF3+corr)', 'H5 (FF5+corr)']
clrs   = ['#2166ac', '#4393c3', '#d73027', '#fc8d59', '#1a9641', '#4dac26']
vals   = [summary[k] for k in order]
bars   = ax.bar(order, vals, color=clrs, alpha=0.87, edgecolor='white', linewidth=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
            f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(summary['D (EW)'], color='#2166ac', linewidth=1.2, linestyle='--', alpha=0.7,
           label=f'기준선 D (EW) = {summary["D (EW)"]:.3f}')
ax.set_ylabel('9-셀 평균 adj-R²', fontsize=13)
ax.set_ylim(0, max(vals) * 1.35)
ax.legend(fontsize=11)
ax.set_xticklabels(order, fontsize=11)
ax.set_title('전체 실험 평균 adj-R² 비교  (GFC·COVID·금리위기 × 3개 추정기)', fontsize=14)

fig.text(0.5, 0.01,
         '* Model D/E: 분산분해 기준선 (실험 A)  |  G3/G5/H3/H5: 다중요인 FF3/FF5 (실험 D)',
         ha='center', fontsize=10, color='#555555')

import matplotlib.patches as mpatches
exp_patches = [
    mpatches.Patch(facecolor='#1f497d', label='실험 A (D, E)'),
    mpatches.Patch(facecolor='#4a3069', label='실험 D (G3/G5/H3/H5)'),
]
ax.legend(handles=exp_patches, fontsize=10, loc='upper right')

plt.tight_layout(rect=[0, 0.04, 1, 1])
out_sum = OUT / 'SUMMARY_r2_all_experiments.png'
fig.savefig(out_sum, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  → {out_sum.name}')

print(f'\n완료. {len(list(OUT.glob("*.png")))}개 파일 → {OUT}')
