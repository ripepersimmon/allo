# 연구 2 — GMV 포트폴리오 비중은 무엇으로 설명되는가 (LW 전용)

**스크립트:** `weight_explain_study.py` · **환경:** conda `allo` (xgboost 3.3.0, shap 0.52.0)
**참고문헌:** `참고문헌.md` — [CST2011], [FP2014], [BBW2011], [LW2004], [JM2003], [DGU2009]

---

## 1. 질문과 설계

**해석 가능한 어떤 자산 특성이 Ledoit-Wolf long-only GMV 포트폴리오 비중의 횡단면을 설명하는가?**

- **공분산:** Ledoit-Wolf 전용 ([LW2004]).
- **타깃:** 비중 수준 `w_i` (long-only GMV). ※ 위기 변화 Δw가 아닌 *수준*.
- **헤드라인 피처(非Σ):** `beta`, `log_dolvol`(사이즈), `amihud`(비유동성), `momentum`, GICS 섹터 더미.
- **벤치마크(Σ-파생, 분리 보고):** `total_var`, `syst_share` — *비제약* GMV에서 `w ∝ Σ⁻¹1`이라 항등식이 되므로 헤드라인에서 분리.
- **시간 구조:** 2005–2024 연말 비중첩 스냅샷 20개(1,921 자산-행). 윈도우 길이≈연 간격이라 일별 중첩 누수 없음.
- **평가:** Leave-One-Year-Out (LOYO) **out-of-sample** R²/MAE.

### 왜 이 설계인가
"비중 수준 설명"은 잘못하면 항등식(C1)에 빠집니다. 방어막은 **피처 선택**: `w ~ total_var`(같은 Σ) = 순환, `w ~ 유동성·사이즈·모멘텀` = 정당한 기술적 질문. 그래서 非Σ를 헤드라인으로, Σ-파생은 "기계적 성분이 얼마나 되나"의 벤치마크로만 둡니다.

---

## 2. 결과 — LOYO out-of-sample R²

`figures/weight_explain/ladder_loyo.png`

| 모델 | 피처 | R² | MAE |
|---|---|---|---|
| **B0** Σ-벤치 | total_var, syst_share | **0.059** | 0.0152 |
| **M1** 非Σ | beta, size, 유동성, 모멘텀 | 0.172 | 0.0154 |
| **M2** +섹터 | M1 + GICS 더미 | 0.176 | 0.0152 |
| **M3** GBR(xgboost) | M2 피처, 비선형 | **0.356** | 0.0101 |
| *(참고)* Σ + 非Σ | 전부 | 0.187 | — |

- **非Σ가 Σ-벤치 대비 +0.128 R² 추가; Σ는 非Σ 대비 +0.011만 추가.** → 해석 가능한 특성이 설명력의 거의 전부.
- **M3가 선형 M2를 2배(0.176→0.356).** → 특성→비중 매핑이 강하게 **비선형**.

### OLS — 非Σ + 섹터 (HC3 강건SE), 주요 항
| 항 | 계수 | t | p |
|---|---|---|---|
| **beta** | **−0.0340** | **−11.1** | ~0 |
| log_dolvol(사이즈) | +0.0016 | +2.1 | 0.039 |
| amihud | +6.47 | +1.4 | 0.165 |
| momentum | −0.0022 | −1.5 | 0.146 |
| sec_Financials | +0.0073 | +4.5 | ~0 |
| sec_ConsStap | +0.0063 | +1.8 | 0.066 |

(Σ-벤치 OLS: `total_var` −3.93[t=−4.3], `syst_share` −0.038[t=−10.9] — 기계적으로 음(−)이나 합쳐도 OOS R²≈0.06뿐.)

---

## 3. SHAP 해석 (TreeExplainer on GBR)

`figures/weight_explain/shap_summary.png`, `shap_dependence_beta.png`

mean |SHAP value| (예측 비중 기여 크기):

| 피처 | mean &#124;SHAP&#124; | 순위 |
|---|---|---|
| **beta** | **0.01263** | 1 (2위의 약 3배) |
| amihud | 0.00424 | 2 |
| log_dolvol | 0.00316 | 3 |
| momentum | 0.00161 | 4 |
| sec_ConsStap | 0.00082 | 5 |

### 3.1 핵심 — beta 효과는 임계(cliff)
`shap_dependence_beta.png`가 명확한 하키스틱을 보임:
- **β ≲ 0.6:** 비중에 강한 *양(+)* 기여, 최대 **+0.12**, β가 낮을수록 가팔라짐.
- **β ≈ 0.7–0.9:** SHAP 0 통과 — 실증적 **임계 베타**.
- **β ≳ 1.0:** β 3.0까지 전 구간에서 *음의 바닥*(~−0.01)에 평평 — 고베타 종목은 일률적으로 비중 0으로 밀림.

이 임계 비선형성이 **GBR(M3)가 선형 M2를 2배로 올린 직접 원인**(단일 선형 베타 기울기로는 cliff를 표현 불가).

### 3.2 beeswarm 요약
`shap_summary.png`: `beta`의 영향 폭이 압도적(저β = 큰 양의 비중 기여). `amihud`(고비유동성→음의 기여)와 `log_dolvol`이 이차·비선형 효과. 섹터는 작음(ConsStap/Financials 소폭 양). SHAP은 `amihud`를 `log_dolvol`보다 높게 랭크(순열중요도와 반대) — 유동성이 주효과보다 상호작용으로 작용함과 일치.

---

## 4. 선행연구와의 연결

| 본 연구 결과 | 선행연구 | 관계 |
|---|---|---|
| beta 지배, 임계 위로 비중→0 | **[CST2011]** Minimum-Variance Portfolio Composition | **이론적 정합.** CST는 단일팩터 하 long-only MVP 비중이 베타의 함수이고 **임계 베타** 위 종목은 해에서 제외됨을 *해석적으로* 유도. SHAP cliff(β≈0.7–0.9)가 그 실증 대응. |
| GMV = 저베타 틸트 | **[FP2014]** Betting Against Beta; **[BBW2011]** 저변동성 이상현상 | 틸트가 방어주 프리미엄에 노출. |
| Σ-벤치 R²≈0.06(항등식 약함) | **[JM2003]** Wrong Constraints Help | long-only 제약이 `w∝Σ⁻¹1` 항등식을 깸. |
| OOS(LOYO) 평가 | **[DGU2009]** Optimal vs Naive | 추정오차가 최적화 이득을 상쇄 → OOS가 정직한 지표. |

**CST2011 대비 본 연구 기여:** CST는 단일팩터 모델 하 임계를 *해석적으로* 유도. 본 연구는 실현 LW-GMV 비중에서 **모델-프리(GBR+SHAP)로 동일 임계 구조를 실증 복원**하고, 분산항을 넘어선 非분산 특성(사이즈·유동성·섹터)의 추가 설명력을 정량화.

---

## 5. 결론

> **LW long-only GMV 포트폴리오는 무엇보다 저베타 포트폴리오이며, β≈0.7–0.9의 날카로운 임계 위 종목은 ~0 비중을 받는다 — Clarke·de Silva·Thorley(2011)의 해석적 결과와 일치. 사이즈·유동성은 이차적·비선형 틸트를 더하고 섹터는 거의 무관. 관계가 선형이 아니라 임계라서 GBR이 선형 OLS의 OOS R²를 2배(0.18→0.36)로 올리며, 해석 가능한 非Σ 특성이 기계적 분산항보다 훨씬 많이 설명한다.**

---

## 6. 재현
```bash
conda activate allo
python weight_explain_study.py                 # ew 프록시, 2005–2024 (본 리포트)
python weight_explain_study.py --proxy spy     # SPY 프록시 강건성 (fetch_spy.py 선행)
python weight_explain_study.py --years 2010 2024
```

## 7. 한계
- **기술적, 비인과** — `w`는 Σ의 결정함수; 특성은 비중과 *연관*.
- **생존편향** — 2024 S&P100 명단을 전 스냅샷 연도에 적용.
- **정적 2024 GICS 섹터.**
- **집중된 타깃** — long-only `w`는 우편향(대부분 ~0, 소수 지배); OLS는 기술적 해석 전용.
- **SHAP는 in-sample** — pooled GBR 적합의 학습 구조 해석이며, R²는 LOYO OOS 수치.
