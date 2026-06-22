# GMVP 비중 구조 분해 — 재현 코드

리포트 *「전역최소분산 포트폴리오의 비중 구조 분해」* 의 결과를 재현하는 코드.
S&P100 종목의 일별 수익률로 Long-only GMV 포트폴리오를 구성하고, 그 비중이
어떤 비(非)공분산 특성으로 설명되는지(베타·사이즈·유동성·모멘텀·섹터),
그리고 위기 국면에서 그 구조가 어떻게 이동하는지를 분석한다.

## 구성

```
fetch_data.py        Yahoo Finance에서 S&P100 + SPY + VIX 수집 → data/
weight_explain.py    비중 횡단면 설명: OLS / XGBoost / SHAP (Table 3·4, Figure 1)
crisis_case.py       VIX 기반 위기 케이스 스터디: pre→peak 비중 구조 (Table 2·5)
src/
  data_loader.py     parquet 로딩, 로그수익률, 달러 거래량
  estimators.py      표본 / Ledoit-Wolf 공분산
  portfolio.py       Long-only GMV, Effective-N
  market.py          시장 프록시 (동일가중 / SPY)
  sectors.py         GICS 섹터 매핑
  crises.py          VIX 히스테리시스 위기 탐지
```

## 실행

```bash
pip install -r requirements.txt
python fetch_data.py        # 데이터 수집 (1회)
python weight_explain.py    # 비중 설명 분석
python crisis_case.py       # 위기 케이스 스터디
python -m src.crises        # 위기 구간 표 (Table 2)
```

기본 시장 프록시는 동일가중 지수이며, `--proxy spy` 로 SPY 강건성 검증을 수행한다.
결과 표는 `tables/`, 그림은 `figures/` 에 저장된다.
데이터(`data/`)와 산출물은 재생성 가능하므로 저장소에 포함하지 않는다.
