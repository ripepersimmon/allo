"""VIX-based crisis definition.

Episodes are detected from the VIX (exogenous to the 100-stock covariance) with a
two-threshold Schmitt trigger, then labelled by the dominant event of the peak year:
  - enter when VIX close > 30, exit when VIX close < 20 (hysteresis)
  - drop episodes shorter than MIN_DAYS trading days
  - merge episodes less than MERGE_GAP trading days apart into one regime

Run fetch_data.py first to create data/VIX.parquet.
"""
from pathlib import Path
import pandas as pd

ENTER, EXIT = 30.0, 20.0
MIN_DAYS, MERGE_GAP = 10, 42

_PEAK_YEAR_NAMES = {
    2010: 'Flash Crash / EU',
    2011: 'EU 부채 / 미국 강등',
    2015: '중국 위안화 절하',
    2018: '2018-Q4 Selloff',
    2020: 'COVID-19',
    2022: '금리 인상 / 우크라이나',
}


def _crisis_label(peak_date) -> str:
    # 2008 has two episodes; split by peak month (Bear ~Mar, Lehman ~Nov).
    if peak_date.year == 2008:
        return 'GFC (Bear)' if peak_date.month < 6 else 'GFC (Lehman)'
    return _PEAK_YEAR_NAMES.get(peak_date.year, f'VIX-spike {peak_date.year}')


def load_vix(start="2005-01-01", end="2024-12-31", path="data/VIX.parquet") -> pd.Series:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found — run python fetch_data.py first.")
    s = pd.read_parquet(p)["Close"].dropna()
    s.index = pd.to_datetime(s.index)
    s.name = "VIX"
    return s.loc[start:end]


def detect_vix_crises(vix: pd.Series | None = None, enter=ENTER, exit=EXIT,
                      min_days=MIN_DAYS, merge_gap=MERGE_GAP) -> pd.DataFrame:
    """Detect crisis episodes. Columns: start, end, days, peak_date, peak_vix, label."""
    if vix is None:
        vix = load_vix()
    vix = vix.dropna()

    # 1. hysteresis trigger
    raw, in_crisis, start = [], False, None
    for d, v in vix.items():
        if not in_crisis and v > enter:
            in_crisis, start = True, d
        elif in_crisis and v < exit:
            raw.append([start, d]); in_crisis = False
    if in_crisis:
        raw.append([start, vix.index[-1]])

    # 2. merge nearby episodes
    pos = {d: i for i, d in enumerate(vix.index)}
    merged = []
    for s, e in raw:
        if merged and merge_gap > 0 and pos[s] - pos[merged[-1][1]] < merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # 3. min-duration filter, peak, label
    rows = []
    for s, e in merged:
        seg = vix.loc[s:e]
        if len(seg) < min_days:
            continue
        peak_d = seg.idxmax()
        rows.append({
            'start': s, 'end': e, 'days': int(len(seg)),
            'peak_date': peak_d, 'peak_vix': round(float(seg.max()), 1),
            'label': _crisis_label(peak_d),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = detect_vix_crises()
    print(df.assign(start=df['start'].dt.date, end=df['end'].dt.date,
                    peak_date=df['peak_date'].dt.date).to_string(index=False))
    out = Path('tables/vix_crisis_periods.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'\nSaved → {out}  ({len(df)} episodes)')
