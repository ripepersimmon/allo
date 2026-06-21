"""VIX-based crisis definition for the crisis case study.

Crisis episodes are detected objectively from the VIX (CBOE volatility index,
exogenous to the 100-stock realized covariance) using a two-threshold Schmitt
trigger, then labelled with the dominant narrative event.

Definition (locked 2026-06-21):
  - ENTER episode when VIX close > 30  (elevated stress)
  - EXIT  episode when VIX close < 20  (hysteresis — avoids on/off flicker)
  - drop episodes shorter than MIN_DAYS trading days (denoise)
  - merge consecutive episodes < MERGE_GAP trading days apart (one named regime)

Run `python fetch_vix.py` once to create sp500/VIX.parquet first.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

# default detection parameters
ENTER, EXIT = 30.0, 20.0
MIN_DAYS, MERGE_GAP = 10, 42

# narrative labels keyed by the calendar year of an episode's VIX peak
_PEAK_YEAR_NAMES = {
    2008: 'GFC',
    2010: 'Flash Crash / EU',
    2011: 'EU debt / US downgrade',
    2015: 'China devaluation',
    2018: 'Volmageddon',
    2020: 'COVID-19',
    2022: 'Rate hikes / Ukraine',
}


def load_vix(start: str = '2005-01-01', end: str = '2024-12-31',
             path: str = 'sp500/VIX.parquet') -> pd.Series:
    """Daily VIX close as a Series. Raises if the parquet is missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'{path} not found — run `python fetch_vix.py` first.')
    s = pd.read_parquet(p)['Close'].dropna()
    s.index = pd.to_datetime(s.index)
    s.name = 'VIX'
    return s.loc[start:end]


def detect_vix_crises(
    vix: pd.Series | None = None,
    enter: float = ENTER,
    exit: float = EXIT,
    min_days: int = MIN_DAYS,
    merge_gap: int = MERGE_GAP,
) -> pd.DataFrame:
    """Detect crisis episodes from VIX via hysteresis + merge.

    Returns a DataFrame with columns:
      start, end (Timestamps), days, peak_date, peak_vix, label.

    Set merge_gap=0 to disable merging (raw hysteresis episodes).
    """
    if vix is None:
        vix = load_vix()
    vix = vix.dropna()

    # 1. Schmitt-trigger hysteresis
    raw, in_crisis, start = [], False, None
    for d, v in vix.items():
        if not in_crisis and v > enter:
            in_crisis, start = True, d
        elif in_crisis and v < exit:
            raw.append([start, d]); in_crisis = False
    if in_crisis:
        raw.append([start, vix.index[-1]])

    # 2. merge episodes closer than merge_gap trading days
    pos = {d: i for i, d in enumerate(vix.index)}
    merged = []
    for s, e in raw:
        if merged and merge_gap > 0 and pos[s] - pos[merged[-1][1]] < merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # 3. min-duration filter + peak + label
    rows = []
    for s, e in merged:
        seg = vix.loc[s:e]
        if len(seg) < min_days:
            continue
        peak_d = seg.idxmax()
        rows.append({
            'start': s, 'end': e, 'days': int(len(seg)),
            'peak_date': peak_d, 'peak_vix': round(float(seg.max()), 1),
            'label': _PEAK_YEAR_NAMES.get(peak_d.year, f'VIX-spike {peak_d.year}'),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = detect_vix_crises()
    pd.set_option('display.width', 120)
    print(f'VIX crisis episodes (enter>{ENTER}, exit<{EXIT}, min {MIN_DAYS}d, '
          f'merge<{MERGE_GAP}td):\n')
    print(df.assign(start=df['start'].dt.date, end=df['end'].dt.date,
                    peak_date=df['peak_date'].dt.date).to_string(index=False))
    out = Path('reports/vix_crisis_periods.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'\nSaved → {out}  ({len(df)} episodes, {df["days"].sum()} crisis days)')
