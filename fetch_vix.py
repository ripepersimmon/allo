"""One-time VIX data fetch. Run from repo root with the `allo` env active:

    conda activate allo
    python fetch_vix.py

Downloads ^VIX daily OHLC via yfinance and saves Close to sp500/VIX.parquet
(matching the SPY convention in fetch_spy.py). VIX is the crisis-detection input
for src/crises.py and crisis_case_study.py.
"""
from pathlib import Path
import pandas as pd


def fetch_vix(start: str = '2004-01-01', end: str = '2024-12-31',
              out_path: str = 'sp500/VIX.parquet') -> None:
    import yfinance as yf
    print(f'Downloading ^VIX {start} → {end}...')
    raw = yf.download('^VIX', start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[['Close']].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f'Saved → {out}  ({len(df)} rows, {df.index.min().date()}–{df.index.max().date()})')


if __name__ == '__main__':
    fetch_vix()
