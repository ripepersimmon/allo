"""Download every market series used by the study from Yahoo Finance.

Writes one parquet per series under data/:
  data/<TICKER>.parquet   OHLCV (adjusted) for each S&P 100 constituent
  data/SPY.parquet        OHLCV (adjusted) — robustness market proxy
  data/VIX.parquet        Close — crisis-detection input

Run once before the analysis scripts:
    python fetch_data.py
"""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, '.')
from src.data_loader import TICKERS

DATA = Path('data')
START, END = '2000-01-01', '2024-12-31'
OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']


def download(symbol: str, auto_adjust=True) -> pd.DataFrame:
    raw = yf.download(symbol, start=START, end=END, progress=False, auto_adjust=auto_adjust)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = 'Date'
    return raw


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    saved = 0
    for tk in TICKERS:
        try:
            download(tk)[OHLCV].to_parquet(DATA / f'{tk}.parquet')
            saved += 1
        except Exception as e:
            print(f'  skip {tk}: {e}')
    print(f'{saved}/{len(TICKERS)} constituents saved')

    download('SPY')[OHLCV].to_parquet(DATA / 'SPY.parquet')
    download('^VIX', auto_adjust=False)[['Close']].to_parquet(DATA / 'VIX.parquet')
    print('SPY, VIX saved → data/')


if __name__ == '__main__':
    main()
