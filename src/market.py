"""Market-proxy utilities shared across all OLS scripts."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def fetch_spy(
    start: str = '2000-01-01',
    end: str = '2024-12-31',
    out_path: str = 'sp500/SPY.parquet',
) -> None:
    """Download SPY OHLCV via yfinance and save to parquet.

    Writes columns [Open, High, Low, Close, Volume] indexed by Date,
    matching the format of all other per-ticker files in sp500/.
    Run once from repo root: python fetch_spy.py
    """
    import yfinance as yf
    print(f'Downloading SPY {start} → {end}...')
    raw = yf.download('SPY', start=start, end=end, progress=False,
                      auto_adjust=True)
    # yfinance ≥1.0 may return MultiIndex columns for single-ticker download
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f'Saved → {out}  ({len(df)} rows)')


def load_spy_returns(
    start: str = '2000-01-01',
    end: str = '2024-12-31',
    method: str = 'log',
    spy_path: str = 'sp500/SPY.parquet',
) -> pd.Series:
    """Load SPY parquet and return a return Series (log or simple).

    Raises FileNotFoundError if sp500/SPY.parquet does not exist.
    Run fetch_spy.py first.
    """
    path = Path(spy_path)
    if not path.exists():
        raise FileNotFoundError(
            f'{spy_path} not found — run python fetch_spy.py first.'
        )
    df = pd.read_parquet(path)
    close = df['Close'].copy()
    close.index = pd.to_datetime(close.index)
    close = close.loc[start:end]
    if method == 'log':
        ret = np.log(close / close.shift(1)).dropna()
    else:
        ret = close.pct_change().dropna()
    ret.name = 'SPY'
    return ret


def get_market_proxy(
    returns_win: pd.DataFrame,
    proxy: str = 'ew',
    spy_returns: pd.Series | None = None,
) -> pd.Series:
    """Return a market-proxy return Series aligned to returns_win.index.

    proxy='ew'  → equal-weighted mean of returns_win (project default).
    proxy='spy' → SPY log returns reindexed to returns_win.index.
                  Rows with no SPY data become NaN; callers should handle
                  this the same way they handle any missing-data day.
                  spy_returns must be supplied when proxy='spy'.
    """
    if proxy == 'ew':
        return returns_win.mean(axis=1)
    if proxy == 'spy':
        if spy_returns is None:
            raise ValueError("spy_returns must be provided when proxy='spy'")
        return spy_returns.reindex(returns_win.index)
    raise ValueError(f"Unknown proxy '{proxy}'. Choose 'ew' or 'spy'.")
