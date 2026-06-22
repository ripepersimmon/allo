"""Market-proxy returns for beta estimation."""
from pathlib import Path

import numpy as np
import pandas as pd


def load_spy_returns(start="2000-01-01", end="2024-12-31",
                     method="log", spy_path="data/SPY.parquet") -> pd.Series:
    """SPY return series (robustness proxy). Run fetch_data.py first."""
    path = Path(spy_path)
    if not path.exists():
        raise FileNotFoundError(f"{spy_path} not found — run python fetch_data.py first.")
    close = pd.read_parquet(path)["Close"]
    close.index = pd.to_datetime(close.index)
    close = close.loc[start:end]
    ret = np.log(close / close.shift(1)) if method == "log" else close.pct_change()
    ret = ret.dropna()
    ret.name = "SPY"
    return ret


def get_market_proxy(returns_win: pd.DataFrame, proxy="ew",
                     spy_returns: pd.Series | None = None) -> pd.Series:
    """Market return aligned to returns_win.index.

    proxy='ew'  → equal-weighted mean of the window (default baseline).
    proxy='spy' → SPY log returns reindexed to the window.
    """
    if proxy == "ew":
        return returns_win.mean(axis=1)
    if proxy == "spy":
        if spy_returns is None:
            raise ValueError("spy_returns required when proxy='spy'")
        return spy_returns.reindex(returns_win.index)
    raise ValueError(f"unknown proxy '{proxy}'")
