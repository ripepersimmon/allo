from pathlib import Path

import numpy as np
import pandas as pd

# S&P 100 constituents (OEX), as of 2024
TICKERS = [
    "AAPL", "ABBV", "ABT",  "ACN",  "ADBE", "AIG",  "AMD",  "AMGN", "AMT",  "AMZN",
    "AXP",  "BA",   "BAC",  "BIIB", "BK",   "BKNG", "BLK",  "BMY",  "BRK-B","C",
    "CAT",  "CHTR", "CL",   "CMCSA","COF",  "COP",  "COST", "CRM",  "CSCO", "CVS",
    "CVX",  "D",    "DHR",  "DIS",  "DOW",  "DUK",  "EMR",  "EXC",  "F",    "FDX",
    "GD",   "GE",   "GILD", "GM",   "GOOGL","GS",   "HD",   "HON",  "IBM",  "INTC",
    "JNJ",  "JPM",  "KHC",  "KO",   "LIN",  "LLY",  "LMT",  "LOW",  "MA",   "MCD",
    "MDLZ", "MDT",  "MET",  "META", "MMM",  "MO",   "MRK",  "MS",   "MSFT", "NEE",
    "NKE",  "NVDA", "ORCL", "OXY",  "PEP",  "PFE",  "PG",   "PM",   "PYPL", "QCOM",
    "RTX",  "SBUX", "SLB",  "SO",   "SPG",  "T",    "TGT",  "TMO",  "TMUS", "TXN",
    "UNH",  "UNP",  "UPS",  "USB",  "V",    "VZ",   "WFC",  "WMT",  "XOM",  "SCHW",
]


def load_prices_from_parquet(
    data_dir: str = "sp500",
    tickers:  list[str] = TICKERS,
    start:    str = "2000-01-01",
    end:      str = "2024-12-31",
) -> pd.DataFrame:
    """
    Load Close prices from per-ticker parquet files (e.g. sp500/AAPL.parquet).
    Tickers with no corresponding file are silently skipped.
    Returns a wide DataFrame (Date × ticker).
    """
    series = {}
    for ticker in tickers:
        path = Path(data_dir) / f"{ticker}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        close = df["Close"].copy()
        close.index = pd.to_datetime(close.index)
        series[ticker] = close

    prices = pd.DataFrame(series)
    prices.index.name = "Date"
    prices = prices.loc[start:end].ffill(limit=5).dropna(how="all")
    return prices


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna(how="all")
    return prices.pct_change().dropna(how="all")


def load_field_from_parquet(
    field: str = "Close",
    data_dir: str = "sp500",
    tickers: list[str] = TICKERS,
    start: str = "2000-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Wide (Date × ticker) DataFrame of a single OHLCV field."""
    series = {}
    for ticker in tickers:
        path = Path(data_dir) / f"{ticker}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if field not in df.columns:
            continue
        s = df[field].copy()
        s.index = pd.to_datetime(s.index)
        series[ticker] = s
    out = pd.DataFrame(series)
    out.index.name = "Date"
    out = out.loc[start:end].ffill(limit=5).dropna(how="all")
    return out


def load_dollar_volume(
    data_dir: str = "sp500",
    tickers: list[str] = TICKERS,
    start: str = "2000-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Wide (Date × ticker) daily dollar volume = Close_ffilled × Volume_raw.

    Close is forward-filled (price continuity); Volume is kept raw so that
    zero-volume / missing-volume days do not produce phantom liquidity.
    """
    series = {}
    for ticker in tickers:
        path = Path(data_dir) / f"{ticker}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "Close" not in df.columns or "Volume" not in df.columns:
            continue
        close_ff = df["Close"].ffill(limit=5)
        volume   = df["Volume"].replace(0, np.nan)
        dv = (close_ff * volume).copy()
        dv.index = pd.to_datetime(dv.index)
        series[ticker] = dv
    out = pd.DataFrame(series)
    out.index.name = "Date"
    out = out.loc[start:end].dropna(how="all")
    return out
