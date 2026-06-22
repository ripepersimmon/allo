"""Price / volume loading for the S&P 100 universe."""
from pathlib import Path

import numpy as np
import pandas as pd

# S&P 100 (OEX) constituents as of 2024.
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

DATA_DIR = "data"


def load_prices_from_parquet(data_dir=DATA_DIR, tickers=TICKERS,
                             start="2000-01-01", end="2024-12-31") -> pd.DataFrame:
    """Wide (Date × ticker) Close-price frame. Missing tickers are skipped."""
    series = {}
    for tk in tickers:
        path = Path(data_dir) / f"{tk}.parquet"
        if not path.exists():
            continue
        close = pd.read_parquet(path)["Close"]
        close.index = pd.to_datetime(close.index)
        series[tk] = close
    prices = pd.DataFrame(series)
    prices.index.name = "Date"
    return prices.loc[start:end].ffill(limit=5).dropna(how="all")


def compute_returns(prices: pd.DataFrame, method="log") -> pd.DataFrame:
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna(how="all")
    return prices.pct_change().dropna(how="all")


def load_dollar_volume(data_dir=DATA_DIR, tickers=TICKERS,
                       start="2000-01-01", end="2024-12-31") -> pd.DataFrame:
    """Wide (Date × ticker) daily dollar volume = Close(ffilled) × Volume(raw).

    Volume is kept raw so missing/zero-volume days don't fabricate liquidity.
    """
    series = {}
    for tk in tickers:
        path = Path(data_dir) / f"{tk}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "Close" not in df or "Volume" not in df:
            continue
        dv = df["Close"].ffill(limit=5) * df["Volume"].replace(0, np.nan)
        dv.index = pd.to_datetime(dv.index)
        series[tk] = dv
    out = pd.DataFrame(series)
    out.index.name = "Date"
    return out.loc[start:end].dropna(how="all")
