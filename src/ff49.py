"""Loader for Fama-French 49 Industry Portfolio daily returns.

Data file: 49_Industry_Portfolios_Daily.csv (value-weighted section only).
Returns are in percentage in the raw file; this loader converts to decimal.
Missing codes -99.99 / -999 are replaced with NaN.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

_FF49_PATH   = '49_Industry_Portfolios_Daily.csv'
_VW_SKIPROWS = 9      # lines 1-9 are file header + section label
_VW_NROWS    = 26214  # VW data rows before the EW section begins

_FF_FACTOR_SKIPROWS = 4  # 3 text lines + 1 blank line → column header is line 5


def load_ff49_returns(start: str, end: str,
                      path: str = _FF49_PATH) -> pd.DataFrame:
    """Load Fama-French 49 Industry VW daily returns as decimal returns.

    Returns a DataFrame indexed by date, columns = 49 industry short names
    (Agric, Food, …, Other).  NaN where the original file shows -99.99 / -999.

    Parameters
    ----------
    start, end : ISO date strings, e.g. '2000-01-01'
    path       : path to the raw CSV (relative to repo root)
    """
    raw = pd.read_csv(
        path,
        skiprows=_VW_SKIPROWS,
        header=0,
        index_col=0,
        nrows=_VW_NROWS,
        encoding='latin-1',
        engine='python',
    )
    raw = raw.apply(pd.to_numeric, errors='coerce')
    raw = raw.mask((raw == -99.99) | (raw == -999.0)) / 100
    raw.columns = raw.columns.str.strip()

    dates = pd.to_datetime(
        raw.index.astype(str).str.strip(),
        format='%Y%m%d',
        errors='coerce',
    )
    valid = ~pd.isnull(dates)
    raw   = raw.loc[valid]
    raw.index = dates[valid]

    return raw.loc[start:end]


def load_ff_factors(path: str, start: str, end: str) -> pd.DataFrame:
    """Load Fama-French factor daily returns (3-factor or 5-factor file).

    Drops the RF (risk-free) column; returns only the factor columns.
    Returns decimal returns (divides raw % values by 100).

    Works for both:
      F-F_Research_Data_Factors_daily.csv      → Mkt-RF, SMB, HML
      F-F_Research_Data_5_Factors_2x3_daily.csv → Mkt-RF, SMB, HML, RMW, CMA
    """
    raw = pd.read_csv(
        path,
        skiprows=_FF_FACTOR_SKIPROWS,
        header=0,
        index_col=0,
        engine='python',
        encoding='latin-1',
    )
    raw = raw.apply(pd.to_numeric, errors='coerce') / 100
    raw.columns = raw.columns.str.strip()
    if 'RF' in raw.columns:
        raw = raw.drop(columns='RF')

    dates = pd.to_datetime(
        raw.index.astype(str).str.strip(),
        format='%Y%m%d',
        errors='coerce',
    )
    valid = ~pd.isnull(dates)
    raw   = raw.loc[valid]
    raw.index = dates[valid]

    return raw.loc[start:end]
