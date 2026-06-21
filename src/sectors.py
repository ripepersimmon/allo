"""GICS sector mapping for S&P 100 constituents (as of 2024).

Hardcoded for reproducibility and auditability. No network dependency at runtime.
Mirrors the design of TICKERS in src/data_loader.py.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

# GICS sector assignment for every ticker in TICKERS
GICS_SECTORS: dict[str, str] = {
    # ── Information Technology ────────────────────────────────────────────────
    'AAPL': 'InfoTech',   'ACN':  'InfoTech',   'ADBE': 'InfoTech',
    'AMD':  'InfoTech',   'CRM':  'InfoTech',   'CSCO': 'InfoTech',
    'IBM':  'InfoTech',   'INTC': 'InfoTech',   'MSFT': 'InfoTech',
    'NVDA': 'InfoTech',   'ORCL': 'InfoTech',   'QCOM': 'InfoTech',
    'TXN':  'InfoTech',
    # ── Communication Services ────────────────────────────────────────────────
    'CHTR': 'CommSvcs',   'CMCSA':'CommSvcs',   'DIS':  'CommSvcs',
    'GOOGL':'CommSvcs',   'META': 'CommSvcs',   'T':    'CommSvcs',
    'TMUS': 'CommSvcs',   'VZ':   'CommSvcs',
    # ── Consumer Discretionary ────────────────────────────────────────────────
    'AMZN': 'ConsDis',    'BKNG': 'ConsDis',    'F':    'ConsDis',
    'GM':   'ConsDis',    'HD':   'ConsDis',     'LOW':  'ConsDis',
    'MCD':  'ConsDis',    'NKE':  'ConsDis',     'SBUX': 'ConsDis',
    'TGT':  'ConsDis',
    # ── Consumer Staples ──────────────────────────────────────────────────────
    'CL':   'ConsStap',   'COST': 'ConsStap',   'KHC':  'ConsStap',
    'KO':   'ConsStap',   'MDLZ': 'ConsStap',   'MO':   'ConsStap',
    'PEP':  'ConsStap',   'PG':   'ConsStap',   'PM':   'ConsStap',
    'WMT':  'ConsStap',
    # ── Energy ───────────────────────────────────────────────────────────────
    'COP':  'Energy',     'CVX':  'Energy',     'OXY':  'Energy',
    'SLB':  'Energy',     'XOM':  'Energy',
    # ── Financials ───────────────────────────────────────────────────────────
    'AIG':  'Financials', 'AXP':  'Financials', 'BAC':  'Financials',
    'BK':   'Financials', 'BLK':  'Financials', 'BRK-B':'Financials',
    'C':    'Financials', 'COF':  'Financials', 'GS':   'Financials',
    'JPM':  'Financials', 'MA':   'Financials', 'MET':  'Financials',
    'MS':   'Financials', 'PYPL': 'Financials', 'SCHW': 'Financials',
    'USB':  'Financials', 'V':    'Financials', 'WFC':  'Financials',
    # ── Health Care ──────────────────────────────────────────────────────────
    'ABBV': 'HealthCare', 'ABT':  'HealthCare', 'AMGN': 'HealthCare',
    'BIIB': 'HealthCare', 'BMY':  'HealthCare', 'CVS':  'HealthCare',
    'DHR':  'HealthCare', 'GILD': 'HealthCare', 'JNJ':  'HealthCare',
    'LLY':  'HealthCare', 'MDT':  'HealthCare', 'MRK':  'HealthCare',
    'PFE':  'HealthCare', 'TMO':  'HealthCare', 'UNH':  'HealthCare',
    # ── Industrials ──────────────────────────────────────────────────────────
    'BA':   'Industrials','CAT':  'Industrials','EMR':  'Industrials',
    'FDX':  'Industrials','GD':   'Industrials','GE':   'Industrials',
    'HON':  'Industrials','LMT':  'Industrials','MMM':  'Industrials',
    'RTX':  'Industrials','UNP':  'Industrials','UPS':  'Industrials',
    # ── Materials ────────────────────────────────────────────────────────────
    'DOW':  'Materials',  'LIN':  'Materials',
    # ── Real Estate ──────────────────────────────────────────────────────────
    'AMT':  'RealEstate', 'SPG':  'RealEstate',
    # ── Utilities ────────────────────────────────────────────────────────────
    'D':    'Utilities',  'DUK':  'Utilities',  'EXC':  'Utilities',
    'NEE':  'Utilities',  'SO':   'Utilities',
}

# Ordered list of sectors (used to fix the reference category for dummy encoding)
SECTOR_ORDER = [
    'InfoTech', 'CommSvcs', 'ConsDis', 'ConsStap', 'Energy',
    'Financials', 'HealthCare', 'Industrials', 'Materials', 'RealEstate',
    'Utilities',
]


def get_sector_dummies(
    tickers: list[str],
    drop_first: bool = True,
) -> pd.DataFrame:
    """One-hot sector dummies for the given tickers (reference = InfoTech if drop_first).

    Returns DataFrame indexed by ticker, columns = sector names minus the
    dropped reference. Tickers missing from GICS_SECTORS get all-zero rows
    (treated as 'Unknown') so the cross-section is never shrunk silently.

    Parameters
    ----------
    tickers     : list of ticker symbols to encode
    drop_first  : if True, drop the first sector (InfoTech) as the reference
                  category to avoid perfect multicollinearity
    """
    sectors = pd.Categorical(
        [GICS_SECTORS.get(tk, 'Unknown') for tk in tickers],
        categories=SECTOR_ORDER + (['Unknown'] if any(tk not in GICS_SECTORS for tk in tickers) else []),
        ordered=False,
    )
    dummies = pd.get_dummies(sectors, prefix='sec', dtype=float)
    dummies.index = tickers
    if drop_first:
        ref = f'sec_{SECTOR_ORDER[0]}'
        if ref in dummies.columns:
            dummies = dummies.drop(columns=ref)
    return dummies
