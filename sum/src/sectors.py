"""GICS sector mapping for the S&P 100 universe (as of 2024).

Hardcoded for reproducibility — no network dependency at runtime.
"""
import pandas as pd

GICS_SECTORS = {
    # Information Technology
    'AAPL': 'InfoTech',   'ACN':  'InfoTech',   'ADBE': 'InfoTech',
    'AMD':  'InfoTech',   'CRM':  'InfoTech',   'CSCO': 'InfoTech',
    'IBM':  'InfoTech',   'INTC': 'InfoTech',   'MSFT': 'InfoTech',
    'NVDA': 'InfoTech',   'ORCL': 'InfoTech',   'QCOM': 'InfoTech',
    'TXN':  'InfoTech',
    # Communication Services
    'CHTR': 'CommSvcs',   'CMCSA':'CommSvcs',   'DIS':  'CommSvcs',
    'GOOGL':'CommSvcs',   'META': 'CommSvcs',   'T':    'CommSvcs',
    'TMUS': 'CommSvcs',   'VZ':   'CommSvcs',
    # Consumer Discretionary
    'AMZN': 'ConsDis',    'BKNG': 'ConsDis',    'F':    'ConsDis',
    'GM':   'ConsDis',    'HD':   'ConsDis',    'LOW':  'ConsDis',
    'MCD':  'ConsDis',    'NKE':  'ConsDis',    'SBUX': 'ConsDis',
    'TGT':  'ConsDis',
    # Consumer Staples
    'CL':   'ConsStap',   'COST': 'ConsStap',   'KHC':  'ConsStap',
    'KO':   'ConsStap',   'MDLZ': 'ConsStap',   'MO':   'ConsStap',
    'PEP':  'ConsStap',   'PG':   'ConsStap',   'PM':   'ConsStap',
    'WMT':  'ConsStap',
    # Energy
    'COP':  'Energy',     'CVX':  'Energy',     'OXY':  'Energy',
    'SLB':  'Energy',     'XOM':  'Energy',
    # Financials
    'AIG':  'Financials', 'AXP':  'Financials', 'BAC':  'Financials',
    'BK':   'Financials', 'BLK':  'Financials', 'BRK-B':'Financials',
    'C':    'Financials', 'COF':  'Financials', 'GS':   'Financials',
    'JPM':  'Financials', 'MA':   'Financials', 'MET':  'Financials',
    'MS':   'Financials', 'PYPL': 'Financials', 'SCHW': 'Financials',
    'USB':  'Financials', 'V':    'Financials', 'WFC':  'Financials',
    # Health Care
    'ABBV': 'HealthCare', 'ABT':  'HealthCare', 'AMGN': 'HealthCare',
    'BIIB': 'HealthCare', 'BMY':  'HealthCare', 'CVS':  'HealthCare',
    'DHR':  'HealthCare', 'GILD': 'HealthCare', 'JNJ':  'HealthCare',
    'LLY':  'HealthCare', 'MDT':  'HealthCare', 'MRK':  'HealthCare',
    'PFE':  'HealthCare', 'TMO':  'HealthCare', 'UNH':  'HealthCare',
    # Industrials
    'BA':   'Industrials','CAT':  'Industrials','EMR':  'Industrials',
    'FDX':  'Industrials','GD':   'Industrials','GE':   'Industrials',
    'HON':  'Industrials','LMT':  'Industrials','MMM':  'Industrials',
    'RTX':  'Industrials','UNP':  'Industrials','UPS':  'Industrials',
    # Materials / Real Estate / Utilities
    'DOW':  'Materials',  'LIN':  'Materials',
    'AMT':  'RealEstate', 'SPG':  'RealEstate',
    'D':    'Utilities',  'DUK':  'Utilities',  'EXC':  'Utilities',
    'NEE':  'Utilities',  'SO':   'Utilities',
}

SECTOR_ORDER = [
    'InfoTech', 'CommSvcs', 'ConsDis', 'ConsStap', 'Energy', 'Financials',
    'HealthCare', 'Industrials', 'Materials', 'RealEstate', 'Utilities',
]


def get_sector_dummies(tickers, drop_first=True) -> pd.DataFrame:
    """One-hot GICS dummies indexed by ticker (reference = InfoTech)."""
    has_unknown = any(tk not in GICS_SECTORS for tk in tickers)
    sectors = pd.Categorical(
        [GICS_SECTORS.get(tk, 'Unknown') for tk in tickers],
        categories=SECTOR_ORDER + (['Unknown'] if has_unknown else []),
    )
    dummies = pd.get_dummies(sectors, prefix='sec', dtype=float)
    dummies.index = tickers
    if drop_first and f'sec_{SECTOR_ORDER[0]}' in dummies:
        dummies = dummies.drop(columns=f'sec_{SECTOR_ORDER[0]}')
    return dummies
