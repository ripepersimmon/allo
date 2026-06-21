"""One-time SPY data fetch. Run from repo root with venv active:
    python fetch_spy.py
Saves sp500/SPY.parquet in the same OHLCV format as all other per-ticker files.
"""
import sys
sys.path.insert(0, '.')

from src.market import fetch_spy

if __name__ == '__main__':
    fetch_spy(start='2000-01-01', end='2024-12-31', out_path='sp500/SPY.parquet')
