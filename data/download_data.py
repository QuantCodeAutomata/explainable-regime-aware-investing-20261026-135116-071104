"""
Data download for Explainable Regime-Aware Investing replication.

Downloads daily adjusted close prices from Yahoo Finance for the five-asset
universe and caches to data/prices.csv.

[paper p.4]: "Daily adjusted close prices from Yahoo Finance, 2005-2026"
"""
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TICKERS, DATA_START, DATA_END

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prices.csv")


def download_prices(tickers=None, start=DATA_START, end=DATA_END, save=True):
    """Download and cache adjusted close prices from Yahoo Finance."""
    import yfinance as yf
    if tickers is None:
        tickers = TICKERS

    yahoo_tickers = list(tickers.values())
    raw = yf.download(yahoo_tickers, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()

    inv_map = {v: k for k, v in tickers.items()}
    prices = prices.rename(columns=inv_map)
    asset_names = [k for k in tickers if k in prices.columns]
    prices = prices[asset_names]
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().ffill(limit=3).dropna()

    if save:
        prices.to_csv(DATA_FILE)
        print(f"Saved {len(prices)} rows to {DATA_FILE}")
    return prices


def load_prices(force_download=False):
    """Load from cache or download fresh."""
    if not force_download and os.path.exists(DATA_FILE):
        prices = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
        print(f"Loaded {len(prices)} rows from {DATA_FILE}")
        return prices
    return download_prices()


if __name__ == "__main__":
    p = download_prices()
    print(p.tail(3))
