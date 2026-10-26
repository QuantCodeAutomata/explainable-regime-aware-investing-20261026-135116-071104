"""
Feature construction for regime-aware investing.

Paper: "Explainable Regime-Aware Investing", Section 4 (p.4)
x_t = [r_t; sigma_t; m_t] in R^{3N}
  r_t    : daily log returns (N-dim)
  sigma_t: 60-day rolling per-asset volatility (N-dim)
  m_t    : 20-day rolling per-asset mean return (N-dim)
All features strictly causal: computed from data up to t-1.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from adjusted close prices.

    [paper p.4]: r_t = log(P_t) - log(P_{t-1})

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, shape (T, N). Index is DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Log returns, shape (T-1, N). First row is NaN-free.
    """
    returns = np.log(prices).diff().dropna()
    return returns


def compute_features(
    returns: pd.DataFrame,
    vol_window: int = 60,
    mom_window: int = 20,
) -> pd.DataFrame:
    """
    Build strictly causal feature matrix x_t = [r_t; sigma_t; m_t] in R^{3N}.

    Strict causality: all features at time t use only data available up to t-1.
    Rolling vol and momentum are shifted by 1 (so sigma_t uses returns up to t-1).

    [paper p.4]: sigma_t is 60-day rolling volatility per asset (using data up to t-1)
    [paper p.4]: m_t is 20-day rolling mean return per asset (using data up to t-1)

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns, shape (T, N).
    vol_window : int
        Rolling window for volatility (default 60 days).
    mom_window : int
        Rolling window for momentum (default 20 days).

    Returns
    -------
    pd.DataFrame
        Feature matrix of shape (T, 3N). Rows with NaN are dropped.
        Column order: [r_1,...,r_N, sigma_1,...,sigma_N, m_1,...,m_N]
    """
    n_assets = returns.shape[1]
    assets = returns.columns.tolist()

    # Paper p.4: "all features at time t must be computed strictly causally
    # using only information available up to t-1."
    # r_t in x_t is the MOST RECENTLY OBSERVED return before trading on day t,
    # i.e. r_{t-1}.  We implement this as returns.shift(1).
    r_lagged = returns.shift(1)                                          # r_{t-1}
    sigma    = returns.rolling(vol_window, min_periods=vol_window).std().shift(1)
    mom      = returns.rolling(mom_window, min_periods=mom_window).mean().shift(1)

    feat_cols = (
        [f"r_{a}"     for a in assets]
        + [f"sigma_{a}" for a in assets]
        + [f"m_{a}"     for a in assets]
    )

    feat = pd.concat([r_lagged, sigma, mom], axis=1)
    feat.columns = feat_cols
    feat = feat.dropna()
    return feat


def get_return_slice(returns: pd.DataFrame, feature_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Extract the N-dimensional return slice aligned with the feature index.

    Parameters
    ----------
    returns : pd.DataFrame
        Full log returns DataFrame, shape (T, N).
    feature_index : pd.DatetimeIndex
        Dates for which features are available.

    Returns
    -------
    pd.DataFrame
        Returns aligned to feature_index.
    """
    return returns.loc[feature_index]


def split_features(
    features: pd.DataFrame,
    returns: pd.DataFrame,
    t0: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split features and returns into training and test sets at date t0.

    Parameters
    ----------
    features : pd.DataFrame
        Full feature matrix.
    returns : pd.DataFrame
        Full returns DataFrame.
    t0 : str
        Split date (ISO format). Training uses dates <= t0; test uses > t0.

    Returns
    -------
    Tuple of (features_train, features_test, returns_train, returns_test)
    """
    feat_train = features[features.index <= t0]
    feat_test  = features[features.index >  t0]
    ret_train  = returns.loc[features.index[features.index <= t0]]
    ret_test   = returns.loc[features.index[features.index >  t0]]
    return feat_train, feat_test, ret_train, ret_test
